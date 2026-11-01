"""Self-improvement loop (train only), runnable with one command:

    python -m contract_intelligence.improvement.loop --iterations 6

Each iteration:
1. diagnose the current configuration's TRAIN run (improvement.diagnose);
2. propose one change for the dominant failure mode (improvement.propose);
3. run the candidate on TRAIN, calibrating its abstention threshold on TRAIN;
4. compare with the current configuration and accept or reject (criteria below);
5. log the iteration (accepted or rejected) and save the best configuration.

Objective (declared before the first iteration):
- primary: character-level Recall@5 on answerable train queries (ranking before
  abstention). A reviewer reads about five passages; recall says whether the
  answer text is among them.
- guardrails: Precision@5 may drop by at most 0.005; balanced abstention accuracy
  may drop by at most 0.02; false-abstention rate must stay <= 0.20.
Acceptance:
  (A) Recall@5 improves by >= 0.01 (about 1.4 of 140 queries) with paired-bootstrap
      P(improvement > 0) >= 0.90, and all guardrails hold; or
  (B) Recall@5 is non-inferior (drop <= 0.005), balanced abstention accuracy improves
      by >= 0.02, and the other guardrails hold. Abstention is the system's second
      required capability, so a clear abstention gain at equal ranking quality counts.
The loop never reads the test split; gold spans are used only to score rankings.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from contract_intelligence.config import PROJECT_ROOT, load_config
from contract_intelligence.evaluation.runner import apply_threshold, score_queries, summarize
from contract_intelligence.improvement.diagnose import PRIMARY_K, diagnose
from contract_intelligence.improvement.propose import RULES, propose
from contract_intelligence.retrieval.calibrate import calibrate_from_scores
from contract_intelligence.retrieval.retriever import Retriever

MIN_GAIN = 0.01
MIN_BOOTSTRAP_PROB = 0.90
MAX_P5_DROP = 0.005
MAX_BALACC_DROP = 0.02
MAX_FALSE_ABSTENTION = 0.20
NON_INFERIOR_DROP = 0.005
MIN_BALACC_GAIN = 0.02
BOOTSTRAP_SAMPLES = 2000

LOOP_DIR = PROJECT_ROOT / "experiments/loop"
LOG_PATH = PROJECT_ROOT / "experiments/improvement_log.jsonl"
BEST_PATH = PROJECT_ROOT / "configs/final.yaml"


def apply_changes(raw: dict, changes: dict) -> dict:
    out = copy.deepcopy(raw)
    for dotted, value in changes.items():
        section, key = dotted.split(".")
        out.setdefault(section, {})[key] = value
    return out


def run_candidate(raw: dict, name: str) -> tuple[dict, list[dict], Path]:
    """Evaluate a config on TRAIN with a TRAIN-calibrated threshold; save config and run."""
    raw = {**copy.deepcopy(raw), "name": name}
    config_path = LOOP_DIR / "configs" / f"{name}.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    config = load_config(config_path)

    retriever = Retriever.from_config(config)
    started = time.perf_counter()
    scored = score_queries(config, retriever, "train")
    seconds = time.perf_counter() - started
    del retriever
    gc.collect()

    scores = [q["top_score"] for q in scored if q["top_score"] is not None]
    floor = (min(scores) if scores else 0.0) - 1.0  # empty results always abstain
    pos = np.array([q["top_score"] if q["top_score"] is not None else floor for q in scored if q["answerable"]])
    neg = np.array([q["top_score"] if q["top_score"] is not None else floor for q in scored if not q["answerable"]])
    threshold = calibrate_from_scores(pos, neg)["threshold"]

    per_query = apply_threshold(scored, threshold)
    summary = summarize(config, "train", per_query, threshold)
    summary["seconds"] = round(seconds, 1)

    raw["retrieval"] = {**raw.get("retrieval", {}), "abstain_threshold": threshold}
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    run_dir = LOOP_DIR / "runs" / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (run_dir / "per_query.jsonl").open("w", encoding="utf-8") as handle:
        for row in per_query:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return summary, per_query, config_path


def metrics(summary: dict) -> dict:
    r, a = summary["retrieval"], summary["abstention"]
    out = {f"R@{k}": r["recall"][str(k)] if str(k) in r["recall"] else r["recall"][k] for k in (1, 3, 5, 10)}
    out |= {f"P@{k}": r["precision"][str(k)] if str(k) in r["precision"] else r["precision"][k] for k in (1, 3, 5, 10)}
    out |= {
        "MRR": r["mrr"],
        "balanced_abstention_acc": a["balanced_accuracy"],
        "false_abstention": a["answerable_wrongly_abstained"],
        "correct_abstention": a["unanswerable_correctly_abstained"],
        "threshold": summary["abstain_threshold"],
    }
    return out


def bootstrap_prob(current: list[dict], candidate: list[dict], seed: int = 0) -> tuple[float, float]:
    """Mean Recall@k gain and P(gain > 0) under a paired bootstrap over train queries."""
    base = {q["query"]: q["recall"][PRIMARY_K] for q in current if q["answerable"]}
    cand = {q["query"]: q["recall"][PRIMARY_K] for q in candidate if q["answerable"]}
    diff = np.array([cand[q] - base[q] for q in base])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(diff), size=(BOOTSTRAP_SAMPLES, len(diff)))
    return float(diff.mean()), float((diff[samples].mean(axis=1) > 0).mean())


def decide(current: dict, candidate: dict, gain: float, prob: float) -> tuple[bool, str]:
    p5_drop = current["P@5"] - candidate["P@5"]
    bal_change = candidate["balanced_abstention_acc"] - current["balanced_abstention_acc"]
    guards = []
    if p5_drop > MAX_P5_DROP:
        guards.append(f"P@5 fell by {p5_drop:.3f} > {MAX_P5_DROP}")
    if candidate["false_abstention"] > MAX_FALSE_ABSTENTION:
        guards.append(f"false abstention {candidate['false_abstention']:.1%} > {MAX_FALSE_ABSTENTION:.0%}")
    if guards:
        return False, "guardrail: " + "; ".join(guards)
    if gain >= MIN_GAIN and prob >= MIN_BOOTSTRAP_PROB and -bal_change <= MAX_BALACC_DROP:
        return True, f"(A) R@5 +{gain:.3f}, P(gain>0)={prob:.2f}, guardrails hold"
    if -bal_change > MAX_BALACC_DROP:
        return False, f"guardrail: balanced abstention accuracy fell by {-bal_change:.3f}"
    if gain >= -NON_INFERIOR_DROP and bal_change >= MIN_BALACC_GAIN:
        return True, f"(B) R@5 non-inferior ({gain:+.3f}), balanced abstention +{bal_change:.3f}"
    return False, f"no sufficient gain: R@5 {gain:+.3f} (P={prob:.2f}), balanced abstention {bal_change:+.3f}"


def read_log() -> list[dict]:
    if not LOG_PATH.exists():
        return []
    return [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_log(entry: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def write_markdown(entries: list[dict]) -> None:
    lines = [
        "# Improvement loop log (train only)", "",
        "Generated by `python -m contract_intelligence.improvement.loop` from "
        "`experiments/improvement_log.jsonl`. Every iteration is listed, accepted or rejected.",
        "",
        f"Primary metric: Recall@{PRIMARY_K}. Guardrails: P@5 drop <= {MAX_P5_DROP}, balanced abstention "
        f"accuracy drop <= {MAX_BALACC_DROP}, false abstention <= {MAX_FALSE_ABSTENTION:.0%}. Accept on "
        f"(A) R@5 gain >= {MIN_GAIN} with bootstrap P >= {MIN_BOOTSTRAP_PROB}, or (B) R@5 drop <= "
        f"{NON_INFERIOR_DROP} with balanced abstention gain >= {MIN_BALACC_GAIN}.",
        "",
        "| # | Kind | Proposer | Change | R@1 | R@3 | R@5 | R@10 | P@5 | MRR | Bal. abst. | False abst. | dR@5 | P(gain>0) | Decision |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        m = e["metrics"]
        change = ", ".join(f"`{k}={v}`" for k, v in e.get("changes", {}).items()) or e.get("config", "")
        gain = f"{e['gain']:+.3f}" if "gain" in e else ""
        prob = f"{e['bootstrap_prob']:.2f}" if "bootstrap_prob" in e else ""
        lines.append(
            f"| {e['iteration']} | {e['kind']} | {e['proposer']} | {change} | {m['R@1']:.3f} | {m['R@3']:.3f} "
            f"| {m['R@5']:.3f} | {m['R@10']:.3f} | {m['P@5']:.3f} | {m['MRR']:.3f} "
            f"| {m['balanced_abstention_acc']:.3f} | {m['false_abstention']:.1%} | {gain} | {prob} | {e['decision']} |"
        )
    lines += ["", "## Iteration details", ""]
    for e in entries:
        lines.append(f"### {e['iteration']}. {e.get('rule', e['kind'])}")
        lines.append("")
        if e.get("hypothesis"):
            lines.append(f"- **Hypothesis:** {e['hypothesis']}")
        if e.get("diagnosis_before"):
            top = sorted(e["diagnosis_before"].items(), key=lambda kv: -kv[1])[:6]
            lines.append("- **Diagnosis that triggered it:** " + ", ".join(f"{k} {v}" for k, v in top))
        lines.append(f"- **Config:** `{e['config']}`")
        lines.append(f"- **Decision:** {e['decision']}: {e['reason']}")
        lines.append(f"- **Train eval time:** {e['seconds']} s; logged {e['timestamp']}")
        lines.append("")
    (PROJECT_ROOT / "experiments/improvement_log.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def entry(iteration, kind, proposer, config_path, summary, decision, reason, **extra) -> dict:
    return {
        "iteration": iteration,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "kind": kind,
        "proposer": proposer,
        "config": config_path.relative_to(PROJECT_ROOT).as_posix(),
        "metrics": metrics(summary),
        "decision": decision,
        "reason": reason,
        "seconds": summary.get("seconds"),
        **extra,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train-only self-improvement loop.")
    parser.add_argument("--start", default="configs/graph_hybrid_rag.yaml")
    parser.add_argument("--reference", default="configs/baseline.yaml")
    parser.add_argument("--iterations", type=int, default=6)
    args = parser.parse_args()

    log = read_log()
    if not log:
        ref_raw = yaml.safe_load((PROJECT_ROOT / args.reference).read_text(encoding="utf-8"))
        summary, _, path = run_candidate(ref_raw, "loop_00_reference_baseline")
        append_log(entry(0, "reference", "Phase 1 baseline", path, summary, "reference",
                         "Phase 1 baseline, re-scored with a train-calibrated threshold for comparison."))
        start_raw = yaml.safe_load((PROJECT_ROOT / args.start).read_text(encoding="utf-8"))
        summary, _, path = run_candidate(start_raw, "loop_01_start")
        append_log(entry(1, "start", "Phase 2 architecture comparison (train)", path, summary, "start",
                         f"Loop starts from {args.start}, the best architecture on train in Phase 2."))
        log = read_log()

    current = next(e for e in reversed(log) if e["decision"] in ("start", "accepted"))
    tried = {e["rule"] for e in log if "rule" in e}
    next_iteration = max(e["iteration"] for e in log) + 1

    for iteration in range(next_iteration, next_iteration + args.iterations):
        current_raw = yaml.safe_load((PROJECT_ROOT / current["config"]).read_text(encoding="utf-8"))
        current_runs = LOOP_DIR / "runs" / current_raw["name"] / "per_query.jsonl"
        current_pq = [json.loads(l) for l in current_runs.read_text(encoding="utf-8").splitlines()]
        diagnosis = diagnose(current_pq)
        rule = propose(diagnosis, tried)
        if rule is None:
            print("No untried rule targets a remaining failure; stopping.")
            break
        tried.add(rule.name)
        name = f"loop_{iteration:02d}_{rule.name}"
        print(f"[{iteration}] proposing {rule.name}: {rule.changes}")
        summary, candidate_pq, path = run_candidate(apply_changes(current_raw, rule.changes), name)
        gain, prob = bootstrap_prob(current_pq, candidate_pq)
        accepted, reason = decide(current["metrics"], metrics(summary), gain, prob)
        record = entry(iteration, "candidate", "rules (improvement/propose.py)", path, summary,
                       "accepted" if accepted else "rejected", reason, rule=rule.name, changes=rule.changes,
                       hypothesis=rule.hypothesis, diagnosis_before=diagnosis,
                       gain=round(gain, 4), bootstrap_prob=round(prob, 3), base_config=current["config"])
        append_log(record)
        print(f"    R@5 {record['metrics']['R@5']:.3f} (gain {gain:+.3f}, P={prob:.2f}) -> {record['decision']}: {reason}")
        if accepted:
            current = record
        write_markdown(read_log())

    shutil.copyfile(PROJECT_ROOT / current["config"], BEST_PATH)
    write_markdown(read_log())
    print(f"Best configuration: {current['config']} -> {BEST_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
