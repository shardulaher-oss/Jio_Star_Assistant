"""Structured, grounded answers from the top retrieved passages (Gemini).

Retrieval stays the source of truth; the LLM only reads the top passages and returns
a structured answer with a verbatim quote. Safeguards:
- if retrieval abstains, the model is never called;
- the model must answer only from the passages, or say "not_found";
- its quote is located verbatim (whitespace-insensitive) in the cited passage and
  converted to exact character offsets in the original contract; a quote that cannot
  be found is reported as ungrounded rather than shown as evidence;
- every response is cached on disk by (model, prompt), so evaluation and demos
  reproduce without calling the API again.

Privacy: the top passages (contract text) are sent to Google's API. That is fine for
the public CUAD corpus; for confidential agreements use a self-hosted model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from contract_intelligence.config import PROJECT_ROOT, AnswerConfig
from contract_intelligence.logging_utils import query_id
from contract_intelligence.schemas import Passage, SearchResult

log = logging.getLogger(__name__)
PROMPT_VERSION = "1"

SYSTEM_INSTRUCTION = (
    "You answer questions about commercial contracts using ONLY the numbered contract "
    "passages provided. Rules: (1) If no passage answers the question, set answer_type "
    "to 'not_found', evidence_passage to 0 and evidence_quote to ''. (2) evidence_quote "
    "must be copied exactly, character for character, from ONE passage: the shortest "
    "complete sentence or clause that answers the question. Do not paraphrase, shorten "
    "with ellipses or merge passages. (3) answer is one or two plain sentences that "
    "state what the quoted text says; do not add facts, interpretation or legal advice. "
    "(4) Use 'yes' or 'no' only when the question is yes/no and the quote settles it."
)


class StructuredAnswer(BaseModel):
    answer: str = Field(description="One or two sentences, strictly based on the quote.")
    answer_type: Literal["yes", "no", "value", "clause", "not_found"]
    evidence_passage: int = Field(description="Number of the passage holding the quote; 0 if not_found.")
    evidence_quote: str = Field(description="Verbatim text copied from that passage; empty if not_found.")
    supporting_passages: list[int] = Field(description="Numbers of all passages relevant to the answer.")
    confidence: Literal["high", "medium", "low"]


@dataclass
class Evidence:
    passage_rank: int
    file_path: str
    start: int  # character offsets of the quote in the original contract
    end: int
    text: str


@dataclass
class GroundedAnswer:
    question: str
    answer: str
    answer_type: str
    confidence: str
    grounded: bool  # quote found verbatim in a retrieved passage
    evidence: Evidence | None
    quote: str
    supporting_passages: list[int] = field(default_factory=list)
    model: str = ""
    cached: bool = False
    latency_ms: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class LLMClient(Protocol):
    def generate(self, prompt: str) -> dict: ...


def load_dotenv(path: Path = PROJECT_ROOT / ".env") -> None:
    """Load KEY=VALUE lines from a git-ignored .env without overriding the environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        match = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if match and not line.lstrip().startswith("#"):
            os.environ.setdefault(match.group(1), match.group(2).strip("\"'"))


class GeminiClient:
    def __init__(self, config: AnswerConfig) -> None:
        load_dotenv()
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY in the environment or in a git-ignored .env file.")
        from google import genai
        from google.genai import types

        # One attempt per SDK call; generate() below owns the retry policy.
        self.client = genai.Client(api_key=key, http_options=types.HttpOptions(
            timeout=config.timeout_seconds * 1000, retry_options=types.HttpRetryOptions(attempts=1)))
        self.config = config

    def generate(self, prompt: str) -> dict:
        from google.genai import errors, types

        thinking = (
            types.ThinkingConfig(thinking_budget=self.config.thinking_budget)
            if self.config.thinking_budget is not None else None
        )
        settings = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=self.config.temperature,
            seed=0,
            response_mime_type="application/json",
            response_schema=StructuredAnswer,
            thinking_config=thinking,
        )
        for attempt in range(5):
            try:
                response = self.client.models.generate_content(
                    model=self.config.model, contents=prompt, config=settings
                )
                usage = response.usage_metadata
                return {
                    **json.loads(response.text),
                    "_usage": {
                        "input_tokens": getattr(usage, "prompt_token_count", None),
                        "output_tokens": getattr(usage, "candidates_token_count", None),
                    },
                }
            except errors.APIError as error:  # rate limits and transient server errors
                if getattr(error, "code", None) not in (429, 500, 503, 504) or attempt == 4:
                    raise
                time.sleep(min(60, 5 * 2**attempt))
        raise RuntimeError("unreachable")


def build_prompt(question: str, passages: list[Passage]) -> str:
    blocks = [
        f"[{i}] source: {p.file_path} (characters {p.start}-{p.end})\n{' '.join(p.text.split())}"
        for i, p in enumerate(passages, 1)
    ]
    return f"Question: {question}\n\nContract passages:\n\n" + "\n\n".join(blocks)


def locate_quote(quote: str, passages: list[Passage], preferred: int) -> Evidence | None:
    """Find the quote verbatim (ignoring whitespace) in the passages, preferred one first."""
    words = quote.strip().strip("\"'“”").split()
    if not words:
        return None
    pattern = re.compile(r"\s+".join(re.escape(w) for w in words))
    order = [preferred] + [i for i in range(1, len(passages) + 1) if i != preferred]
    for rank in order:
        if 1 <= rank <= len(passages):
            passage = passages[rank - 1]
            match = pattern.search(passage.text)
            if match:
                return Evidence(rank, passage.file_path, passage.start + match.start(),
                                passage.start + match.end(), match.group(0))
    return None


class AnswerGenerator:
    def __init__(self, config: AnswerConfig, client: LLMClient | None = None) -> None:
        self.config = config
        self._client = client
        self.cache_path = PROJECT_ROOT / config.cache_path
        self.cache: dict[str, dict] = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                self.cache[record["key"]] = record["response"]

    @property
    def client(self) -> LLMClient:
        if self._client is None:  # created lazily: fully cached runs need no API key
            if not self.config.allow_external:
                raise PermissionError(
                    "answer.allow_external is false: contract text may not be sent to an external "
                    "LLM provider. Use retrieval results only, or configure a self-hosted model."
                )
            if self.config.provider != "gemini":
                raise ValueError(f"Unknown answer provider: {self.config.provider}")
            self._client = GeminiClient(self.config)
        return self._client

    def _key(self, prompt: str) -> str:
        payload = f"{PROMPT_VERSION}\x00{self.config.model}\x00{self.config.temperature}\x00{prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def answer(self, result: SearchResult) -> GroundedAnswer:
        if result.abstained or not result.passages:
            return GroundedAnswer(result.query, "No relevant clause was found in the indexed contracts.",
                                  "not_found", "high", False, None, "", model="(not called)",
                                  note="Retrieval abstained, so the LLM was not called.")
        passages = result.passages[: self.config.max_passages]
        prompt = build_prompt(result.query, passages)
        key = self._key(prompt)
        started = time.perf_counter()
        cached = key in self.cache
        if not cached:
            response = self.client.generate(prompt)
            self.cache[key] = response
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"key": key, "model": self.config.model, "response": response},
                                        ensure_ascii=False) + "\n")
        response = self.cache[key]
        latency = (time.perf_counter() - started) * 1000
        log.info("answer", extra={"event": "answer", "model": self.config.model, "cached": cached,
                                  "passages_sent": 0 if cached else len(passages),
                                  "query_id": query_id(result.query), "latency_ms": round(latency, 1)})

        parsed = StructuredAnswer.model_validate({k: v for k, v in response.items() if not k.startswith("_")})
        common = dict(question=result.query, confidence=parsed.confidence, model=self.config.model,
                      cached=cached, latency_ms=round(latency, 1),
                      supporting_passages=[i for i in parsed.supporting_passages if 1 <= i <= len(passages)])
        if parsed.answer_type == "not_found":
            return GroundedAnswer(answer="The retrieved passages do not answer this question.",
                                  answer_type="not_found", grounded=False, evidence=None, quote="",
                                  note="The model found no answer in the top passages.", **common)
        evidence = locate_quote(parsed.evidence_quote, passages, parsed.evidence_passage)
        return GroundedAnswer(
            answer=parsed.answer, answer_type=parsed.answer_type, grounded=evidence is not None,
            evidence=evidence, quote=parsed.evidence_quote,
            note="" if evidence else "The quoted text was not found verbatim in the passages; "
                                     "treat this answer as unverified.",
            **common,
        )
