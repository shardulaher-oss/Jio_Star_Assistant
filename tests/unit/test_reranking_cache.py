from pathlib import Path

from contract_intelligence.retrieval.reranking import CrossEncoderReranker


class CountingModel:
    def __init__(self) -> None:
        self.pairs = 0

    def predict(self, pairs, batch_size, show_progress_bar):
        self.pairs += len(pairs)
        return [float(len(p)) for _, p in pairs]


def make(tmp_path: Path) -> tuple[CrossEncoderReranker, CountingModel]:
    reranker = CrossEncoderReranker("fake/model", cache_dir=tmp_path)
    model = CountingModel()
    reranker._model = model
    return reranker, model


def test_scores_are_cached_and_reused(tmp_path: Path) -> None:
    reranker, model = make(tmp_path)
    first = reranker.score("q", ["a b", "ccc"])
    second = reranker.score("q", ["ccc", "a  b", "dddd"])  # whitespace-normalised hit
    assert list(first) == [3.0, 3.0]
    assert list(second) == [3.0, 3.0, 4.0]
    assert model.pairs == 3


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    reranker, _ = make(tmp_path)
    reranker.score("q", ["a b"])
    reloaded, model = make(tmp_path)
    assert list(reloaded.score("q", ["a b"])) == [3.0]
    assert model.pairs == 0


def test_cache_key_includes_the_query(tmp_path: Path) -> None:
    reranker, model = make(tmp_path)
    reranker.score("q1", ["same"])
    reranker.score("q2", ["same"])
    assert model.pairs == 2
