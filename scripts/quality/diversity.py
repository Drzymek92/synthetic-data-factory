import math
import re
from collections import Counter
from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_TOKEN = re.compile(r"\b\w+\b")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def distinct_n(texts: list[str], n: int = 2) -> float:
    """Distinct-n: unique n-grams / total n-grams across the corpus (0-1).
    Higher = more lexically varied, less repetitive phrasing.
    """
    total = 0
    seen: set[tuple] = set()
    for t in texts:
        toks = _tokens(t)
        grams = [tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        total += len(grams)
        seen.update(grams)
    return round(len(seen) / total, 4) if total else 0.0


def vocabulary_size(texts: list[str]) -> int:
    vocab: set[str] = set()
    for t in texts:
        vocab.update(_tokens(t))
    return len(vocab)


def mean_pairwise_distance(texts: list[str]) -> float:
    """Mean (1 - cosine similarity) over all TF-IDF vector pairs (0-1).
    Higher = records are more semantically spread out (more diverse).
    """
    cleaned = [t for t in texts if (t or "").strip()]
    if len(cleaned) < 2:
        return 0.0
    matrix = TfidfVectorizer(stop_words="english").fit_transform(cleaned)
    sim = cosine_similarity(matrix)
    n = sim.shape[0]
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += 1.0 - float(sim[i, j])
            count += 1
    return round(total / count, 4) if count else 0.0


def label_distribution(records: list[dict], field: str) -> dict[str, int]:
    return dict(Counter(str(r.get(field)) for r in records))


def normalized_entropy(counts: dict) -> float:
    """Shannon entropy of a label distribution normalized to 0-1, where 1 means
    perfectly balanced classes and 0 means everything in one class.
    """
    values = [c for c in counts.values() if c > 0]
    total = sum(values)
    k = len(values)
    if total == 0 or k <= 1:
        return 0.0
    entropy = -sum((c / total) * math.log(c / total) for c in values)
    return round(entropy / math.log(k), 4)


def diversity_report(
    records: list[dict],
    text_fn: Callable[[dict], str],
    label_fields: list[str],
    ngram_n: int = 2,
) -> dict:
    texts = [(text_fn(r) or "") for r in records]
    label_stats = {}
    for f in label_fields:
        dist = label_distribution(records, f)
        label_stats[f] = {"distribution": dist, "balance": normalized_entropy(dist)}
    return {
        "n_records": len(records),
        f"distinct_{ngram_n}": distinct_n(texts, ngram_n),
        "vocabulary_size": vocabulary_size(texts),
        "mean_pairwise_distance": mean_pairwise_distance(texts),
        "labels": label_stats,
    }
