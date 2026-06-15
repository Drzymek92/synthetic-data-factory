from typing import Callable

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _texts(records: list[dict], text_fn: Callable[[dict], str]) -> list[str]:
    return [(text_fn(r) or "").strip() for r in records]


def find_near_duplicates(
    records: list[dict],
    text_fn: Callable[[dict], str],
    threshold: float = 0.92,
) -> list[tuple[int, int, float]]:
    """Return (i, j, similarity) for every record pair whose TF-IDF cosine
    similarity is >= threshold. TF-IDF keeps this dependency-light (scikit-learn
    only); swap in embeddings here for a stronger semantic backend.
    """
    texts = _texts(records, text_fn)
    if len(texts) < 2:
        return []
    matrix = TfidfVectorizer(stop_words="english").fit_transform(texts)
    sim = cosine_similarity(matrix)
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = float(sim[i, j])
            if score >= threshold:
                pairs.append((i, j, round(score, 4)))
    return pairs


def dedupe_from_pairs(
    records: list[dict],
    pairs: list[tuple[int, int, float]],
) -> tuple[list[dict], list[int]]:
    """Greedily drop near-duplicates from precomputed pairs. Keeps the first record
    of each duplicate cluster; returns (kept_records, removed_indices). Use this when
    the pairs are already computed (e.g. for reporting) to avoid a second TF-IDF pass.
    """
    removed: set[int] = set()
    for i, j, _ in pairs:
        if i not in removed:
            removed.add(j)  # keep the earlier record, drop the later one
    kept = [r for idx, r in enumerate(records) if idx not in removed]
    return kept, sorted(removed)


def dedupe(
    records: list[dict],
    text_fn: Callable[[dict], str],
    threshold: float = 0.92,
) -> tuple[list[dict], list[int]]:
    """Greedily drop near-duplicates. Keeps the first record of each duplicate
    cluster; returns (kept_records, removed_indices) against the original list.
    """
    pairs = find_near_duplicates(records, text_fn, threshold)
    return dedupe_from_pairs(records, pairs)
