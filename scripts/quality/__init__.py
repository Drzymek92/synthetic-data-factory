"""Reusable dataset quality / evaluation harness.

Domain-agnostic: every function operates on a list of record dicts plus the field
names to analyse, so the same harness scores synthetic data AND real labeled
exports (e.g. FTS/Playment job exports). Nothing here depends on how the data
was produced.
"""

from scripts.quality.validation import validate_records, field_coverage
from scripts.quality.dedup import find_near_duplicates, dedupe, dedupe_from_pairs
from scripts.quality.diversity import diversity_report
from scripts.quality.judge import judge_dataset
from scripts.quality.report import build_report, write_report

__all__ = [
    "validate_records",
    "field_coverage",
    "find_near_duplicates",
    "dedupe",
    "dedupe_from_pairs",
    "diversity_report",
    "judge_dataset",
    "build_report",
    "write_report",
]
