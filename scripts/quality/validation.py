from typing import Type

from pydantic import BaseModel, ValidationError


def validate_records(records: list[dict], model: Type[BaseModel]) -> tuple[list[dict], list[dict]]:
    """Split raw records into (valid, errors) against a Pydantic model.

    Each error entry is {"index", "record", "error"} so failures can be reported,
    not silently dropped.
    """
    valid: list[dict] = []
    errors: list[dict] = []
    for i, rec in enumerate(records):
        try:
            valid.append(model(**rec).model_dump(mode="json"))
        except (ValidationError, TypeError) as exc:
            errors.append({"index": i, "record": rec, "error": str(exc)})
    return valid, errors


def field_coverage(records: list[dict], fields: list[str]) -> dict[str, float]:
    """Fraction of records (0-1) with a non-null, non-empty value for each field."""
    n = len(records)
    if n == 0:
        return {f: 0.0 for f in fields}
    coverage: dict[str, float] = {}
    for f in fields:
        filled = sum(
            1 for r in records if r.get(f) not in (None, "", [], {})
        )
        coverage[f] = round(filled / n, 4)
    return coverage
