from scripts.quality.validation import validate_records, field_coverage


def test_validate_records_splits_valid_and_invalid(schema, valid_records):
    records = valid_records + [{"subject": "x", "body": "y"}]  # missing required fields
    valid, errors = validate_records(records, schema)
    assert len(valid) == len(valid_records)
    assert len(errors) == 1
    assert errors[0]["index"] == len(valid_records)


def test_field_coverage_full(valid_records):
    coverage = field_coverage(valid_records, ["subject", "category"])
    assert coverage["subject"] == 1.0
    assert coverage["category"] == 1.0


def test_field_coverage_counts_nulls(valid_records):
    coverage = field_coverage(valid_records, ["product"])
    # one of three records has product=None
    assert coverage["product"] < 1.0


def test_field_coverage_empty_input():
    assert field_coverage([], ["subject"]) == {"subject": 0.0}
