import pytest
from pydantic import ValidationError

from scripts.domain import load_domain, build_schema, field_spec_text


def test_load_domain_has_required_keys(domain_spec):
    for key in ("name", "fields", "text_field", "label_fields"):
        assert key in domain_spec


def test_load_domain_missing_raises():
    with pytest.raises(FileNotFoundError):
        load_domain("does_not_exist")


def test_build_schema_accepts_valid_record(schema, valid_records):
    obj = schema(**valid_records[0])
    assert obj.model_dump(mode="json")["category"] == "billing"


def test_build_schema_rejects_bad_enum(schema, valid_records):
    bad = dict(valid_records[0])
    bad["category"] = "not_a_category"
    with pytest.raises(ValidationError):
        schema(**bad)


def test_build_schema_enforces_str_constraints(schema, valid_records):
    bad = dict(valid_records[0])
    bad["body"] = "too short"
    with pytest.raises(ValidationError):
        schema(**bad)


def test_build_schema_allows_optional_null(schema, valid_records):
    rec = dict(valid_records[1])
    assert schema(**rec).model_dump(mode="json")["product"] is None


def test_field_spec_text_mentions_fields_and_enums(domain_spec):
    text = field_spec_text(domain_spec)
    assert "subject" in text
    assert "one of: billing" in text
    assert "or null" in text  # product is optional


# --- second domain: proves the config-driven design (zero Python to add) and the
#     int field type + ge/le bounds, which the support_tickets domain never exercises.

REVIEW = {
    "title": "Solid value",
    "review_body": "Used it daily for a month and it holds up well, no complaints worth raising at all.",
    "rating": 4,
    "sentiment": "positive",
    "recommended": "yes",
    "product_category": "electronics",
    "reviewer_name": "Sam Lee",
    "variant": None,
}


def test_product_reviews_domain_loads_and_builds():
    spec = load_domain("product_reviews")
    assert spec["text_field"] == "review_body"
    model = build_schema(spec)
    assert model(**REVIEW).model_dump(mode="json")["rating"] == 4


def test_product_reviews_int_bounds_enforced():
    model = build_schema(load_domain("product_reviews"))
    bad = dict(REVIEW)
    bad["rating"] = 9  # outside ge:1 le:5
    with pytest.raises(ValidationError):
        model(**bad)


# --- YAML boolean-enum trap: unquoted [Yes, No]/[On, Off]/[True, False] are parsed by
#     PyYAML as Python bools; coercing enum values to str must keep build_schema and
#     field_spec_text from crashing (TypeError) on such a spec.

def _bool_enum_spec() -> dict:
    return {
        "name": "bool_enum_domain",
        "fields": {
            "flag": {"type": "enum", "values": [True, False]},  # what PyYAML makes of [Yes, No]
            "body": {"type": "str"},
        },
        "text_field": "body",
        "label_fields": ["flag"],
    }


def test_build_schema_survives_bool_enum_values():
    model = build_schema(_bool_enum_spec())
    # bools coerce to their str form; the member value round-trips as a string
    assert model(flag="True", body="x").model_dump(mode="json")["flag"] == "True"
    with pytest.raises(ValidationError):
        model(flag="Yes", body="x")  # "Yes" was never a member — only "True"/"False"


def test_field_spec_text_survives_bool_enum_values():
    text = field_spec_text(_bool_enum_spec())
    assert "one of: True, False" in text
