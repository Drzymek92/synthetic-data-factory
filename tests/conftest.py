import pytest

from scripts.domain import load_domain, build_schema


@pytest.fixture
def domain_spec() -> dict:
    return load_domain("support_tickets")


@pytest.fixture
def schema(domain_spec):
    return build_schema(domain_spec)


@pytest.fixture
def valid_records() -> list[dict]:
    return [
        {
            "subject": "Charged twice for March",
            "body": "I was billed twice on March 3rd for the Pro plan and need the duplicate refunded please.",
            "category": "billing",
            "priority": "high",
            "sentiment": "negative",
            "customer_name": "Dana Whitfield",
            "product": "Pro Plan",
        },
        {
            "subject": "Thanks for the quick fix",
            "body": "Your team resolved my login problem in under ten minutes, really impressed with the service.",
            "category": "account",
            "priority": "low",
            "sentiment": "positive",
            "customer_name": "Marco Reyes",
            "product": None,
        },
        {
            "subject": "App keeps crashing on launch",
            "body": "Since the latest update the mobile app crashes immediately when I open it on my Android phone.",
            "category": "technical",
            "priority": "urgent",
            "sentiment": "negative",
            "customer_name": "Priya Nair",
            "product": "Mobile App",
        },
    ]


@pytest.fixture
def duplicate_records(valid_records) -> list[dict]:
    near_dup = dict(valid_records[0])
    near_dup["customer_name"] = "Someone Else"
    near_dup["body"] = valid_records[0]["body"]  # identical body -> near-duplicate
    return valid_records + [near_dup]
