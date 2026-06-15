import scripts.llm_client as llm_client
from scripts.config import GenConfig
from scripts.generator import generate


def test_generate_collects_and_trims_to_n(monkeypatch, domain_spec, valid_records):
    def fake(prompt, system=None, model=None, temperature=0.0, **kwargs):
        return {"records": [dict(r) for r in valid_records]}  # 3 valid per batch

    monkeypatch.setattr(llm_client, "llm_json", fake)
    out = generate(GenConfig(n_records=5, batch_size=3), domain_spec, valid_records)
    assert len(out) == 5                       # trimmed to n_records
    assert all("category" in r for r in out)   # schema-valid records


def test_generate_drops_invalid_records(monkeypatch, domain_spec, valid_records):
    def fake(prompt, system=None, model=None, temperature=0.0, **kwargs):
        bad = {"subject": "x", "body": "too short"}  # missing required fields
        return {"records": [dict(valid_records[0]), bad]}

    monkeypatch.setattr(llm_client, "llm_json", fake)
    out = generate(GenConfig(n_records=1, batch_size=2), domain_spec, valid_records)
    assert len(out) == 1
    assert out[0]["category"] == "billing"     # the valid record survived


def test_generate_recovers_from_batch_failure(monkeypatch, domain_spec, valid_records):
    calls = {"n": 0}

    def flaky(prompt, system=None, model=None, temperature=0.0, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("rate limited")
        return {"records": [dict(valid_records[0])]}

    monkeypatch.setattr(llm_client, "llm_json", flaky)
    out = generate(GenConfig(n_records=1, batch_size=1, max_retries=3), domain_spec, valid_records)
    assert len(out) == 1                        # recovered on the retry


def test_generate_accepts_alternate_wrapper_key(monkeypatch, domain_spec, valid_records):
    def fake(prompt, system=None, model=None, temperature=0.0, **kwargs):
        return {"tickets": [dict(valid_records[0])]}  # not "records" — still parsed

    monkeypatch.setattr(llm_client, "llm_json", fake)
    out = generate(GenConfig(n_records=1, batch_size=1), domain_spec, valid_records)
    assert len(out) == 1
