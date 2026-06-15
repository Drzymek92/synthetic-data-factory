import scripts.llm_client as llm_client
from scripts.quality.judge import judge_dataset, judge_record


def _fixed(scores):
    """Return a fake llm_json that always yields the given score dict."""
    def fake(prompt, system=None, model=None, temperature=0.0, **kwargs):
        return dict(scores)
    return fake


def test_judge_record_reads_cfg_dimensions(monkeypatch, valid_records):
    # The judge must use whatever dimensions the domain cfg declares, not a fixed set.
    monkeypatch.setattr(llm_client, "llm_json", _fixed({"a": 5, "b": 1, "issue": ""}))
    cfg = {"system": "score it", "dimensions": ["a", "b"]}
    out = judge_record(valid_records[0], cfg)
    assert out == {"a": 5, "b": 1, "issue": ""}


def test_judge_dataset_aggregates_means(monkeypatch, valid_records, domain_spec):
    monkeypatch.setattr(
        llm_client, "llm_json",
        _fixed({"realism": 4, "coherence": 5, "label_fit": 3, "issue": ""}),
    )
    result = judge_dataset(valid_records, domain_spec["judge"])
    assert result["judged"] == len(valid_records)
    assert result["means"] == {"realism": 4.0, "coherence": 5.0, "label_fit": 3.0}
    assert result["overall_mean"] == 4.0
    assert result["flagged"] == []  # mean 4.0, no issue -> not flagged


def test_judge_dataset_flags_low_scores(monkeypatch, valid_records, domain_spec):
    monkeypatch.setattr(
        llm_client, "llm_json",
        _fixed({"realism": 1, "coherence": 2, "label_fit": 1, "issue": ""}),
    )
    result = judge_dataset(valid_records, domain_spec["judge"])
    assert len(result["flagged"]) == len(valid_records)  # record mean 1.33 < 3


def test_judge_dataset_flags_on_issue_despite_high_score(monkeypatch, valid_records, domain_spec):
    monkeypatch.setattr(
        llm_client, "llm_json",
        _fixed({"realism": 5, "coherence": 5, "label_fit": 5, "issue": "off-topic"}),
    )
    result = judge_dataset(valid_records, domain_spec["judge"])
    assert len(result["flagged"]) == len(valid_records)  # high mean but issue set


def test_judge_dataset_respects_sample_size(monkeypatch, valid_records, domain_spec):
    calls = {"n": 0}

    def counting(prompt, system=None, model=None, temperature=0.0, **kwargs):
        calls["n"] += 1
        return {"realism": 4, "coherence": 4, "label_fit": 4, "issue": ""}

    monkeypatch.setattr(llm_client, "llm_json", counting)
    result = judge_dataset(valid_records, domain_spec["judge"], sample_size=1)
    assert calls["n"] == 1          # only the sampled record was scored
    assert result["judged"] == 1


def test_judge_dataset_empty_returns_zero():
    result = judge_dataset([], {"system": "x", "dimensions": ["realism"]})
    assert result["judged"] == 0
    assert result["overall_mean"] == 0.0


def test_judge_dataset_skips_failed_records(monkeypatch, valid_records, domain_spec):
    def boom(prompt, system=None, model=None, temperature=0.0, **kwargs):
        raise RuntimeError("api down")

    monkeypatch.setattr(llm_client, "llm_json", boom)
    result = judge_dataset(valid_records, domain_spec["judge"])
    assert result["judged"] == 0          # all failed, none counted
    assert result["overall_mean"] == 0.0
