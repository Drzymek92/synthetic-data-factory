import json

from scripts.export import to_chat_jsonl, write_jsonl, export_dataset
from scripts.config import ExportConfig


def test_to_chat_jsonl_shape(domain_spec, valid_records):
    rows = to_chat_jsonl(valid_records, domain_spec["chat_export"])
    assert len(rows) == len(valid_records)
    msgs = rows[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    assert "Charged twice" in msgs[1]["content"]            # user = subject + body
    assert "category: billing" in msgs[2]["content"]        # assistant = labels


def test_to_chat_jsonl_handles_null_optional(domain_spec, valid_records):
    # record[1] has product=None; templates don't reference it, so no 'None' should leak
    rows = to_chat_jsonl([valid_records[1]], domain_spec["chat_export"])
    assert "None" not in rows[0]["messages"][2]["content"]


def test_write_jsonl_roundtrip(tmp_path, domain_spec, valid_records):
    rows = to_chat_jsonl(valid_records, domain_spec["chat_export"])
    path = write_jsonl(rows, tmp_path / "out.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(valid_records)
    assert json.loads(lines[0])["messages"][0]["role"] == "system"


def test_export_dataset_writes_enabled_formats(tmp_path, domain_spec, valid_records):
    cfg = ExportConfig(csv=True, chat_jsonl=True)
    written = export_dataset(valid_records, domain_spec, cfg, tmp_path, "20260615_000000")
    suffixes = sorted(p.suffix for p in written)
    assert suffixes == [".csv", ".jsonl"]
    assert all(p.exists() for p in written)
