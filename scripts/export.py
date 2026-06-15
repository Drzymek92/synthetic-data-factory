import json
from pathlib import Path

import pandas as pd

from scripts.config import ExportConfig
from scripts.logger import get_logger

logger = get_logger("export")


class _SafeDict(dict):
    """Renders missing template keys as '' and None values as '' (so optional
    fields don't leak the literal 'None' into fine-tuning examples)."""

    def __missing__(self, key):
        return ""


def _fmt(template: str, record: dict) -> str:
    safe = _SafeDict({k: ("" if v is None else v) for k, v in record.items()})
    return template.format_map(safe)


def to_chat_jsonl(records: list[dict], chat_cfg: dict) -> list[dict]:
    """Map each record to a chat fine-tuning example using the domain's templates:
    {"messages": [{system}, {user}, {assistant}]}.
    """
    system = chat_cfg.get("system", "")
    user_t = chat_cfg["user_template"]
    assistant_t = chat_cfg["assistant_template"]
    rows: list[dict] = []
    for r in records:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": _fmt(user_t, r)})
        messages.append({"role": "assistant", "content": _fmt(assistant_t, r)})
        rows.append({"messages": messages})
    return rows


def write_jsonl(rows: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_csv(records: list[dict], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8")
    return path


def export_dataset(
    records: list[dict],
    spec: dict,
    export_cfg: ExportConfig,
    out_dir: Path,
    timestamp: str,
    tag: str = "",
) -> list[Path]:
    """Write the dataset in every enabled format. Returns the paths written.
    `tag` is inserted into the filename (e.g. "cleaned") to distinguish variants."""
    name = spec["name"] + (f"_{tag}" if tag else "")
    written: list[Path] = []

    if export_cfg.csv:
        p = write_csv(records, out_dir / f"{name}_{timestamp}.csv")
        logger.info(f"Wrote dataset CSV: {p}")
        written.append(p)

    if export_cfg.chat_jsonl:
        chat_cfg = spec.get("chat_export")
        if not chat_cfg:
            logger.warning(f"chat_jsonl enabled but domain '{name}' has no chat_export block; skipping")
        else:
            rows = to_chat_jsonl(records, chat_cfg)
            p = write_jsonl(rows, out_dir / f"{name}_chat_{timestamp}.jsonl")
            logger.info(f"Wrote fine-tuning chat JSONL ({len(rows)} examples): {p}")
            written.append(p)

    return written
