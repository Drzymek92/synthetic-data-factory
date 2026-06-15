import json
import random
import statistics

from scripts.logger import get_logger

logger = get_logger("judge")


def judge_record(record: dict, judge_cfg: dict, model: str | None = None) -> dict:
    """Score one record on the domain's judge dimensions. `judge_cfg` is the domain
    spec's `judge` block: {"system": str, "dimensions": [str, ...]}."""
    from scripts.llm_client import llm_json

    dims = judge_cfg["dimensions"]
    prompt = "Score this record:\n" + json.dumps(record, ensure_ascii=False, indent=2)
    scores = llm_json(prompt, system=judge_cfg["system"], model=model, temperature=0.0)
    out = {d: int(scores.get(d, 0)) for d in dims}
    out["issue"] = str(scores.get("issue", ""))
    return out


def judge_dataset(
    records: list[dict],
    judge_cfg: dict,
    sample_size: int | None = None,
    seed: int = 42,
    model: str | None = None,
) -> dict:
    """LLM-as-judge over the dataset, using the domain's `judge` block for the
    reviewer prompt and dimensions. Optionally scores a random sample for cost.
    Returns per-dimension means, overall mean, and flagged low-scoring records.
    """
    dim_names = judge_cfg["dimensions"]
    if not records:
        return {"judged": 0, "means": {}, "overall_mean": 0.0, "flagged": []}

    pool = list(records)
    if sample_size and sample_size < len(pool):
        pool = random.Random(seed).sample(pool, sample_size)

    dims: dict[str, list[int]] = {d: [] for d in dim_names}
    flagged: list[dict] = []
    for i, rec in enumerate(pool):
        try:
            s = judge_record(rec, judge_cfg, model=model)
        except Exception as exc:
            logger.warning(f"Judge failed on record {i}: {exc}")
            continue
        for d in dim_names:
            dims[d].append(s[d])
        record_mean = statistics.mean(s[d] for d in dim_names)
        if record_mean < 3 or s["issue"]:
            flagged.append({"record": rec, "scores": s})
        logger.info(f"Judged {i + 1}/{len(pool)} (mean {record_mean:.1f})")

    means = {d: round(statistics.mean(v), 3) for d, v in dims.items() if v}
    overall = round(statistics.mean(means.values()), 3) if means else 0.0
    return {
        "judged": len(next(iter(dims.values()))),
        "means": means,
        "overall_mean": overall,
        "flagged": flagged,
    }
