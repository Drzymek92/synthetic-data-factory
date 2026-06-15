import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Ensure the project root (parent of scripts/) is importable as `scripts`, ahead of
# any installed site-packages `scripts` package, when run as `python scripts/main.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from scripts.config import AppConfig, load_config
from scripts.domain import build_schema
from scripts.export import export_dataset
from scripts.generator import generate
from scripts.logger import get_logger
from scripts.quality import (
    validate_records,
    field_coverage,
    find_near_duplicates,
    dedupe_from_pairs,
    diversity_report,
    judge_dataset,
    write_report,
)

logger = get_logger("main")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def load_records(path: Path) -> list[dict]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data["records"] if isinstance(data, dict) and "records" in data else data
    if path.suffix in (".csv", ".tsv"):
        sep = "\t" if path.suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep).to_dict(orient="records")
    raise ValueError(f"Unsupported input format: {path.suffix}")


def evaluate(records: list[dict], cfg: AppConfig) -> dict:
    """Run the full quality harness on any list of records, driven by config."""
    logger.info(f"Evaluating {len(records)} records (domain: {cfg.domain})")
    spec = cfg.domain_spec
    schema = build_schema(spec)
    qcfg = cfg.quality
    text_fn = lambda r: r.get(cfg.text_field, "")

    valid, errors = validate_records(records, schema)
    logger.info(f"Validation: {len(valid)} valid, {len(errors)} rejected")

    pairs = find_near_duplicates(valid, text_fn, qcfg.dedup_threshold)
    deduped, removed = dedupe_from_pairs(valid, pairs)
    logger.info(f"Dedup: {len(pairs)} near-duplicate pairs, {len(removed)} removed")

    div = diversity_report(deduped, text_fn, cfg.label_fields, qcfg.ngram_n)
    logger.info(f"Diversity: distinct_{qcfg.ngram_n}={div.get(f'distinct_{qcfg.ngram_n}')}")

    judge = {}
    if qcfg.run_judge:
        judge_cfg = spec.get("judge")
        if not judge_cfg:
            logger.warning(f"run_judge enabled but domain '{cfg.domain}' has no judge block; skipping judge")
        else:
            judge = judge_dataset(deduped, judge_cfg, qcfg.judge_sample_size, model=cfg.model)
            logger.info(f"Judge: overall mean {judge.get('overall_mean')}")

    judged = bool(judge.get("judged"))
    overall = judge.get("overall_mean", 0.0)
    passed = (not judged) or overall >= qcfg.min_quality_score
    verdict = (
        f"{'PASS' if passed else 'REVIEW'} — {len(deduped)} clean records"
        + (f", mean quality {overall}/5 (threshold {qcfg.min_quality_score})." if judged else ".")
    )

    return {
        "clean_records": deduped,
        "display_field": next(iter(spec["fields"])),
        "summary": {
            "domain": cfg.domain,
            "input_records": len(records),
            "valid_records": len(valid),
            "clean_records": len(deduped),
            "duplicates_removed": len(removed),
        },
        "validation": {
            "valid": len(valid),
            "errors": len(errors),
            "coverage": field_coverage(valid, list(spec["fields"])),
        },
        "dedup": {"pairs": len(pairs), "removed": len(removed), "threshold": qcfg.dedup_threshold},
        "diversity": div,
        "judge": judge,
        "verdict": verdict,
    }


def run(cfg: AppConfig, evaluate_path: Path | None) -> None:
    timestamp = _timestamp()
    out_dir = Path(cfg.output_dir)
    spec = cfg.domain_spec

    if evaluate_path:
        records = load_records(evaluate_path)
        logger.info(f"Loaded {len(records)} records from {evaluate_path}")
    else:
        seed_file = spec.get("seed_file")
        seeds = json.loads(Path(seed_file).read_text(encoding="utf-8")) if seed_file and Path(seed_file).exists() else []
        records = generate(cfg.gen, spec, seeds, model=cfg.model)
        logger.info(f"Generated {len(records)} records")

    metrics = evaluate(records, cfg)
    clean = metrics.pop("clean_records")

    # Generate mode writes the dataset as-is; evaluate mode writes the *cleaned*
    # (deduped/validated) set under a "_cleaned" tag so the input isn't shadowed.
    tag = "cleaned" if evaluate_path else ""
    export_dataset(clean, spec, cfg.export, out_dir, timestamp, tag=tag)

    report_path = write_report(metrics, out_dir, timestamp, title=f"{cfg.domain} — Quality Report")
    logger.info(f"Wrote report: {report_path}")
    logger.info(metrics["verdict"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthetic Data Factory — config-driven generate and/or evaluate.")
    parser.add_argument("--config", default="config/default.yaml", help="path to the global config YAML")
    parser.add_argument("--domain", help="override the active domain (config/domains/<domain>.yaml)")
    parser.add_argument("--n", type=int, help="override number of records to generate")
    parser.add_argument("--model", help="override the model name")
    parser.add_argument("--evaluate", help="evaluate an existing .json/.csv dataset instead of generating")
    parser.add_argument("--no-judge", action="store_true", help="skip the LLM-as-judge stage (no API calls)")
    parser.add_argument("--judge-sample", type=int, help="judge only N sampled records")
    args = parser.parse_args()

    overrides = {
        "domain": args.domain,
        "n_records": args.n,
        "model": args.model,
        "judge_sample_size": args.judge_sample,
        "run_judge": False if args.no_judge else None,
    }
    cfg = load_config(args.config, overrides)
    evaluate_path = Path(args.evaluate) if args.evaluate else None

    logger.info(f"Pipeline started (domain: {cfg.domain})")
    try:
        run(cfg, evaluate_path)
        logger.info("Pipeline completed successfully")
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
