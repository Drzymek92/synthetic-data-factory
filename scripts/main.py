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
from scripts.domain import build_schema, is_relational
from scripts.export import export_dataset, export_relational
from scripts.generator import generate
from scripts.logger import get_logger
from scripts.relational import (
    generate_relational,
    check_referential_integrity,
    validate_entities,
    evaluate_entities,
    build_relational_report,
    export_duckdb,
    apply_scale,
)
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


def run_relational(cfg: AppConfig, timestamp: str, write_duckdb: bool = True) -> None:
    """Generate a referentially-linked multi-table dataset for a relational domain."""
    out_dir = Path(cfg.output_dir)
    spec = cfg.domain_spec

    seeds_by_entity: dict[str, list] = {}
    for ename, espec in spec["entities"].items():
        seed_file = espec.get("seed_file")
        if seed_file and Path(seed_file).exists():
            seeds_by_entity[ename] = json.loads(Path(seed_file).read_text(encoding="utf-8"))

    tables = generate_relational(cfg.gen, spec, seeds_by_entity, model=cfg.model)
    total = sum(len(rows) for rows in tables.values())
    logger.info(f"Generated {total} records across {len(tables)} entities")

    validation = validate_entities(tables, spec)
    violations = check_referential_integrity(tables, spec)
    if violations:
        logger.warning(f"Referential integrity: {len(violations)} FK violations")
    else:
        logger.info("Referential integrity: PASS (all FKs resolve)")

    quality = evaluate_entities(tables, spec, cfg.quality, model=cfg.model)
    for ename, q in quality.items():
        if "near_duplicate_pairs" in q:
            logger.info(f"{ename}: {q['near_duplicate_pairs']} near-duplicate pairs (flagged, not removed)")
        if "judge" in q:
            logger.info(f"{ename}: judge overall mean {q['judge'].get('overall_mean')}")

    export_relational(tables, spec, out_dir, timestamp)
    if write_duckdb:
        try:
            export_duckdb(tables, spec, out_dir, timestamp)
        except Exception as exc:
            logger.warning(f"DuckDB export failed (CSV/JSON still written): {exc}")

    report = build_relational_report(tables, spec, violations, validation, quality)
    report_path = out_dir / f"{spec['name']}_relational_report_{timestamp}.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    logger.info(f"Wrote report: {report_path}")
    verdict = "PASS" if not violations else "REVIEW"
    logger.info(f"{verdict} — {total} records, {len(violations)} FK violations.")


def run(cfg: AppConfig, evaluate_path: Path | None, write_duckdb: bool = True) -> None:
    timestamp = _timestamp()
    out_dir = Path(cfg.output_dir)
    spec = cfg.domain_spec

    if is_relational(spec):
        if evaluate_path:
            logger.warning("--evaluate is not supported for relational domains; generating instead")
        run_relational(cfg, timestamp, write_duckdb=write_duckdb)
        return

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
    parser.add_argument("--scale", type=float,
                        help="(relational) multiply every top-level entity count for bigger/smaller samples")
    parser.add_argument("--count", nargs="+", metavar="ENTITY=N",
                        help="(relational) override specific entity counts, e.g. --count offers=200 orders=5000")
    parser.add_argument("--no-duckdb", action="store_true",
                        help="(relational) skip the .duckdb export (CSV + bundle JSON still written)")
    parser.add_argument("--smoke", action="store_true",
                        help="one tiny live LLM call to verify gateway connectivity + credentials, then exit")
    args = parser.parse_args()

    if args.smoke:
        from scripts.llm_client import smoke_test
        try:
            reply = smoke_test(model=args.model)
            logger.info(f"Smoke test passed — gateway reachable, reply: {reply!r}")
        except Exception as exc:
            logger.error(f"Smoke test FAILED: {type(exc).__name__}: {exc}")
            sys.exit(1)
        return

    overrides = {
        "domain": args.domain,
        "n_records": args.n,
        "model": args.model,
        "judge_sample_size": args.judge_sample,
        "run_judge": False if args.no_judge else None,
    }
    cfg = load_config(args.config, overrides)
    evaluate_path = Path(args.evaluate) if args.evaluate else None

    if is_relational(cfg.domain_spec) and (args.scale or args.count):
        count_overrides = {}
        for kv in (args.count or []):
            key, _, val = kv.partition("=")
            count_overrides[key.strip()] = int(val)
        apply_scale(cfg.domain_spec, args.scale, count_overrides)

    logger.info(f"Pipeline started (domain: {cfg.domain})")
    try:
        run(cfg, evaluate_path, write_duckdb=not args.no_duckdb)
        logger.info("Pipeline completed successfully")
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
