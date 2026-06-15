import json
import random
from typing import Type

from pydantic import BaseModel, ValidationError

from scripts.config import GenConfig
from scripts.domain import build_schema, field_spec_text
from scripts.logger import get_logger

logger = get_logger("generator")


def _build_prompt(n: int, spec: dict, seeds: list[dict], hints: list[str]) -> str:
    gen = spec.get("generation", {})
    instructions = gen.get("instructions", "")
    hint_block = "\n".join(f"- {h}" for h in hints)
    seed_block = json.dumps(seeds[:2], indent=2, ensure_ascii=False) if seeds else "(none)"
    return (
        f"Generate {n} {spec['name'].replace('_', ' ')} records.\n\n"
        f"{field_spec_text(spec)}\n\n"
        f"{instructions}\n"
        f"Use these scenarios as loose inspiration (do not copy them verbatim):\n{hint_block}\n\n"
        f"Example of the target style and shape:\n{seed_block}\n\n"
        f'Return ONLY a JSON object of the form {{"records": [ ... ]}} containing '
        f"exactly {n} record objects."
    )


def _parse_batch(raw: dict | list) -> list[dict]:
    if isinstance(raw, dict):
        for key in ("records", "tickets", "data", "items"):
            if isinstance(raw.get(key), list):
                return raw[key]
        return [raw]
    if isinstance(raw, list):
        return raw
    return []


def _validate(records: list[dict], schema: Type[BaseModel]) -> list[dict]:
    valid: list[dict] = []
    for rec in records:
        try:
            valid.append(schema(**rec).model_dump(mode="json"))
        except (ValidationError, TypeError) as exc:
            logger.warning(f"Dropping invalid generated record: {exc}")
    return valid


def generate(config: GenConfig, spec: dict, seeds: list[dict] | None = None,
             model: str | None = None) -> list[dict]:
    """Generate `config.n_records` schema-valid records for the given domain spec.

    Batched generation with rotating scenario hints for diversity; every record is
    validated against the schema built from the spec; short batches are topped up
    up to `config.max_retries` extra attempts.
    """
    from scripts.llm_client import llm_json

    schema = build_schema(spec)
    system = spec.get("generation", {}).get("system", "Return only valid JSON.")
    all_hints = spec.get("generation", {}).get("scenario_hints", [])
    seeds = seeds or []
    rng = random.Random(config.seed)

    collected: list[dict] = []
    attempts = 0
    max_attempts = (config.n_records // config.batch_size + 1) + config.max_retries

    while len(collected) < config.n_records and attempts < max_attempts:
        attempts += 1
        need = min(config.batch_size, config.n_records - len(collected))
        hints = rng.sample(all_hints, k=min(need + 1, len(all_hints))) if all_hints else []
        prompt = _build_prompt(need, spec, seeds, hints)
        try:
            raw = llm_json(prompt, system=system, model=model, temperature=config.temperature)
        except Exception as exc:
            logger.warning(f"Batch {attempts} generation failed: {exc}")
            continue
        batch = _validate(_parse_batch(raw), schema)
        collected.extend(batch)
        logger.info(f"Batch {attempts}: +{len(batch)} valid ({len(collected)}/{config.n_records})")

    if len(collected) < config.n_records:
        logger.warning(f"Generated {len(collected)}/{config.n_records} after {attempts} attempts")
    return collected[: config.n_records]
