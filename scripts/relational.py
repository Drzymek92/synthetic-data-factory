"""Relational (multi-table) generation for domains with an `entities` block.

The LLM only fills each entity's CONTENT fields (via the existing single-entity
`generator.generate`); this module assigns `id`s and foreign keys deterministically in
Python by drawing real parent IDs, which is what guarantees referential integrity.
Entities are generated parents-first (topological order over relationships + per_parent).
"""
from dataclasses import replace
from datetime import date, timedelta
import random

from scripts.config import GenConfig
from scripts.domain import (
    build_schema,
    entity_content_spec,
    entity_full_spec,
    id_field_of,
    _enum_values,
)
from scripts.generator import generate as generate_records
from scripts.logger import get_logger
from scripts.quality import (
    validate_records,
    find_near_duplicates,
    diversity_report,
    judge_dataset,
)
from scripts.quality.diversity import label_distribution, normalized_entropy

logger = get_logger("relational")


def apply_scale(spec: dict, scale: float | None = None,
                count_overrides: dict[str, int] | None = None) -> None:
    """Mutate top-level entity `count`s for bigger/smaller samples. `count_overrides`
    (entity → N) wins; otherwise `scale` multiplies each `count` (min 1). `per_parent`
    entities have no `count` — they scale automatically with their parent's row count."""
    count_overrides = count_overrides or {}
    for ename, espec in spec["entities"].items():
        if ename in count_overrides:
            espec["count"] = max(0, int(count_overrides[ename]))
        elif scale and "count" in espec:
            espec["count"] = max(1, round(espec["count"] * scale))


def generation_order(spec: dict) -> list[str]:
    """Topologically order entities so every referenced parent precedes its children.

    Dependencies come from each entity's `relationships[*].ref` and its `per_parent.parent`.
    Raises on a cycle or an unresolved dependency.
    """
    entities = spec["entities"]
    deps: dict[str, set[str]] = {}
    for name, espec in entities.items():
        d = {rel["ref"] for rel in espec.get("relationships", {}).values()}
        pp = espec.get("per_parent")
        if pp:
            d.add(pp["parent"])
        deps[name] = {x for x in d if x in entities and x != name}

    order: list[str] = []
    resolved: set[str] = set()
    while len(order) < len(entities):
        progressed = False
        for name in entities:
            if name not in resolved and deps[name] <= resolved:
                order.append(name)
                resolved.add(name)
                progressed = True
        if not progressed:
            raise ValueError(
                f"Cyclic or unresolved entity dependencies among: {set(entities) - resolved}"
            )
    return order


def _is_synthetic(espec: dict) -> bool:
    """Whether an entity's content fields are filled deterministically in Python rather
    than by the LLM. SYNTHETIC IS THE DEFAULT (fast, reproducible, free); an entity opts
    into LLM realism with `generate: llm`. `fill: random` / `synthetic: true` are accepted
    as explicit aliases for the synthetic default."""
    return espec.get("generate", "synthetic") != "llm"


def _synthetic_value(fname: str, fdef: dict, i: int, rng: random.Random):
    """One deterministic field value, honouring the optional distribution config:
    enum `weights`, int `choices`, and the `date` type (anchor + offset range)."""
    ftype = fdef.get("type", "str")
    if ftype == "int":
        choices = fdef.get("choices")
        if choices:
            return int(rng.choice(choices))
        lo = fdef.get("ge", 0)
        return rng.randint(lo, fdef.get("le", lo + 100))
    if ftype == "enum":
        values = _enum_values(fdef["values"])
        weights = fdef.get("weights")
        if weights:
            # coerce keys to str — unquoted yes/no/on/off keys parse as YAML bools
            wmap = {str(k): float(v) for k, v in weights.items()}
            return rng.choices(values, weights=[wmap.get(v, 0.0) for v in values], k=1)[0]
        return rng.choice(values)
    if ftype == "date":
        anchor = fdef.get("anchor", "today")
        base = date.today() if anchor == "today" else date.fromisoformat(str(anchor))
        lo, hi = fdef.get("min_offset_days", 0), fdef.get("max_offset_days", 0)
        return (base + timedelta(days=rng.randint(min(lo, hi), max(lo, hi)))).isoformat()
    return f"{fname}_{i + 1}"  # str — synthetic fill is not for realistic prose


def _synthetic_fill(espec: dict, n: int, rng: random.Random) -> list[dict]:
    fields = espec.get("fields", {})
    return [{fn: _synthetic_value(fn, fd, i, rng) for fn, fd in fields.items()} for i in range(n)]


def _gen_content(entity_name: str, espec: dict, gen_cfg: GenConfig, count: int,
                 seeds: list[dict] | None, model: str | None,
                 rng: random.Random) -> list[dict]:
    if count <= 0:
        return []
    if _is_synthetic(espec):
        return _synthetic_fill(espec, count, rng)
    flat = entity_content_spec(entity_name, espec)
    return generate_records(replace(gen_cfg, n_records=count), flat, seeds or [], model=model)


def _resolve_row(tables: dict, spec: dict, entity: str, id_value) -> dict | None:
    id_field = id_field_of(spec["entities"][entity])
    for row in tables.get(entity, []):
        if row.get(id_field) == id_value:
            return row
    return None


def _assign_fks(record: dict, ename: str, espec: dict, tables: dict, spec: dict,
                rng: random.Random, fixed: dict | None = None,
                fallbacks: dict | None = None) -> None:
    fixed = fixed or {}
    rels = espec.get("relationships", {})
    # Set fixed FKs first so a `match` constraint can read them (e.g. the per_parent fk).
    for fk_field, val in fixed.items():
        record[fk_field] = val

    for fk_field, rel in rels.items():
        if fk_field in fixed:
            continue
        ref = rel["ref"]
        candidates = tables.get(ref, [])

        # Constrained FK: restrict candidates to parents sharing `field` with the row
        # referenced by another (already-assigned) FK `via`, for cross-entity coherence.
        # (Keys are `field`/`via`, not `on`/`parent`: a bare YAML `on:` parses as a bool.)
        match = rel.get("match")
        if match and candidates:
            shared, via = match["field"], match["via"]
            parent_row = _resolve_row(tables, spec, rels[via]["ref"], record.get(via))
            if parent_row is not None and shared in parent_row:
                want = parent_row[shared]
                filtered = [r for r in candidates if r.get(shared) == want]
                if filtered:
                    candidates = filtered
                elif fallbacks is not None:
                    fallbacks[fk_field] = fallbacks.get(fk_field, 0) + 1  # summarized by caller

        if not candidates:
            record[fk_field] = None
            continue
        record[fk_field] = rng.choice(candidates)[id_field_of(spec["entities"][ref])]


def _apply_copies(record: dict, espec: dict, tables: dict, spec: dict) -> None:
    """Denormalize parent fields into this record per the entity's `copy` block:
    copy: {target_field: {from: <fk on this record>, field: <col on referenced parent>}}."""
    rels = espec.get("relationships", {})
    for tgt, cdef in espec.get("copy", {}).items():
        ref = rels.get(cdef["from"], {}).get("ref")
        row = _resolve_row(tables, spec, ref, record.get(cdef["from"])) if ref else None
        record[tgt] = row.get(cdef["field"]) if row else None


def _with_ids(rows: list[dict], espec: dict) -> list[dict]:
    id_field = id_field_of(espec)
    prefix = espec.get("id_prefix") or espec["__name__"][:3].upper()
    return [{id_field: f"{prefix}-{i:04d}", **row} for i, row in enumerate(rows, start=1)]


def generate_relational(gen_cfg: GenConfig, spec: dict,
                        seeds_by_entity: dict[str, list[dict]] | None = None,
                        model: str | None = None,
                        rng: random.Random | None = None) -> dict[str, list[dict]]:
    """Generate every entity in dependency order, returning {entity_name: [records]}.

    Top-level entities use their `count`; `per_parent` entities emit min..max children
    per parent row (the parent FK is set to that specific parent, other FKs drawn at
    random from their referenced tables).
    """
    seeds_by_entity = seeds_by_entity or {}
    rng = rng or random.Random(gen_cfg.seed)
    tables: dict[str, list[dict]] = {}

    for ename in generation_order(spec):
        espec = spec["entities"][ename]
        per_parent = espec.get("per_parent")
        rows: list[dict] = []
        fallbacks: dict[str, int] = {}   # constrained-FK fallbacks, summarized once per entity

        if per_parent:
            parent = per_parent["parent"]
            lo, hi = per_parent.get("min", 1), per_parent.get("max", 1)
            fk_field = per_parent.get("fk") or _infer_parent_fk(espec, parent)
            parent_id_field = id_field_of(spec["entities"][parent])
            parents = tables.get(parent, [])
            counts = [rng.randint(lo, hi) for _ in parents]
            content = _gen_content(ename, espec, gen_cfg, sum(counts),
                                   seeds_by_entity.get(ename), model, rng)
            idx = 0
            for prow, k in zip(parents, counts):
                for _ in range(k):
                    if idx >= len(content):
                        break
                    rec = content[idx]
                    idx += 1
                    fixed = {fk_field: prow[parent_id_field]} if fk_field else {}
                    _assign_fks(rec, ename, espec, tables, spec, rng, fixed=fixed, fallbacks=fallbacks)
                    _apply_copies(rec, espec, tables, spec)
                    rows.append(rec)
            if idx < sum(counts):
                logger.warning(
                    f"{ename}: only {idx}/{sum(counts)} child records generated (LLM shortfall)"
                )
        else:
            count = espec.get("count", gen_cfg.n_records)
            content = _gen_content(ename, espec, gen_cfg, count,
                                   seeds_by_entity.get(ename), model, rng)
            for rec in content:
                _assign_fks(rec, ename, espec, tables, spec, rng, fallbacks=fallbacks)
                _apply_copies(rec, espec, tables, spec)
                rows.append(rec)

        for fk_field, n in fallbacks.items():
            logger.warning(
                f"{ename}.{fk_field}: {n} pick(s) fell back to unconstrained "
                f"(no parent matched the `match` constraint)"
            )

        espec_named = {**espec, "__name__": ename}
        tables[ename] = _with_ids(rows, espec_named)
        logger.info(f"Generated {len(tables[ename])} {ename} records")

    return tables


def _infer_parent_fk(espec: dict, parent: str) -> str | None:
    """The FK field whose relationship references the per_parent parent entity."""
    for fk_field, rel in espec.get("relationships", {}).items():
        if rel.get("ref") == parent:
            return fk_field
    return None


def check_referential_integrity(tables: dict[str, list[dict]], spec: dict) -> list[dict]:
    """Return one violation entry per FK value that doesn't resolve to a parent id."""
    violations: list[dict] = []
    for ename, espec in spec["entities"].items():
        for fk_field, rel in espec.get("relationships", {}).items():
            ref = rel["ref"]
            ref_id_field = id_field_of(spec["entities"][ref])
            ref_ids = {r.get(ref_id_field) for r in tables.get(ref, [])}
            for i, row in enumerate(tables.get(ename, [])):
                val = row.get(fk_field)
                if val is None or val not in ref_ids:
                    violations.append(
                        {"entity": ename, "row": i, "fk": fk_field, "value": val, "ref": ref}
                    )
    return violations


def validate_entities(tables: dict[str, list[dict]], spec: dict) -> dict[str, dict]:
    """Per-entity schema validation of the FINAL records (content + id + FK columns)."""
    results: dict[str, dict] = {}
    for ename, espec in spec["entities"].items():
        schema = build_schema(entity_full_spec(ename, espec, spec))
        valid, errors = validate_records(tables.get(ename, []), schema)
        results[ename] = {"valid": len(valid), "errors": len(errors)}
    return results


def _duck_type(ftype: str) -> str:
    return {"int": "BIGINT", "date": "DATE"}.get(ftype, "VARCHAR")


def export_duckdb(tables: dict[str, list[dict]], spec: dict, out_dir, timestamp: str):
    """Write the linked tables to a single queryable .duckdb file — a real relational
    schema with a PRIMARY KEY on each entity's id and FOREIGN KEY constraints on its
    relationship columns. Tables are created parents-first so the FK targets exist.
    Lets other projects load a ready fixture and JOIN across it with no glue code.
    """
    import duckdb
    import pandas as pd
    from pathlib import Path

    path = Path(out_dir) / f"{spec['name']}_{timestamp}.duckdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    con = duckdb.connect(str(path))
    try:
        for ename in generation_order(spec):
            espec = spec["entities"][ename]
            full = entity_full_spec(ename, espec, spec)
            id_field = id_field_of(espec)
            cols = list(full["fields"].keys())

            coldefs = []
            for c in cols:
                t = _duck_type(full["fields"][c].get("type", "str"))
                coldefs.append(f'"{c}" {t}' + (" PRIMARY KEY" if c == id_field else ""))
            for fk, rel in espec.get("relationships", {}).items():
                ref_id = id_field_of(spec["entities"][rel["ref"]])
                coldefs.append(f'FOREIGN KEY ("{fk}") REFERENCES "{rel["ref"]}"("{ref_id}")')
            con.execute(f'CREATE TABLE "{ename}" ({", ".join(coldefs)})')

            df = pd.DataFrame(tables.get(ename, [])).reindex(columns=cols)
            con.register("df_tmp", df)
            select = ", ".join(
                f'CAST("{c}" AS {_duck_type(full["fields"][c].get("type", "str"))}) AS "{c}"'
                for c in cols
            )
            con.execute(f'INSERT INTO "{ename}" SELECT {select} FROM df_tmp')
            con.unregister("df_tmp")
    finally:
        con.close()

    logger.info(f"Wrote DuckDB ({len(tables)} tables, PK+FK constraints): {path}")
    return path


def evaluate_entities(tables: dict[str, list[dict]], spec: dict, qcfg,
                      model: str | None = None) -> dict[str, dict]:
    """Per-entity quality pass. Runs the reusable harness on each entity that opts in
    via `text_field` / `label_fields` / `judge`.

    IMPORTANT: dedup is FLAG-ONLY here — it counts near-duplicate pairs but never removes
    rows. Deleting a row would either orphan its children (FK-target entity) or drop a
    required child, breaking the referential integrity guaranteed at generation time.
    """
    results: dict[str, dict] = {}
    for ename, espec in spec["entities"].items():
        rows = tables.get(ename, [])
        text_field = espec.get("text_field")
        label_fields = espec.get("label_fields", [])
        entry: dict = {}

        if text_field:
            text_fn = lambda r, tf=text_field: r.get(tf, "")
            pairs = find_near_duplicates(rows, text_fn, qcfg.dedup_threshold)
            entry["near_duplicate_pairs"] = len(pairs)   # flagged, not removed
            entry["diversity"] = diversity_report(rows, text_fn, label_fields, qcfg.ngram_n)
        elif label_fields:
            entry["labels"] = {
                f: {"distribution": label_distribution(rows, f),
                    "balance": normalized_entropy(label_distribution(rows, f))}
                for f in label_fields
            }

        judge_cfg = espec.get("judge")
        if judge_cfg and qcfg.run_judge:
            entry["judge"] = judge_dataset(rows, judge_cfg, qcfg.judge_sample_size, model=model)

        if entry:
            results[ename] = entry
    return results


def build_relational_report(tables: dict, spec: dict, violations: list[dict],
                            validation: dict[str, dict],
                            quality: dict[str, dict] | None = None) -> str:
    """Compact markdown summary of a relational generation run."""
    lines = [f"# {spec['name']} — Relational Quality Report", ""]
    lines.append(f"- Entities: {len(spec['entities'])}")
    lines.append(f"- Referential integrity: {'PASS' if not violations else f'{len(violations)} VIOLATIONS'}")
    lines += ["", "## Entities", "", "| entity | rows | valid | errors |", "|---|---|---|---|"]
    for ename in tables:
        v = validation.get(ename, {})
        lines.append(f"| {ename} | {len(tables[ename])} | {v.get('valid', 0)} | {v.get('errors', 0)} |")
    if violations:
        lines += ["", "## FK violations (first 20)", "", "| entity | row | fk | value | ref |", "|---|---|---|---|---|"]
        for v in violations[:20]:
            lines.append(f"| {v['entity']} | {v['row']} | {v['fk']} | {v['value']} | {v['ref']} |")

    if quality:
        lines += [
            "", "## Quality (per entity)",
            "_Dedup is flag-only in relational mode — near-duplicates are counted, never removed,"
            " to preserve referential integrity._", "",
            "| entity | near-dup pairs | distinct-n | mean pairwise dist | judge mean |",
            "|---|---|---|---|---|",
        ]
        for ename, q in quality.items():
            div = q.get("diversity", {})
            distinct = next((v for k, v in div.items() if k.startswith("distinct_")), "—")
            mpd = div.get("mean_pairwise_distance", "—")
            judge_mean = q.get("judge", {}).get("overall_mean", "—")
            ndp = q.get("near_duplicate_pairs", "—")
            lines.append(f"| {ename} | {ndp} | {distinct} | {mpd} | {judge_mean} |")
    return "\n".join(lines) + "\n"
