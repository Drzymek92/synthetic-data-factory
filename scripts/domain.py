from enum import Enum
from pathlib import Path
from typing import Optional, Type

import yaml
from pydantic import BaseModel, Field, create_model

_STR_CONSTRAINTS = ("min_length", "max_length")
_INT_CONSTRAINTS = ("ge", "le")


def load_domain(name: str, domains_dir: Path = Path("config/domains")) -> dict:
    path = domains_dir / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Domain spec not found: {path}")
    spec = yaml.safe_load(path.read_text(encoding="utf-8"))
    if is_relational(spec):
        _require(spec, ["name", "entities"])
        _validate_relationships(spec)
    else:
        _require(spec, ["name", "fields", "text_field", "label_fields"])
    return spec


def is_relational(spec: dict) -> bool:
    """A relational (multi-table) spec declares an `entities` block instead of a
    top-level flat `fields` schema."""
    return isinstance(spec, dict) and "entities" in spec


def id_field_of(entity_spec: dict) -> str:
    return entity_spec.get("id_field", "id")


def _validate_relationships(spec: dict) -> None:
    """Fail fast on a spec whose relationships/per_parent point at unknown entities."""
    entities = spec["entities"]
    for ename, espec in entities.items():
        for fk, rel in espec.get("relationships", {}).items():
            ref = rel.get("ref")
            if ref not in entities:
                raise ValueError(
                    f"Entity '{ename}' relationship '{fk}' references unknown entity '{ref}'"
                )
        pp = espec.get("per_parent")
        if pp and pp.get("parent") not in entities:
            raise ValueError(
                f"Entity '{ename}' per_parent references unknown parent '{pp.get('parent')}'"
            )


def entity_content_spec(entity_name: str, entity_spec: dict) -> dict:
    """A flat, `build_schema`-compatible view of an entity's LLM-generated CONTENT
    fields only. Excludes `id` and FK fields — those are assigned in Python."""
    return {
        "name": entity_name,
        "fields": entity_spec.get("fields", {}),
        "generation": entity_spec.get("generation", {}),
        "text_field": entity_spec.get("text_field"),
        "label_fields": entity_spec.get("label_fields", []),
    }


def entity_full_spec(entity_name: str, entity_spec: dict, spec: dict | None = None) -> dict:
    """A flat spec for validating FINAL records: content fields plus the `id`, every FK
    field, and every `copy` (denormalized) field, so all columns are schema-checked.

    Copy-field types are resolved from the source entity's field definition when the full
    `spec` is provided (so a copied `int` price validates as an int), else default to str.
    """
    fields = dict(entity_spec.get("fields", {}))
    fields[id_field_of(entity_spec)] = {"type": "str", "min_length": 1}
    rels = entity_spec.get("relationships", {})
    for fk in rels:
        fields.setdefault(fk, {"type": "str", "min_length": 1})
    for tgt, cdef in entity_spec.get("copy", {}).items():
        fdef: dict = {"type": "str"}
        if spec is not None:
            ref = rels.get(cdef["from"], {}).get("ref")
            src = (spec.get("entities", {}).get(ref, {}).get("fields", {}) or {}).get(cdef["field"])
            if src:
                fdef = dict(src)
        fields.setdefault(tgt, fdef)
    return {
        "name": entity_name,
        "fields": fields,
        "text_field": entity_spec.get("text_field"),
        "label_fields": entity_spec.get("label_fields", []),
    }


def _require(spec: dict, keys: list[str]) -> None:
    missing = [k for k in keys if k not in spec]
    if missing:
        raise ValueError(f"Domain spec '{spec.get('name', '?')}' missing keys: {missing}")


def _enum_values(values: list) -> list[str]:
    """Coerce enum values to str so a domain spec doesn't crash the schema build.

    PyYAML parses unquoted YAML 1.1 booleans (Yes/No, On/Off, True/False) as Python
    bools, which break both Enum construction and the prompt's str join. Coercing to
    str prevents the crash; note it can't recover the intended label (str(True) ==
    "True", not "Yes") — quote such enum values in the YAML to keep them verbatim.
    """
    return [str(v) for v in values]


def _enum_for(field_name: str, values: list) -> Type[Enum]:
    cls_name = "".join(p.capitalize() for p in field_name.split("_"))
    return Enum(cls_name, {v: v for v in _enum_values(values)}, type=str)


def build_schema(spec: dict) -> Type[BaseModel]:
    """Build a Pydantic model from a domain spec's `fields` block at runtime."""
    fields: dict = {}
    for fname, fdef in spec["fields"].items():
        ftype = fdef.get("type", "str")
        optional = fdef.get("optional", False)

        if ftype == "enum":
            base = _enum_for(fname, fdef["values"])
            constraints: dict = {}
        elif ftype == "int":
            base = int
            constraints = {k: fdef[k] for k in _INT_CONSTRAINTS if k in fdef}
        elif ftype == "date":  # ISO date string (validated as a non-empty str)
            base = str
            constraints = {"min_length": 1}
        else:  # str (default)
            base = str
            constraints = {k: fdef[k] for k in _STR_CONSTRAINTS if k in fdef}

        if optional:
            fields[fname] = (Optional[base], Field(default=None, **constraints))
        else:
            fields[fname] = (base, Field(**constraints))

    model_name = "".join(p.capitalize() for p in spec["name"].split("_"))
    return create_model(model_name, **fields)


def field_spec_text(spec: dict) -> str:
    """Render the field list into the human-readable block injected into the prompt."""
    lines = ["Each record is an object with these fields:"]
    for fname, fdef in spec["fields"].items():
        ftype = fdef.get("type", "str")
        desc = fdef.get("description", "")
        if ftype == "enum":
            detail = f"one of: {', '.join(_enum_values(fdef['values']))}"
        elif ftype == "int":
            bounds = [f"{k} {fdef[k]}" for k in _INT_CONSTRAINTS if k in fdef]
            detail = "integer" + (f", {', '.join(bounds)}" if bounds else "")
        elif ftype == "date":
            detail = "ISO date string (YYYY-MM-DD)"
        else:
            bounds = []
            if "min_length" in fdef or "max_length" in fdef:
                bounds.append(f"{fdef.get('min_length', 0)}-{fdef.get('max_length', '?')} chars")
            if fdef.get("optional"):
                bounds.append("or null")
            detail = "string" + (f", {', '.join(bounds)}" if bounds else "")
        suffix = f": {desc}" if desc else ""
        lines.append(f"- {fname} ({detail}){suffix}")
    return "\n".join(lines)
