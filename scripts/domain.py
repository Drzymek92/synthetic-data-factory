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
    _require(spec, ["name", "fields", "text_field", "label_fields"])
    return spec


def _require(spec: dict, keys: list[str]) -> None:
    missing = [k for k in keys if k not in spec]
    if missing:
        raise ValueError(f"Domain spec '{spec.get('name', '?')}' missing keys: {missing}")


def _enum_for(field_name: str, values: list[str]) -> Type[Enum]:
    cls_name = "".join(p.capitalize() for p in field_name.split("_"))
    return Enum(cls_name, {v: v for v in values}, type=str)


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
            detail = f"one of: {', '.join(fdef['values'])}"
        elif ftype == "int":
            bounds = [f"{k} {fdef[k]}" for k in _INT_CONSTRAINTS if k in fdef]
            detail = "integer" + (f", {', '.join(bounds)}" if bounds else "")
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
