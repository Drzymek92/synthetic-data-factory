from dataclasses import dataclass, field
from pathlib import Path

import yaml

from scripts.domain import load_domain


@dataclass
class GenConfig:
    n_records: int = 50
    batch_size: int = 5
    temperature: float = 0.9
    seed: int = 42
    max_retries: int = 3
    # Pool-and-recombine defaults for `generate: llm` entities (relational mode). The LLM
    # authors a small pool, then Python recombines it up to the required count — amortizing
    # the per-row API cost. Standardised here so every project/domain inherits the policy;
    # a per-entity `pool:` block overrides any of these (or `pool: false` opts out).
    pool_fraction: float = 0.10   # author >= this share of the required count with the LLM
    pool_min_size: int = 20       # never fewer than this many real LLM rows (small-N safety)
    pool_recombine: str = "sample"  # "sample" = whole-row (coherent) | "shuffle" = per-field


@dataclass
class QualityConfig:
    dedup_threshold: float = 0.92
    ngram_n: int = 2
    run_judge: bool = True
    judge_sample_size: int | None = None
    min_quality_score: float = 3.5


@dataclass
class ExportConfig:
    csv: bool = True
    chat_jsonl: bool = True


@dataclass
class AppConfig:
    domain: str
    gen: GenConfig
    quality: QualityConfig
    export: ExportConfig
    model: str | None
    output_dir: str
    domain_spec: dict = field(default_factory=dict)

    # Convenience pass-throughs to the active domain spec.
    @property
    def text_field(self) -> str:
        return self.domain_spec["text_field"]

    @property
    def label_fields(self) -> list[str]:
        return list(self.domain_spec["label_fields"])


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def load_config(
    config_path: str | Path = "config/default.yaml",
    overrides: dict | None = None,
) -> AppConfig:
    """Load default.yaml + the active domain spec, then apply CLI overrides.

    `overrides` keys (all optional): domain, n_records, run_judge, judge_sample_size,
    model. Only set values override the file; None/absent are ignored.
    """
    overrides = {k: v for k, v in (overrides or {}).items() if v is not None}
    raw = _load_yaml(config_path)

    gen_raw = raw.get("generation", {})
    qual_raw = raw.get("quality", {})
    exp_raw = raw.get("export", {})

    domain_name = overrides.get("domain", raw.get("domain", "support_tickets"))
    domain_spec = load_domain(domain_name)

    pool_raw = gen_raw.get("pool", {}) or {}
    gen = GenConfig(
        n_records=overrides.get("n_records", gen_raw.get("n_records", GenConfig.n_records)),
        batch_size=gen_raw.get("batch_size", GenConfig.batch_size),
        temperature=gen_raw.get("temperature", GenConfig.temperature),
        seed=gen_raw.get("seed", GenConfig.seed),
        max_retries=gen_raw.get("max_retries", GenConfig.max_retries),
        pool_fraction=pool_raw.get("fraction", GenConfig.pool_fraction),
        pool_min_size=pool_raw.get("min_size", GenConfig.pool_min_size),
        pool_recombine=pool_raw.get("recombine", GenConfig.pool_recombine),
    )
    quality = QualityConfig(
        dedup_threshold=qual_raw.get("dedup_threshold", QualityConfig.dedup_threshold),
        ngram_n=qual_raw.get("ngram_n", QualityConfig.ngram_n),
        run_judge=overrides.get("run_judge", qual_raw.get("run_judge", QualityConfig.run_judge)),
        judge_sample_size=overrides.get("judge_sample_size", qual_raw.get("judge_sample_size")),
        min_quality_score=qual_raw.get("min_quality_score", QualityConfig.min_quality_score),
    )
    export = ExportConfig(
        csv=exp_raw.get("csv", ExportConfig.csv),
        chat_jsonl=exp_raw.get("chat_jsonl", ExportConfig.chat_jsonl),
    )

    return AppConfig(
        domain=domain_name,
        gen=gen,
        quality=quality,
        export=export,
        model=overrides.get("model", (raw.get("model") or {}).get("name")),
        output_dir=(raw.get("paths") or {}).get("output_dir", "scripts/outputs"),
        domain_spec=domain_spec,
    )
