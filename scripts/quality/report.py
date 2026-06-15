from datetime import datetime
from pathlib import Path


def _bar(fraction: float, width: int = 20) -> str:
    filled = round(max(0.0, min(1.0, fraction)) * width)
    return "█" * filled + "░" * (width - filled)


def build_report(metrics: dict, title: str = "Dataset Quality Report") -> str:
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"_Generated {metrics.get('generated_at', '')}_")
    lines.append("")

    summ = metrics.get("summary", {})
    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    for k, v in summ.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    val = metrics.get("validation", {})
    if val:
        lines.append("## Validation")
        lines.append("")
        lines.append(f"- Valid records: **{val.get('valid', 0)}**")
        lines.append(f"- Rejected (schema): **{val.get('errors', 0)}**")
        coverage = val.get("coverage", {})
        if coverage:
            lines.append("- Field coverage:")
            for f, c in coverage.items():
                lines.append(f"  - `{f}`: {_bar(c)} {c:.0%}")
        lines.append("")

    dup = metrics.get("dedup", {})
    if dup:
        lines.append("## Deduplication")
        lines.append("")
        lines.append(f"- Near-duplicate pairs found: **{dup.get('pairs', 0)}**")
        lines.append(f"- Records removed: **{dup.get('removed', 0)}**")
        lines.append(f"- Threshold (cosine): {dup.get('threshold', '')}")
        lines.append("")

    div = metrics.get("diversity", {})
    if div:
        lines.append("## Diversity")
        lines.append("")
        for key in [k for k in div if k.startswith("distinct_")]:
            lines.append(f"- {key}: **{div[key]}**")
        lines.append(f"- Vocabulary size: **{div.get('vocabulary_size', 0)}**")
        lines.append(f"- Mean pairwise distance: **{div.get('mean_pairwise_distance', 0)}**")
        labels = div.get("labels", {})
        for field, stats in labels.items():
            lines.append("")
            lines.append(f"### Label `{field}` (balance {stats.get('balance', 0)})")
            lines.append("")
            dist = stats.get("distribution", {})
            total = sum(dist.values()) or 1
            for label, count in sorted(dist.items(), key=lambda x: -x[1]):
                lines.append(f"- `{label}`: {_bar(count / total)} {count} ({count / total:.0%})")
        lines.append("")

    judge = metrics.get("judge", {})
    if judge and judge.get("judged"):
        lines.append("## LLM-as-Judge")
        lines.append("")
        lines.append(f"- Records judged: **{judge.get('judged', 0)}**")
        lines.append(f"- Overall mean score (1-5): **{judge.get('overall_mean', 0)}**")
        for dim, score in judge.get("means", {}).items():
            lines.append(f"  - {dim}: {score}")
        flagged = judge.get("flagged", [])
        if flagged:
            lines.append("")
            lines.append(f"### Flagged records ({len(flagged)})")
            lines.append("")
            display_field = metrics.get("display_field", "")
            dims = list(judge.get("means", {}).keys())
            for item in flagged[:10]:
                s = item.get("scores", {})
                issue = s.get("issue") or "low score"
                label = item.get("record", {}).get(display_field) or "(record)"
                dim_str = "/".join(f"{d[:1]}{s.get(d)}" for d in dims)
                lines.append(f"- **{label}** — {issue} ({dim_str})")
        lines.append("")

    verdict = metrics.get("verdict")
    if verdict:
        lines.append("## Verdict")
        lines.append("")
        lines.append(verdict)
        lines.append("")

    return "\n".join(lines)


def write_report(metrics: dict, out_dir: Path, timestamp: str, title: str = "Dataset Quality Report") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"quality_report_{timestamp}.md"
    metrics.setdefault("generated_at", datetime.now().isoformat(timespec="seconds"))
    path.write_text(build_report(metrics, title), encoding="utf-8")
    return path
