"""Aggregation and rendering of benchmark results (JSON / CSV / Markdown)."""

from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence

# Metric name -> (pretty label, "lower"/"higher" is better). Metrics not listed
# still show up in the CSV/JSON, just without a direction arrow.
METRIC_DIRECTIONS: Dict[str, str] = {
    "num_scenes": "neutral",
    "inference_seconds": "lower",
    "seconds_per_view": "lower",
    "peak_memory_mb": "lower",
    "cloud/valid_ratio": "higher",
    "gt/covered_ratio": "higher",
    "consistency/rel_depth_err_median": "lower",
    "consistency/rel_depth_err_mean": "lower",
    "consistency/inlier_1.03": "higher",
    "consistency/inlier_1.05": "higher",
    "consistency/inlier_1.25": "higher",
    "consistency/overlap_ratio": "neutral",
    "sanity/pose_pointmap_rel_err": "lower",
    "sanity/depth_pointmap_rel_err": "lower",
    "cam0/mae": "lower",
    "cam0/median_ae": "lower",
    "cam0/rel_ae": "lower",
    "cam0/delta_1.03": "higher",
    "cam0/delta_1.05": "higher",
    "cam0/delta_1.25": "higher",
    "cam0/inlier_l2_0.02": "higher",
    "cam0/inlier_l2_0.05": "higher",
    "cam0/inlier_l2_0.1": "higher",
    "sim3/mae_rel_extent": "lower",
    "sim3/mae_lsq_rel_extent": "lower",
    "sim3/median_ae_lsq_rel_extent": "lower",
    "sim3/accuracy_median_rel_extent": "lower",
    "sim3/completeness_median_rel_extent": "lower",
    "sim3/chamfer_mean_rel_extent": "lower",
    "metric/scale_abs_rel": "lower",
    "depth/abs_rel": "lower",
    "depth/delta_1.03": "higher",
    "depth/delta_1.25": "higher",
    "pose/rot_err_deg_median": "lower",
    "pose/trans_ang_err_deg_median": "lower",
    "pose/rra_5deg": "higher",
    "pose/rta_5deg": "higher",
    "pose/ate_rmse_rel": "lower",
}

# num_scenes leads every table: two models are only comparable when they
# completed the same scenes, and a silent difference there is the easiest way
# to read a benchmark backwards.
RUNTIME_COLUMNS = (
    "num_scenes",
    "inference_seconds",
    "seconds_per_view",
    "peak_memory_mb",
)
GT_FREE_COLUMNS = (
    "num_scenes",
    "cloud/valid_ratio",
    "consistency/overlap_ratio",
    "consistency/rel_depth_err_median",
    "consistency/inlier_1.03",
    "consistency/inlier_1.05",
    "consistency/inlier_1.25",
    "sanity/pose_pointmap_rel_err",
)
GT_COLUMNS = (
    "num_scenes",
    # Coverage belongs next to accuracy: every accuracy number below is
    # computed only on the pixels the model predicted, so a model can always
    # trade coverage for accuracy.
    "gt/covered_ratio",
    "cam0/rel_ae",
    "cam0/delta_1.03",
    "cam0/delta_1.25",
    "cam0/inlier_l2_0.05",
    "sim3/mae_rel_extent",
    "sim3/accuracy_median_rel_extent",
    "sim3/completeness_median_rel_extent",
    "depth/abs_rel",
    "depth/delta_1.25",
    "pose/rot_err_deg_median",
    "pose/trans_ang_err_deg_median",
    "pose/ate_rmse_rel",
    "metric/scale_abs_rel",
)


def _finite(values: Iterable[Optional[float]]) -> List[float]:
    out = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            out.append(number)
    return out


def aggregate_by_model(records: Sequence[dict]) -> Dict[str, Dict[str, float]]:
    """Average every numeric metric over the scenes each model completed."""
    per_model: Dict[str, Dict[str, List[float]]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        bucket = per_model.setdefault(record["model"], {})
        for key, value in record.items():
            if key in ("model", "scene", "status", "reason"):
                continue
            bucket.setdefault(key, []).append(value)

    aggregated: Dict[str, Dict[str, float]] = {}
    for model, metrics in per_model.items():
        row: Dict[str, float] = {}
        for key, values in metrics.items():
            finite = _finite(values)
            if finite:
                row[key] = sum(finite) / len(finite)
        row["num_scenes"] = float(
            len([r for r in records if r["model"] == model and r["status"] == "ok"])
        )
        aggregated[model] = row
    return aggregated


def _scene_coverage_warning(records: Sequence[dict]) -> Optional[str]:
    """Warn when the models were not averaged over the same set of scenes.

    Averages taken over different scenes are not comparable, and nothing else
    in the report would reveal it.
    """
    completed: Dict[str, set] = {}
    for record in records:
        completed.setdefault(record["model"], set())
        if record.get("status") == "ok":
            completed[record["model"]].add(record.get("scene"))
    non_empty = {m: sc for m, sc in completed.items() if sc}
    if len(non_empty) < 2 or len(set(map(frozenset, non_empty.values()))) == 1:
        return None
    detail = ", ".join(
        f"{model} ({len(scenes)})" for model, scenes in sorted(non_empty.items())
    )
    return (
        "Not comparable as-is: the models completed different scenes, so the "
        f"averages below are taken over different data - {detail}. Re-run the "
        "failures, or filter results.csv to the scenes every model completed."
    )


def _format(value: Optional[float]) -> str:
    if value is None:
        return "-"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return "n/a"
    if number != 0 and (abs(number) < 1e-3 or abs(number) >= 1e5):
        return f"{number:.2e}"
    return f"{number:.4g}"


def _header(column: str) -> str:
    direction = METRIC_DIRECTIONS.get(column)
    if direction == "lower":
        return f"{column} ↓"
    if direction == "higher":
        return f"{column} ↑"
    return column


def _markdown_table(
    aggregated: Dict[str, Dict[str, float]], columns: Sequence[str]
) -> str:
    present = [c for c in columns if any(c in row for row in aggregated.values())]
    if not present:
        return "_No data._\n"

    lines = ["| model | " + " | ".join(_header(c) for c in present) + " |"]
    lines.append("| --- |" + " --- |" * len(present))
    for model in sorted(aggregated):
        row = aggregated[model]
        cells = [_format(row.get(column)) for column in present]
        lines.append(f"| {model} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def write_reports(
    records: Sequence[dict],
    output_dir: str,
    title: str = "3D point-map benchmark",
    extra_notes: Sequence[str] = (),
) -> Dict[str, str]:
    """Write ``results.json``, ``results.csv`` and ``report.md``.

    Returns:
        Mapping from artefact name to the path written.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = {
        "json": os.path.join(output_dir, "results.json"),
        "csv": os.path.join(output_dir, "results.csv"),
        "markdown": os.path.join(output_dir, "report.md"),
    }

    with open(paths["json"], "w", encoding="utf-8") as handle:
        json.dump(list(records), handle, indent=2, sort_keys=True, default=str)

    fieldnames: List[str] = []
    for record in records:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(paths["csv"], "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    aggregated = aggregate_by_model(records)
    failures = [r for r in records if r.get("status") != "ok"]

    sections = [f"# {title}\n"]
    for note in extra_notes:
        sections.append(f"> {note}\n")
    mismatch = _scene_coverage_warning(records)
    if mismatch:
        sections.append(f"> **{mismatch}**\n")

    sections.append("\n## Runtime and memory\n")
    sections.append(_markdown_table(aggregated, RUNTIME_COLUMNS))

    sections.append("\n## Ground-truth-free consistency\n")
    sections.append(
        "Points of each view are reprojected into every other view and the "
        "depths compared. Scale-invariant, so metric and non-metric models are "
        "directly comparable.\n\n"
    )
    sections.append(_markdown_table(aggregated, GT_FREE_COLUMNS))

    if any(any(c in row for c in GT_COLUMNS) for row in aggregated.values()):
        sections.append("\n## Ground-truth accuracy\n")
        sections.append(
            "`cam0/*` compares in the first camera's frame after normalising "
            "each cloud by its own average point distance (pose errors count). "
            "`sim3/*` first aligns the prediction to the GT with a similarity "
            "transform (shape only). `metric/*` is only meaningful for models "
            "that predict real-world scale.\n\n"
        )
        sections.append(_markdown_table(aggregated, GT_COLUMNS))

    if failures:
        sections.append("\n## Skipped / failed runs\n")
        sections.append("| model | scene | status | reason |")
        sections.append("| --- | --- | --- | --- |")
        for record in failures:
            reason = str(record.get("reason", "")).replace("\n", " ")[:200]
            sections.append(
                f"| {record.get('model')} | {record.get('scene')} | "
                f"{record.get('status')} | {reason} |"
            )
        sections.append("")

    with open(paths["markdown"], "w", encoding="utf-8") as handle:
        handle.write("\n".join(sections))

    return paths
