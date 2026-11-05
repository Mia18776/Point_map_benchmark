#!/usr/bin/env python3
"""Build pmb_runs/final/report.html - the results as a web page with explanations.

Every number is read from the same result files summarize_results.py uses;
nothing is typed in. Re-run after new results:

    python ~/pmb_jobs/make_report_html.py
"""

import glob
import json
import os

import numpy as np

from pmb_final import FINAL, MODEL_NAMES as NAMES, final_rows, main_datasets, ok_rows

OUT = os.path.join(FINAL, "report.html")
METRIC_MODELS = {"mapanything", "mapanything_apache", "mast3r"}


def fnum(x, d):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def dataset_block(name):
    ok = final_rows(name)
    models = []
    for model, part in ok.groupby("model"):
        auc = part["pose/auc_5"]
        models.append({
            "key": model, "name": NAMES.get(model, model), "sets": int(len(part)),
            "auc": fnum(auc.mean(), 1), "auc_se": fnum(auc.std() / len(auc) ** 0.5, 1),
            "rays": fnum(part["rays/err_deg"].mean(), 2),
            "acc": fnum(part["sim3/accuracy_median_rel_extent"].mean(), 4),
            "comp": fnum(part["sim3/completeness_median_rel_extent"].mean(), 4),
            "depth": fnum(part["depth/abs_rel"].mean(), 3),
            "scale": fnum(part["metric/scale_abs_rel"].mean(), 2) if model in METRIC_MODELS else None,
            "spv": fnum(part["seconds_per_view"].mean(), 3),
            "mem": fnum(part["peak_memory_mb"].mean() / 1024, 1),
        })
    models.sort(key=lambda m: -m["auc"])
    scenes = sorted(ok.wai_scene.unique())
    grid = ok.pivot_table(index="wai_scene", columns="model", values="pose/auc_5")
    heat = {m["key"]: [fnum(grid.loc[s, m["key"]], 0) for s in scenes] for m in models}
    return {"models": models, "scenes": scenes, "heat": heat}


OFFICIAL = [("official/pointmaps_abs_rel", 4), ("official/pointmaps_inlier_thres_103", 3),
            ("official/z_depth_abs_rel", 4), ("official/z_depth_inlier_thres_103", 3),
            ("official/pose_ate_rmse", 4), ("official/metric_scale_abs_rel", 3)]


def official_block(name):
    """Mean of each official/* metric per model (upstream's definitions)."""
    ok = final_rows(name)
    if "official/z_depth_abs_rel" not in ok:
        return None
    out = []
    for model, part in ok.groupby("model"):
        row = {"name": NAMES.get(model, model), "auc": fnum(part["pose/auc_5"].mean(), 1),
               "rays": fnum(part["rays/err_deg"].mean(), 3)}
        for col, d in OFFICIAL:
            value = part[col].mean() if col in part else None
            if col == "official/metric_scale_abs_rel" and model not in METRIC_MODELS:
                value = None  # upstream reports it for every model; meaningless here
            row[col.split("/")[1]] = fnum(value, d)
        out.append(row)
    return sorted(out, key=lambda r: -r["auc"])


def calibrated_block(name):
    """Images-only vs GT-intrinsics runs of the same sets, per model."""
    path = os.path.join(FINAL, "calibrated", name)
    if not os.path.exists(os.path.join(path, "results.csv")):
        return None
    cal, img = ok_rows(path), final_rows(name)
    out = []
    for model in sorted(cal.model.unique()):
        m = img[img.model == model].merge(cal[cal.model == model], on="scene", suffixes=("_i", "_c"))
        d = m["pose/auc_5_c"] - m["pose/auc_5_i"]
        out.append({
            "name": NAMES.get(model, model), "sets": int(len(m)),
            "auc_i": fnum(m["pose/auc_5_i"].mean(), 1), "auc_c": fnum(m["pose/auc_5_c"].mean(), 1),
            "delta": fnum(d.mean(), 1), "delta_se": fnum(d.std() / len(d) ** 0.5, 1),
            "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
            "rays_i": fnum(m["rays/err_deg_i"].mean(), 2), "rays_c": fnum(m["rays/err_deg_c"].mean(), 2),
            "depth_i": fnum(m["official/z_depth_abs_rel_i"].mean(), 3),
            "depth_c": fnum(m["official/z_depth_abs_rel_c"].mean(), 3),
        })
    return sorted(out, key=lambda r: -r["delta"])


def agreement_summary(datasets):
    """Cross-model agreement, from agreement_analysis.py (needs the saved poses)."""
    import agreement_analysis

    prefixes = {"eth3d": ("ETH3D", "run2-eth3d"), "tav2_wb": ("TartanAirV2-WB", "run2-tav2_wb")}
    summary = {}
    for _, key, _, _, _ in datasets:
        if key in prefixes:
            label, prefix = prefixes[key]
            try:
                agreement_analysis.analyse(label, prefix, summary)
            except (FileNotFoundError, KeyError, IndexError):
                continue  # no saved poses for this dataset
    return [{"label": k, **v} for k, v in summary.items()]


def crosscheck(datasets):
    rows = []
    eth3d = [d[0] for d in datasets if d[1] == "eth3d"]
    if not eth3d:
        return rows
    main = ok_rows(os.path.join(FINAL, "main", eth3d[0]))
    for path in sorted(glob.glob(os.path.join(FINAL, "official_crosscheck", "*"))):
        model = os.path.basename(path).split("_")[1]
        per_scene = json.load(open(glob.glob(os.path.join(path, "*per_scene_results.json"))[0]))
        ours = main[main.model == model]
        rows.append({
            "name": NAMES[model],
            "off_auc": round(float(np.mean([v for m in per_scene.values() for v in m["pose_auc_5"]])), 4),
            "our_auc": round(float(ours["pose/auc_5"].mean()), 4),
            "off_ray": round(float(np.mean([v for m in per_scene.values() for v in m["ray_dirs_err_deg"]])), 4),
            "our_ray": round(float(ours["rays/err_deg"].mean()), 4),
        })
    return rows


def main():
    datasets = main_datasets()
    blocks = []
    for name, key, en, cn, n_sets in datasets:
        block = dataset_block(name)
        block.update(key=key, label=en, kind=cn, sets=int(n_sets),
                     official=official_block(name), calibrated=calibrated_block(name))
        blocks.append(block)
    data = {"datasets": blocks, "crosscheck": crosscheck(datasets),
            "agreement": agreement_summary(datasets)}
    template = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html"),
                    encoding="utf-8").read()
    html = template.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(OUT)


if __name__ == "__main__":
    main()
