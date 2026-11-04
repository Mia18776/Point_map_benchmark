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
        block.update(key=key, label=en, kind=cn, sets=int(n_sets))
        blocks.append(block)
    data = {"datasets": blocks, "crosscheck": crosscheck(datasets)}
    template = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_template.html"),
                    encoding="utf-8").read()
    html = template.replace("/*__DATA__*/null", json.dumps(data, ensure_ascii=False))
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(OUT)


if __name__ == "__main__":
    main()
