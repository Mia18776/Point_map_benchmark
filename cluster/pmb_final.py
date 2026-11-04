"""Shared helpers for the scripts that turn results under pmb_runs/final/ into reports.

Datasets are discovered from final/main/ rather than listed by hand, so a new
dataset linked there by finalize.py shows up in every report automatically.
"""

import json
import os

import pandas as pd

RUNS = "/home/woody/iwfa/iwfa137h/pmb_runs"
FINAL = os.path.join(RUNS, "final")

# wai_dataset -> (short folder name, English label, Chinese description)
DATASETS = {
    "eth3d": ("eth3d", "ETH3D", "真实场景"),
    "scannetpp": ("scannetpp", "ScanNet++ v2", "真实室内场景"),
    "tav2_wb": ("tav2", "TartanAirV2-WB", "合成场景"),
}
ORDER = list(DATASETS)

MODEL_NAMES = {"mapanything": "MapAnything", "mapanything_apache": "MapAnything (Apache)",
               "mast3r": "MASt3R", "pi3": "Pi3", "pi3x": "Pi3-X", "vggt": "VGGT", "da3": "DA3"}


def run_config(path):
    """The run configuration of a result folder (finished or killed)."""
    for name, key in (("results.json", "config"), ("run_config.json", None)):
        p = os.path.join(path, name)
        if os.path.exists(p):
            data = json.load(open(p, encoding="utf-8"))
            return data[key] if key else data
    return {}


def ok_rows(path):
    df = pd.read_csv(os.path.join(path, "results.csv"))
    return df[df.status == "ok"]


def main_datasets():
    """``[(folder name, wai_dataset, label, chinese, n_sets)]`` under final/main, in ORDER."""
    found = []
    main_dir = os.path.join(FINAL, "main")
    for name in sorted(os.listdir(main_dir)):
        path = os.path.join(main_dir, name)
        if not os.path.exists(os.path.join(path, "results.csv")):
            continue
        dataset = run_config(path).get("wai_dataset") or name.split("_")[0]
        _, label, chinese = DATASETS.get(dataset, (dataset, dataset, ""))
        n_sets = ok_rows(path).scene.nunique()
        found.append((name, dataset, label, chinese, n_sets))
    return sorted(found, key=lambda d: ORDER.index(d[1]) if d[1] in ORDER else len(ORDER))


def final_rows(name):
    """Results at the default settings: an unmasked MapAnything run, where one is
    linked under final/mapanything_nomask/<name>, replaces the masked one."""
    rows = ok_rows(os.path.join(FINAL, "main", name))
    nomask_path = os.path.join(FINAL, "mapanything_nomask", name)
    if os.path.exists(os.path.join(nomask_path, "results.csv")):
        nomask = ok_rows(nomask_path)
        rows = pd.concat([rows[~rows.model.isin(nomask.model.unique())], nomask])
    return rows
