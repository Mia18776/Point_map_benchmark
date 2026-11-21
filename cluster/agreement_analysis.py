#!/usr/bin/env python3
"""Can the models' agreement with each other tell, without ground truth, which one to trust?

For every set, each model's cameras are compared with every other model's on
the same views: for each pair of views, the angle between the two models'
relative rotations and between their relative translation directions; the
larger of the two, averaged over view pairs, is the disagreement of the two
models on that set. A model's "outlier score" on a set is its median
disagreement with the other models - low means it agrees with the majority.

Ground truth is used only to check the idea: does a low outlier score go with
a low pose error, and does picking the lowest-outlier model per set beat the
best single model?

    python ~/pmb_jobs/agreement_analysis.py   # writes final/AGREEMENT.md and prints it
"""

import glob
import os
import re

import numpy as np
import pandas as pd

RUNS = "/home/woody/iwfa/iwfa137h/pmb_runs"
LOGS = os.path.expanduser("~/pmb_jobs/logs")
OUT = os.path.join(RUNS, "final", "AGREEMENT.md")


def relative(poses):
    """Relative camera-to-camera transforms for all view pairs i < j."""
    inv = np.linalg.inv(poses)
    idx = [(i, j) for i in range(len(poses)) for j in range(i + 1, len(poses))]
    return np.stack([inv[i] @ poses[j] for i, j in idx])


def angle_between_rotations(a, b):
    r = np.einsum("nij,nkj->nik", a, b)  # a @ b^T
    cos = np.clip((np.trace(r, axis1=1, axis2=2) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos))


def angle_between_directions(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1)))


def disagreement(rel_a, rel_b):
    rot = angle_between_rotations(rel_a[:, :3, :3], rel_b[:, :3, :3])
    trans = angle_between_directions(rel_a[:, :3, 3], rel_b[:, :3, 3])
    return float(np.mean(np.maximum(rot, trans)))


def job_dirs(prefix):
    """model -> run folder, from logs/<prefix>-<model>-<jobid>.out."""
    out = {}
    for log in glob.glob(os.path.join(LOGS, f"{prefix}-*.out")):
        m = re.match(rf"{re.escape(prefix)}-(.+)-(\d+)\.out$", os.path.basename(log))
        if m and os.path.isdir(os.path.join(RUNS, m.group(2), "poses")):
            out[m.group(1)] = os.path.join(RUNS, m.group(2))
    return out


def analyse(dataset, prefix, summary=None):
    runs = job_dirs(prefix)
    scores = pd.concat(
        [pd.read_csv(os.path.join(d, "results.csv")).assign(model=m) for m, d in runs.items()]
    )
    scores = scores[scores.status == "ok"].set_index(["scene", "model"])
    models = sorted(runs)
    rows = []
    # Only sets every model finished and saved poses for: comparing a model
    # with "the others" needs all of them on the same views.
    scenes = sorted(
        set.intersection(*(
            {os.path.basename(p)[:-4] for p in glob.glob(os.path.join(runs[m], "poses", m, "*.npz"))}
            for m in models
        ))
        & set.intersection(*(set(scores.xs(m, level="model").index) for m in models))
    )
    for scene in scenes:
        rel = {}
        for m in models:
            data = np.load(os.path.join(runs[m], "poses", m, f"{scene}.npz"))
            rel[m] = relative(data["pred_poses_c2w"])
        for m in models:
            others = [disagreement(rel[m], rel[o]) for o in models if o != m]
            rows.append({"scene": scene, "model": m, "outlier": float(np.median(others)),
                         "auc": scores.loc[(scene, m), "pose/auc_5"],
                         "consistency": scores.loc[(scene, m), "consistency/inlier_1.03"]})
    df = pd.DataFrame(rows)

    lines = [f"## {dataset} ({df.scene.nunique()} sets, {len(models)} models)", ""]
    rho = df.groupby("model").apply(lambda x: x["outlier"].corr(x["auc"], method="spearman")).round(2)
    rho_all = df["outlier"].corr(df["auc"], method="spearman")
    lines += ["Spearman correlation between a model's outlier score and its pose AUC@5 on the "
              f"same set (negative = disagreeing with the others goes with worse poses): "
              f"all {rho_all:.2f}; per model " + ", ".join(f"{m} {v}" for m, v in rho.items()) + ".", ""]

    piv_auc = df.pivot(index="scene", columns="model", values="auc")
    piv_out = df.pivot(index="scene", columns="model", values="outlier")
    piv_con = df.pivot(index="scene", columns="model", values="consistency")
    pick_agree = piv_out.idxmin(axis=1)
    pick_cons = piv_con.idxmax(axis=1)
    best_model = piv_auc.mean().idxmax()
    table = pd.DataFrame([
        {"strategy": f"always the best single model ({best_model})", "mean pose AUC@5": piv_auc[best_model].mean()},
        {"strategy": "per set: the model that agrees most with the others", "mean pose AUC@5": np.mean([piv_auc.loc[s, m] for s, m in pick_agree.items()])},
        {"strategy": "per set: the most self-consistent model", "mean pose AUC@5": np.mean([piv_auc.loc[s, m] for s, m in pick_cons.items()])},
        {"strategy": "per set: the truly best model (oracle, needs GT)", "mean pose AUC@5": piv_auc.max(axis=1).mean()},
    ]).round(1)
    lines += ["| strategy | mean pose AUC@5 |", "| --- | --- |"]
    lines += [f"| {r.strategy} | {r._2} |" for r in table.itertuples()]
    lines += ["", "Models picked by agreement: " + ", ".join(f"{m} {n}" for m, n in pick_agree.value_counts().items()), ""]

    # Failure detection: the model with the highest outlier score in a set - is it the worst?
    worst_true = piv_auc.idxmin(axis=1)
    worst_flag = piv_out.idxmax(axis=1)
    hit = (worst_true == worst_flag).mean()
    lines += [f"The most disagreeing model is the set's worst model in {hit:.0%} of sets "
              f"(chance: {1 / len(models):.0%}).", ""]
    if summary is not None:
        summary[dataset] = {
            "rho": round(float(rho_all), 2), "worst_hit": round(float(hit) * 100),
            "chance": round(100 / len(models)), "sets": int(df.scene.nunique()),
            "strategies": [{"name": r.strategy, "auc": float(r._2)} for r in table.itertuples()],
        }
    return lines


def main():
    lines = ["# Cross-model agreement as a ground-truth-free signal", "",
             "Generated by `~/pmb_jobs/agreement_analysis.py` from the poses saved by the",
             "run2 jobs (images only, default settings).", ""]
    lines += analyse("ETH3D", "run2-eth3d")
    lines += analyse("TartanAirV2-WB", "run2-tav2_wb")
    text = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


if __name__ == "__main__":
    main()
