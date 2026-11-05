"""Metrics computed with map-anything's own functions.

These live in their own module because they need torch and the ``mapanything``
package, while :mod:`pointmap_bench.geometry` and :mod:`pointmap_bench.metrics`
stay NumPy-only so the bulk of the benchmark can be unit-tested without any of
the heavy dependencies installed.

Nothing here re-implements a definition, and that is the whole point: AUC and
the ray-direction error mean exactly what they mean in map-anything's
``benchmarking/dense_n_view`` benchmark, so a number produced here can be put
next to a number from their tables. Re-deriving a metric "the same way" is how
two benchmarks end up quietly measuring different things.

Covered here:

* ``pose/auc_{t}`` - AUC of the per-pair relative-pose error up to ``t``
  degrees, where the error of a pair is ``max(rotation_err, translation_err)``.
  Reported as a percentage, as upstream does.
* ``rays/err_deg`` - angular error between the predicted and GT unit ray
  directions, i.e. how wrong the recovered intrinsics are.

Note that upstream's translation term folds the direction ambiguity
(``min(a, |180 - a|)`` on top of an ``acos|cos|``), so it lives in [0, 90] and
treats a 180-degree-flipped baseline as correct. ``pose/trans_ang_err_deg_*``
in :mod:`pointmap_bench.metrics` is the stricter, sign-sensitive version. They
disagree on purpose; both are reported.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def _torch():
    """Import torch lazily so this module is importable without it."""
    import torch

    return torch


def available() -> bool:
    """Whether map-anything's metric helpers can be imported here."""
    try:  # pragma: no cover - trivial import probe
        import torch  # noqa: F401
        from mapanything.utils.geometry import get_rays_in_camera_frame  # noqa: F401
        from mapanything.utils.metrics import (  # noqa: F401
            calculate_auc_np,
            l2_distance_of_unit_ray_directions_to_angular_error,
            se3_to_relative_pose_error,
        )
    except ImportError:
        return False
    return True


def pose_auc(
    pred_poses_c2w: np.ndarray,
    gt_poses_c2w: np.ndarray,
    thresholds: Sequence[int] = (5, 30),
) -> Dict[str, float]:
    """Relative-pose AUC over all camera pairs, upstream's definition.

    Args:
        pred_poses_c2w: (V, 4, 4) predicted camera-to-world poses.
        gt_poses_c2w: (V, 4, 4) GT camera-to-world poses.
        thresholds: Degree thresholds to integrate up to. 5 matches
            map-anything's ``pose_auc_5``; 30 is the threshold most multi-view
            pose papers report.

    Returns:
        ``{"pose/auc_<t>": percentage}``, empty when there are fewer than two
        views or the helpers are unavailable.
    """
    from mapanything.utils.metrics import calculate_auc_np, se3_to_relative_pose_error

    torch = _torch()
    pred = torch.as_tensor(np.asarray(pred_poses_c2w, dtype=np.float64))
    gt = torch.as_tensor(np.asarray(gt_poses_c2w, dtype=np.float64))
    num_views = int(pred.shape[0])
    if num_views < 2:
        return {}

    rot_err, trans_err = se3_to_relative_pose_error(
        pred_se3=pred, gt_se3=gt, num_frames=num_views
    )
    rot_err = rot_err.cpu().numpy()
    trans_err = trans_err.cpu().numpy()

    out: Dict[str, float] = {}
    for threshold in thresholds:
        auc, _ = calculate_auc_np(rot_err, trans_err, max_threshold=int(threshold))
        out[f"pose/auc_{int(threshold)}"] = float(auc) * 100.0
    return out


def ray_direction_error_deg(
    pred_intrinsics: np.ndarray,
    gt_intrinsics: np.ndarray,
    height: int,
    width: int,
    pred_ray_directions: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Angular error between predicted and GT unit ray directions, in degrees.

    This is upstream's ``ray_dirs_err_deg``: the L2 distance between the
    predicted and the GT unit ray of every pixel, converted to an angle. It is
    the cleanest read on whether a model recovered the camera's field of view,
    independent of any depth it predicted.

    Upstream scores the rays the model *predicted* (``preds[i]["ray_directions"]``
    in ``benchmarking/dense_n_view/benchmark.py``), not rays rebuilt from a
    pinhole fit to them; the fit discards whatever the model got wrong that a
    pinhole cannot express, which lowers the error. So ``pred_ray_directions``
    is used whenever the model provides it, and ``pred_intrinsics`` only as a
    fallback for one that does not.

    Args:
        pred_intrinsics: (V, 3, 3) predicted pinhole intrinsics.
        gt_intrinsics: (V, 3, 3) GT pinhole intrinsics on the same grid.
        height: Image height in pixels.
        width: Image width in pixels.
        pred_ray_directions: Optional (V, H, W, 3) predicted ray directions in
            each camera frame; normalised here.
    """
    from mapanything.utils.geometry import get_rays_in_camera_frame
    from mapanything.utils.metrics import (
        l2_distance_of_unit_ray_directions_to_angular_error,
    )

    torch = _torch()
    pred = torch.as_tensor(np.asarray(pred_intrinsics, dtype=np.float64))
    gt = torch.as_tensor(np.asarray(gt_intrinsics, dtype=np.float64))
    if pred.shape != gt.shape:
        raise ValueError(
            f"Intrinsics shapes differ: {tuple(pred.shape)} vs {tuple(gt.shape)}"
        )

    if pred_ray_directions is not None:
        pred_rays = torch.as_tensor(np.asarray(pred_ray_directions, dtype=np.float64))
        expected = (gt.shape[0], height, width, 3)
        if tuple(pred_rays.shape) != expected:
            raise ValueError(
                f"Ray directions have shape {tuple(pred_rays.shape)}, expected {expected}"
            )
        pred_rays = torch.nn.functional.normalize(pred_rays, dim=-1)
    else:
        _, pred_rays = get_rays_in_camera_frame(
            pred, height, width, normalize_to_unit_sphere=True
        )
    _, gt_rays = get_rays_in_camera_frame(
        gt, height, width, normalize_to_unit_sphere=True
    )
    # Per-view mean, then the mean across views - the order upstream uses.
    l2 = torch.norm(gt_rays - pred_rays, dim=-1)
    per_view = l2_distance_of_unit_ray_directions_to_angular_error(l2)
    per_view = per_view.reshape(per_view.shape[0], -1).mean(dim=-1)
    return {"rays/err_deg": float(per_view.mean())}


def evaluate_official(
    prediction,
    gt_intrinsics: np.ndarray,
    gt_poses_c2w: np.ndarray,
) -> Dict[str, float]:
    """All upstream-defined metrics for a prediction, or {} if unavailable.

    Missing dependencies are not an error here: the rest of the benchmark is
    still meaningful without these, and a hard failure would turn an optional
    comparison into a broken run.
    """
    if not available():
        return {}

    out: Dict[str, float] = {}
    out.update(pose_auc(prediction.poses_c2w, gt_poses_c2w))
    height, width = prediction.depth_z.shape[1], prediction.depth_z.shape[2]
    out.update(
        ray_direction_error_deg(
            prediction.intrinsics,
            gt_intrinsics,
            height,
            width,
            pred_ray_directions=getattr(prediction, "ray_directions", None),
        )
    )
    return out


# --------------------------------------------------------------------------
# Every metric of upstream's dense_n_view benchmark, run by upstream's code
# --------------------------------------------------------------------------

_UPSTREAM_BENCHMARK = None


def _upstream_benchmark_module():
    """Import benchmarking/dense_n_view/benchmark.py from the map-anything checkout.

    It is a script, not part of the installed package, but importing it only
    defines functions (its Hydra entry point runs under ``__main__``). Its
    ``get_all_info_for_metric_computation`` is what turns a batch and the
    model's outputs into the normalised quantities every upstream metric is
    computed from, so it is used as is rather than re-derived.
    """
    global _UPSTREAM_BENCHMARK
    if _UPSTREAM_BENCHMARK is None:
        import importlib.util
        import os

        from .models import mapanything_root

        path = os.path.join(mapanything_root(), "benchmarking", "dense_n_view", "benchmark.py")
        spec = importlib.util.spec_from_file_location("_mapanything_dense_n_view_benchmark", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _UPSTREAM_BENCHMARK = module
    return _UPSTREAM_BENCHMARK


def upstream_dense_n_view_metrics(upstream_views, raw_outputs) -> Dict[str, float]:
    """The per-set metrics of upstream's dense_n_view benchmark, as ``official/*``.

    Mirrors the body of upstream's evaluation loop for one multi-view set
    (benchmark.py, "Compute the metrics across all views"): the same inputs,
    the same helper functions, the same per-view mean then per-set value.
    These are upstream's definitions, which differ from the benchmark's own
    columns on purpose in places - e.g. ``official/z_depth_abs_rel`` scores
    depth after the joint avg-dis normalisation of all views, where
    ``depth/abs_rel`` first rescales by the median depth ratio; and
    ``official/pose_ate_rmse`` is in normalised units, ``pose/ate_rmse`` in
    the GT's metres. ``official/metric_scale_abs_rel`` is reported for every
    model, as upstream does, but means something only for metric-scale ones.

    Args:
        upstream_views: ``PreparedViews.upstream_views`` - each view's GT as
            upstream's DataLoader batches it.
        raw_outputs: ``Prediction.raw_outputs`` - the model's own output dicts.
    """
    from mapanything.utils.metrics import evaluate_ate, m_rel_ae, thresh_inliers

    # evaluate_ate's Umeyama step calls np.linalg.linalg.svd, a private alias
    # numpy 2.x removed; it is the same function as np.linalg.svd.
    if not hasattr(np.linalg, "linalg"):
        np.linalg.linalg = np.linalg

    batch = [dict(view) for view in upstream_views]
    preds = [dict(out) for out in raw_outputs]
    gt_info, pr_info, valid_masks = _upstream_benchmark_module().get_all_info_for_metric_computation(
        batch, preds
    )

    pts_rel, pts_in, z_rel, z_in, gt_poses, pr_poses = [], [], [], [], [], []
    for v in range(len(batch)):
        mask = valid_masks[v][0].numpy()
        gt_pts, pr_pts = gt_info["pts3d"][v][0].numpy(), pr_info["pts3d"][v][0].numpy()
        gt_z, pr_z = gt_info["z_depths"][v][0].numpy(), pr_info["z_depths"][v][0].numpy()
        pts_rel.append(m_rel_ae(gt=gt_pts, pred=pr_pts, mask=mask))
        pts_in.append(thresh_inliers(gt=gt_pts, pred=pr_pts, mask=mask, thresh=1.03))
        z_rel.append(m_rel_ae(gt=gt_z, pred=pr_z, mask=mask))
        z_in.append(thresh_inliers(gt=gt_z, pred=pr_z, mask=mask, thresh=1.03))
        gt_poses.append(gt_info["poses"][v][0])
        pr_poses.append(pr_info["poses"][v][0])

    out = {
        "official/pointmaps_abs_rel": float(np.mean(pts_rel)),
        "official/pointmaps_inlier_thres_103": float(np.mean(pts_in)),
        "official/z_depth_abs_rel": float(np.mean(z_rel)),
        "official/z_depth_inlier_thres_103": float(np.mean(z_in)),
        "official/pose_ate_rmse": float(evaluate_ate(gt_traj=gt_poses, est_traj=pr_poses)),
    }
    if gt_info["metric_scale"] is not None:
        gt_scale = gt_info["metric_scale"][0].numpy()
        pr_scale = pr_info["metric_scale"][0].numpy()
        out["official/metric_scale_abs_rel"] = float(
            (np.abs(pr_scale - gt_scale) / gt_scale).item()
        )
    return out
