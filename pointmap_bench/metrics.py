"""Metrics for evaluating predicted 3D point maps.

Two families of metrics are implemented:

1. **Ground-truth metrics** (:func:`evaluate_with_gt`). Because feed-forward
   reconstruction models predict geometry in their *own* world frame and, for
   non-metric models, up to an unknown global scale, every GT metric states
   exactly which alignment it uses:

   * ``cam0/*``  - both point clouds are expressed in the frame of the **first
     camera** and normalised by their own average point distance. This removes
     the world-frame and the scale ambiguity but *not* the pose error, so these
     numbers measure the joint quality of geometry and relative poses. This is
     the same protocol MapAnything's own ``dense_n_view`` benchmark uses.
   * ``sim3/*``  - the prediction is aligned to the GT with a closed-form
     Umeyama similarity transform over all valid points. This is the most
     forgiving alignment and isolates shape quality from global placement.
   * ``metric/*`` - no alignment at all beyond the frame change; only
     meaningful for models that claim metric (real-world) scale.

2. **Ground-truth-free metrics** (:func:`evaluate_without_gt`), which quantify
   how self-consistent a multi-view reconstruction is by reprojecting each
   view's points into every other view and comparing depths. These are
   scale-invariant and need no annotations.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .geometry import (
    apply_sim3,
    depth_to_world_points,
    invert_se3,
    optimal_scale,
    project_points,
    relative_poses,
    rotation_angle_deg,
    transform_points,
    translation_angle_deg,
    umeyama_sim3,
)

try:  # SciPy gives a large speed-up for the Chamfer distance; it is optional.
    from scipy.spatial import cKDTree as _KDTree
except ImportError:  # pragma: no cover - exercised only when SciPy is absent
    _KDTree = None

# Relative thresholds (ratio based) used for depth/point-norm inlier ratios.
RATIO_THRESHOLDS = (1.03, 1.05, 1.25)
# Absolute thresholds, in units of "average scene distance", for L2 point error.
L2_THRESHOLDS = (0.02, 0.05, 0.10)


def _finite_mask(points: np.ndarray) -> np.ndarray:
    """True where every coordinate of a (..., 3) point array is finite."""
    return np.isfinite(points).all(axis=-1)


def _safe(value: float) -> float:
    """Convert a possibly-empty reduction into a plain float (NaN when empty)."""
    value = float(value)
    return value if np.isfinite(value) else float("nan")


def _subsample(num_points: int, max_points: int, seed: int = 0) -> np.ndarray:
    """Deterministic uniform subsample of ``num_points`` indices."""
    if num_points <= max_points:
        return np.arange(num_points)
    rng = np.random.default_rng(seed)
    return rng.choice(num_points, size=max_points, replace=False)


def nearest_neighbour_distances(
    query: np.ndarray, reference: np.ndarray, chunk_size: int = 4096
) -> np.ndarray:
    """Distance from every query point to its nearest reference point.

    Uses a SciPy KD-tree when available and falls back to a chunked brute-force
    NumPy computation otherwise (so the package has no hard SciPy dependency).
    """
    query = np.asarray(query, dtype=np.float64).reshape(-1, 3)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
    if query.size == 0 or reference.size == 0:
        return np.empty((0,), dtype=np.float64)

    if _KDTree is not None:
        dist, _ = _KDTree(reference).query(query, k=1, workers=-1)
        return np.asarray(dist, dtype=np.float64)

    out = np.empty(query.shape[0], dtype=np.float64)
    ref_sq = (reference**2).sum(axis=1)
    for start in range(0, query.shape[0], chunk_size):
        block = query[start : start + chunk_size]
        # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2; the ||a||^2 term does not
        # change the argmin but is needed for the returned distance.
        sq = (block**2).sum(axis=1)[:, None] - 2.0 * block @ reference.T + ref_sq[None]
        out[start : start + chunk_size] = np.sqrt(np.maximum(sq.min(axis=1), 0.0))
    return out


def chamfer_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    max_points: int = 50000,
    seed: int = 0,
) -> Dict[str, float]:
    """Accuracy / completeness between two already-aligned point clouds.

    * *accuracy*: distance from each predicted point to the closest GT point.
    * *completeness*: distance from each GT point to the closest predicted point.

    Both clouds are uniformly subsampled to ``max_points`` for tractability.
    """
    pred = np.asarray(pred_points, dtype=np.float64).reshape(-1, 3)
    gt_pts = np.asarray(gt_points, dtype=np.float64).reshape(-1, 3)
    pred = pred[_subsample(pred.shape[0], max_points, seed)]
    gt_pts = gt_pts[_subsample(gt_pts.shape[0], max_points, seed + 1)]

    if pred.shape[0] == 0 or gt_pts.shape[0] == 0:
        return {
            "accuracy_mean": float("nan"),
            "accuracy_median": float("nan"),
            "completeness_mean": float("nan"),
            "completeness_median": float("nan"),
            "chamfer_mean": float("nan"),
        }

    acc = nearest_neighbour_distances(pred, gt_pts)
    comp = nearest_neighbour_distances(gt_pts, pred)
    return {
        "accuracy_mean": _safe(acc.mean()),
        "accuracy_median": _safe(np.median(acc)),
        "completeness_mean": _safe(comp.mean()),
        "completeness_median": _safe(np.median(comp)),
        "chamfer_mean": _safe(0.5 * (acc.mean() + comp.mean())),
    }


def _avg_distance_scale(points: np.ndarray, valid: np.ndarray) -> float:
    """Average L2 norm of the valid points, i.e. MapAnything's ``avg_dis`` norm."""
    if not valid.any():
        return float("nan")
    return _safe(np.linalg.norm(points[valid], axis=-1).mean())


def _ratio_inliers(pred_norm: np.ndarray, gt_norm: np.ndarray) -> Dict[str, float]:
    """Inlier ratios based on max(a/b, b/a) of point distances from the origin."""
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.maximum(pred_norm / gt_norm, gt_norm / pred_norm)
    ratio = np.where(np.isfinite(ratio), ratio, np.inf)
    return {f"delta_{t}": _safe((ratio < t).mean()) for t in RATIO_THRESHOLDS}


def evaluate_pointmaps_in_first_camera_frame(
    pred_points: np.ndarray,
    pred_poses_c2w: np.ndarray,
    gt_points: np.ndarray,
    gt_poses_c2w: np.ndarray,
    valid: np.ndarray,
) -> Dict[str, float]:
    """Scale-normalised point-map error in the frame of the first camera.

    Args:
        pred_points: (V, H, W, 3) predicted world-frame points.
        pred_poses_c2w: (V, 4, 4) predicted camera-to-world poses.
        gt_points: (V, H, W, 3) GT world-frame points.
        gt_poses_c2w: (V, 4, 4) GT camera-to-world poses.
        valid: (V, H, W) boolean mask of pixels valid in *both* prediction and GT.

    Returns:
        Dict with ``cam0/*`` and ``metric/*`` entries.
    """
    pred_cam0 = transform_points(invert_se3(pred_poses_c2w[0]), pred_points)
    gt_cam0 = transform_points(invert_se3(gt_poses_c2w[0]), gt_points)

    scale_pred = _avg_distance_scale(pred_cam0, valid)
    scale_gt = _avg_distance_scale(gt_cam0, valid)
    if not np.isfinite(scale_pred) or not np.isfinite(scale_gt) or scale_pred <= 0:
        return {"cam0/num_points": 0.0}

    pred_n = pred_cam0[valid] / scale_pred
    gt_n = gt_cam0[valid] / scale_gt

    err = np.linalg.norm(pred_n - gt_n, axis=-1)
    gt_dist = np.linalg.norm(gt_n, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(gt_dist > 0, err / gt_dist, np.nan)

    out: Dict[str, float] = {
        "cam0/mae": _safe(err.mean()),
        "cam0/median_ae": _safe(np.median(err)),
        "cam0/rel_ae": _safe(np.nanmean(rel)),
        "cam0/num_points": float(err.size),
    }
    for name, value in _ratio_inliers(np.linalg.norm(pred_n, axis=-1), gt_dist).items():
        out[f"cam0/{name}"] = value
    for thresh in L2_THRESHOLDS:
        out[f"cam0/inlier_l2_{thresh}"] = _safe((err < thresh).mean())

    # Metric-scale agreement: only meaningful for models predicting real scale.
    ratio = scale_pred / scale_gt
    out["metric/scale_ratio"] = _safe(ratio)
    out["metric/scale_abs_rel"] = _safe(abs(ratio - 1.0))
    return out


def evaluate_pointmaps_sim3(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    valid: np.ndarray,
    max_points: int = 200000,
    chamfer_max_points: int = 50000,
    seed: int = 0,
) -> Dict[str, float]:
    """Point-map error after a global Umeyama similarity alignment.

    The alignment is estimated on (a subsample of) the valid pixel
    correspondences, then applied to the full predicted cloud. Errors are
    reported in GT units, plus a scale-normalised variant so that scenes of
    different physical size can be averaged.
    """
    pred_valid = pred_points[valid]
    gt_valid = gt_points[valid]
    if pred_valid.shape[0] < 3:
        return {"sim3/num_points": float(pred_valid.shape[0])}

    idx = _subsample(pred_valid.shape[0], max_points, seed)
    scale, rot, trans = umeyama_sim3(pred_valid[idx], gt_valid[idx])

    pred_aligned = apply_sim3(pred_valid, scale, rot, trans)
    err = np.linalg.norm(pred_aligned - gt_valid, axis=-1)

    # Scene extent used to make the errors comparable across scenes.
    gt_extent = _safe(np.linalg.norm(gt_valid - gt_valid.mean(axis=0), axis=-1).mean())

    out: Dict[str, float] = {
        "sim3/scale": _safe(scale),
        "sim3/mae": _safe(err.mean()),
        "sim3/median_ae": _safe(np.median(err)),
        "sim3/num_points": float(err.size),
    }
    if gt_extent > 0:
        out["sim3/mae_rel_extent"] = _safe(err.mean() / gt_extent)
        for thresh in L2_THRESHOLDS:
            out[f"sim3/inlier_l2_{thresh}"] = _safe((err < thresh * gt_extent).mean())

    chamfer = chamfer_metrics(pred_aligned, gt_valid, chamfer_max_points, seed)
    for key, value in chamfer.items():
        out[f"sim3/{key}"] = value
        if gt_extent > 0:
            out[f"sim3/{key}_rel_extent"] = _safe(value / gt_extent)
    return out


def evaluate_poses(
    pred_poses_c2w: np.ndarray, gt_poses_c2w: np.ndarray
) -> Dict[str, float]:
    """Relative-pose accuracy plus absolute trajectory error.

    Relative pose errors are computed over all camera pairs, which makes them
    invariant to the (arbitrary) world frame and, for the translation part, to
    the global scale.
    """
    pred_poses_c2w = np.asarray(pred_poses_c2w, dtype=np.float64)
    gt_poses_c2w = np.asarray(gt_poses_c2w, dtype=np.float64)
    num_views = pred_poses_c2w.shape[0]
    out: Dict[str, float] = {}

    if num_views >= 2:
        rel_pred, _, _ = relative_poses(pred_poses_c2w)
        rel_gt, _, _ = relative_poses(gt_poses_c2w)
        rot_err = rotation_angle_deg(rel_pred[:, :3, :3], rel_gt[:, :3, :3])
        trans_err = translation_angle_deg(rel_pred[:, :3, 3], rel_gt[:, :3, 3])
        valid_trans = np.isfinite(trans_err)

        out["pose/rot_err_deg_mean"] = _safe(rot_err.mean())
        out["pose/rot_err_deg_median"] = _safe(np.median(rot_err))
        out["pose/rra_5deg"] = _safe((rot_err < 5.0).mean())
        if valid_trans.any():
            out["pose/trans_ang_err_deg_mean"] = _safe(trans_err[valid_trans].mean())
            out["pose/trans_ang_err_deg_median"] = _safe(
                np.median(trans_err[valid_trans])
            )
            out["pose/rta_5deg"] = _safe((trans_err[valid_trans] < 5.0).mean())

    if num_views >= 3:
        # ATE after a similarity alignment of the camera centres.
        pred_centres = pred_poses_c2w[:, :3, 3]
        gt_centres = gt_poses_c2w[:, :3, 3]
        scale, rot, trans = umeyama_sim3(pred_centres, gt_centres)
        aligned = apply_sim3(pred_centres, scale, rot, trans)
        residual = np.linalg.norm(aligned - gt_centres, axis=-1)
        traj_extent = _safe(
            np.linalg.norm(gt_centres - gt_centres.mean(axis=0), axis=-1).mean()
        )
        out["pose/ate_rmse"] = _safe(np.sqrt((residual**2).mean()))
        if traj_extent > 0:
            out["pose/ate_rmse_rel"] = _safe(out["pose/ate_rmse"] / traj_extent)
    return out


def evaluate_depth(
    pred_depth: np.ndarray, gt_depth: np.ndarray, valid: np.ndarray
) -> Dict[str, float]:
    """Per-view z-depth accuracy after a single global median-scale alignment."""
    pred = pred_depth[valid]
    gt_vals = gt_depth[valid]
    keep = (gt_vals > 0) & (pred > 0) & np.isfinite(pred) & np.isfinite(gt_vals)
    pred, gt_vals = pred[keep], gt_vals[keep]
    if pred.size == 0:
        return {"depth/num_points": 0.0}

    scale = float(np.median(gt_vals) / np.median(pred))
    pred = pred * scale
    abs_rel = np.abs(pred - gt_vals) / gt_vals
    ratio = np.maximum(pred / gt_vals, gt_vals / pred)

    out = {
        "depth/scale": _safe(scale),
        "depth/abs_rel": _safe(abs_rel.mean()),
        "depth/rmse": _safe(np.sqrt(((pred - gt_vals) ** 2).mean())),
        "depth/num_points": float(pred.size),
    }
    for thresh in RATIO_THRESHOLDS:
        out[f"depth/delta_{thresh}"] = _safe((ratio < thresh).mean())
    return out


def multiview_consistency(
    points_world: np.ndarray,
    depth_z: np.ndarray,
    intrinsics: np.ndarray,
    poses_c2w: np.ndarray,
    valid: np.ndarray,
    max_points_per_view: int = 20000,
    max_pairs: int = 30,
    seed: int = 0,
) -> Dict[str, float]:
    """Ground-truth-free multi-view depth consistency.

    For an ordered pair of views (i, j) the valid 3D points of view j are
    transformed into camera i, projected with view i's predicted intrinsics and
    the resulting depth is compared against view i's own predicted depth at the
    landing pixel. A globally consistent reconstruction agrees with itself; a
    per-view-inconsistent one does not.

    The comparison is a *ratio*, so the metric is invariant to the global scale
    and can be compared across metric and non-metric models.

    Caveat: points of view j that are genuinely occluded in view i count as
    disagreements. Robust statistics (median, inlier ratios) are therefore
    reported rather than a mean, and ``overlap_ratio`` records how much of view
    j actually landed inside view i.
    """
    num_views = points_world.shape[0]
    if num_views < 2:
        return {"consistency/num_pairs": 0.0}

    height, width = depth_z.shape[1], depth_z.shape[2]
    pairs = [(i, j) for i in range(num_views) for j in range(num_views) if i != j]
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(seed)
        pairs = [pairs[k] for k in rng.choice(len(pairs), max_pairs, replace=False)]

    rel_errors, inliers, overlaps = [], {t: [] for t in RATIO_THRESHOLDS}, []
    for i, j in pairs:
        src = points_world[j][valid[j]]
        if src.shape[0] == 0:
            continue
        idx = _subsample(src.shape[0], max_points_per_view, seed)
        src = src[idx]

        pts_in_i = transform_points(invert_se3(poses_c2w[i]), src)
        u, v, z = project_points(pts_in_i, intrinsics[i])

        in_front = z > 0
        col = np.rint(np.where(in_front, u, 0.0)).astype(np.int64)
        row = np.rint(np.where(in_front, v, 0.0)).astype(np.int64)
        inside = in_front & (col >= 0) & (col < width) & (row >= 0) & (row < height)
        overlaps.append(float(inside.mean()))
        if not inside.any():
            continue

        col, row, z = col[inside], row[inside], z[inside]
        ref_depth = depth_z[i][row, col]
        keep = valid[i][row, col] & (ref_depth > 0) & np.isfinite(ref_depth)
        if not keep.any():
            continue
        z, ref_depth = z[keep], ref_depth[keep]

        rel_errors.append(np.abs(z - ref_depth) / ref_depth)
        ratio = np.maximum(z / ref_depth, ref_depth / z)
        for thresh in RATIO_THRESHOLDS:
            inliers[thresh].append(float((ratio < thresh).mean()))

    if not rel_errors:
        return {"consistency/num_pairs": 0.0}

    all_rel = np.concatenate(rel_errors)
    out = {
        "consistency/num_pairs": float(len(rel_errors)),
        "consistency/rel_depth_err_median": _safe(np.median(all_rel)),
        "consistency/rel_depth_err_mean": _safe(all_rel.mean()),
        "consistency/overlap_ratio": _safe(float(np.mean(overlaps))),
    }
    for thresh in RATIO_THRESHOLDS:
        out[f"consistency/inlier_{thresh}"] = _safe(float(np.mean(inliers[thresh])))
    return out


def pointcloud_stats(points_world: np.ndarray, valid: np.ndarray) -> Dict[str, float]:
    """Cheap descriptive statistics of the produced point cloud."""
    total = float(valid.size)
    num_valid = float(valid.sum())
    out = {
        "cloud/num_points": num_valid,
        "cloud/valid_ratio": _safe(num_valid / total) if total > 0 else float("nan"),
    }
    if num_valid >= 2:
        pts = points_world[valid]
        centre = pts.mean(axis=0)
        out["cloud/extent_mean"] = _safe(
            np.linalg.norm(pts - centre, axis=-1).mean()
        )
        out["cloud/bbox_diagonal"] = _safe(
            float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        )
    return out


def evaluate_without_gt(prediction, max_pairs: int = 30) -> Dict[str, float]:
    """All GT-free metrics for a :class:`~pointmap_bench.prediction.Prediction`."""
    out: Dict[str, float] = {}
    out.update(pointcloud_stats(prediction.points_world, prediction.mask))
    out.update(
        multiview_consistency(
            prediction.points_world,
            prediction.depth_z,
            prediction.intrinsics,
            prediction.poses_c2w,
            prediction.mask,
            max_pairs=max_pairs,
        )
    )
    return out


def evaluate_with_gt(
    prediction,
    gt_depth_z: np.ndarray,
    gt_intrinsics: np.ndarray,
    gt_poses_c2w: np.ndarray,
    gt_mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """All GT metrics for a prediction against a posed depth ground truth.

    Args:
        prediction: A :class:`~pointmap_bench.prediction.Prediction`.
        gt_depth_z: (V, H, W) GT z-depth on the *same* pixel grid as the
            prediction. Non-positive or non-finite values mark invalid pixels.
        gt_intrinsics: (V, 3, 3) GT intrinsics for that grid.
        gt_poses_c2w: (V, 4, 4) GT camera-to-world poses.
        gt_mask: Optional (V, H, W) extra validity mask.
    """
    gt_depth_z = np.asarray(gt_depth_z, dtype=np.float64)
    if gt_depth_z.shape != prediction.depth_z.shape:
        raise ValueError(
            "GT depth shape "
            f"{gt_depth_z.shape} does not match prediction shape "
            f"{prediction.depth_z.shape}. Run the benchmark with a common "
            "--image-size so every model shares the GT pixel grid."
        )

    gt_valid = np.isfinite(gt_depth_z) & (gt_depth_z > 0)
    if gt_mask is not None:
        gt_valid &= np.asarray(gt_mask, dtype=bool)

    gt_points = depth_to_world_points(
        np.where(gt_valid, gt_depth_z, 0.0), gt_intrinsics, gt_poses_c2w
    )
    valid = gt_valid & prediction.mask & _finite_mask(prediction.points_world)

    out: Dict[str, float] = {"gt/valid_ratio": _safe(valid.mean())}
    if not valid.any():
        return out

    out.update(
        evaluate_pointmaps_in_first_camera_frame(
            prediction.points_world,
            prediction.poses_c2w,
            gt_points,
            gt_poses_c2w,
            valid,
        )
    )
    out.update(evaluate_pointmaps_sim3(prediction.points_world, gt_points, valid))
    out.update(evaluate_poses(prediction.poses_c2w, gt_poses_c2w))
    out.update(evaluate_depth(prediction.depth_z, gt_depth_z, valid))
    return out


def scale_only_alignment_error(
    pred_points: np.ndarray, gt_points: np.ndarray, valid: np.ndarray
) -> float:
    """Mean L2 error after a least-squares scale-only alignment.

    Helper kept separate from :func:`evaluate_with_gt` because it assumes the
    two clouds already share a common rotation and origin.
    """
    scale = optimal_scale(pred_points[valid], gt_points[valid])
    if not np.isfinite(scale):
        return float("nan")
    err = np.linalg.norm(scale * pred_points[valid] - gt_points[valid], axis=-1)
    return _safe(err.mean())
