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
) -> Dict[str, float]:
    """Angular error between predicted and GT unit ray directions, in degrees.

    This is upstream's ``ray_dirs_err_deg``: build the unit ray of every pixel
    from each set of intrinsics, take the L2 distance between them, and convert
    it to an angle. It is the cleanest read on whether a model recovered the
    camera's field of view, independent of any depth it predicted.

    Args:
        pred_intrinsics: (V, 3, 3) predicted pinhole intrinsics.
        gt_intrinsics: (V, 3, 3) GT pinhole intrinsics on the same grid.
        height: Image height in pixels.
        width: Image width in pixels.
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
        ray_direction_error_deg(prediction.intrinsics, gt_intrinsics, height, width)
    )
    return out
