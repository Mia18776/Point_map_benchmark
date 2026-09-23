"""Unified container for a multi-view point-map prediction.

Every model in this benchmark returns something slightly different, so each
runner converts its raw output into a single :class:`Prediction`. All metric and
export code only ever sees this class, which is what makes the comparison
apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch


@dataclass
class Prediction:
    """A V-view reconstruction on a common (H, W) pixel grid.

    Attributes:
        points_world: (V, H, W, 3) point map in the model's own world frame.
        points_cam: (V, H, W, 3) point map in each view's camera frame.
        depth_z: (V, H, W) z-depth in the camera frame.
        intrinsics: (V, 3, 3) pinhole intrinsics for the (H, W) grid.
        poses_c2w: (V, 4, 4) OpenCV camera-to-world poses.
        mask: (V, H, W) boolean validity mask.
        images: (V, H, W, 3) un-normalised RGB in [0, 1], for colouring exports.
        confidence: Optional (V, H, W) per-pixel confidence.
        is_metric: Whether the model claims real-world (metric) scale.
        info: Free-form diagnostics (runtime, memory, model id, ...).
    """

    points_world: np.ndarray
    points_cam: np.ndarray
    depth_z: np.ndarray
    intrinsics: np.ndarray
    poses_c2w: np.ndarray
    mask: np.ndarray
    images: np.ndarray
    confidence: Optional[np.ndarray] = None
    is_metric: bool = False
    info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        num_views, height, width = self.depth_z.shape
        expected = {
            "points_world": (num_views, height, width, 3),
            "points_cam": (num_views, height, width, 3),
            "intrinsics": (num_views, 3, 3),
            "poses_c2w": (num_views, 4, 4),
            "mask": (num_views, height, width),
            "images": (num_views, height, width, 3),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual != shape:
                raise ValueError(f"{name} has shape {actual}, expected {shape}")
        if self.confidence is not None and self.confidence.shape != (
            num_views,
            height,
            width,
        ):
            raise ValueError(f"confidence has shape {self.confidence.shape}")
        self.mask = self.mask.astype(bool)

    @property
    def num_views(self) -> int:
        return int(self.depth_z.shape[0])

    def filter_by_confidence(self, percentile: float) -> "Prediction":
        """Return a copy whose mask drops the lowest-confidence pixels.

        Args:
            percentile: Percentage of valid pixels to remove (0 disables it).
        """
        if self.confidence is None or percentile <= 0:
            return self
        valid = self.mask & np.isfinite(self.confidence)
        if not valid.any():
            return self
        threshold = float(np.percentile(self.confidence[valid], percentile))
        new_mask = self.mask & (self.confidence >= threshold)
        return Prediction(
            points_world=self.points_world,
            points_cam=self.points_cam,
            depth_z=self.depth_z,
            intrinsics=self.intrinsics,
            poses_c2w=self.poses_c2w,
            mask=new_mask,
            images=self.images,
            confidence=self.confidence,
            is_metric=self.is_metric,
            info=dict(self.info),
        )


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a tensor to a float64 NumPy array (bf16/fp16 safe)."""
    return tensor.detach().to(torch.float32).cpu().numpy().astype(np.float64)


def _squeeze_trailing_singleton(array: np.ndarray) -> np.ndarray:
    """Drop a trailing length-1 axis, so per-pixel maps are always (H, W).

    The upstream wrappers disagree here: VGGT and Depth Anything 3 emit a
    confidence of shape (H, W) while Pi3 and Pi3-X emit (H, W, 1). Normalising
    it once keeps :class:`Prediction`'s shape contract simple and stops Pi3 from
    failing validation at the very end of an otherwise successful run.
    """
    if array.ndim >= 1 and array.shape[-1] == 1:
        return array[..., 0]
    return array


def prediction_from_wrapper_output(
    outputs: List[Dict[str, torch.Tensor]],
    images: np.ndarray,
    is_metric: bool,
    info: Optional[Dict[str, Any]] = None,
    batch_index: int = 0,
) -> Prediction:
    """Convert MapAnything's common wrapper output format into a Prediction.

    Every external wrapper in ``mapanything.models.external`` returns a list of
    per-view dicts with the keys ``pts3d``, ``pts3d_cam``, ``ray_directions``,
    ``depth_along_ray``, ``cam_trans``, ``cam_quats`` and (usually) ``conf``,
    each batched as (B, ...). This function slices out one batch element and
    derives the quantities the benchmark needs.

    Intrinsics are recovered from the predicted unit ray directions with
    MapAnything's own least-squares fit, which is exactly how the repository
    reports intrinsics for models that do not output them explicitly.
    """
    from mapanything.utils.geometry import (
        recover_pinhole_intrinsics_from_ray_directions,
    )

    from .geometry import pose_from_quat_trans

    points_world, points_cam, intrinsics, quats, trans, confidences = [], [], [], [], [], []
    for view in outputs:
        points_world.append(_to_numpy(view["pts3d"][batch_index]))
        points_cam.append(_to_numpy(view["pts3d_cam"][batch_index]))

        if "intrinsics" in view:
            intrinsics.append(_to_numpy(view["intrinsics"][batch_index]))
        else:
            rays = view["ray_directions"][batch_index].detach().to(torch.float32)
            intrinsics.append(
                _to_numpy(recover_pinhole_intrinsics_from_ray_directions(rays))
            )

        quats.append(_to_numpy(view["cam_quats"][batch_index]))
        trans.append(_to_numpy(view["cam_trans"][batch_index]))
        confidences.append(
            _squeeze_trailing_singleton(_to_numpy(view["conf"][batch_index]))
            if "conf" in view
            else None
        )

    points_world = np.stack(points_world)
    points_cam = np.stack(points_cam)
    depth_z = points_cam[..., 2]

    poses_c2w = pose_from_quat_trans(np.stack(quats), np.stack(trans))
    confidence = (
        np.stack(confidences) if all(c is not None for c in confidences) else None
    )

    mask = (
        np.isfinite(points_world).all(axis=-1)
        & np.isfinite(points_cam).all(axis=-1)
        & (depth_z > 0)
    )

    return Prediction(
        points_world=points_world,
        points_cam=points_cam,
        depth_z=depth_z,
        intrinsics=np.stack(intrinsics),
        poses_c2w=poses_c2w,
        mask=mask,
        images=images,
        confidence=confidence,
        is_metric=is_metric,
        info=dict(info or {}),
    )
