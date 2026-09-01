"""Exporting predictions to inspectable files.

Point clouds are written as binary little-endian PLY with no third-party
dependency, so the qualitative comparison works even in a minimal environment.
GLB export reuses MapAnything's own ``predictions_to_glb`` when trimesh is
available.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from .prediction import Prediction

_PLY_HEADER = """ply
format binary_little_endian 1.0
element vertex {count}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""


def write_ply(
    path: str,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> int:
    """Write an XYZ+RGB point cloud as a binary PLY.

    Args:
        path: Destination ``.ply`` file.
        points: (N, 3) float coordinates.
        colors: Optional (N, 3) RGB, either float in [0, 1] or uint8.

    Returns:
        The number of points written.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]

    if colors is None:
        rgb = np.full((points.shape[0], 3), 200, dtype=np.uint8)
    else:
        colors = np.asarray(colors).reshape(-1, 3)[finite]
        if colors.dtype != np.uint8:
            colors = np.clip(colors, 0.0, 1.0) * 255.0
        rgb = colors.astype(np.uint8)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    vertex = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertex["x"], vertex["y"], vertex["z"] = points[:, 0], points[:, 1], points[:, 2]
    vertex["red"], vertex["green"], vertex["blue"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    with open(path, "wb") as handle:
        handle.write(_PLY_HEADER.format(count=points.shape[0]).encode("ascii"))
        handle.write(vertex.tobytes())
    return int(points.shape[0])


def export_point_cloud(
    prediction: Prediction, path: str, max_points: Optional[int] = 2_000_000
) -> int:
    """Write a prediction's valid, colour-mapped points to a PLY file."""
    points = prediction.points_world[prediction.mask]
    colors = prediction.images[prediction.mask]
    if max_points is not None and points.shape[0] > max_points:
        rng = np.random.default_rng(0)
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points, colors = points[keep], colors[keep]
    return write_ply(path, points, colors)


def export_glb(prediction: Prediction, path: str, as_mesh: bool = False) -> bool:
    """Write a GLB scene via MapAnything's exporter. Returns False if unavailable."""
    try:
        from mapanything.utils.viz import predictions_to_glb
    except ImportError:
        return False

    scene = predictions_to_glb(
        {
            "world_points": prediction.points_world.astype(np.float32),
            "images": prediction.images.astype(np.float32),
            "final_masks": prediction.mask,
        },
        as_mesh=as_mesh,
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    scene.export(path)
    return True


def export_npz(prediction: Prediction, path: str) -> None:
    """Dump the raw prediction so metrics can be recomputed without re-running."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    payload = {
        "points_world": prediction.points_world.astype(np.float32),
        "points_cam": prediction.points_cam.astype(np.float32),
        "depth_z": prediction.depth_z.astype(np.float32),
        "intrinsics": prediction.intrinsics.astype(np.float32),
        "poses_c2w": prediction.poses_c2w.astype(np.float32),
        "mask": prediction.mask,
        "images": (prediction.images * 255).astype(np.uint8),
    }
    if prediction.confidence is not None:
        payload["confidence"] = prediction.confidence.astype(np.float32)
    np.savez_compressed(path, **payload)
