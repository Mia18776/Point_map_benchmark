#!/usr/bin/env python3
"""Render a small synthetic scene with exact ground truth.

The scene is a textured box-shaped room with a few spheres inside it, rendered
by analytic ray casting. Because the geometry is closed-form there is no
renderer dependency and the depth maps are exact to floating-point precision,
which makes this the right tool for verifying that the whole benchmark pipeline
(image loading, GT resizing, alignment, metrics) is wired up correctly.

It is a *pipeline* test, not a substitute for real evaluation data: the
appearance statistics are nothing like real photographs, so absolute model
scores on it are not meaningful.

Usage::

    python scripts/make_synthetic_scene.py --output-dir data/synthetic --num-views 6
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from PIL import Image

# Room half-extents (metres) centred on the origin, and the sphere obstacles.
ROOM_MIN = np.array([-3.0, -2.0, -3.0])
ROOM_MAX = np.array([3.0, 2.0, 3.0])
SPHERES = (
    (np.array([-1.0, 1.2, 0.5]), 0.8),
    (np.array([1.2, 1.4, -0.6]), 0.6),
    (np.array([0.2, 1.6, 1.6]), 0.4),
)


def look_at_pose(eye: np.ndarray, target: np.ndarray, up=(0.0, -1.0, 0.0)):
    """Build an OpenCV camera-to-world pose looking from ``eye`` at ``target``.

    OpenCV convention: +X right, +Y down, +Z forward. ``up`` therefore defaults
    to -Y in world coordinates for a world whose +Y points down as well.
    """
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    up_vec = np.asarray(up, dtype=np.float64)
    right = np.cross(forward, up_vec)
    if np.linalg.norm(right) < 1e-8:
        up_vec = np.array([0.0, 0.0, 1.0])
        right = np.cross(forward, up_vec)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)

    pose = np.eye(4)
    pose[:3, 0] = right
    pose[:3, 1] = down
    pose[:3, 2] = forward
    pose[:3, 3] = eye
    return pose


def _ray_box_exit(origin: np.ndarray, directions: np.ndarray):
    """Distance along each ray at which it leaves the axis-aligned room."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t_min = (ROOM_MIN - origin) / directions
        t_max = (ROOM_MAX - origin) / directions
    t_near = np.minimum(t_min, t_max)
    t_far = np.maximum(t_min, t_max)
    # Leaving the box happens at the smallest of the three "far" slab crossings.
    exit_t = np.nanmin(np.where(np.isfinite(t_far), t_far, np.inf), axis=-1)
    # Guard against numerically degenerate rays that never enter the box.
    entry_t = np.nanmax(np.where(np.isfinite(t_near), t_near, -np.inf), axis=-1)
    exit_t = np.where(exit_t > np.maximum(entry_t, 0.0), exit_t, np.inf)
    return exit_t


def _ray_sphere(origin: np.ndarray, directions: np.ndarray, centre, radius):
    """Nearest positive ray-sphere intersection distance (inf when missed)."""
    offset = origin - centre
    b = 2.0 * directions @ offset
    c = float(offset @ offset) - radius * radius
    disc = b * b - 4.0 * c  # a == 1 because directions are unit vectors
    hit = disc > 0
    sqrt_disc = np.sqrt(np.where(hit, disc, 0.0))
    t0 = (-b - sqrt_disc) / 2.0
    t1 = (-b + sqrt_disc) / 2.0
    t = np.where(t0 > 1e-4, t0, t1)
    return np.where(hit & (t > 1e-4), t, np.inf)


def _procedural_colour(points: np.ndarray, object_id: np.ndarray) -> np.ndarray:
    """A deterministic, view-independent texture with plenty of gradient detail."""
    base = np.array(
        [
            [0.75, 0.72, 0.68],  # room walls
            [0.85, 0.35, 0.30],  # sphere 0
            [0.30, 0.55, 0.85],  # sphere 1
            [0.45, 0.75, 0.40],  # sphere 2
        ]
    )
    colour = base[np.clip(object_id, 0, len(base) - 1)]

    # Checkerboard plus a couple of sine bands, so the images have both sharp
    # edges and smooth shading for the networks to latch onto.
    checker = ((np.floor(points * 4.0).sum(axis=-1) % 2) == 0).astype(np.float64)
    bands = 0.5 + 0.5 * np.sin(points[..., 0] * 6.0) * np.cos(points[..., 2] * 5.0)
    shade = 0.55 + 0.25 * checker + 0.20 * bands
    return np.clip(colour * shade[..., None], 0.0, 1.0)


def render_view(pose_c2w: np.ndarray, intrinsics: np.ndarray, width: int, height: int):
    """Ray-cast one view. Returns ``(rgb_uint8, depth_z)``."""
    cols = np.arange(width, dtype=np.float64)[None, :]
    rows = np.arange(height, dtype=np.float64)[:, None]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    dirs_cam = np.stack(
        [
            np.broadcast_to((cols - cx) / fx, (height, width)),
            np.broadcast_to((rows - cy) / fy, (height, width)),
            np.ones((height, width)),
        ],
        axis=-1,
    )
    dirs_cam = dirs_cam / np.linalg.norm(dirs_cam, axis=-1, keepdims=True)

    rotation = pose_c2w[:3, :3]
    origin = pose_c2w[:3, 3]
    dirs_world = dirs_cam @ rotation.T
    flat_dirs = dirs_world.reshape(-1, 3)

    best_t = _ray_box_exit(origin, flat_dirs)
    object_id = np.zeros(flat_dirs.shape[0], dtype=np.int64)
    for index, (centre, radius) in enumerate(SPHERES, start=1):
        t_sphere = _ray_sphere(origin, flat_dirs, centre, radius)
        closer = t_sphere < best_t
        best_t = np.where(closer, t_sphere, best_t)
        object_id = np.where(closer, index, object_id)

    # A camera inside the room always hits something, but keep the renderer
    # total in case of a degenerate ray: such pixels get depth 0 (= invalid).
    hit = np.isfinite(best_t)
    safe_t = np.where(hit, best_t, 0.0)
    points = origin + flat_dirs * safe_t[:, None]
    colour = np.where(hit[:, None], _procedural_colour(points, object_id), 0.0)

    # z-depth in the camera frame (not the distance along the ray).
    points_cam = (points - origin) @ rotation
    depth_z = np.where(hit, points_cam[:, 2], 0.0)

    rgb = (colour.reshape(height, width, 3) * 255).astype(np.uint8)
    return rgb, depth_z.reshape(height, width)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--num-views", type=int, default=6)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--height", type=int, default=336)
    parser.add_argument(
        "--focal", type=float, default=None, help="Focal length in pixels (default: width)"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    focal = args.focal if args.focal is not None else float(args.width)
    intrinsics = np.array(
        [
            [focal, 0.0, (args.width - 1) / 2.0],
            [0.0, focal, (args.height - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )

    image_dir = os.path.join(args.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    depths, poses, names = [], [], []
    for index in range(args.num_views):
        angle = np.pi * 0.55 * index / max(args.num_views - 1, 1) - np.pi * 0.275
        eye = np.array(
            [2.0 * np.sin(angle), 0.3 + 0.15 * rng.standard_normal(), -2.0 * np.cos(angle)]
        )
        target = np.array([0.0, 0.6, 1.0]) + 0.2 * rng.standard_normal(3)
        pose = look_at_pose(eye, target)

        rgb, depth = render_view(pose, intrinsics, args.width, args.height)
        name = f"{index:03d}.png"
        Image.fromarray(rgb).save(os.path.join(image_dir, name))

        depths.append(depth)
        poses.append(pose)
        names.append(name)

    np.savez_compressed(
        os.path.join(args.output_dir, "gt.npz"),
        depth_z=np.stack(depths).astype(np.float32),
        intrinsics=np.stack([intrinsics] * args.num_views).astype(np.float32),
        poses_c2w=np.stack(poses).astype(np.float32),
        image_names=np.array(names),
    )
    print(
        f"Wrote {args.num_views} views to {image_dir} and ground truth to "
        f"{os.path.join(args.output_dir, 'gt.npz')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
