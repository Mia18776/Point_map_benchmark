"""Self-contained NumPy geometry helpers used by the point-map benchmark.

Everything here is deliberately dependency-free (NumPy only) so that it can be
unit-tested without any of the heavy model dependencies installed.

Conventions used throughout the whole project (identical to MapAnything /
DUSt3R / VGGT):

* Camera frame is OpenCV style: +X right, +Y down, +Z forward.
* ``poses_c2w`` are 4x4 camera-to-world matrices.
* ``intrinsics`` are 3x3 pinhole matrices [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  expressed in pixels of the *same* image grid as the depth map / point map.
* Quaternions are scalar-last (x, y, z, w).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "quat_xyzw_to_rotmat",
    "pose_from_quat_trans",
    "invert_se3",
    "transform_points",
    "depth_to_camera_points",
    "depth_to_world_points",
    "project_points",
    "optimal_scale",
    "umeyama_sim3",
    "robust_umeyama_sim3",
    "apply_sim3",
    "rotation_angle_deg",
    "translation_angle_deg",
    "relative_poses",
]


def quat_xyzw_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert scalar-last unit quaternions to rotation matrices.

    Args:
        quat: (..., 4) array of (x, y, z, w) quaternions. It does not need to be
            normalised; normalisation happens internally.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    quat = np.asarray(quat, dtype=np.float64)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    x, y, z, w = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]

    rot = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        axis=-1,
    )
    return rot.reshape(quat.shape[:-1] + (3, 3))


def pose_from_quat_trans(quat: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Build (..., 4, 4) SE(3) matrices from quaternions and translations."""
    quat = np.asarray(quat, dtype=np.float64)
    trans = np.asarray(trans, dtype=np.float64)
    rot = quat_xyzw_to_rotmat(quat)
    batch_shape = rot.shape[:-2]
    pose = np.zeros(batch_shape + (4, 4), dtype=np.float64)
    pose[..., :3, :3] = rot
    pose[..., :3, 3] = trans
    pose[..., 3, 3] = 1.0
    return pose


def invert_se3(pose: np.ndarray) -> np.ndarray:
    """Invert (..., 4, 4) rigid transforms using the closed-form inverse."""
    pose = np.asarray(pose, dtype=np.float64)
    rot = pose[..., :3, :3]
    trans = pose[..., :3, 3]
    rot_t = np.swapaxes(rot, -1, -2)
    out = np.zeros_like(pose)
    out[..., :3, :3] = rot_t
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", rot_t, trans)
    out[..., 3, 3] = 1.0
    return out


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a single 4x4 transform to an array of points of shape (..., 3)."""
    transform = np.asarray(transform, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    if transform.shape != (4, 4):
        raise ValueError(f"transform must be (4, 4), got {transform.shape}")
    rot = transform[:3, :3]
    trans = transform[:3, 3]
    return points @ rot.T + trans


def depth_to_camera_points(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """Back-project z-depth maps into camera-frame point maps.

    Args:
        depth: (H, W) or (V, H, W) z-depth (distance along the optical axis).
        intrinsics: (3, 3) or (V, 3, 3) pinhole intrinsics.

    Returns:
        (H, W, 3) or (V, H, W, 3) camera-frame points.
    """
    depth = np.asarray(depth, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    squeeze = depth.ndim == 2
    if squeeze:
        depth = depth[None]
        intrinsics = intrinsics[None]
    if depth.ndim != 3:
        raise ValueError(f"depth must be (H, W) or (V, H, W), got {depth.shape}")

    _, height, width = depth.shape
    # Pixel centres follow the same convention as MapAnything: integer pixel
    # indices (not the +0.5 convention), so a round-trip through
    # `project_points` is exact.
    xs = np.arange(width, dtype=np.float64)[None, None, :]
    ys = np.arange(height, dtype=np.float64)[None, :, None]

    fx = intrinsics[:, 0, 0][:, None, None]
    fy = intrinsics[:, 1, 1][:, None, None]
    cx = intrinsics[:, 0, 2][:, None, None]
    cy = intrinsics[:, 1, 2][:, None, None]

    x_cam = (xs - cx) * depth / fx
    y_cam = (ys - cy) * depth / fy
    pts = np.stack([x_cam, y_cam, depth], axis=-1)
    return pts[0] if squeeze else pts


def depth_to_world_points(
    depth: np.ndarray, intrinsics: np.ndarray, poses_c2w: np.ndarray
) -> np.ndarray:
    """Back-project z-depth maps directly into the world frame."""
    depth = np.asarray(depth, dtype=np.float64)
    squeeze = depth.ndim == 2
    pts_cam = depth_to_camera_points(depth, intrinsics)
    poses_c2w = np.asarray(poses_c2w, dtype=np.float64)
    if squeeze:
        return transform_points(poses_c2w, pts_cam)

    if poses_c2w.shape[0] != pts_cam.shape[0]:
        raise ValueError("Number of poses does not match number of views")
    rot = poses_c2w[:, :3, :3]
    trans = poses_c2w[:, :3, 3]
    return np.einsum("vij,vhwj->vhwi", rot, pts_cam) + trans[:, None, None, :]


def project_points(points_cam: np.ndarray, intrinsics: np.ndarray):
    """Project camera-frame points to pixel coordinates.

    Args:
        points_cam: (..., 3) points in the camera frame.
        intrinsics: (3, 3) pinhole intrinsics.

    Returns:
        Tuple (u, v, z) where u/v are pixel coordinates (integer-pixel-centre
        convention) and z is the depth along the optical axis. Points with
        z <= 0 are behind the camera; their u/v values are meaningless and must
        be masked out by the caller.
    """
    points_cam = np.asarray(points_cam, dtype=np.float64)
    intrinsics = np.asarray(intrinsics, dtype=np.float64)
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = points_cam[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * points_cam[..., 0] / z + cx
        v = fy * points_cam[..., 1] / z + cy
    return u, v, z


def optimal_scale(src: np.ndarray, dst: np.ndarray) -> float:
    """Least-squares scale s minimising ||s * src - dst||^2 (no rotation).

    Used for scale-only alignment, i.e. when the two point sets already live in
    a common reference frame up to an unknown global scale.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    denom = float((src * src).sum())
    if denom <= 0 or not np.isfinite(denom):
        return float("nan")
    return float((src * dst).sum() / denom)


def umeyama_sim3(src: np.ndarray, dst: np.ndarray, with_scale: bool = True):
    """Umeyama (1991) closed-form similarity alignment mapping src onto dst.

    Minimises ``sum_i || s * R @ src_i + t - dst_i ||^2``.

    Args:
        src: (N, 3) source points.
        dst: (N, 3) target points.
        with_scale: If False the scale is fixed to 1 (rigid alignment).

    Returns:
        Tuple (s, R, t) with s a float, R a (3, 3) rotation with det(R) = +1,
        and t a (3,) translation.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape != dst.shape:
        raise ValueError("src and dst must have the same shape")
    num = src.shape[0]
    if num < 3:
        raise ValueError("At least 3 correspondences are required")

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Cross-covariance with dst on the left, so that R maps src -> dst.
    cov = dst_c.T @ src_c / num
    u_mat, sing, vt_mat = np.linalg.svd(cov)

    # Reflection guard: force a proper rotation.
    sign = np.ones(3)
    if np.linalg.det(u_mat) * np.linalg.det(vt_mat) < 0:
        sign[2] = -1.0
    rot = u_mat @ np.diag(sign) @ vt_mat

    if with_scale:
        var_src = float((src_c**2).sum() / num)
        scale = float((sing * sign).sum() / var_src) if var_src > 0 else 1.0
    else:
        scale = 1.0

    trans = mu_dst - scale * rot @ mu_src
    return scale, rot, trans


def robust_umeyama_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    trim_ratio: float = 0.2,
    num_iterations: int = 3,
    num_candidates: int = 64,
    max_scoring_points: int = 20000,
    seed: int = 0,
):
    """Outlier-tolerant similarity alignment of src onto dst.

    :func:`umeyama_sim3` is a least-squares estimator, so a small cluster of
    gross outliers - sky pixels, a diverging patch of one prediction - can drag
    the alignment and corrupt every error computed on top of it.

    Trimming alone does not fix that: when the initial fit is dominated by the
    outliers, the smallest residuals no longer identify the inliers. So the
    estimate is seeded by RANSAC over the (known) correspondences - the plain
    fit plus a number of minimal three-point fits, each scored by its trimmed
    squared residual - and the winner is then refined by trim-and-refit rounds.

    Args:
        src: (N, 3) source points.
        dst: (N, 3) target points, in correspondence with ``src``.
        trim_ratio: Fraction of the worst correspondences to discard. 0 makes
            this exactly :func:`umeyama_sim3`.
        num_iterations: Number of trim-and-refit rounds after seeding.
        num_candidates: Minimal-sample hypotheses tried alongside the plain fit.
        max_scoring_points: Cap on the points used to score a hypothesis.
        seed: Seed of the deterministic sampling.

    Returns:
        Tuple (s, R, t), the same as :func:`umeyama_sim3`.
    """
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    plain = umeyama_sim3(src, dst)
    num = src.shape[0]
    keep_count = max(3, int(round(num * (1.0 - trim_ratio))))
    if trim_ratio <= 0.0 or num_iterations <= 0 or keep_count >= num:
        return plain

    rng = np.random.default_rng(seed)
    score_idx = (
        np.arange(num)
        if num <= max_scoring_points
        else rng.choice(num, size=max_scoring_points, replace=False)
    )
    score_src, score_dst = src[score_idx], dst[score_idx]
    score_keep = max(3, int(round(score_src.shape[0] * (1.0 - trim_ratio))))

    def trimmed_cost(candidate) -> float:
        """Sum of the smallest ``score_keep`` squared residuals (inf if degenerate)."""
        scale, rot, trans = candidate
        if not np.isfinite(scale) or not np.isfinite(rot).all():
            return float("inf")
        residual = ((apply_sim3(score_src, scale, rot, trans) - score_dst) ** 2).sum(
            axis=-1
        )
        return float(np.partition(residual, score_keep - 1)[:score_keep].sum())

    best, best_cost = plain, trimmed_cost(plain)
    for _ in range(num_candidates):
        sample = rng.choice(num, size=3, replace=False)
        try:
            candidate = umeyama_sim3(src[sample], dst[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        cost = trimmed_cost(candidate)
        if cost < best_cost:
            best, best_cost = candidate, cost

    scale, rot, trans = best
    for _ in range(num_iterations):
        residual = np.linalg.norm(apply_sim3(src, scale, rot, trans) - dst, axis=-1)
        keep = np.argpartition(residual, keep_count - 1)[:keep_count]
        scale, rot, trans = umeyama_sim3(src[keep], dst[keep])
    return scale, rot, trans


def apply_sim3(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray):
    """Apply ``s * R @ p + t`` to points of shape (..., 3)."""
    points = np.asarray(points, dtype=np.float64)
    return scale * (points @ np.asarray(rot, dtype=np.float64).T) + np.asarray(
        trans, dtype=np.float64
    )


def rotation_angle_deg(rot_a: np.ndarray, rot_b: np.ndarray) -> np.ndarray:
    """Geodesic angle in degrees between batches of rotation matrices."""
    rot_a = np.asarray(rot_a, dtype=np.float64)
    rot_b = np.asarray(rot_b, dtype=np.float64)
    rel = np.einsum("...ij,...kj->...ik", rot_a, rot_b)  # rot_a @ rot_b^T
    trace = np.trace(rel, axis1=-2, axis2=-1)
    cos = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def translation_angle_deg(trans_a: np.ndarray, trans_b: np.ndarray) -> np.ndarray:
    """Angle in degrees between translation directions (scale-invariant).

    Zero-length translations yield NaN, which callers must filter out.
    """
    trans_a = np.asarray(trans_a, dtype=np.float64)
    trans_b = np.asarray(trans_b, dtype=np.float64)
    norm_a = np.linalg.norm(trans_a, axis=-1)
    norm_b = np.linalg.norm(trans_b, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cos = (trans_a * trans_b).sum(axis=-1) / (norm_a * norm_b)
    cos = np.where((norm_a > 0) & (norm_b > 0), cos, np.nan)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def relative_poses(poses_c2w: np.ndarray):
    """All ordered pairs i < j of relative poses ``inv(T_i) @ T_j``.

    Returns:
        Tuple (rel, idx_i, idx_j) with rel of shape (P, 4, 4).
    """
    poses_c2w = np.asarray(poses_c2w, dtype=np.float64)
    num = poses_c2w.shape[0]
    idx_i, idx_j = np.triu_indices(num, k=1)
    rel = invert_se3(poses_c2w[idx_i]) @ poses_c2w[idx_j]
    return rel, idx_i, idx_j
