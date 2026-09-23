"""Unit tests for the NumPy geometry helpers.

These pin down the conventions the whole benchmark relies on: OpenCV camera
frame, camera-to-world poses, scalar-last quaternions and integer-pixel-centre
projection.
"""

import numpy as np
import pytest

from pointmap_bench.geometry import (
    apply_sim3,
    depth_to_camera_points,
    depth_to_world_points,
    invert_se3,
    optimal_scale,
    pose_from_quat_trans,
    project_points,
    quat_xyzw_to_rotmat,
    relative_poses,
    rotation_angle_deg,
    transform_points,
    translation_angle_deg,
    robust_umeyama_sim3,
    umeyama_sim3,
)


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """A uniformly random proper rotation via QR of a Gaussian matrix."""
    q_mat, r_mat = np.linalg.qr(rng.standard_normal((3, 3)))
    q_mat = q_mat @ np.diag(np.sign(np.diag(r_mat)))
    if np.linalg.det(q_mat) < 0:
        q_mat[:, 0] *= -1
    return q_mat


def test_identity_quaternion_gives_identity_rotation():
    assert np.allclose(quat_xyzw_to_rotmat(np.array([0.0, 0.0, 0.0, 1.0])), np.eye(3))


def test_quaternion_matches_known_90_degree_rotation():
    # 90 degrees about +Z maps +X to +Y.
    quat = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    rot = quat_xyzw_to_rotmat(quat)
    assert np.allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)
    assert np.isclose(np.linalg.det(rot), 1.0)


def test_quaternion_batching_matches_elementwise():
    rng = np.random.default_rng(0)
    quats = rng.standard_normal((5, 4))
    batched = quat_xyzw_to_rotmat(quats)
    for index in range(5):
        assert np.allclose(batched[index], quat_xyzw_to_rotmat(quats[index]))


def test_invert_se3_is_a_true_inverse():
    rng = np.random.default_rng(1)
    poses = np.stack(
        [
            pose_from_quat_trans(
                rng.standard_normal(4), rng.standard_normal(3)
            )
            for _ in range(4)
        ]
    )
    product = poses @ invert_se3(poses)
    assert np.allclose(product, np.eye(4), atol=1e-12)


def test_backprojection_and_projection_round_trip():
    height, width = 12, 17
    intrinsics = np.array([[120.0, 0.0, 8.0], [0.0, 110.0, 5.5], [0.0, 0.0, 1.0]])
    rng = np.random.default_rng(2)
    depth = rng.uniform(0.5, 8.0, size=(height, width))

    points_cam = depth_to_camera_points(depth, intrinsics)
    u, v, z = project_points(points_cam, intrinsics)

    cols = np.broadcast_to(np.arange(width, dtype=float), (height, width))
    rows = np.broadcast_to(np.arange(height, dtype=float)[:, None], (height, width))
    assert np.allclose(u, cols, atol=1e-9)
    assert np.allclose(v, rows, atol=1e-9)
    assert np.allclose(z, depth, atol=1e-12)


def test_world_backprojection_matches_manual_transform():
    rng = np.random.default_rng(3)
    intrinsics = np.array([[80.0, 0.0, 4.0], [0.0, 80.0, 3.0], [0.0, 0.0, 1.0]])
    depth = rng.uniform(1.0, 4.0, size=(2, 7, 9))
    poses = np.stack(
        [pose_from_quat_trans(rng.standard_normal(4), rng.standard_normal(3)) for _ in range(2)]
    )
    intrinsics_batched = np.stack([intrinsics, intrinsics])

    world = depth_to_world_points(depth, intrinsics_batched, poses)
    for view in range(2):
        expected = transform_points(
            poses[view], depth_to_camera_points(depth[view], intrinsics)
        )
        assert np.allclose(world[view], expected, atol=1e-12)


def test_umeyama_recovers_a_known_similarity():
    rng = np.random.default_rng(4)
    src = rng.standard_normal((200, 3)) * 2.0
    rot_true = random_rotation(rng)
    scale_true, trans_true = 3.7, np.array([1.0, -2.0, 0.5])
    dst = scale_true * (src @ rot_true.T) + trans_true

    scale, rot, trans = umeyama_sim3(src, dst)
    assert np.isclose(scale, scale_true, rtol=1e-10)
    assert np.allclose(rot, rot_true, atol=1e-9)
    assert np.allclose(trans, trans_true, atol=1e-8)
    assert np.allclose(apply_sim3(src, scale, rot, trans), dst, atol=1e-8)


def test_umeyama_never_returns_a_reflection():
    rng = np.random.default_rng(5)
    src = rng.standard_normal((50, 3))
    dst = src * np.array([1.0, 1.0, -1.0])  # a mirror, not a rotation
    _, rot, _ = umeyama_sim3(src, dst)
    assert np.isclose(np.linalg.det(rot), 1.0, atol=1e-9)


def test_umeyama_without_scale_keeps_unit_scale():
    rng = np.random.default_rng(6)
    src = rng.standard_normal((30, 3))
    rot_true = random_rotation(rng)
    dst = src @ rot_true.T + np.array([0.3, 0.0, -0.2])
    scale, rot, _ = umeyama_sim3(src, dst, with_scale=False)
    assert scale == 1.0
    assert np.allclose(rot, rot_true, atol=1e-9)


def test_optimal_scale_recovers_a_pure_scaling():
    rng = np.random.default_rng(7)
    src = rng.standard_normal((100, 3))
    assert np.isclose(optimal_scale(src, 2.5 * src), 2.5, rtol=1e-12)


def test_rotation_angle_matches_known_value():
    quat = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])  # 90 deg
    angle = rotation_angle_deg(quat_xyzw_to_rotmat(quat), np.eye(3))
    assert np.isclose(angle, 90.0, atol=1e-9)


def test_translation_angle_is_scale_invariant():
    vec_a = np.array([1.0, 0.0, 0.0])
    vec_b = np.array([0.0, 5.0, 0.0])
    assert np.isclose(translation_angle_deg(vec_a, vec_b), 90.0, atol=1e-9)
    assert np.isclose(translation_angle_deg(vec_a, 100.0 * vec_a), 0.0, atol=1e-7)
    assert np.isnan(translation_angle_deg(vec_a, np.zeros(3)))


def test_relative_poses_are_frame_invariant():
    rng = np.random.default_rng(8)
    poses = np.stack(
        [pose_from_quat_trans(rng.standard_normal(4), rng.standard_normal(3)) for _ in range(4)]
    )
    world_change = pose_from_quat_trans(rng.standard_normal(4), rng.standard_normal(3))

    rel_a, idx_i, idx_j = relative_poses(poses)
    rel_b, _, _ = relative_poses(world_change @ poses)
    assert np.allclose(rel_a, rel_b, atol=1e-10)
    assert len(idx_i) == len(idx_j) == 6


def test_transform_points_rejects_wrong_shape():
    with pytest.raises(ValueError):
        transform_points(np.eye(3), np.zeros((5, 3)))


def test_robust_umeyama_ignores_gross_outliers():
    """The trimmed fit recovers the transform the inliers agree on."""
    rng = np.random.default_rng(7)
    src = rng.normal(size=(500, 3))
    rot = quat_xyzw_to_rotmat(np.array([0.1, 0.2, 0.3, 0.9]))
    dst = 2.5 * (src @ rot.T) + np.array([1.0, -2.0, 0.5])

    corrupted = dst.copy()
    corrupted[:50] += rng.normal(scale=50.0, size=(50, 3))

    scale, rot_fit, trans = robust_umeyama_sim3(src, corrupted, trim_ratio=0.2)
    assert scale == pytest.approx(2.5, rel=1e-6)
    np.testing.assert_allclose(rot_fit, rot, atol=1e-6)
    np.testing.assert_allclose(trans, np.array([1.0, -2.0, 0.5]), atol=1e-6)

    # The plain fit is measurably worse on the same data.
    plain_scale, _, _ = umeyama_sim3(src, corrupted)
    assert abs(plain_scale - 2.5) > abs(scale - 2.5)


def test_robust_umeyama_matches_plain_fit_without_trimming():
    rng = np.random.default_rng(3)
    src = rng.normal(size=(100, 3))
    dst = 1.7 * src + np.array([0.2, 0.3, 0.4])
    trimmed = robust_umeyama_sim3(src, dst, trim_ratio=0.0)
    plain = umeyama_sim3(src, dst)
    assert trimmed[0] == pytest.approx(plain[0])
    np.testing.assert_allclose(trimmed[1], plain[1], atol=1e-12)
    np.testing.assert_allclose(trimmed[2], plain[2], atol=1e-12)
