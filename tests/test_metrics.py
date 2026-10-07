"""Unit tests for the benchmark metrics.

The tests build a small ray-cast scene with *exact* ground truth and then check
that the metrics behave the way their definitions promise:

* a perfect prediction scores perfectly,
* a prediction that differs from the GT only by a global similarity transform
  still scores perfectly under the scale/frame-invariant metrics but is caught
  by the metric-scale metric,
* added noise strictly degrades every metric.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
)

from make_synthetic_scene import look_at_pose, render_view  # noqa: E402

from pointmap_bench import metrics as metrics_module  # noqa: E402
from pointmap_bench.geometry import (  # noqa: E402
    apply_sim3,
    depth_to_camera_points,
    depth_to_world_points,
    invert_se3,
    pose_from_quat_trans,
    quat_xyzw_to_rotmat,
)
from pointmap_bench.metrics import (  # noqa: E402
    chamfer_metrics,
    evaluate_poses,
    evaluate_with_gt,
    evaluate_without_gt,
    multiview_consistency,
    nearest_neighbour_distances,
)
from pointmap_bench.prediction import Prediction  # noqa: E402

WIDTH, HEIGHT, NUM_VIEWS = 128, 96, 4


@pytest.fixture(scope="module")
def scene():
    """Ray-cast a few views of the synthetic room and return exact GT arrays."""
    focal = float(WIDTH)
    intrinsics = np.array(
        [
            [focal, 0.0, (WIDTH - 1) / 2.0],
            [0.0, focal, (HEIGHT - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    depths, poses, images = [], [], []
    for index in range(NUM_VIEWS):
        angle = np.pi * 0.4 * index / (NUM_VIEWS - 1) - np.pi * 0.2
        eye = np.array([2.0 * np.sin(angle), 0.3, -2.0 * np.cos(angle)])
        pose = look_at_pose(eye, np.array([0.0, 0.6, 1.0]))
        rgb, depth = render_view(pose, intrinsics, WIDTH, HEIGHT)
        depths.append(depth)
        poses.append(pose)
        images.append(rgb.astype(np.float64) / 255.0)

    return {
        "depth_z": np.stack(depths),
        "intrinsics": np.stack([intrinsics] * NUM_VIEWS),
        "poses_c2w": np.stack(poses),
        "images": np.stack(images),
    }


def make_prediction(scene, points_world=None, depth_z=None, poses_c2w=None):
    """Wrap (possibly perturbed) geometry in a Prediction on the scene's grid."""
    depth_z = scene["depth_z"] if depth_z is None else depth_z
    poses_c2w = scene["poses_c2w"] if poses_c2w is None else poses_c2w
    points_cam = depth_to_camera_points(depth_z, scene["intrinsics"])
    if points_world is None:
        points_world = depth_to_world_points(depth_z, scene["intrinsics"], poses_c2w)
    return Prediction(
        points_world=points_world,
        points_cam=points_cam,
        depth_z=depth_z,
        intrinsics=scene["intrinsics"],
        poses_c2w=poses_c2w,
        mask=depth_z > 0,
        images=scene["images"],
        is_metric=True,
    )


def test_render_produces_usable_depth(scene):
    assert scene["depth_z"].shape == (NUM_VIEWS, HEIGHT, WIDTH)
    assert np.isfinite(scene["depth_z"]).all()
    assert (scene["depth_z"] > 0).mean() > 0.99


def test_perfect_prediction_scores_perfectly(scene):
    prediction = make_prediction(scene)
    result = evaluate_with_gt(
        prediction, scene["depth_z"], scene["intrinsics"], scene["poses_c2w"]
    )

    assert result["cam0/mae"] == pytest.approx(0.0, abs=1e-9)
    assert result["cam0/rel_ae"] == pytest.approx(0.0, abs=1e-9)
    assert result["cam0/delta_1.03"] == pytest.approx(1.0)
    assert result["cam0/inlier_l2_0.02"] == pytest.approx(1.0)
    assert result["sim3/mae"] == pytest.approx(0.0, abs=1e-8)
    assert result["sim3/scale"] == pytest.approx(1.0, rel=1e-9)
    assert result["metric/scale_abs_rel"] == pytest.approx(0.0, abs=1e-9)
    assert result["depth/abs_rel"] == pytest.approx(0.0, abs=1e-9)
    assert result["depth/delta_1.03"] == pytest.approx(1.0)
    assert result["pose/rot_err_deg_median"] == pytest.approx(0.0, abs=1e-9)
    assert result["pose/ate_rmse"] == pytest.approx(0.0, abs=1e-8)


def test_similarity_transformed_prediction_is_scale_and_frame_invariant(scene):
    """A globally rescaled+rotated reconstruction is still geometrically perfect."""
    scale = 3.0
    rot = quat_xyzw_to_rotmat(np.array([0.1, -0.3, 0.2, 0.9]))
    trans = np.array([5.0, -1.0, 2.0])

    world_transform = np.eye(4)
    world_transform[:3, :3] = rot
    world_transform[:3, 3] = trans

    gt_points = depth_to_world_points(
        scene["depth_z"], scene["intrinsics"], scene["poses_c2w"]
    )
    pred_points = apply_sim3(gt_points, scale, rot, trans)
    pred_poses = scene["poses_c2w"].copy()
    pred_poses[:, :3, :3] = rot @ scene["poses_c2w"][:, :3, :3]
    pred_poses[:, :3, 3] = scale * (scene["poses_c2w"][:, :3, 3] @ rot.T) + trans

    prediction = make_prediction(
        scene,
        points_world=pred_points,
        depth_z=scene["depth_z"] * scale,
        poses_c2w=pred_poses,
    )
    result = evaluate_with_gt(
        prediction, scene["depth_z"], scene["intrinsics"], scene["poses_c2w"]
    )

    # Frame- and scale-invariant metrics stay perfect ...
    assert result["cam0/mae"] == pytest.approx(0.0, abs=1e-9)
    assert result["cam0/delta_1.03"] == pytest.approx(1.0)
    assert result["sim3/scale"] == pytest.approx(1.0 / scale, rel=1e-8)
    assert result["sim3/mae"] == pytest.approx(0.0, abs=1e-7)
    assert result["pose/rot_err_deg_median"] == pytest.approx(0.0, abs=1e-7)
    assert result["depth/abs_rel"] == pytest.approx(0.0, abs=1e-9)
    # ... but the metric-scale metric correctly reports a 3x scale error.
    assert result["metric/scale_ratio"] == pytest.approx(scale, rel=1e-8)
    assert result["metric/scale_abs_rel"] == pytest.approx(scale - 1.0, rel=1e-8)


def test_noise_strictly_degrades_the_metrics(scene):
    rng = np.random.default_rng(0)
    clean = make_prediction(scene)
    noisy_depth = scene["depth_z"] * (
        1.0 + 0.05 * rng.standard_normal(scene["depth_z"].shape)
    )
    noisy = make_prediction(scene, depth_z=noisy_depth)

    clean_result = evaluate_with_gt(
        clean, scene["depth_z"], scene["intrinsics"], scene["poses_c2w"]
    )
    noisy_result = evaluate_with_gt(
        noisy, scene["depth_z"], scene["intrinsics"], scene["poses_c2w"]
    )

    assert noisy_result["cam0/mae"] > clean_result["cam0/mae"]
    assert noisy_result["cam0/rel_ae"] > clean_result["cam0/rel_ae"]
    assert noisy_result["cam0/delta_1.03"] < clean_result["cam0/delta_1.03"]
    assert noisy_result["depth/abs_rel"] > clean_result["depth/abs_rel"]
    assert noisy_result["sim3/mae"] > clean_result["sim3/mae"]


def test_pose_errors_detect_a_rotated_camera(scene):
    perturbed = scene["poses_c2w"].copy()
    angle = np.deg2rad(10.0)
    delta = quat_xyzw_to_rotmat(np.array([0.0, np.sin(angle / 2), 0.0, np.cos(angle / 2)]))
    perturbed[-1, :3, :3] = delta @ perturbed[-1, :3, :3]

    result = evaluate_poses(perturbed, scene["poses_c2w"])
    assert result["pose/rot_err_deg_mean"] > 1.0
    # Three of the six view pairs involve the perturbed camera.
    assert result["pose/rra_5deg"] == pytest.approx(0.5)
    assert evaluate_poses(scene["poses_c2w"], scene["poses_c2w"])[
        "pose/rot_err_deg_mean"
    ] == pytest.approx(0.0, abs=1e-9)


def test_self_consistent_reconstruction_scores_high_consistency(scene):
    prediction = make_prediction(scene)
    result = multiview_consistency(
        prediction.points_world,
        prediction.depth_z,
        prediction.intrinsics,
        prediction.poses_c2w,
        prediction.mask,
    )
    # Occluded pixels legitimately disagree, hence a threshold below 1.
    assert result["consistency/inlier_1.05"] > 0.75
    assert result["consistency/rel_depth_err_median"] < 0.02
    assert result["consistency/overlap_ratio"] > 0.3


def test_inconsistent_reconstruction_scores_lower(scene):
    consistent = make_prediction(scene)
    broken_depth = scene["depth_z"].copy()
    broken_depth[1] *= 1.4  # one view disagrees with the rest
    broken = make_prediction(scene, depth_z=broken_depth)

    good = evaluate_without_gt(consistent)
    bad = evaluate_without_gt(broken)
    assert bad["consistency/inlier_1.05"] < good["consistency/inlier_1.05"]
    assert bad["consistency/rel_depth_err_median"] > good["consistency/rel_depth_err_median"]


def test_consistency_is_invariant_to_global_scale(scene):
    base = make_prediction(scene)
    scaled_poses = scene["poses_c2w"].copy()
    scaled_poses[:, :3, 3] *= 7.0
    scaled = make_prediction(
        scene, depth_z=scene["depth_z"] * 7.0, poses_c2w=scaled_poses
    )

    base_result = evaluate_without_gt(base)
    scaled_result = evaluate_without_gt(scaled)
    assert scaled_result["consistency/inlier_1.03"] == pytest.approx(
        base_result["consistency/inlier_1.03"], abs=1e-9
    )


def test_chamfer_is_zero_for_identical_clouds_even_when_subsampled():
    """Subsampling must not invent a distance floor.

    Subsampling both sides makes the query and reference points land on
    different samples of the same surface, so identical clouds score the
    sampling spacing instead of zero - and the floor depends on how many
    points a model produced, which biases the comparison toward sparse
    predictions. This test uses more points than max_points on purpose; the
    original one stayed under the threshold and so never reached the code
    that was wrong.
    """
    rng = np.random.default_rng(2)
    cloud = rng.standard_normal((5000, 3))
    result = chamfer_metrics(cloud, cloud.copy(), max_points=200)
    assert result["accuracy_mean"] == pytest.approx(0.0, abs=1e-12)
    assert result["completeness_mean"] == pytest.approx(0.0, abs=1e-12)
    assert result["chamfer_mean"] == pytest.approx(0.0, abs=1e-12)


def test_chamfer_still_measures_a_real_offset_when_subsampled():
    """A thin surface moved along its normal must report exactly that offset."""
    rng = np.random.default_rng(3)
    count = 5000
    plane = np.column_stack(
        [rng.uniform(-5, 5, count), rng.uniform(-5, 5, count), np.zeros(count)]
    )
    result = chamfer_metrics(
        plane + np.array([0.0, 0.0, 0.5]), plane, max_points=500
    )
    assert result["accuracy_mean"] == pytest.approx(0.5, abs=1e-6)
    assert result["completeness_mean"] == pytest.approx(0.5, abs=1e-6)


def test_chamfer_of_identical_clouds_is_zero():
    rng = np.random.default_rng(1)
    cloud = rng.standard_normal((500, 3))
    result = chamfer_metrics(cloud, cloud)
    assert result["accuracy_mean"] == pytest.approx(0.0, abs=1e-12)
    assert result["completeness_mean"] == pytest.approx(0.0, abs=1e-12)
    assert result["chamfer_mean"] == pytest.approx(0.0, abs=1e-12)


def test_chamfer_matches_a_known_offset():
    cloud = np.zeros((10, 3))
    shifted = cloud + np.array([0.0, 0.0, 0.25])
    result = chamfer_metrics(cloud, shifted)
    assert result["accuracy_mean"] == pytest.approx(0.25)
    assert result["completeness_mean"] == pytest.approx(0.25)


def test_numpy_nearest_neighbour_fallback_matches_kdtree(monkeypatch):
    rng = np.random.default_rng(2)
    query = rng.standard_normal((97, 3))
    reference = rng.standard_normal((211, 3))

    reference_result = nearest_neighbour_distances(query, reference)
    monkeypatch.setattr(metrics_module, "_KDTree", None)
    fallback_result = nearest_neighbour_distances(query, reference, chunk_size=16)

    brute_force = np.sqrt(
        ((query[:, None, :] - reference[None, :, :]) ** 2).sum(-1)
    ).min(axis=1)
    assert np.allclose(reference_result, brute_force, atol=1e-9)
    assert np.allclose(fallback_result, brute_force, atol=1e-9)


def test_evaluate_with_gt_rejects_mismatched_grids(scene):
    prediction = make_prediction(scene)
    with pytest.raises(ValueError, match="does not match prediction shape"):
        evaluate_with_gt(
            prediction,
            scene["depth_z"][:, :-2],
            scene["intrinsics"],
            scene["poses_c2w"],
        )


def test_prediction_validates_shapes(scene):
    with pytest.raises(ValueError, match="poses_c2w"):
        Prediction(
            points_world=depth_to_world_points(
                scene["depth_z"], scene["intrinsics"], scene["poses_c2w"]
            ),
            points_cam=depth_to_camera_points(scene["depth_z"], scene["intrinsics"]),
            depth_z=scene["depth_z"],
            intrinsics=scene["intrinsics"],
            poses_c2w=pose_from_quat_trans(
                np.zeros((NUM_VIEWS - 1, 4)) + np.array([0, 0, 0, 1.0]),
                np.zeros((NUM_VIEWS - 1, 3)),
            ),
            mask=scene["depth_z"] > 0,
            images=scene["images"],
        )


def test_completeness_penalises_geometry_the_model_did_not_predict(scene):
    """A model that predicts half the image must not score a perfect completeness.

    Accuracy is measured on the pixels a model predicted, but completeness has
    to be measured against the whole ground truth - otherwise masking away the
    hard half of a scene looks like flawless reconstruction.
    """
    args = (scene["depth_z"], scene["intrinsics"], scene["poses_c2w"])
    full = evaluate_with_gt(make_prediction(scene), *args)

    partial_prediction = make_prediction(scene)
    keep = np.ones_like(partial_prediction.mask)
    keep[:, :, WIDTH // 2 :] = False
    partial_prediction.mask = partial_prediction.mask & keep
    partial = evaluate_with_gt(partial_prediction, *args)

    assert full["gt/covered_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert partial["gt/covered_ratio"] < 0.6
    # The half it did predict is still exactly right.
    assert partial["sim3/accuracy_mean"] == pytest.approx(0.0, abs=1e-6)
    # The half it skipped is not free.
    assert full["sim3/completeness_mean"] == pytest.approx(0.0, abs=1e-6)
    assert partial["sim3/completeness_mean"] > 1e-3


def test_similarity_alignment_resists_outliers(scene):
    """A few diverging points must not drag the alignment of everything else."""
    points_world = depth_to_world_points(
        scene["depth_z"], scene["intrinsics"], scene["poses_c2w"]
    ).copy()
    rng = np.random.default_rng(0)
    outlier = rng.random(points_world.shape[:3]) < 0.1
    points_world[outlier] += np.array([80.0, -60.0, 120.0])

    result = evaluate_with_gt(
        make_prediction(scene, points_world=points_world),
        scene["depth_z"],
        scene["intrinsics"],
        scene["poses_c2w"],
    )

    # The 90% of points that are exact stay exact under the robust fit.
    assert result["sim3/median_ae"] == pytest.approx(0.0, abs=1e-6)
    assert result["sim3/inlier_l2_0.02"] > 0.85
    # The plain least-squares fit is dragged off that geometry by the outliers.
    # The comparison is on the medians: the least-squares fit minimises squared
    # error, so it posts the lower *mean* by smearing the outliers over every
    # other point, which is exactly the behaviour being guarded against.
    assert result["sim3/median_ae_lsq"] > 1.0
    assert result["sim3/mae_lsq"] < result["sim3/mae"]


def test_metric_scale_is_not_reported_for_scale_ambiguous_models(scene):
    metric_prediction = make_prediction(scene)
    ambiguous_prediction = make_prediction(scene)
    ambiguous_prediction.is_metric = False
    args = (scene["depth_z"], scene["intrinsics"], scene["poses_c2w"])

    assert "metric/scale_abs_rel" in evaluate_with_gt(metric_prediction, *args)
    assert "metric/scale_abs_rel" not in evaluate_with_gt(ambiguous_prediction, *args)


def test_sanity_check_catches_a_flipped_pose_convention(scene):
    """The harness check must fire when poses and point maps disagree.

    A wrapper returning world-to-camera where camera-to-world is expected would
    otherwise show up as "this model is bad" rather than "this comparison is
    broken", which is the failure mode this check exists to prevent.
    """
    healthy = evaluate_without_gt(make_prediction(scene))
    assert healthy["sanity/pose_pointmap_rel_err"] == pytest.approx(0.0, abs=1e-9)
    assert healthy["sanity/depth_pointmap_rel_err"] == pytest.approx(0.0, abs=1e-9)

    broken_prediction = make_prediction(scene)
    broken_prediction.poses_c2w = invert_se3(broken_prediction.poses_c2w)
    broken = evaluate_without_gt(broken_prediction)
    assert broken["sanity/pose_pointmap_rel_err"] > 0.1


def test_rel_extent_normaliser_does_not_depend_on_the_model_mask(scene):
    """The scene extent is a property of the scene, not of the prediction.

    Normalising by the extent of the per-model intersection would hand a model
    that masks aggressively a smaller denominator, making identical geometry
    look worse - the same tilt the coverage and completeness fixes removed.
    """
    args = (scene["depth_z"], scene["intrinsics"], scene["poses_c2w"])

    noisy_points = depth_to_world_points(
        scene["depth_z"], scene["intrinsics"], scene["poses_c2w"]
    ) + 0.01

    full = make_prediction(scene, points_world=noisy_points)
    half = make_prediction(scene, points_world=noisy_points)
    keep = np.ones_like(half.mask)
    keep[:, :, WIDTH // 2 :] = False
    half.mask = half.mask & keep

    full_result = evaluate_with_gt(full, *args)
    half_result = evaluate_with_gt(half, *args)

    # accuracy_mean / accuracy_mean_rel_extent recovers the normaliser used.
    def normaliser(result):
        return result["sim3/accuracy_mean"] / result["sim3/accuracy_mean_rel_extent"]

    assert half_result["gt/covered_ratio"] < 0.6
    assert normaliser(half_result) == pytest.approx(normaliser(full_result), rel=1e-9)
