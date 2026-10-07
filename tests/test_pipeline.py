"""End-to-end checks of the data path that does not need any model weights.

These verify the part of the benchmark that is easiest to get subtly wrong: that
the RGB images fed to a model, the ground-truth depth maps and the ground-truth
intrinsics all end up on exactly the same pixel grid, so that a predicted point
map and the GT point map can be compared pixel by pixel.

They need the ``mapanything`` package (for its image loading and cropping
utilities) and are skipped otherwise.
"""

import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
)

from PIL import Image  # noqa: E402

from make_synthetic_scene import look_at_pose, render_view  # noqa: E402

from pointmap_bench.data import (  # noqa: E402
    Scene,
    check_common_size,
    discover_scenes,
    list_images,
    load_ground_truth,
    load_views,
    sorted_stride_sampling,
    views_to_rgb,
)
from pointmap_bench.export import write_ply  # noqa: E402
from pointmap_bench.geometry import (  # noqa: E402
    depth_to_camera_points,
    depth_to_world_points,
)
from pointmap_bench.metrics import evaluate_with_gt  # noqa: E402
from pointmap_bench.prediction import Prediction  # noqa: E402
from pointmap_bench.report import aggregate_by_model, write_reports  # noqa: E402

pytest.importorskip("mapanything", reason="requires the map-anything package")

WIDTH, HEIGHT, NUM_VIEWS = 448, 336, 3


@pytest.fixture(scope="module")
def synthetic_scene(tmp_path_factory):
    """Render a synthetic scene to disk exactly like the CLI helper does."""
    root = tmp_path_factory.mktemp("scene")
    image_dir = root / "images"
    image_dir.mkdir()

    focal = float(WIDTH)
    intrinsics = np.array(
        [
            [focal, 0.0, (WIDTH - 1) / 2.0],
            [0.0, focal, (HEIGHT - 1) / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    depths, poses, names = [], [], []
    for index in range(NUM_VIEWS):
        angle = np.pi * 0.4 * index / (NUM_VIEWS - 1) - np.pi * 0.2
        pose = look_at_pose(
            np.array([2.0 * np.sin(angle), 0.3, -2.0 * np.cos(angle)]),
            np.array([0.0, 0.6, 1.0]),
        )
        rgb, depth = render_view(pose, intrinsics, WIDTH, HEIGHT)
        name = f"{index:03d}.png"
        Image.fromarray(rgb).save(image_dir / name)
        depths.append(depth)
        poses.append(pose)
        names.append(name)

    np.savez_compressed(
        root / "gt.npz",
        depth_z=np.stack(depths).astype(np.float32),
        intrinsics=np.stack([intrinsics] * NUM_VIEWS).astype(np.float32),
        poses_c2w=np.stack(poses).astype(np.float32),
        image_names=np.array(names),
    )
    return {
        "root": str(root),
        "image_dir": str(image_dir),
        "gt_path": str(root / "gt.npz"),
        "depth_z": np.stack(depths).astype(np.float32).astype(np.float64),
        "intrinsics": np.stack([intrinsics] * NUM_VIEWS),
        "poses_c2w": np.stack(poses),
    }


def test_common_size_validation():
    check_common_size((448, 336))  # 448 = 32*14 = 28*16, 336 = 24*14 = 21*16
    with pytest.raises(ValueError, match="divisible by patch size"):
        check_common_size((518, 392))


def test_load_views_respects_the_requested_grid(synthetic_scene):
    paths = list_images(synthetic_scene["image_dir"])
    assert len(paths) == NUM_VIEWS

    views, resolution = load_views(
        paths, norm_type="dinov2", patch_size=14, image_size=(WIDTH, HEIGHT)
    )
    assert resolution == (WIDTH, HEIGHT)
    assert views[0]["img"].shape == (1, 3, HEIGHT, WIDTH)
    assert views[0]["data_norm_type"] == ["dinov2"]

    images = views_to_rgb(views, "dinov2")
    assert images.shape == (NUM_VIEWS, HEIGHT, WIDTH, 3)
    assert images.min() >= 0.0 and images.max() <= 1.0


def test_denormalisation_recovers_the_rendered_pixels(synthetic_scene):
    """load_images -> normalise -> rgb() must round-trip back to the source PNG."""
    paths = list_images(synthetic_scene["image_dir"])
    views, _ = load_views(
        paths, norm_type="dinov2", patch_size=14, image_size=(WIDTH, HEIGHT)
    )
    recovered = views_to_rgb(views, "dinov2")
    original = np.stack(
        [np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0 for path in paths]
    )
    assert np.abs(recovered - original).max() < 2e-2


def test_ground_truth_stays_pixel_aligned_with_the_images(synthetic_scene):
    paths = list_images(synthetic_scene["image_dir"])
    ground_truth = load_ground_truth(
        synthetic_scene["gt_path"], paths, target_size=(WIDTH, HEIGHT)
    )

    assert ground_truth["depth_z"].shape == (NUM_VIEWS, HEIGHT, WIDTH)
    # Rendering already happened at the target resolution, so the crop-resize
    # must be a no-op: any drift here would silently bias every GT metric.
    assert np.allclose(ground_truth["depth_z"], synthetic_scene["depth_z"], atol=1e-4)
    assert np.allclose(
        ground_truth["intrinsics"], synthetic_scene["intrinsics"], atol=1e-3
    )
    assert np.allclose(ground_truth["poses_c2w"], synthetic_scene["poses_c2w"])


def test_ground_truth_is_reordered_to_match_the_image_list(synthetic_scene):
    paths = list_images(synthetic_scene["image_dir"])[::-1]
    ground_truth = load_ground_truth(
        synthetic_scene["gt_path"], paths, target_size=(WIDTH, HEIGHT)
    )
    assert np.allclose(
        ground_truth["depth_z"][0], synthetic_scene["depth_z"][NUM_VIEWS - 1], atol=1e-4
    )


def test_gt_evaluation_of_a_perfect_prediction_through_the_real_loader(synthetic_scene):
    """The full GT path scores an exact prediction as exact."""
    paths = list_images(synthetic_scene["image_dir"])
    views, resolution = load_views(
        paths, norm_type="dinov2", patch_size=14, image_size=(WIDTH, HEIGHT)
    )
    ground_truth = load_ground_truth(synthetic_scene["gt_path"], paths, resolution)

    depth = ground_truth["depth_z"]
    prediction = Prediction(
        points_world=depth_to_world_points(
            depth, ground_truth["intrinsics"], ground_truth["poses_c2w"]
        ),
        points_cam=depth_to_camera_points(depth, ground_truth["intrinsics"]),
        depth_z=depth,
        intrinsics=ground_truth["intrinsics"],
        poses_c2w=ground_truth["poses_c2w"],
        mask=depth > 0,
        images=views_to_rgb(views, "dinov2"),
        is_metric=True,
    )

    result = evaluate_with_gt(
        prediction, depth, ground_truth["intrinsics"], ground_truth["poses_c2w"]
    )
    assert result["cam0/rel_ae"] == pytest.approx(0.0, abs=1e-9)
    assert result["cam0/delta_1.03"] == pytest.approx(1.0)
    assert result["metric/scale_abs_rel"] == pytest.approx(0.0, abs=1e-9)
    assert result["gt/valid_ratio"] > 0.99


def test_scene_discovery_finds_images_and_ground_truth(synthetic_scene):
    scenes = discover_scenes(os.path.dirname(synthetic_scene["root"]))
    matching = [s for s in scenes if s.name == os.path.basename(synthetic_scene["root"])]
    assert matching, "the rendered scene should be discovered"
    scene = matching[0]
    assert scene.num_views == NUM_VIEWS
    assert scene.gt_path is not None and os.path.isfile(scene.gt_path)


def test_ply_export_round_trips(tmp_path):
    points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [np.nan, 0.0, 0.0]])
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 0.5, 1.0], [0.0, 0.0, 0.0]])
    path = tmp_path / "cloud.ply"

    written = write_ply(str(path), points, colors)
    assert written == 2  # the non-finite point is dropped

    with open(path, "rb") as handle:
        header = handle.read(200).decode("ascii", errors="ignore")
        assert "element vertex 2" in header
        handle.seek(header.index("end_header\n") + len("end_header\n"))
        data = np.frombuffer(
            handle.read(),
            dtype=[
                ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("red", "u1"), ("green", "u1"), ("blue", "u1"),
            ],
        )
    assert np.allclose(np.stack([data["x"], data["y"], data["z"]], axis=1), points[:2])
    assert data["red"][0] == 255 and data["blue"][1] == 255


def test_report_writing_and_aggregation(tmp_path):
    records = [
        {"model": "a", "scene": "s1", "status": "ok", "inference_seconds": 1.0,
         "cloud/valid_ratio": 0.9},
        {"model": "a", "scene": "s2", "status": "ok", "inference_seconds": 3.0,
         "cloud/valid_ratio": 0.7},
        {"model": "b", "scene": "s1", "status": "skipped", "reason": "not installed"},
    ]
    aggregated = aggregate_by_model(records)
    assert aggregated["a"]["inference_seconds"] == pytest.approx(2.0)
    assert aggregated["a"]["num_scenes"] == 2
    assert "b" not in aggregated

    paths = write_reports(records, str(tmp_path))
    assert all(os.path.isfile(path) for path in paths.values())
    markdown = open(paths["markdown"], encoding="utf-8").read()
    assert "Skipped / failed runs" in markdown
    assert "not installed" in markdown


def test_pi3_style_confidence_shape_is_accepted():
    """Pi3 and Pi3-X emit a (H, W, 1) confidence where VGGT emits (H, W).

    The benchmark has to normalise that, otherwise Pi3 raises a shape error
    after a full, successful inference and produces no results at all.
    """
    import torch

    from pointmap_bench.prediction import prediction_from_wrapper_output

    num_views, height, width = 2, 112, 112
    outputs = []
    for _ in range(num_views):
        points = torch.rand(1, height, width, 3) + 1.0
        rays = torch.rand(1, height, width, 3) + torch.tensor([0.0, 0.0, 4.0])
        outputs.append(
            {
                "pts3d": points,
                "pts3d_cam": points,
                "ray_directions": torch.nn.functional.normalize(rays, dim=-1),
                "cam_quats": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
                "cam_trans": torch.zeros(1, 3),
                "conf": torch.rand(1, height, width, 1),
            }
        )

    prediction = prediction_from_wrapper_output(
        outputs,
        images=np.zeros((num_views, height, width, 3), dtype=np.float32),
        is_metric=False,
    )
    assert prediction.confidence.shape == (num_views, height, width)


def test_report_warns_when_models_ran_different_scenes(tmp_path):
    records = [
        {"model": "a", "scene": "s0", "status": "ok", "cloud/valid_ratio": 1.0},
        {"model": "a", "scene": "s1", "status": "ok", "cloud/valid_ratio": 1.0},
        {"model": "b", "scene": "s0", "status": "ok", "cloud/valid_ratio": 1.0},
        {"model": "b", "scene": "s1", "status": "error", "reason": "boom"},
    ]
    paths = write_reports(records, str(tmp_path))
    report = open(paths["markdown"], encoding="utf-8").read()
    assert "Not comparable as-is" in report
    assert "num_scenes" in report

    matched = [r for r in records if not (r["model"] == "b" and r["scene"] == "s1")]
    matched = matched + [dict(matched[2], scene="s1")]
    report = open(write_reports(matched, str(tmp_path))["markdown"], encoding="utf-8").read()
    assert "Not comparable as-is" not in report


def test_scene_records_which_views_it_used(synthetic_scene):
    """A result that cannot say which views it saw cannot be reproduced."""
    paths = list_images(synthetic_scene["image_dir"])
    scene = Scene(name="s", image_paths=paths, sampling=sorted_stride_sampling(1, None))

    assert scene.view_names == [os.path.basename(p) for p in paths]
    provenance = scene.provenance()
    assert provenance["num_views"] == len(paths)
    assert provenance["sampling/strategy"] == "sorted_filename_stride"
    assert provenance["sampling/stride"] == 1

    # Same views, same digest; a different selection or order is a different run.
    assert Scene(name="other", image_paths=list(paths)).views_digest == scene.views_digest
    assert Scene(name="s", image_paths=paths[:-1]).views_digest != scene.views_digest
    assert Scene(name="s", image_paths=paths[::-1]).views_digest != scene.views_digest


def test_report_carries_the_run_configuration(tmp_path):
    records = [
        {
            "model": "a",
            "scene": "s1",
            "status": "ok",
            "inference_seconds": 1.0,
            "num_views": 3,
            "views": "000.png 001.png 002.png",
            "views_digest": "abc123def456",
            "sampling/strategy": "sorted_filename_stride",
            "sampling/stride": 2,
        }
    ]
    config = {"image_size": "448x336", "stride": 2, "code_revision": "deadbee"}
    paths = write_reports(records, str(tmp_path), config=config)

    markdown = open(paths["markdown"], encoding="utf-8").read()
    assert "Run configuration" in markdown
    assert "deadbee" in markdown and "448x336" in markdown
    assert "Views used" in markdown and "abc123def456" in markdown
    assert "stride=2" in markdown

    with open(paths["json"], encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["config"]["code_revision"] == "deadbee"
    assert payload["records"][0]["views_digest"] == "abc123def456"

    # Provenance must never be averaged into the score table.
    aggregated = aggregate_by_model(records)
    assert "views_digest" not in aggregated["a"]
    assert "sampling/stride" not in aggregated["a"]
    assert aggregated["a"]["num_views"] == 3


def _pose_from(rotation, translation):
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = translation
    return pose


def test_official_pose_auc_and_ray_error_match_a_perfect_prediction(synthetic_scene):
    """Upstream-defined metrics, computed with upstream's own functions."""
    from pointmap_bench.official_metrics import pose_auc, ray_direction_error_deg

    poses = synthetic_scene["poses_c2w"]
    intrinsics = synthetic_scene["intrinsics"]

    perfect = pose_auc(poses, poses, thresholds=(5, 30))
    assert perfect["pose/auc_5"] == pytest.approx(100.0)
    assert perfect["pose/auc_30"] == pytest.approx(100.0)

    rays = ray_direction_error_deg(intrinsics, intrinsics, HEIGHT, WIDTH)
    assert rays["rays/err_deg"] == pytest.approx(0.0, abs=1e-9)


def test_official_metrics_degrade_when_the_prediction_is_wrong(synthetic_scene):
    poses = synthetic_scene["poses_c2w"]
    intrinsics = synthetic_scene["intrinsics"]
    from pointmap_bench.geometry import quat_xyzw_to_rotmat
    from pointmap_bench.official_metrics import pose_auc, ray_direction_error_deg

    # Rotate one camera by ~16 degrees: every pair touching it is now wrong.
    tilt = quat_xyzw_to_rotmat(np.array([0.0, 0.14, 0.0, 0.99]))
    broken = poses.copy()
    broken[1] = _pose_from(tilt @ poses[1][:3, :3], poses[1][:3, 3])
    degraded = pose_auc(broken, poses, thresholds=(5, 30))
    assert degraded["pose/auc_5"] < 100.0
    assert degraded["pose/auc_30"] < 100.0

    # A 5% focal error is a ray-direction error of order a degree.
    wrong_focal = intrinsics.copy()
    wrong_focal[:, 0, 0] *= 1.05
    wrong_focal[:, 1, 1] *= 1.05
    rays = ray_direction_error_deg(wrong_focal, intrinsics, HEIGHT, WIDTH)
    assert 0.1 < rays["rays/err_deg"] < 5.0


def test_gt_evaluation_includes_the_official_columns(synthetic_scene):
    """evaluate_with_gt must surface the upstream metrics alongside its own."""
    from pointmap_bench.geometry import depth_to_camera_points, depth_to_world_points

    depth = synthetic_scene["depth_z"]
    intrinsics = synthetic_scene["intrinsics"]
    poses = synthetic_scene["poses_c2w"]
    points_cam = depth_to_camera_points(depth, intrinsics)
    prediction = Prediction(
        points_world=depth_to_world_points(depth, intrinsics, poses),
        points_cam=points_cam,
        depth_z=depth,
        intrinsics=intrinsics,
        poses_c2w=poses,
        mask=depth > 0,
        images=np.zeros(depth.shape + (3,), dtype=np.float32),
        is_metric=True,
    )
    result = evaluate_with_gt(prediction, depth, intrinsics, poses)
    assert result["pose/auc_5"] == pytest.approx(100.0)
    assert result["rays/err_deg"] == pytest.approx(0.0, abs=1e-9)
