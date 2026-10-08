"""Tests for reading WAI dataset settings from map-anything's configs, and for
working out which files a WAI multi-view set needs before downloading them.

Neither needs any WAI data: the configs are written to a temporary directory
laid out like map-anything's ``configs/dataset``, and the sampler runs against a
stub dataset that behaves like ``mapanything.datasets.wai.*``.
"""

import sys
import types
from pathlib import Path

import pytest

from pointmap_bench.sources import available_wai_datasets, wai_dataset_spec

DEFAULTS = """
num_views: 2
principal_point_centered: false
train:
  variable_num_views: true
test:
  variable_num_views: false
"""

TEST_CONFIG = """
dataset_str:
  "{cls}(
    split='${{dataset.{name}_wai.test.split}}',
    resolution=${{dataset.{name}_wai.test.dataset_resolution}},
    principal_point_centered=${{dataset.{name}_wai.test.principal_point_centered}},
    seed=${{dataset.{name}_wai.test.seed}},
    transform='${{dataset.{name}_wai.test.transform}}',
    data_norm_type='${{dataset.{name}_wai.test.data_norm_type}}',
    ROOT='${{dataset.{name}_wai.test.ROOT}}',
    dataset_metadata_dir='${{dataset.{name}_wai.test.dataset_metadata_dir}}',
    variable_num_views=${{dataset.{name}_wai.test.variable_num_views}},
    num_views=${{dataset.{name}_wai.test.num_views}},
    covisibility_thres=${{dataset.{name}_wai.test.covisibility_thres}})"
split: 'test'
dataset_resolution: ${{dataset.resolution_test_{name}}}
principal_point_centered: ${{dataset.principal_point_centered}}
seed: 777
transform: 'imgnorm'
data_norm_type: ${{model.data_norm_type}}
ROOT: ${{root_data_dir}}/{dirname}
dataset_metadata_dir: ${{mapanything_dataset_metadata_dir}}
variable_num_views: ${{dataset.test.variable_num_views}}
num_views: ${{dataset.num_views}}
covisibility_thres: {thres}
"""


def write_configs(root: Path, datasets) -> Path:
    config_dir = root / "configs" / "dataset"
    config_dir.mkdir(parents=True)
    (config_dir / "default.yaml").write_text(DEFAULTS)
    for name, cls, dirname, thres in datasets:
        test_dir = config_dir / f"{name}_wai" / "test"
        test_dir.mkdir(parents=True)
        (test_dir / "default.yaml").write_text(
            TEST_CONFIG.format(name=name, cls=cls, dirname=dirname, thres=thres)
        )
    return root


@pytest.fixture
def configs(tmp_path):
    root = write_configs(
        tmp_path,
        [("eth3d", "ETH3DWAI", "eth3d", 0.025), ("scannetpp", "ScanNetPPWAI", "scannetppv2", 0.25)],
    )
    # A train-only dataset: has a config folder but no test split.
    (root / "configs" / "dataset" / "dl3dv_wai" / "train").mkdir(parents=True)
    return str(root)


def test_only_datasets_with_a_test_config_are_available(configs):
    assert available_wai_datasets(configs) == ["eth3d", "scannetpp"]


def test_spec_is_read_from_the_upstream_config(configs):
    spec = wai_dataset_spec("scannetpp", configs)
    assert spec.class_name == "ScanNetPPWAI"
    assert spec.data_dirname == "scannetppv2"
    # Literals come from the test config, ${dataset.*} references from the
    # shared defaults, and arguments the benchmark supplies are left out.
    assert spec.kwargs == {
        "split": "test",
        "principal_point_centered": False,
        "seed": 777,
        "transform": "imgnorm",
        "variable_num_views": False,
        "covisibility_thres": 0.25,
    }
    assert wai_dataset_spec("eth3d", configs).kwargs["covisibility_thres"] == 0.025


def test_unknown_dataset_lists_the_available_ones(configs):
    with pytest.raises(ValueError, match=r"\['eth3d', 'scannetpp'\]"):
        wai_dataset_spec("dl3dv", configs)


def test_an_argument_that_does_not_resolve_is_refused_not_guessed(configs):
    path = Path(configs) / "configs" / "dataset" / "eth3d_wai" / "test" / "default.yaml"
    path.write_text(path.read_text().replace("seed: 777", "seed: ${machine.seed}"))
    with pytest.raises(ValueError, match="'seed'"):
        wai_dataset_spec("eth3d", configs)


# --------------------------------------------------------------------------
# wai_sample_plan
# --------------------------------------------------------------------------

FRAMES = ["f0", "f1", "f2", "f3", "f4"]
MODALITIES = ["image", "depth", "pred_mask/moge2"]


def _scene_meta():
    return {
        "frame_names": {name: i for i, name in enumerate(FRAMES)},
        "frames": [
            {
                "frame_name": name,
                "image": f"images/{name}.png",
                "depth": f"depth/{name}.exr",
                "moge": f"moge/v0/mask/{name}.png",
            }
            for name in FRAMES
        ],
    }


@pytest.fixture
def stub_module(monkeypatch):
    """A module shaped like mapanything.datasets.wai.<name>.

    ``load_frame`` resolves each modality to a path and reads it through
    ``wai.core.load_data``, which is how map-anything's own one works.
    """
    core = pytest.importorskip("mapanything.utils.wai.core")
    module = types.ModuleType("stub_wai_dataset")
    paths = {"image": "image", "depth": "depth", "pred_mask/moge2": "moge"}

    def load_frame(scene_root, frame_key, modalities=None, scene_meta=None):
        frame = scene_meta["frames"][scene_meta["frame_names"][frame_key]]
        return {m: core.load_data(Path(scene_root, frame[paths[m]])) for m in modalities}

    module.load_frame = load_frame
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


def make_dataset(module, indices, index_to_frame=lambda i: FRAMES[i]):
    class StubDataset:
        def _sample_view_indices(self, num_views, num_in_scene, covisibility):
            return list(indices)

        def _getitem_fn(self, key):
            meta = _scene_meta()
            chosen = self._sample_view_indices(len(indices), len(FRAMES), None)
            for i in chosen:
                data = module.load_frame(
                    "/data/scene", index_to_frame(i), modalities=MODALITIES, scene_meta=meta
                )
                data["image"].permute(1, 2, 0)  # what the real datasets do next

    StubDataset.__module__ = module.__name__
    return StubDataset()


def test_plan_lists_the_sampled_frames_and_every_file_they_need(stub_module):
    from pointmap_bench.sources import wai_sample_plan

    plan = wai_sample_plan(make_dataset(stub_module, [3, 0, 3]), index=0)

    assert plan["scene_root"] == "/data/scene"
    assert plan["frames"] == ["f3", "f0", "f3"]  # view order, repeats kept
    assert plan["files"] == [  # de-duplicated, relative, nested modality included
        "depth/f0.exr",
        "depth/f3.exr",
        "images/f0.png",
        "images/f3.png",
        "moge/v0/mask/f0.png",
        "moge/v0/mask/f3.png",
    ]


def test_plan_puts_everything_it_patched_back(stub_module):
    from mapanything.utils.wai import core

    from pointmap_bench.sources import wai_sample_plan

    load_frame, load_data = stub_module.load_frame, core.load_data
    dataset = make_dataset(stub_module, [1, 2])
    wai_sample_plan(dataset, index=0)

    assert stub_module.load_frame is load_frame
    assert core.load_data is load_data
    assert "_sample_view_indices" not in vars(dataset)


def test_plan_refuses_a_dataset_whose_frame_order_it_does_not_understand(stub_module):
    from pointmap_bench.sources import wai_sample_plan

    reversed_order = make_dataset(stub_module, [1, 2], index_to_frame=lambda i: FRAMES[-1 - i])
    with pytest.raises(RuntimeError, match="frame order"):
        wai_sample_plan(reversed_order, index=0)


def test_plan_refuses_a_dataset_that_bypasses_load_frame(stub_module):
    from pointmap_bench.sources import wai_sample_plan

    dataset = make_dataset(stub_module, [0, 1])
    dataset._getitem_fn = lambda key: None
    with pytest.raises(RuntimeError, match="loading code has changed"):
        wai_sample_plan(dataset, index=0)


def test_dataset_without_a_test_split_is_named_in_the_error(tmp_path):
    root = write_configs(tmp_path, [("eth3d", "ETH3DWAI", "eth3d", 0.025)])
    with pytest.raises(ValueError, match="No WAI test config"):
        wai_dataset_spec("tav2_wb", str(root))
