"""Where a multi-view sample comes from.

The benchmark started out able to read exactly one thing: a folder of images.
That is the right default for looking at your own data, but it is not how the
reconstruction literature evaluates, and the gap is not cosmetic - **which
views a model is given is part of the protocol**, not a detail of file
handling. A folder read in filename order with a fixed stride cannot reproduce
a number from a paper, however correct every metric downstream is.

So the input side is a small interface with two implementations:

* :class:`FolderViewSource` - a directory of images plus an optional
  ``gt.npz``. Views are chosen by sorted filename and stride.
* :class:`WaiViewSource` - a map-anything WAI dataset (ETH3D, ScanNet++v2,
  TartanAirV2-WB, ...), which samples views by random walk over a precomputed
  covisibility matrix so the sampled views form one connected component. This
  is the protocol map-anything's own ``dense_n_view`` benchmark uses.

Both produce the same :class:`PreparedViews`, so everything downstream -
inference, metrics, export, reporting - is unchanged by the choice. Swapping
the data source must not be able to change what a metric means.

Note on normalisation: a WAI dataset bakes its ``data_norm_type`` in at
construction, while models in this benchmark need three different ones
(dinov2 / dust3r / identity). :class:`WaiViewSource` therefore builds one
dataset per normalisation through the factory it is given, and caches them.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .data import Scene, load_ground_truth, load_views, views_to_rgb


@dataclass
class PreparedViews:
    """One multi-view sample, ready to hand to a model.

    Attributes:
        views: Per-view dicts in map-anything's input format, batched as
            ``(1, ...)`` - what every model wrapper expects.
        images: (V, H, W, 3) un-normalised RGB in [0, 1], for coloured exports.
        resolution: (width, height) the views were loaded at.
        norm_type: The normalisation actually applied to ``views``.
        scene_name: Scene identifier, also used for MASt3R's cache keys.
        view_names: Per-view names, in the order the model sees them.
        ground_truth: ``depth_z`` / ``intrinsics`` / ``poses_c2w`` on the same
            pixel grid as ``views``, or None when the source has no GT.
        provenance: What this sample is and how its views were chosen; copied
            onto the result record so a score can be reproduced.
    """

    views: List[dict]
    images: np.ndarray
    resolution: Tuple[int, int]
    norm_type: str
    scene_name: str
    view_names: List[str]
    ground_truth: Optional[Dict[str, np.ndarray]] = None
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_views(self) -> int:
        return len(self.views)


def views_digest(view_names) -> str:
    """Short stable hash of an ordered view list."""
    return hashlib.sha1("|".join(view_names).encode("utf-8")).hexdigest()[:12]


class ViewSource:
    """Something that can produce a multi-view sample for a given model.

    Implementations must honour ``norm_type``, ``patch_size`` and
    ``image_size`` exactly: a source that quietly hands back a different
    normalisation or grid would make the comparison meaningless in a way no
    metric could detect.
    """

    name: str

    def prepare(
        self,
        norm_type: str,
        patch_size: int,
        resolution_set: int = 518,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> PreparedViews:
        raise NotImplementedError


class FolderViewSource(ViewSource):
    """A folder of images, with optional posed-depth ground truth."""

    def __init__(self, scene: Scene):
        self.scene = scene
        self.name = scene.name

    def prepare(
        self,
        norm_type: str,
        patch_size: int,
        resolution_set: int = 518,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> PreparedViews:
        views, resolution = load_views(
            self.scene.image_paths,
            norm_type=norm_type,
            patch_size=patch_size,
            resolution_set=resolution_set,
            image_size=image_size,
        )
        images = views_to_rgb(views, norm_type)

        ground_truth = None
        if self.scene.gt_path:
            # Loaded at the resolution this model actually ran at, so the GT
            # point map and the predicted point map share a pixel grid.
            ground_truth = load_ground_truth(
                self.scene.gt_path, self.scene.image_paths, resolution
            )

        return PreparedViews(
            views=views,
            images=images,
            resolution=resolution,
            norm_type=norm_type,
            scene_name=self.scene.name,
            view_names=self.scene.view_names,
            ground_truth=ground_truth,
            provenance=self.scene.provenance(),
        )


# Only these keys may reach a model. The list is deliberately short, for two
# independent reasons:
#
# 1. MapAnything.infer validates its input against an allow-list
#    (ALLOWED_VIEW_KEYS in mapanything/utils/inference.py) and raises on
#    anything else. A WAI view carries depthmap, camera_intrinsics,
#    camera_pose, label, pts3d, valid_mask and more, so passing it through
#    would break the one model the benchmark is named after, while the hydra
#    wrappers - which do not validate - carried on.
# 2. More importantly, a WAI view *contains the ground truth*, and
#    MapAnything.infer accepts intrinsics, depth and poses as optional
#    geometric priors. Forwarding those would hand the model the answer and
#    quietly turn the benchmark into a measurement of nothing.
#
# This benchmark evaluates images-only reconstruction, so images are all a
# model gets.
_MODEL_INPUT_KEYS = ("img", "data_norm_type", "true_shape", "instance", "idx")


def _as_batched_view(view: dict) -> dict:
    """Turn one WAI dataset view into a model-wrapper input view.

    The dataset yields un-batched tensors for training; the wrappers expect a
    leading batch dimension and ``data_norm_type`` as a list. Everything
    outside :data:`_MODEL_INPUT_KEYS` is dropped - see the note above.
    """
    import torch

    out = {key: view[key] for key in _MODEL_INPUT_KEYS if key in view}
    img = view["img"]
    out["img"] = img if img.dim() == 4 else img[None]
    norm_type = view["data_norm_type"]
    out["data_norm_type"] = (
        norm_type if isinstance(norm_type, (list, tuple)) else [norm_type]
    )
    true_shape = view.get("true_shape")
    if true_shape is not None:
        tensor = torch.as_tensor(np.asarray(true_shape))
        out["true_shape"] = tensor if tensor.dim() == 2 else tensor[None]
    return out


# --------------------------------------------------------------------------
# WAI dataset settings, read from map-anything's own configs
#
# The seed and covisibility threshold decide which views the random walk picks,
# so they are part of the protocol. Rather than copying them here (and letting
# them drift), they are read from configs/dataset/<name>_wai/test/default.yaml
# in the map-anything checkout - the file upstream's dense_n_view benchmark
# itself is built from. A dataset is usable as soon as upstream has a test
# config for it; no code change is needed here.
# --------------------------------------------------------------------------

# Constructor arguments the benchmark supplies itself. Everything else must
# resolve to a literal in the upstream config, or the spec refuses to build.
_WAI_RUNTIME_ARGS = ("resolution", "num_views", "data_norm_type", "ROOT", "dataset_metadata_dir")

_INTERPOLATION = re.compile(r"\$\{([^}]+)\}")


@dataclass
class WaiDatasetSpec:
    """How upstream constructs one WAI test dataset.

    Attributes:
        name: Config name, e.g. ``eth3d`` for ``configs/dataset/eth3d_wai``.
        class_name: Dataset class exported by ``mapanything.datasets``.
        kwargs: Constructor arguments resolved from the upstream config,
            excluding :data:`_WAI_RUNTIME_ARGS`.
        data_dirname: Folder name of the data under upstream's data root
            (``ROOT: ${root_data_dir}/<data_dirname>``), which is also the
            folder name in the ``facebook/map-anything-benchmarking`` repo.
        config_path: The config file this was read from, for provenance.
    """

    name: str
    class_name: str
    kwargs: Dict[str, Any]
    data_dirname: str
    config_path: str


def _wai_config_dir(mapanything_root: Optional[str] = None) -> str:
    if mapanything_root is None:
        from .models import mapanything_root as find_root

        mapanything_root = find_root()
    return os.path.join(mapanything_root, "configs", "dataset")


def available_wai_datasets(mapanything_root: Optional[str] = None) -> List[str]:
    """WAI datasets that upstream defines a test split for."""
    config_dir = _wai_config_dir(mapanything_root)
    return sorted(
        entry[: -len("_wai")]
        for entry in os.listdir(config_dir)
        if entry.endswith("_wai")
        and os.path.isfile(os.path.join(config_dir, entry, "test", "default.yaml"))
    )


def _lookup(tree: Dict[str, Any], dotted: str):
    node: Any = tree
    for key in dotted.split("."):
        if not isinstance(node, dict) or key not in node:
            raise KeyError(dotted)
        node = node[key]
    return node


def wai_dataset_spec(name: str, mapanything_root: Optional[str] = None) -> WaiDatasetSpec:
    """Read how upstream builds the ``name`` WAI test dataset.

    Each argument of the config's ``dataset_str`` is resolved through the
    test config and, for ``${dataset.*}`` references, the shared
    ``configs/dataset/default.yaml``. An argument that resolves to neither a
    literal nor one of :data:`_WAI_RUNTIME_ARGS` raises instead of being
    guessed.
    """
    import yaml

    config_dir = _wai_config_dir(mapanything_root)
    config_path = os.path.join(config_dir, f"{name}_wai", "test", "default.yaml")
    if not os.path.isfile(config_path):
        raise ValueError(
            f"No WAI test config for '{name}' ({config_path}). "
            f"Available: {available_wai_datasets(mapanything_root)}"
        )
    with open(config_path) as fh:
        test_cfg = yaml.safe_load(fh)
    with open(os.path.join(config_dir, "default.yaml")) as fh:
        dataset_defaults = yaml.safe_load(fh)

    dataset_str = " ".join(str(test_cfg["dataset_str"]).split())
    match = re.match(r"(\w+)\((.*)\)$", dataset_str)
    if not match:
        raise ValueError(f"Cannot parse dataset_str in {config_path}: {dataset_str}")
    class_name, arg_text = match.groups()

    own_prefix = f"dataset.{name}_wai.test."
    kwargs: Dict[str, Any] = {}
    for arg, ref in re.findall(r"(\w+)\s*=\s*'?\$\{([^}]+)\}'?", arg_text):
        if arg in _WAI_RUNTIME_ARGS:
            continue
        if not ref.startswith(own_prefix):
            raise ValueError(f"{config_path}: unexpected reference ${{{ref}}} for {arg}")
        value = test_cfg[ref[len(own_prefix):]]
        inner = _INTERPOLATION.fullmatch(value) if isinstance(value, str) else None
        if inner and inner.group(1).startswith("dataset."):
            try:
                value = _lookup(dataset_defaults, inner.group(1)[len("dataset."):])
            except KeyError:
                pass
        if isinstance(value, str) and _INTERPOLATION.search(value):
            raise ValueError(
                f"{config_path}: argument '{arg}' = {value!r} does not resolve to a "
                "literal, and the benchmark does not know how to supply it."
            )
        kwargs[arg] = value

    root_match = re.fullmatch(r"\$\{root_data_dir\}/(.+)", str(test_cfg.get("ROOT", "")))
    if not root_match:
        raise ValueError(f"{config_path}: cannot read the data folder name from ROOT")

    return WaiDatasetSpec(
        name=name,
        class_name=class_name,
        kwargs=kwargs,
        data_dirname=root_match.group(1),
        config_path=config_path,
    )


def wai_dataset_factory(spec: WaiDatasetSpec, root: str, metadata_dir: str):
    """Build the ``dataset_factory`` :class:`WaiViewSource` expects.

    ``max_num_retries=0`` matters: by default a WAI dataset that fails to load
    a sample silently retries with a *random other scene*, which would turn a
    missing file into a quietly different benchmark.
    """

    def factory(norm_type: str, resolution: Tuple[int, int], num_views: int):
        import mapanything.datasets

        cls = getattr(mapanything.datasets, spec.class_name)
        return cls(
            **spec.kwargs,
            resolution=tuple(resolution),
            num_views=num_views,
            data_norm_type=norm_type,
            ROOT=root,
            dataset_metadata_dir=metadata_dir,
            max_num_retries=0,
        )

    return factory


class _SamplingDone(Exception):
    """Stops a WAI dataset as soon as it starts loading the first view."""


def wai_sample_plan(dataset, index: int) -> Dict[str, Any]:
    """Which frames and files ``dataset[(index, 0)]`` would read, without reading them.

    Runs the dataset's real ``_getitem_fn`` - same seed, same RNG calls, same
    covisibility walk - but stops it at the first ``load_frame`` call. The
    files each chosen frame needs are then resolved with map-anything's own
    ``load_frame`` while ``wai.core.load_data`` only records paths, so nested
    modalities (``rendered_depth``, ``pred_mask/moge2``, ...) are found without
    this code knowing about them. Only ``scene_meta.json`` and the covisibility
    matrix need to be on disk.

    Returns:
        Dict with ``scene_root``, ``frames`` (names, in view order) and
        ``files`` (paths relative to ``scene_root``, sorted, de-duplicated).
    """
    import importlib
    import inspect

    from mapanything.utils.wai import core

    module = importlib.import_module(type(dataset).__module__)
    real_load_frame = module.load_frame
    real_sample = dataset._sample_view_indices
    seen: Dict[str, Any] = {}

    def record_sample(*args, **kwargs):
        seen["indices"] = [int(i) for i in real_sample(*args, **kwargs)]
        return seen["indices"]

    def stop_at_load(*args, **kwargs):
        seen["call"] = inspect.signature(real_load_frame).bind(*args, **kwargs).arguments
        raise _SamplingDone

    dataset._sample_view_indices = record_sample
    module.load_frame = stop_at_load
    try:
        dataset._getitem_fn((index, 0))
    except _SamplingDone:
        pass
    finally:
        module.load_frame = real_load_frame
        del dataset._sample_view_indices
    if "indices" not in seen or "call" not in seen:
        raise RuntimeError(
            f"{type(dataset).__name__} did not sample views through "
            "_sample_view_indices and load_frame; its loading code has changed."
        )

    call = seen["call"]
    scene_root = str(call["scene_root"])
    scene_meta = call["scene_meta"]
    modalities = call.get("modalities")
    frame_names = list(scene_meta["frame_names"])
    frames = [frame_names[i] for i in seen["indices"]]
    # Guard the one assumption made here: view index i is the i-th frame.
    first = call["frame_key"]
    if str(first) != str(frames[0]):
        raise RuntimeError(
            f"{type(dataset).__name__} loaded frame {first!r} for view index "
            f"{seen['indices'][0]}, expected {frames[0]!r}; its frame order has changed."
        )

    files = set()
    real_load_data = core.load_data

    def record_path(fname, *args, **kwargs):
        files.add(os.path.relpath(str(fname), scene_root))

    core.load_data = record_path
    try:
        for frame in dict.fromkeys(frames):
            real_load_frame(scene_root, frame, modalities=modalities, scene_meta=scene_meta)
    finally:
        core.load_data = real_load_data

    return {"scene_root": scene_root, "frames": frames, "files": sorted(files)}


class WaiViewSource(ViewSource):
    """One multi-view sample from a map-anything WAI dataset.

    Args:
        dataset_factory: ``(norm_type, resolution, num_views) -> dataset``.
            Called once per normalisation, because a WAI dataset fixes its
            ``data_norm_type`` at construction; results are cached.
        index: Index of the multi-view set to evaluate.
        num_views: Views per set, passed to the factory.
        name: Scene name used in the report.
        dataset_name: Dataset the sample came from, recorded as provenance.
        seed: Dataset seed, recorded as provenance - the covisibility walk is
            random, so a result without its seed cannot be repeated.
    """

    def __init__(
        self,
        dataset_factory: Callable[[str, Tuple[int, int], int], Any],
        index: int,
        num_views: int,
        name: Optional[str] = None,
        dataset_name: str = "wai",
        seed: Optional[int] = None,
    ):
        self.dataset_factory = dataset_factory
        self.index = int(index)
        self.num_views = int(num_views)
        self.name = name or f"{dataset_name}_{index:05d}"
        self.dataset_name = dataset_name
        self.seed = seed
        self._cache: Dict[Tuple[str, Tuple[int, int]], Any] = {}

    def _dataset(self, norm_type: str, resolution: Tuple[int, int]):
        key = (norm_type, resolution)
        if key not in self._cache:
            self._cache[key] = self.dataset_factory(norm_type, resolution, self.num_views)
        return self._cache[key]

    def prepare(
        self,
        norm_type: str,
        patch_size: int,
        resolution_set: int = 518,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> PreparedViews:
        if image_size is None:
            raise ValueError(
                "WaiViewSource needs an explicit --image-size: a WAI dataset is "
                "constructed for one resolution, so there is no per-model native "
                "mapping to fall back on."
            )
        from mapanything.utils.image import rgb

        dataset = self._dataset(norm_type, tuple(image_size))
        raw_views = dataset[(self.index, 0)]

        views = [_as_batched_view(view) for view in raw_views]
        images = np.stack(
            [rgb(view["img"][0], norm_type) for view in views]
        ).astype(np.float32)

        depth = np.stack(
            [np.asarray(v["depthmap"], dtype=np.float64).squeeze(-1) for v in raw_views]
        )
        ground_truth = {
            "depth_z": depth,
            "intrinsics": np.stack(
                [np.asarray(v["camera_intrinsics"], dtype=np.float64) for v in raw_views]
            ),
            "poses_c2w": np.stack(
                [np.asarray(v["camera_pose"], dtype=np.float64) for v in raw_views]
            ),
        }

        view_names = [str(v.get("instance", index)) for index, v in enumerate(raw_views)]
        height, width = depth.shape[1], depth.shape[2]
        provenance = {
            "num_views": len(views),
            "views_digest": views_digest(view_names),
            "views": " ".join(view_names),
            "sampling/strategy": "covisibility_random_walk",
            "sampling/dataset": self.dataset_name,
            "sampling/set_index": self.index,
            "sampling/seed": self.seed,
        }
        return PreparedViews(
            views=views,
            images=images,
            resolution=(width, height),
            norm_type=norm_type,
            scene_name=self.name,
            view_names=view_names,
            ground_truth=ground_truth,
            provenance=provenance,
        )
