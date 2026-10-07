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


def _as_batched_view(view: dict) -> dict:
    """Turn one WAI dataset view into a model-wrapper input view.

    The dataset yields un-batched tensors for training; the wrappers expect a
    leading batch dimension and ``data_norm_type`` as a list.
    """
    import torch

    out = dict(view)
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
