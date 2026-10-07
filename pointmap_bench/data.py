"""Image and ground-truth loading, shared by every model in the benchmark.

The important guarantee provided here: for a given scene, the RGB images, the
GT depth maps and the GT intrinsics all go through *the same* crop-and-resize
operation (MapAnything's ``crop_resize_if_necessary``), so a predicted point map
and the GT point map are defined on exactly the same pixel grid and can be
compared pixel by pixel without any resampling.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import PIL.Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

# Patch sizes of the backbones used in this benchmark. A shared image size must
# be divisible by all of them for every model to accept the same grid.
PATCH_SIZES = (14, 16)
COMMON_SIZE_MULTIPLE = 112  # lcm(14, 16)


@dataclass
class Scene:
    """A set of images (and optionally ground truth) to reconstruct.

    ``sampling`` records *how* these particular views were chosen. Which views
    a model is given changes its scores, so a result that does not carry its
    view selection with it cannot be reproduced or compared against anyone
    else's run.
    """

    name: str
    image_paths: List[str]
    gt_path: Optional[str] = None
    sampling: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_views(self) -> int:
        return len(self.image_paths)

    @property
    def view_names(self) -> List[str]:
        """Basenames of the views, in the order they are fed to the model."""
        return [os.path.basename(path) for path in self.image_paths]

    @property
    def views_digest(self) -> str:
        """Short stable hash of the ordered view list.

        Two runs that share a digest saw exactly the same images in exactly the
        same order; two that do not are not comparable, however similar their
        settings look.
        """
        joined = "|".join(self.view_names).encode("utf-8")
        return hashlib.sha1(joined).hexdigest()[:12]

    def provenance(self) -> Dict[str, Any]:
        """The fields a result record needs in order to be reproducible."""
        out: Dict[str, Any] = {
            "num_views": self.num_views,
            "views_digest": self.views_digest,
            "views": " ".join(self.view_names),
        }
        for key, value in self.sampling.items():
            out[f"sampling/{key}"] = value
        return out


def list_images(folder_or_list, stride: int = 1, limit: Optional[int] = None):
    """Collect image paths from a folder (sorted) or pass a list through.

    Args:
        folder_or_list: Directory containing images, or an explicit list of paths.
        stride: Keep every ``stride``-th image.
        limit: Optional cap on the number of images kept.
    """
    if isinstance(folder_or_list, (list, tuple)):
        paths = list(folder_or_list)
    else:
        if not os.path.isdir(folder_or_list):
            raise NotADirectoryError(f"Not a directory: {folder_or_list}")
        paths = sorted(
            os.path.join(folder_or_list, name)
            for name in os.listdir(folder_or_list)
            if name.endswith(IMAGE_EXTENSIONS)
        )
    paths = paths[::stride]
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise ValueError(f"No images found in {folder_or_list}")
    return paths


def sorted_stride_sampling(stride: int = 1, limit: Optional[int] = None):
    """Describe the default view-selection strategy.

    It is deliberately spelled out rather than left implicit: "whatever order
    the filenames happened to be in" is a choice that changes the scores, and
    it has to travel with the results.
    """
    return {
        "strategy": "sorted_filename_stride",
        "stride": int(stride),
        "max_views": limit,
    }


def discover_scenes(root: str, stride: int = 1, limit: Optional[int] = None):
    """Build scenes from a dataset root.

    Two layouts are supported:

    * ``root`` directly contains images -> a single scene named after ``root``.
    * ``root`` contains sub-directories -> one scene per sub-directory. A
      ``gt.npz`` next to the images (or ``<scene>/gt.npz``) is picked up
      automatically as ground truth.
    """
    root = os.path.abspath(root)
    sampling = sorted_stride_sampling(stride, limit)
    direct = [n for n in os.listdir(root) if n.endswith(IMAGE_EXTENSIONS)]
    if direct:
        gt_path = os.path.join(root, "gt.npz")
        return [
            Scene(
                name=os.path.basename(root.rstrip(os.sep)),
                image_paths=list_images(root, stride, limit),
                gt_path=gt_path if os.path.isfile(gt_path) else None,
                sampling=dict(sampling),
            )
        ]

    scenes = []
    for name in sorted(os.listdir(root)):
        scene_dir = os.path.join(root, name)
        if not os.path.isdir(scene_dir):
            continue
        image_dir = scene_dir
        if not any(n.endswith(IMAGE_EXTENSIONS) for n in os.listdir(scene_dir)):
            nested = os.path.join(scene_dir, "images")
            if not os.path.isdir(nested):
                continue
            image_dir = nested
        gt_path = os.path.join(scene_dir, "gt.npz")
        scenes.append(
            Scene(
                name=name,
                image_paths=list_images(image_dir, stride, limit),
                gt_path=gt_path if os.path.isfile(gt_path) else None,
                sampling=dict(sampling),
            )
        )
    if not scenes:
        raise ValueError(f"No scenes with images found under {root}")
    return scenes


def check_common_size(size: Tuple[int, int]) -> None:
    """Raise if a user-requested shared image size is not usable by all models."""
    width, height = size
    for value, label in ((width, "width"), (height, "height")):
        for patch in PATCH_SIZES:
            if value % patch != 0:
                raise ValueError(
                    f"Image {label} {value} is not divisible by patch size {patch}. "
                    f"Use a multiple of {COMMON_SIZE_MULTIPLE} "
                    f"(e.g. 448x336, 560x448) so all backbones accept the same grid."
                )


def load_views(
    image_paths: Sequence[str],
    norm_type: str,
    patch_size: int,
    resolution_set: int = 518,
    image_size: Optional[Tuple[int, int]] = None,
    verbose: bool = False,
):
    """Load and normalise images with MapAnything's own preprocessing.

    Args:
        image_paths: Explicit list of image files, in the order to be fed to the model.
        norm_type: Normalisation expected by the model ("dinov2", "dust3r", "identity").
        patch_size: Backbone patch size, used when ``image_size`` is given.
        resolution_set: 518 (patch 14) or 512 (patch 16) mapping table, used when
            ``image_size`` is None.
        image_size: Optional (width, height) forced on every model. Required when
            comparing models against a shared ground-truth pixel grid.

    Returns:
        Tuple ``(views, (width, height))``.
    """
    from mapanything.utils.image import load_images

    if image_size is not None:
        views = load_images(
            list(image_paths),
            resize_mode="fixed_size",
            size=tuple(image_size),
            norm_type=norm_type,
            patch_size=patch_size,
            verbose=verbose,
        )
    else:
        views = load_images(
            list(image_paths),
            resize_mode="fixed_mapping",
            norm_type=norm_type,
            patch_size=patch_size,
            resolution_set=resolution_set,
            verbose=verbose,
        )

    if len(views) != len(image_paths):
        raise RuntimeError(
            f"Loaded {len(views)} views from {len(image_paths)} paths; some images "
            "could not be read, which would silently misalign the ground truth."
        )

    height, width = int(views[0]["true_shape"][0][0]), int(views[0]["true_shape"][0][1])
    for view in views[1:]:
        if (int(view["true_shape"][0][0]), int(view["true_shape"][0][1])) != (
            height,
            width,
        ):
            raise RuntimeError("All views must share the same resolution")
    return views, (width, height)


def views_to_rgb(views, norm_type: str) -> np.ndarray:
    """Undo the normalisation to get (V, H, W, 3) RGB in [0, 1] for exports."""
    from mapanything.utils.image import rgb

    return np.stack([rgb(view["img"][0], norm_type) for view in views]).astype(
        np.float32
    )


def load_ground_truth(
    gt_path: str,
    image_paths: Sequence[str],
    target_size: Tuple[int, int],
) -> Dict[str, np.ndarray]:
    """Load posed-depth ground truth and align it to the model's pixel grid.

    The ``.npz`` file must contain:

    * ``depth_z``:    (V, H0, W0) float, z-depth in metres. 0 / NaN = invalid.
    * ``intrinsics``: (V, 3, 3) float, pinhole intrinsics at (H0, W0).
    * ``poses_c2w``:  (V, 4, 4) float, OpenCV camera-to-world poses.
    * ``image_names`` (optional): (V,) strings used to match the GT entries to
      ``image_paths`` by basename. Without it the GT is assumed to already be in
      the same order as ``image_paths``.

    Args:
        gt_path: Path to the ``.npz`` file.
        image_paths: The images actually fed to the model, in order.
        target_size: (width, height) grid the model ran at.

    Returns:
        Dict with ``depth_z`` (V, H, W), ``intrinsics`` (V, 3, 3) and
        ``poses_c2w`` (V, 4, 4), all matching ``target_size``.
    """
    from mapanything.utils.cropping import crop_resize_if_necessary

    with np.load(gt_path, allow_pickle=False) as data:
        required = ("depth_z", "intrinsics", "poses_c2w")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{gt_path} is missing required keys: {missing}")
        depth_all = np.asarray(data["depth_z"], dtype=np.float64)
        intrinsics_all = np.asarray(data["intrinsics"], dtype=np.float64)
        poses_all = np.asarray(data["poses_c2w"], dtype=np.float64)
        names = (
            [str(n) for n in data["image_names"]] if "image_names" in data else None
        )

    if names is not None:
        index_of = {os.path.basename(n): i for i, n in enumerate(names)}
        try:
            order = [index_of[os.path.basename(p)] for p in image_paths]
        except KeyError as exc:
            raise KeyError(
                f"Image {exc} has no matching entry in {gt_path}'s image_names"
            ) from exc
        depth_all = depth_all[order]
        intrinsics_all = intrinsics_all[order]
        poses_all = poses_all[order]
    elif len(depth_all) != len(image_paths):
        raise ValueError(
            f"{gt_path} holds {len(depth_all)} views but {len(image_paths)} images "
            "were selected, and no image_names array is available to match them."
        )

    depths, intrinsics = [], []
    for path, depth, intr in zip(image_paths, depth_all, intrinsics_all):
        image = PIL.Image.open(path).convert("RGB")
        if image.size != (depth.shape[1], depth.shape[0]):
            raise ValueError(
                f"GT depth for {path} is {depth.shape[1]}x{depth.shape[0]} but the "
                f"image is {image.size[0]}x{image.size[1]}; they must match."
            )
        # Same Lanczos-resize + intrinsics-aware crop that load_images applies
        # to the RGB, so depth, intrinsics and image stay pixel-aligned.
        _, depth_out, intr_out = crop_resize_if_necessary(
            image, resolution=tuple(target_size), depthmap=depth, intrinsics=intr
        )
        depths.append(np.asarray(depth_out, dtype=np.float64))
        intrinsics.append(np.asarray(intr_out, dtype=np.float64))

    depth_stack = np.stack(depths)
    width, height = target_size
    if depth_stack.shape[1:] != (height, width):
        raise RuntimeError(
            f"Resized GT depth is {depth_stack.shape[1:]}, expected {(height, width)}"
        )

    return {
        "depth_z": depth_stack,
        "intrinsics": np.stack(intrinsics),
        "poses_c2w": poses_all,
    }
