"""Model registry and runners.

Every model is driven through MapAnything's unified interface, so the four
models really do receive identical inputs and are converted from identical
output conventions:

* ``mapanything`` is loaded from the Hugging Face hub and run through its
  ``infer()`` API (the officially recommended path, which also returns the
  model's own validity mask).
* ``mast3r``, ``pi3`` and ``da3`` are loaded through MapAnything's Hydra model
  configs and run through their ``mapanything.models.external.*`` wrappers,
  which already translate each upstream model into the common per-view dict
  format (``pts3d``, ``pts3d_cam``, ``ray_directions``, ``cam_quats``, ...).

Nothing here re-implements a model. That is deliberate: re-implementing
inference is where benchmark bugs come from.
"""

from __future__ import annotations

import contextlib
import importlib.util
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .prediction import Prediction, _to_numpy, prediction_from_wrapper_output
from .sources import PreparedViews


@dataclass
class ModelSpec:
    """Everything the benchmark needs to know about one model."""

    key: str
    display_name: str
    # "hf" -> MapAnything.from_pretrained, "hydra" -> MapAnything Hydra config.
    loader: str
    norm_type: str
    patch_size: int
    resolution_set: int
    hf_repo: Optional[str] = None
    hydra_config: Optional[str] = None
    # Model claims real-world (metric) scale; controls whether metric/* metrics
    # are meaningful for it.
    is_metric: bool = False
    # The upstream wrapper calls torch.cuda.* unconditionally at construction.
    requires_cuda: bool = False
    # MASt3R's sparse global aligner needs per-view label/instance strings.
    needs_view_labels: bool = False
    # Python modules that must be importable for this model to run.
    required_modules: Tuple[str, ...] = ()
    install_hint: str = ""
    notes: str = ""
    extra_overrides: List[str] = field(default_factory=list)


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "mapanything": ModelSpec(
        key="mapanything",
        display_name="MapAnything",
        loader="hf",
        hf_repo="facebook/map-anything",
        norm_type="dinov2",
        patch_size=14,
        resolution_set=518,
        is_metric=True,
        install_hint="pip install -e /path/to/map-anything",
        notes="CC-BY-NC 4.0 weights; run through model.infer().",
    ),
    "mapanything_apache": ModelSpec(
        key="mapanything_apache",
        display_name="MapAnything (Apache)",
        loader="hf",
        hf_repo="facebook/map-anything-apache",
        norm_type="dinov2",
        patch_size=14,
        resolution_set=518,
        is_metric=True,
        install_hint="pip install -e /path/to/map-anything",
        notes="Apache-2.0 weights, trained on the permissive data subset.",
    ),
    "mast3r": ModelSpec(
        key="mast3r",
        display_name="MASt3R + Sparse GA",
        loader="hydra",
        hydra_config="mast3r",
        norm_type="dust3r",
        patch_size=16,
        resolution_set=512,
        is_metric=True,
        needs_view_labels=True,
        required_modules=("mast3r", "dust3r"),
        install_hint=(
            "pip install 'mapanything[mast3r]' 'mapanything[dust3r]' and download "
            "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth into "
            "--mast3r-checkpoint-dir"
        ),
        notes=(
            "Pairwise model + sparse global alignment; the optimisation makes it "
            "far slower than the feed-forward models."
        ),
    ),
    "pi3": ModelSpec(
        key="pi3",
        display_name="Pi3",
        loader="hydra",
        hydra_config="pi3",
        norm_type="identity",
        patch_size=14,
        resolution_set=518,
        is_metric=False,
        requires_cuda=True,
        install_hint="Vendored in mapanything; weights pulled from HF 'yyfz233/Pi3'.",
        notes="pi-cubed; permutation-equivariant, scale-ambiguous (affine-invariant) output.",
    ),
    "pi3x": ModelSpec(
        key="pi3x",
        display_name="Pi3-X",
        loader="hydra",
        hydra_config="pi3x",
        norm_type="identity",
        patch_size=14,
        resolution_set=518,
        is_metric=False,
        requires_cuda=True,
        install_hint="Vendored in mapanything.",
        notes="MapAnything's Pi3 variant; included as an extra baseline.",
    ),
    "da3": ModelSpec(
        key="da3",
        display_name="Depth Anything 3",
        loader="hydra",
        hydra_config="da3",
        norm_type="dinov2",
        patch_size=14,
        resolution_set=518,
        is_metric=False,
        requires_cuda=True,
        required_modules=("depth_anything_3",),
        install_hint="pip install 'mapanything[depth-anything-3]'",
        notes="HF checkpoint depth-anything/DA3-GIANT-1.1, run images-only.",
    ),
    "vggt": ModelSpec(
        key="vggt",
        display_name="VGGT",
        loader="hydra",
        hydra_config="vggt",
        norm_type="identity",
        patch_size=14,
        resolution_set=518,
        is_metric=False,
        requires_cuda=True,
        install_hint="Vendored in mapanything.",
        notes="Optional extra reference baseline, not part of the default set.",
    ),
}

DEFAULT_MODELS = ("mapanything", "mast3r", "pi3", "da3")


class ModelUnavailable(RuntimeError):
    """Raised when a model cannot run in the current environment."""


def mapanything_root() -> str:
    """Locate the map-anything source checkout that holds ``configs/``.

    Hydra model configs live at the repository root, not inside the installed
    package, so the benchmark needs the source tree. Set ``MAPANYTHING_ROOT`` to
    override the auto-detection.
    """
    env_root = os.environ.get("MAPANYTHING_ROOT")
    candidates = []
    if env_root:
        candidates.append(env_root)
    try:
        import mapanything

        candidates.append(os.path.dirname(os.path.dirname(os.path.abspath(mapanything.__file__))))
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ModelUnavailable(
            "The 'mapanything' package is not importable. Install the map-anything "
            "repository first: pip install -e /path/to/map-anything"
        ) from exc

    for root in candidates:
        if os.path.isfile(os.path.join(root, "configs", "train.yaml")):
            return root
    raise ModelUnavailable(
        "Could not find the map-anything 'configs/' directory. Point "
        "MAPANYTHING_ROOT at your map-anything source checkout."
    )


def missing_modules(spec: ModelSpec) -> List[str]:
    """Names of required modules that are not importable."""
    return [m for m in spec.required_modules if importlib.util.find_spec(m) is None]


def check_availability(spec: ModelSpec, device: str) -> Optional[str]:
    """Return a human-readable reason the model cannot run, or None if it can."""
    if importlib.util.find_spec("mapanything") is None:
        return "missing module 'mapanything'. pip install -e /path/to/map-anything"
    # Hydra-configured models compose map-anything's configs/train.yaml.
    if spec.loader == "hydra" and importlib.util.find_spec("hydra") is None:
        return "missing module 'hydra'. pip install hydra-core"

    missing = missing_modules(spec)
    if missing:
        return f"missing module(s) {', '.join(missing)}. {spec.install_hint}"
    if spec.requires_cuda and not torch.cuda.is_available():
        return (
            f"{spec.display_name} requires a CUDA device "
            "(its wrapper queries torch.cuda at init)"
        )
    if spec.requires_cuda and not device.startswith("cuda"):
        return f"{spec.display_name} requires --device cuda"
    return None


def _compose_hydra_config(config_name: str, overrides: Sequence[str]):
    """Compose ``configs/train.yaml`` with ``model=<config_name>`` applied."""
    import hydra
    from hydra.core.global_hydra import GlobalHydra

    root = mapanything_root()
    config_dir = os.path.join(root, "configs")

    GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = hydra.compose(
            config_name="train",
            overrides=[f"model={config_name}", *overrides],
        )
    return cfg


def load_model(spec: ModelSpec, device: str, mast3r_checkpoint_dir: Optional[str] = None):
    """Instantiate a model on ``device`` in eval mode.

    Returns:
        Tuple ``(model, norm_type)``; ``norm_type`` comes from the model's own
        config wherever possible instead of being hard-coded.
    """
    if spec.loader == "hf":
        from mapanything.models import MapAnything

        model = MapAnything.from_pretrained(spec.hf_repo).to(device).eval()
        norm_type = getattr(
            getattr(model, "encoder", None), "data_norm_type", None
        ) or spec.norm_type
        return model, norm_type

    if spec.loader != "hydra":
        raise ValueError(f"Unknown loader '{spec.loader}' for {spec.key}")

    from mapanything.models import init_model

    overrides = ["machine=default", *spec.extra_overrides]
    if spec.key == "mast3r":
        if not mast3r_checkpoint_dir:
            raise ModelUnavailable(
                "MASt3R needs --mast3r-checkpoint-dir pointing at the directory that "
                "holds MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
            )
        checkpoint_dir = os.path.abspath(mast3r_checkpoint_dir)
        checkpoint = os.path.join(
            checkpoint_dir, "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
        )
        if not os.path.isfile(checkpoint):
            raise ModelUnavailable(f"MASt3R checkpoint not found: {checkpoint}")
        # The wrapper calls tempfile.mkdtemp(dir=cache_dir), so it must exist.
        os.makedirs(os.path.join(checkpoint_dir, "mast3r_cache"), exist_ok=True)
        overrides.append(f"machine.root_pretrained_checkpoints_dir={checkpoint_dir}")

    cfg = _compose_hydra_config(spec.hydra_config, overrides)
    model = init_model(
        model_str=cfg.model.model_str,
        model_config=cfg.model.model_config,
        torch_hub_force_reload=False,
    )
    model = model.to(device).eval()
    return model, str(cfg.model.data_norm_type)


def _prepare_views(views, device: str, spec: ModelSpec, scene_name: str, view_names):
    """Move views to the device and add the extra keys some wrappers require."""
    prepared = []
    for index, view in enumerate(views):
        entry = dict(view)
        entry["img"] = view["img"].to(device)
        entry["data_norm_type"] = list(view["data_norm_type"])
        if spec.needs_view_labels:
            # MASt3R's sparse global aligner keys its on-disk cache off these,
            # and load_images stores `instance` as a bare str (not a list).
            entry["label"] = [scene_name]
            entry["instance"] = [os.path.basename(str(view_names[index]))]
        prepared.append(entry)
    return prepared


def _prediction_from_mapanything(outputs, images, spec: ModelSpec, info) -> Prediction:
    """Convert MapAnything's ``infer()`` output into the common Prediction."""
    points_world = np.stack([_to_numpy(o["pts3d"][0]) for o in outputs])
    points_cam = np.stack([_to_numpy(o["pts3d_cam"][0]) for o in outputs])
    depth_z = np.stack([_to_numpy(o["depth_z"][0]).squeeze(-1) for o in outputs])
    intrinsics = np.stack([_to_numpy(o["intrinsics"][0]) for o in outputs])
    poses_c2w = np.stack([_to_numpy(o["camera_poses"][0]) for o in outputs])

    # infer() only emits "mask" when apply_mask=True - see the `if apply_mask:`
    # block in mapanything/utils/inference.py. With --no-mapanything-mask the
    # key is simply absent, and reading it unconditionally made the flag the
    # README recommends for a raw-density comparison fail every scene with a
    # KeyError. Absent means "nothing was masked", so every pixel starts valid
    # and only the geometric checks below remove any.
    if "mask" in outputs[0]:
        mask = np.stack([_to_numpy(o["mask"][0]).squeeze(-1) for o in outputs]) > 0.5
    else:
        mask = np.ones(depth_z.shape, dtype=bool)
    mask &= np.isfinite(points_world).all(axis=-1) & (depth_z > 0)

    confidence = (
        np.stack([_to_numpy(o["conf"][0]) for o in outputs])
        if "conf" in outputs[0]
        else None
    )
    return Prediction(
        points_world=points_world,
        points_cam=points_cam,
        depth_z=depth_z,
        intrinsics=intrinsics,
        poses_c2w=poses_c2w,
        mask=mask,
        images=images,
        confidence=confidence,
        is_metric=spec.is_metric,
        info=info,
    )


@contextlib.contextmanager
def _measure(device: str):
    """Time a block and record peak CUDA memory over it."""
    stats: Dict[str, float] = {}
    on_cuda = device.startswith("cuda") and torch.cuda.is_available()
    if on_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    try:
        yield stats
    finally:
        if on_cuda:
            torch.cuda.synchronize()
            stats["peak_memory_mb"] = torch.cuda.max_memory_allocated() / (1024**2)
        stats["seconds"] = time.perf_counter() - start


def run_model(
    spec: ModelSpec,
    prepared: PreparedViews,
    model: Any,
    device: str = "cuda",
    mapanything_apply_mask: bool = True,
    mapanything_minibatch_size: Optional[int] = 1,
) -> Prediction:
    """Run one already-loaded model on one already-prepared multi-view sample.

    Loading the views is the caller's job, through a
    :class:`~pointmap_bench.sources.ViewSource`: the sample has to be built for
    *this* model's normalisation and grid, and keeping that out of here is what
    lets a folder of images and a WAI dataset feed the same code path.

    Args:
        spec: The model to run.
        prepared: Views, ground truth and provenance for one scene.
        model: Loaded model, in eval mode on ``device``.
        device: Torch device string.
        mapanything_apply_mask: Pass through to ``MapAnything.infer``; keeps the
            model's recommended edge/ambiguity masking.
        mapanything_minibatch_size: Memory-efficient inference minibatch size.

    Returns:
        A :class:`~pointmap_bench.prediction.Prediction`.
    """
    reason = check_availability(spec, device)
    if reason:
        raise ModelUnavailable(reason)

    views = _prepare_views(
        prepared.views, device, spec, prepared.scene_name, prepared.view_names
    )
    images = prepared.images

    info: Dict[str, Any] = {
        "model": spec.key,
        "display_name": spec.display_name,
        "num_views": prepared.num_views,
        "resolution": f"{prepared.resolution[0]}x{prepared.resolution[1]}",
        "norm_type": prepared.norm_type,
        "device": device,
    }

    is_mapanything = spec.loader == "hf"
    with torch.no_grad():
        with _measure(device) as run_stats:
            if is_mapanything:
                outputs = model.infer(
                    views,
                    memory_efficient_inference=True,
                    minibatch_size=mapanything_minibatch_size,
                    use_amp=True,
                    amp_dtype="bf16",
                    apply_mask=mapanything_apply_mask,
                    mask_edges=mapanything_apply_mask,
                )
            else:
                # MASt3R re-enables grad internally for its global alignment;
                # torch.enable_grad() inside this block takes precedence.
                outputs = model(views)

    info["inference_seconds"] = run_stats["seconds"]
    info["seconds_per_view"] = run_stats["seconds"] / max(prepared.num_views, 1)
    if "peak_memory_mb" in run_stats:
        info["peak_memory_mb"] = run_stats["peak_memory_mb"]

    if is_mapanything:
        return _prediction_from_mapanything(outputs, images, spec, info)
    return prediction_from_wrapper_output(
        outputs, images=images, is_metric=spec.is_metric, info=info
    )
