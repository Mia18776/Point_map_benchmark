#!/usr/bin/env python3
"""Benchmark MapAnything, MASt3R, Pi3 and Depth Anything 3 on 3D point-map generation.

Examples
--------
Qualitative + runtime comparison on one folder of images (no ground truth)::

    python scripts/run_benchmark.py --images /data/scene0/images \
        --models mapanything pi3 da3 --output-dir runs/scene0 --export-ply

Quantitative comparison against posed-depth ground truth, with every model
forced onto the same pixel grid::

    python scripts/run_benchmark.py --scenes-root /data/eval \
        --image-size 448x336 --output-dir runs/eval \
        --mast3r-checkpoint-dir /data/checkpoints
"""

from __future__ import annotations

import argparse
import datetime
import os
import subprocess
import sys
import traceback

import torch

# Windows consoles default to a legacy code page (e.g. GBK) that cannot encode
# the characters used in model names and reports; never let that abort a run.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointmap_bench.data import (  # noqa: E402
    Scene,
    check_common_size,
    discover_scenes,
    list_images,
    sorted_stride_sampling,
)
from pointmap_bench.export import (  # noqa: E402
    export_glb,
    export_npz,
    export_point_cloud,
)
from pointmap_bench.metrics import evaluate_with_gt, evaluate_without_gt  # noqa: E402
from pointmap_bench.models import (  # noqa: E402
    DEFAULT_MODELS,
    MODEL_REGISTRY,
    ModelUnavailable,
    check_availability,
    load_model,
    run_model,
)
from pointmap_bench.report import write_reports  # noqa: E402
from pointmap_bench.sources import (  # noqa: E402
    FolderViewSource,
    WaiViewSource,
    wai_dataset_factory,
    wai_dataset_spec,
)


def parse_image_size(value: str):
    """Parse a ``WIDTHxHEIGHT`` string into a validated (width, height) tuple."""
    try:
        width, height = (int(part) for part in value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Expected WIDTHxHEIGHT (e.g. 448x336), got '{value}'"
        ) from exc
    try:
        check_common_size((width, height))
    except ValueError as exc:
        # argparse replaces a bare ValueError from a type= callable with its
        # own generic "invalid value" text, throwing away the explanation of
        # which patch size the number is not divisible by.
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return width, height


def positive_int(value: str) -> int:
    """An int >= 1, rejected at parse time rather than deep inside a slice."""
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer, got '{value}'") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(
            f"Must be 1 or greater, got {number}. A stride of 0 is a slice error, "
            "a negative stride silently reverses the view order, and a cap of 0 "
            "selects no images at all."
        )
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--images", help="Folder with the images of a single scene")
    source.add_argument(
        "--scenes-root",
        help="Folder containing one sub-folder per scene (each may hold a gt.npz)",
    )
    source.add_argument(
        "--wai-dataset",
        help=(
            "Evaluate on a map-anything WAI test dataset (eth3d, scannetpp, "
            "tav2_wb, ...: any configs/dataset/<name>_wai/test in map-anything), "
            "with views sampled by its covisibility random walk"
        ),
    )
    parser.add_argument(
        "--gt", help="Ground-truth .npz for --images (see README for the format)"
    )
    parser.add_argument("--wai-root", help="Folder holding the WAI scenes")
    parser.add_argument(
        "--wai-metadata-dir",
        help="Folder holding test/<dataset>_scene_list_test.npy",
    )
    parser.add_argument(
        "--wai-num-views", type=positive_int, default=8, help="Views per WAI set"
    )
    parser.add_argument(
        "--wai-scenes", nargs="+", help="Only evaluate these WAI scenes (default: all)"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        choices=sorted(MODEL_REGISTRY),
        help=f"Models to evaluate (default: {' '.join(DEFAULT_MODELS)})",
    )
    parser.add_argument("--output-dir", default="runs/benchmark")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--image-size",
        type=parse_image_size,
        help=(
            "Force every model onto the same WIDTHxHEIGHT grid (multiple of 112, "
            "e.g. 448x336). Required for ground-truth evaluation."
        ),
    )
    parser.add_argument(
        "--stride", type=positive_int, default=1, help="Use every Nth image"
    )
    parser.add_argument(
        "--max-views", type=positive_int, help="Cap the views per scene"
    )
    parser.add_argument(
        "--mast3r-checkpoint-dir",
        help="Directory holding MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
    )
    parser.add_argument(
        "--confidence-percentile",
        type=float,
        default=0.0,
        help="Drop this percentage of the lowest-confidence pixels before scoring",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=30,
        help="View pairs sampled for the GT-free consistency metric",
    )
    parser.add_argument("--export-ply", action="store_true")
    parser.add_argument("--export-glb", action="store_true")
    parser.add_argument("--export-npz", action="store_true")
    parser.add_argument(
        "--no-mapanything-mask",
        action="store_true",
        help="Disable MapAnything's edge/ambiguity masking (keeps raw dense output)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print model availability in this environment and exit",
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Abort on the first model error"
    )
    return parser


def code_revision() -> str:
    """Short git revision of this checkout, or "unknown" outside a repo."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(
            ["git", "-C", root, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover - environment
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    revision = out.stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", root, "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    if dirty.returncode == 0 and dirty.stdout.strip():
        revision += "-dirty"
    return revision or "unknown"


def run_configuration(args) -> dict:
    """Every setting that changes the numbers, collected for the report.

    A score without its configuration cannot be reproduced or compared, so
    this travels with the results rather than living in someone's shell
    history.
    """
    return {
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "code_revision": code_revision(),
        "models": list(args.models),
        "device": args.device,
        "image_size": (
            f"{args.image_size[0]}x{args.image_size[1]}" if args.image_size else "native"
        ),
        "stride": args.stride,
        "max_views": args.max_views,
        "wai_dataset": args.wai_dataset,
        "wai_num_views": args.wai_num_views if args.wai_dataset else None,
        "confidence_percentile": args.confidence_percentile,
        "max_pairs": args.max_pairs,
        "mapanything_mask": not args.no_mapanything_mask,
    }


def print_model_availability(device: str) -> None:
    print(f"{'model':<20} {'available':<10} detail")
    print("-" * 88)
    for key in sorted(MODEL_REGISTRY):
        spec = MODEL_REGISTRY[key]
        reason = check_availability(spec, device)
        status = "no" if reason else "yes"
        print(f"{key:<20} {status:<10} {reason or spec.notes}")


def build_folder_sources(args):
    """``(name, source, provenance)`` for each scene of an image folder."""
    if args.images:
        scenes = [
            Scene(
                name=os.path.basename(os.path.abspath(args.images).rstrip(os.sep)),
                image_paths=list_images(args.images, args.stride, args.max_views),
                gt_path=args.gt,
                sampling=sorted_stride_sampling(args.stride, args.max_views),
            )
        ]
    else:
        scenes = discover_scenes(args.scenes_root, args.stride, args.max_views)
    return [(scene.name, FolderViewSource(scene), scene.provenance()) for scene in scenes]


def build_wai_sources(args):
    """``(name, source, provenance)`` for one multi-view set per WAI scene.

    Set ``i`` is scene ``i`` of the natsorted test split, sampled with seed
    ``seed + i`` - the first pass of upstream's ``N @ dataset`` evaluation.
    """
    import functools

    missing = [
        flag
        for flag, value in (
            ("--wai-root", args.wai_root),
            ("--wai-metadata-dir", args.wai_metadata_dir),
            ("--image-size", args.image_size),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"--wai-dataset also needs {' '.join(missing)}")

    try:
        spec = wai_dataset_spec(args.wai_dataset)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    # One dataset per (normalisation, resolution), shared by every scene.
    factory = functools.lru_cache(maxsize=None)(
        wai_dataset_factory(spec, args.wai_root, args.wai_metadata_dir)
    )
    all_scenes = [
        str(s) for s in factory("dinov2", args.image_size, args.wai_num_views).scenes
    ]
    unknown = set(args.wai_scenes or []) - set(all_scenes)
    if unknown:
        raise SystemExit(f"Unknown {args.wai_dataset} scenes: {sorted(unknown)}")

    seed = spec.kwargs.get("seed")
    sources = []
    for index, name in enumerate(all_scenes):
        if args.wai_scenes and name not in args.wai_scenes:
            continue
        source = WaiViewSource(
            factory,
            index=index,
            num_views=args.wai_num_views,
            name=name,
            dataset_name=args.wai_dataset,
            seed=seed,
        )
        provenance = {
            "num_views": args.wai_num_views,
            "sampling/strategy": "covisibility_random_walk",
            "sampling/dataset": args.wai_dataset,
            "sampling/set_index": index,
            "sampling/seed": seed,
            "sampling/config": spec.config_path,
        }
        sources.append((name, source, provenance))
    return sources


def main() -> int:
    args = build_parser().parse_args()

    if args.list_models:
        print_model_availability(args.device)
        return 0

    if not (args.images or args.scenes_root or args.wai_dataset):
        build_parser().error("one of --images, --scenes-root or --wai-dataset is required")

    if args.wai_dataset:
        sources = build_wai_sources(args)
    else:
        sources = build_folder_sources(args)

    has_gt = args.wai_dataset is not None or any(
        isinstance(source, FolderViewSource) and source.scene.gt_path
        for _, source, _ in sources
    )
    if has_gt and args.image_size is None:
        print(
            "[warn] Ground truth was found but --image-size is not set. Each model "
            "will run at its own native resolution; the GT is resized to match each "
            "model, which is fair but not pixel-identical across models. Pass "
            "--image-size 448x336 for a strictly shared grid.",
            file=sys.stderr,
        )

    summary = [f"{name} ({provenance['num_views']} views)" for name, _, provenance in sources]
    print(f"Scenes: {summary}")
    print(f"Models: {args.models}")
    print(f"Device: {args.device}")

    records = []
    for model_key in args.models:
        spec = MODEL_REGISTRY[model_key]
        reason = check_availability(spec, args.device)
        if reason:
            print(f"[skip] {model_key}: {reason}")
            records.extend(
                {
                    "model": model_key,
                    "scene": name,
                    "status": "skipped",
                    "reason": reason,
                    **provenance,
                }
                for name, _, provenance in sources
            )
            continue

        print(f"\n=== {spec.display_name} ===")
        try:
            model, norm_type = load_model(spec, args.device, args.mast3r_checkpoint_dir)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            message = f"{type(exc).__name__}: {exc}"
            print(f"[error] failed to load {model_key}: {message}")
            if args.fail_fast:
                raise
            records.extend(
                {
                    "model": model_key,
                    "scene": name,
                    "status": "load_error",
                    "reason": message,
                    **provenance,
                }
                for name, _, provenance in sources
            )
            continue

        for name, source, provenance in sources:
            record = {
                "model": model_key,
                "scene": name,
                "status": "ok",
                **provenance,
            }
            try:
                # The sample is built for *this* model's normalisation and
                # grid; the source decides which views that means.
                prepared = source.prepare(
                    norm_type=norm_type,
                    patch_size=spec.patch_size,
                    resolution_set=spec.resolution_set,
                    image_size=args.image_size,
                )
                record.update(prepared.provenance)

                prediction = run_model(
                    spec,
                    prepared,
                    model=model,
                    device=args.device,
                    mapanything_apply_mask=not args.no_mapanything_mask,
                )
                prediction = prediction.filter_by_confidence(
                    args.confidence_percentile
                )
                record.update(prediction.info)
                record.update(
                    evaluate_without_gt(prediction, max_pairs=args.max_pairs)
                )

                if prepared.ground_truth:
                    record.update(
                        evaluate_with_gt(
                            prediction,
                            prepared.ground_truth["depth_z"],
                            prepared.ground_truth["intrinsics"],
                            prepared.ground_truth["poses_c2w"],
                        )
                    )

                # Exports are side artefacts. A failure here (full disk, a
                # long path, trimesh raising) must not discard an inference
                # that already produced a full set of metrics.
                scene_dir = os.path.join(args.output_dir, name)
                try:
                    if args.export_ply:
                        record["exported_points"] = export_point_cloud(
                            prediction, os.path.join(scene_dir, f"{model_key}.ply")
                        )
                    if args.export_glb:
                        export_glb(
                            prediction, os.path.join(scene_dir, f"{model_key}.glb")
                        )
                    if args.export_npz:
                        export_npz(
                            prediction, os.path.join(scene_dir, f"{model_key}.npz")
                        )
                except Exception as exc:  # noqa: BLE001 - recorded, not fatal
                    record["export_error"] = f"{type(exc).__name__}: {exc}"
                    print(f"  [warn] {name}: export failed: {exc}")

                print(
                    f"  {name}: {record.get('inference_seconds', float('nan')):.2f}s"
                    f"  valid={record.get('cloud/valid_ratio', float('nan')):.3f}"
                    f"  consistency@1.03={record.get('consistency/inlier_1.03', float('nan')):.3f}"
                )
            except ModelUnavailable as exc:
                record.update(status="skipped", reason=str(exc))
                print(f"  [skip] {name}: {exc}")
            except Exception as exc:  # noqa: BLE001 - reported, not swallowed
                record.update(status="error", reason=f"{type(exc).__name__}: {exc}")
                print(f"  [error] {name}: {exc}")
                traceback.print_exc()
                if args.fail_fast:
                    raise
            records.append(record)

        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    notes = [
        "Every model is run through MapAnything's unified wrappers, so inputs and "
        "output conventions are identical across models.",
    ]
    if args.image_size:
        notes.append(f"All models ran at {args.image_size[0]}x{args.image_size[1]}.")
    else:
        notes.append(
            "Each model ran at its own native resolution; compare runtimes with that "
            "in mind."
        )
    if args.no_mapanything_mask:
        notes.append("MapAnything's edge/ambiguity masking was disabled.")

    paths = write_reports(
        records, args.output_dir, extra_notes=notes, config=run_configuration(args)
    )
    print("\nWrote:")
    for name, path in paths.items():
        print(f"  {name}: {path}")

    failures = [r for r in records if r["status"] == "error"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
