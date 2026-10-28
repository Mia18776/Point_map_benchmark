#!/usr/bin/env python3
"""Download only the views the benchmark will actually sample from a WAI dataset.

map-anything's benchmarking data (``facebook/map-anything-benchmarking`` on the
Hugging Face hub) ships one zip per scene - ~10 GB for ETH3D, ~500 GB for
ScanNet++ v2. The views of a multi-view set are chosen by a seeded random walk
over the scene's covisibility matrix, so which frames are needed is known
before any image is read. This script:

1. reads each scene's ``scene_meta.json`` and covisibility matrix out of the
   remote zip (HTTP range requests, a few KB per scene),
2. runs map-anything's own dataset class - built from upstream's test config by
   the same code the benchmark uses - until it has chosen the views, and asks
   map-anything's ``load_frame`` which files those views need, and
3. downloads just those files.

Because step 2 is the real sampler with the real seed, the benchmark later
picks exactly the frames on disk, and the views match the full-data protocol.
Nothing here is specific to one dataset: any ``<name>`` with a
``configs/dataset/<name>_wai/test`` config in map-anything works. Usage::

    python scripts/prepare_wai_subset.py --dataset eth3d \\
        --root /data/wai/eth3d --metadata-dir /data/wai_metadata \\
        --num-views 8 --image-size 448x336
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pointmap_bench.sources import (  # noqa: E402
    wai_dataset_factory,
    wai_dataset_spec,
    wai_sample_plan,
)

HF_REPO = "datasets/facebook/map-anything-benchmarking"


def _extract(archive: zipfile.ZipFile, member: str, root: str) -> None:
    target = os.path.join(root, member)
    info = archive.getinfo(member)
    if os.path.isfile(target) and os.path.getsize(target) == info.file_size:
        return
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = target + ".part"
    with archive.open(member) as src, open(tmp, "wb") as dst:
        while chunk := src.read(1 << 22):
            dst.write(chunk)
    os.replace(tmp, target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", required=True, help="e.g. eth3d, scannetpp, tav2_wb")
    parser.add_argument("--root", required=True, help="Where this dataset's WAI scenes go")
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--num-views", type=int, default=8)
    parser.add_argument("--image-size", default="448x336", help="WIDTHxHEIGHT")
    parser.add_argument("--scenes", nargs="*", help="Subset of scenes (default: all)")
    args = parser.parse_args()

    from huggingface_hub import HfFileSystem
    from natsort import natsorted

    spec = wai_dataset_spec(args.dataset)
    fs = HfFileSystem()
    remote_dir = f"{HF_REPO}/{spec.data_dirname}"
    width, height = (int(v) for v in args.image_size.lower().split("x"))

    def remote_zip(scene):
        handle = fs.open(f"{remote_dir}/{scene}.zip", "rb", block_size=1 << 20)
        return handle, zipfile.ZipFile(handle)

    # Scene list: upstream's aggregate_scene_names.py natsorts the scenes of a
    # split, and the order fixes scene index -> seed. The hub folder holds the
    # test split, so its natsorted zip names are that list.
    all_scenes = natsorted(
        os.path.basename(p)[: -len(".zip")]
        for p in fs.ls(remote_dir, detail=False)
        if p.endswith(".zip")
    )
    print(f"{len(all_scenes)} {spec.data_dirname} scenes ({spec.class_name}, {spec.kwargs})")

    # 1. Scene metadata and covisibility for every scene (tiny).
    for scene in all_scenes:
        handle, archive = remote_zip(scene)
        with handle:
            for member in archive.namelist():
                if member.endswith("/scene_meta.json") or (
                    "/covisibility/" in member and not member.endswith("/")
                ):
                    _extract(archive, member, args.root)

    # Each dataset class reads the list from <data folder>_scene_list_test.npy.
    split_dir = os.path.join(args.metadata_dir, "test")
    os.makedirs(split_dir, exist_ok=True)
    np.save(
        os.path.join(split_dir, f"{spec.data_dirname}_scene_list_test.npy"),
        np.array(all_scenes, dtype=object),
    )

    # 2. Ask the real dataset which frames and files each set needs.
    dataset = wai_dataset_factory(spec, args.root, args.metadata_dir)(
        "dinov2", (width, height), args.num_views
    )
    if [str(s) for s in dataset.scenes] != all_scenes:
        raise RuntimeError(
            f"{spec.class_name} did not pick up the scene list written to {split_dir}"
        )

    manifest = {
        "dataset": args.dataset,
        "config": spec.config_path,
        "num_views": args.num_views,
        "image_size": f"{width}x{height}",
        "scenes": {},
    }
    total = 0.0
    for index, scene in enumerate(all_scenes):
        if args.scenes and scene not in args.scenes:
            continue
        plan = wai_sample_plan(dataset, index)
        manifest["scenes"][scene] = {
            "set_index": index,
            "views": [str(f) for f in plan["frames"]],
            "files": plan["files"],
        }

        # 3. Download only those files.
        handle, archive = remote_zip(scene)
        with handle:
            members = [f"{scene}/{rel}" for rel in plan["files"]]
            size = sum(archive.getinfo(m).file_size for m in members) / 1e6
            total += size
            print(
                f"[{index}] {scene}: {len(set(plan['frames']))} frames, "
                f"{len(members)} files, {size:.0f} MB"
            )
            for member in members:
                _extract(archive, member, args.root)

    with open(os.path.join(args.root, f"subset_{args.num_views}v.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"done, {total / 1e3:.1f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
