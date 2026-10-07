# pointmap-benchmark

A reproducible benchmark for **feed-forward 3D point-map generation**, comparing

| model | how it is run | scale |
| --- | --- | --- |
| [MapAnything](https://github.com/facebookresearch/map-anything) | `MapAnything.from_pretrained(...).infer()` | metric |
| [MASt3R](https://github.com/naver/mast3r) + sparse global alignment | `MASt3RSGAWrapper` | metric |
| [Pi3 (π³)](https://github.com/yyfz/Pi3) | `Pi3Wrapper` | up to scale |
| [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) | `DA3Wrapper` | up to scale |

VGGT and Pi3-X are registered as optional extra baselines.

Everything is driven through **MapAnything's unified model interface**. That is
the point of this repository: each of these models has a different native API,
a different image normalisation, a different output parameterisation and a
different camera convention, and comparing them is only meaningful if those
differences are handled correctly. MapAnything's `mapanything.models.external.*`
wrappers already do that translation, and this benchmark builds on them rather
than reimplementing inference, which is where benchmark bugs come from.

## What is measured

### Without ground truth

Works on any folder of images. All of these are **scale-invariant**, so metric
and non-metric models are directly comparable.

| metric | meaning |
| --- | --- |
| `inference_seconds`, `seconds_per_view` | wall-clock inference time (CUDA-synchronised, model loading excluded) |
| `peak_memory_mb` | peak CUDA memory allocated during inference |
| `cloud/valid_ratio` | fraction of pixels with a usable 3D point |
| `consistency/rel_depth_err_median` | median relative depth disagreement when each view's points are reprojected into the other views |
| `consistency/inlier_{1.03,1.05,1.25}` | fraction of reprojected points agreeing within that ratio |
| `consistency/overlap_ratio` | fraction of reprojected points that land inside the target view (context, not a score) |
| `sanity/pose_pointmap_rel_err` | harness check: does `poses_c2w @ points_cam` reproduce `points_world`? |
| `sanity/depth_pointmap_rel_err` | harness check: does back-projecting `depth_z` through `intrinsics` reproduce `points_cam`? |

The consistency metric is the core GT-free signal: a reconstruction that is
globally coherent agrees with itself when you look at the same surface from a
different camera. Genuinely occluded points count as disagreements, so robust
statistics (median, inlier ratios) are reported rather than a mean.

Read it as a necessary condition, not a ranking: a degenerate prediction that
puts every pixel on one fronto-parallel plane is perfectly self-consistent. It
catches reconstructions that contradict themselves; it cannot certify one that
is merely wrong.

The two `sanity/*` numbers score the benchmark, not the model. They check
identities that hold for any correct wrapper, whatever the model predicts, and
must be ~0 for every model. A large value means some convention is being
translated wrongly (a world-to-camera pose where camera-to-world is expected,
say), and that model's other numbers must not be trusted - which is much better
than reading a broken harness as "this model is bad".

### With ground truth

Needs posed depth (see [Ground-truth format](#ground-truth-format)). Three
alignment regimes are reported, each answering a different question:

| prefix | alignment | answers |
| --- | --- | --- |
| `cam0/*` | both clouds moved into the **first camera's frame**, each normalised by its own average point distance | how good is the reconstruction *including* relative-pose error, ignoring global scale |
| `sim3/*` | closed-form Umeyama similarity fit of prediction to GT | how good is the *shape*, ignoring where it was placed |
| `metric/*` | none beyond the frame change | does the model get real-world scale right (only meaningful for metric models) |

Plus `depth/*` (per-view z-depth after a single median-scale alignment) and
`pose/*` (pairwise relative rotation/translation errors, RRA/RTA at 5°, and ATE
after similarity alignment of the camera centres).

Two columns are computed by **map-anything's own functions** rather than
re-derived here, so they can be read directly against the numbers in its
`benchmarking/dense_n_view` tables:

| metric | meaning |
| --- | --- |
| `pose/auc_5`, `pose/auc_30` | AUC of the per-pair relative-pose error (`max(rotation, translation)`) up to 5° / 30°, as a percentage — upstream's `pose_auc_5` |
| `rays/err_deg` | angular error between predicted and GT unit ray directions, i.e. how wrong the recovered intrinsics are — upstream's `ray_dirs_err_deg` |

They live in `pointmap_bench/official_metrics.py`, the one module that needs
torch and `mapanything`; everything else stays NumPy-only. If those imports are
unavailable the two columns are skipped rather than failing the run.

One definition deliberately differs: upstream's translation term folds the
direction ambiguity, so it lies in [0°, 90°] and scores a 180°-flipped baseline
as correct. `pose/trans_ang_err_deg_*` here is the stricter, sign-sensitive
version. Both are reported — `pose/auc_*` uses upstream's.

`sim3/*` also reports Chamfer accuracy (prediction → GT) and completeness
(GT → prediction), both raw and normalised by the GT scene extent so scenes of
different physical size can be averaged.

Two details that decide whether the comparison is fair:

* **Accuracy is measured on the pixels a model predicted; completeness is
  measured against the whole ground truth.** Otherwise a model that masks away
  the hard half of a scene would score a perfect completeness on the half it
  kept. `gt/covered_ratio` reports how much of the GT each model actually
  predicted and sits next to the accuracy columns in the report, because
  accuracy and coverage trade off directly.
* **The similarity fit is robust** (RANSAC-seeded trimmed least squares), so a
  diverging patch in one prediction cannot drag the alignment and corrupt every
  error built on it. `sim3/median_ae_lsq*` reports the plain least-squares fit
  for comparison; a large gap between the two medians means the prediction
  carries real outlier mass. Compare the medians, not the means - a
  least-squares fit can post the lower mean precisely by smearing its outliers
  over every other point.

## Install

The benchmark needs a **source checkout** of map-anything, because the Hydra
model configs live at its repository root:

```bash
git clone https://github.com/facebookresearch/map-anything
pip install -e ./map-anything

git clone https://github.com/<you>/pointmap-benchmark
cd pointmap-benchmark
pip install -r requirements.txt
```

If `mapanything` is installed somewhere the benchmark cannot auto-detect, set
`MAPANYTHING_ROOT=/path/to/map-anything`.

Model-specific extras (each one is optional; missing models are skipped with a
clear reason instead of crashing the run):

```bash
# Pi3 and VGGT are vendored inside mapanything - nothing to install.
pip install "mapanything[depth-anything-3]"              # Depth Anything 3
pip install "mapanything[mast3r]" "mapanything[dust3r]"  # MASt3R
```

MASt3R additionally needs its metric checkpoint downloaded manually:

```bash
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth \
  -P /path/to/checkpoints
```

Check what will actually run in your environment:

```bash
python scripts/run_benchmark.py --list-models
```

> **CUDA note.** Pi3, Depth Anything 3 and VGGT query `torch.cuda` when they are
> constructed, so they need a CUDA GPU. MapAnything also runs on CPU/MPS.

## Usage

Qualitative + runtime comparison on one folder of images:

```bash
python scripts/run_benchmark.py \
    --images /data/scene0/images \
    --models mapanything pi3 da3 \
    --output-dir runs/scene0 \
    --export-ply --export-glb
```

Quantitative comparison against ground truth, with every model forced onto the
same pixel grid:

```bash
python scripts/run_benchmark.py \
    --scenes-root /data/eval \
    --image-size 448x336 \
    --mast3r-checkpoint-dir /data/checkpoints \
    --output-dir runs/eval
```

Outputs land in `--output-dir`:

* `report.md` — Markdown tables, aggregated per model, arrows marking metric direction
* `results.csv` — one row per (model, scene), including the view provenance below
* `results.json` — `{"config": {...}, "records": [...]}`: the run configuration
  plus the same per-record data
* `<scene>/<model>.ply` / `.glb` / `.npz` — point clouds and raw predictions (with the export flags)

### Reproducibility

Which views a model is given changes its scores, so every run records what it
actually saw. `report.md` opens with two tables — **Run configuration** (code
revision, device, image size, stride, thresholds, timestamp) and **Views used**
(per scene: view count, a `views_digest`, and the sampling strategy) — and the
same fields are on every CSV/JSON record.

Two runs are comparable when their digests match. They are not when the digests
differ, however similar the settings look.

The default selection is `sorted_filename_stride`: sort by filename, take every
`--stride`-th image, cap at `--max-views`. It is recorded explicitly rather than
left implicit, because "whatever order the filenames happened to be in" is a
choice that moves the numbers. It is *not* the protocol map-anything's own
benchmark uses, which samples views by covisibility — see [Caveats](#caveats).

### Why `--image-size` matters

By default each model runs at its own native resolution mapping (518-based for
the ViT/14 models, 512-based for MASt3R), which is how each model was trained
and how it performs best. But then the models do not share a pixel grid.

`--image-size` forces a shared grid. It must be divisible by both patch sizes
in play (14 and 16), i.e. a multiple of **112** — for example `448x336`,
`560x448` or `672x504`. For ground-truth comparisons, use it: it makes the GT
depth resampling identical for every model.

### Useful flags

| flag | effect |
| --- | --- |
| `--stride N`, `--max-views N` | subsample the views of each scene |
| `--confidence-percentile P` | drop the lowest-confidence P% of pixels before scoring |
| `--no-mapanything-mask` | disable MapAnything's edge/ambiguity masking (raw dense output) |
| `--max-pairs N` | view pairs sampled for the consistency metric (default 30) |
| `--fail-fast` | abort on the first model error instead of recording it |

## Data layout

Single scene:

```
scene0/
├── images/*.png
└── gt.npz          # optional
```

Multiple scenes (`--scenes-root`):

```
eval/
├── scene0/{images/*.png, gt.npz}
└── scene1/{images/*.png, gt.npz}
```

Images are sorted by filename; that order is the view order.

## Ground-truth format

`gt.npz` (all arrays at the **original image resolution**):

| key | shape | meaning |
| --- | --- | --- |
| `depth_z` | `(V, H0, W0)` | z-depth in metres along the optical axis. `0` or non-finite = invalid |
| `intrinsics` | `(V, 3, 3)` | pinhole `[[fx,0,cx],[0,fy,cy],[0,0,1]]` in pixels |
| `poses_c2w` | `(V, 4, 4)` | OpenCV camera-to-world (`+X` right, `+Y` down, `+Z` forward) |
| `image_names` | `(V,)` *(optional)* | file names used to match GT entries to images; without it the GT must already be in sorted-filename order |

The GT depth and intrinsics are passed through the *same*
`crop_resize_if_necessary` that MapAnything applies to the RGB images, so the
GT point map and the predicted point map are defined on identical pixels. This
alignment is verified by `tests/test_pipeline.py`.

## Verifying the code

The metrics and the GT data path are unit-tested against a synthetic scene with
analytically exact ground truth (a ray-cast textured room), so correctness does
not depend on downloading any model weights:

```bash
pip install pytest
python -m pytest tests -q
```

The tests pin down, among others: a perfect prediction scores perfectly; a
prediction differing only by a global similarity transform still scores
perfectly under `cam0/*` and `sim3/*` but is caught by `metric/*`; added noise
strictly degrades every metric; the consistency metric is invariant to global
scale and drops when one view is made inconsistent; a model that predicts only
half the image is penalised on completeness but not on accuracy; the similarity
fit survives 10% gross outliers that defeat a plain least-squares fit; the
`sanity/*` check fires on a flipped pose convention; and the GT depth survives
the crop-resize pipeline unchanged.

Generate the same synthetic scene as real data to smoke-test the full run:

```bash
python scripts/make_synthetic_scene.py --output-dir data/synthetic --num-views 6
python scripts/run_benchmark.py --images data/synthetic/images \
    --gt data/synthetic/gt.npz --image-size 448x336 --output-dir runs/synthetic
```

Note that the synthetic scene is a *pipeline* test. Its appearance statistics
are nothing like real photographs, so absolute model scores on it are not
meaningful — use real data for real conclusions.

## Caveats

* **MASt3R is not feed-forward.** It runs pairwise inference plus a sparse
  global alignment optimisation, so its runtime is not comparable to the others
  and grows quickly with the number of views. Its wrapper also requires a batch
  size of 1.
* **Occlusion penalises the consistency metric.** Scenes with large
  view-dependent occlusion will show lower inlier ratios for every model; read
  `consistency/overlap_ratio` alongside it.
* **MapAnything's masking is on by default**, matching its recommended usage.
  This removes edge and ambiguous pixels that other models keep, which raises
  its accuracy and lowers its `cloud/valid_ratio` and `gt/covered_ratio`. Read
  those two columns next to every accuracy number, use `--no-mapanything-mask`
  for a raw-density comparison, or equalise coverage across models with
  `--confidence-percentile`.
* **Models that failed on some scenes are averaged over fewer scenes.** The
  report prints `num_scenes` in every table and warns at the top when the
  models did not complete the same set, because averages over different scenes
  are not comparable and nothing else would reveal it.
* **Metric scale.** `metric/*` is only a fair criticism of models that claim
  metric output. Pi3 is affine-invariant by design; DA3's scale depends on the
  checkpoint.
* Model weights come from the Hugging Face hub on first use and carry their own
  licences — notably `facebook/map-anything` is CC-BY-NC 4.0; use
  `--models mapanything_apache` for the Apache-2.0 weights.

## Licence

Apache-2.0. This project depends on, and follows the conventions of,
[map-anything](https://github.com/facebookresearch/map-anything) (Apache-2.0);
the model weights it downloads are covered by their own licences.
