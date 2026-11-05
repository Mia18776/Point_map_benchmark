# Cluster workflow (FAU NHR, TinyGPU A100)

The scripts used to run this benchmark on the FAU NHR cluster and to turn the
results into reports. They are a copy of `~/pmb_jobs/` on the cluster, kept here
as a backup; the working copy is still `~/pmb_jobs/`. Paths are hard-coded for
user `iwfa137h` (`/home/woody/iwfa/iwfa137h/...`).

## Folders on the cluster

| Folder | Holds |
| --- | --- |
| `~/pmb_jobs/` | these scripts, plus `logs/` (one log per Slurm job) |
| `/home/woody/iwfa/iwfa137h/pmb_data/` | WAI datasets (ETH3D, TartanAirV2-WB) |
| `/home/woody/iwfa/iwfa137h/pmb_runs/<jobid>/` | results of one job |
| `/home/woody/iwfa/iwfa137h/pmb_runs/final/` | merged results and the reports |

## Files

| File | Does |
| --- | --- |
| `setup_gpu_env.sh` | one-time: conda env `pmb-gpu`, model extras, weights |
| `run_benchmark.slurm` | one Slurm job of `scripts/run_benchmark.py` (env vars `DATASET`, `MODELS`, `IMAGE_SIZE`, `EXTRA_ARGS`, `CODE_DIR`) |
| `run_official.slurm` | map-anything's own `dense_n_view` benchmark, as a cross-check |
| `numpy_compat/sitecustomize.py` | restores `np.linalg.linalg`, which upstream's metrics still call under numpy 2.x |
| `finalize.py` | one command after the jobs finish: merge them, link into `final/`, regenerate every report |
| `pmb_final.py` | shared helpers of the report scripts (dataset discovery, default-settings rows) |
| `summarize_results.py` | writes `final/RESULTS.md` (one page) and `final/DETAILS.md` (everything) |
| `make_report_html.py` + `report_template.html` | writes `final/report.html` |
| `results_intro.md`, `results_conclusions.md` | hand-written text of `RESULTS.md` ("what was tested", "conclusions") |
| `findings.md` | hand-written analysis appended to `DETAILS.md` |
| `agreement_analysis.py` | cross-model pose agreement as a GT-free signal, from the poses saved with `--export-poses`; writes `final/AGREEMENT.md` |
| `metrics_manual.py` | the metrics manual (every results.csv column: meaning, direction, function, origin); refuses to run if a column is missing |

## Typical run

```bash
cd ~/pmb_jobs
for m in mapanything mapanything_apache mast3r pi3 da3 vggt pi3x; do
  DATASET=eth3d IMAGE_SIZE=native MODELS=$m EXTRA_ARGS="" \
    sbatch --export=ALL --job-name=full-eth3d-$m run_benchmark.slurm
done
# ... when they have finished:
python ~/pmb_jobs/finalize.py --logs full-eth3d
```
