#!/bin/bash -l
#SBATCH --job-name=pmb-official
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=03:00:00
#SBATCH --output=/home/hpc/iwfa/iwfa137h/pmb_jobs/logs/%x-%j.out
#SBATCH --export=NONE
#
# map-anything's own dense_n_view benchmark, as a cross-check of this one.
# Same data, same 130 @ ETH3D sets, 8 views, upstream's 518 resolutions:
#
#   MODEL=pi3 sbatch --export=ALL run_official.slurm
#
# Only ETH3D: upstream's config also lists ScanNet++, which is not on disk.

unset SLURM_EXPORT_ENV
set -euo pipefail

WOODY=/home/woody/iwfa/iwfa137h
MODEL=${MODEL:?set MODEL, e.g. pi3}
OUT=$WOODY/pmb_runs/official/${MODEL}-${SLURM_JOB_ID}

source $WOODY/miniconda3/etc/profile.d/conda.sh
conda activate pmb-gpu
export HF_HOME=$WOODY/hf_cache HF_HUB_OFFLINE=1 TORCH_HOME=$WOODY/torch_cache
export HYDRA_FULL_ERROR=1 PYTHONUNBUFFERED=1
# Upstream calls np.linalg.linalg.svd, removed in numpy 2.x; see numpy_compat/.
export PYTHONPATH=$HOME/pmb_jobs/numpy_compat${PYTHONPATH:+:$PYTHONPATH}

cd $HOME/map-anything
echo "job $SLURM_JOB_ID on $(hostname), official benchmark, model $MODEL, code $(git rev-parse --short HEAD)"
python benchmarking/dense_n_view/benchmark.py \
    machine=default \
    machine.root_data_dir=$WOODY/pmb_data/official_root \
    machine.mapanything_dataset_metadata_dir=$WOODY/pmb_data/wai_metadata \
    machine.root_pretrained_checkpoints_dir=$WOODY/checkpoints \
    machine.root_experiments_dir=$OUT \
    machine.root_uniception_pretrained_checkpoints_dir=$WOODY/checkpoints \
    machine.external_benchmark_data_root_data_dir=$WOODY/pmb_data \
    dataset=benchmark_518_eth3d_snpp_tav2 \
    'dataset.test_dataset="+ 130 @ ${dataset.eth3d_wai.test.dataset_str}"' \
    dataset.num_workers=4 \
    dataset.num_views=8 \
    batch_size=1 \
    model=$MODEL \
    hydra.run.dir=$OUT
echo "results in $OUT"
