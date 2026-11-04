#!/bin/bash -l
#SBATCH --job-name=pmb-bench
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/hpc/iwfa/iwfa137h/pmb_jobs/logs/%x-%j.out
#SBATCH --export=NONE
#
# Point-map benchmark on one GPU.
#
#   sbatch run_benchmark.slurm                       # synthetic scene, default models
#   DATASET=eth3d sbatch --export=ALL run_benchmark.slurm   # WAI dataset (eth3d|scannetpp|tav2_wb)
#   SCENES_ROOT=/path/to/eval sbatch --export=ALL run_benchmark.slurm
#   MODELS="mapanything pi3" sbatch --export=ALL run_benchmark.slurm
#   DATASET=eth3d IMAGE_SIZE=native sbatch --export=ALL run_benchmark.slurm   # per-model resolution
#   sbatch -p v100 --gres=gpu:v100:1 run_benchmark.slurm    # other GPU type
#   CODE_DIR=$HOME/Point_map_benchmark_fix sbatch --export=ALL run_benchmark.slurm
#
# SCENES_ROOT holds one sub-folder per scene (images/*.png, optional gt.npz).

unset SLURM_EXPORT_ENV
set -euo pipefail

WOODY=/home/woody/iwfa/iwfa137h
SCENES_ROOT=${SCENES_ROOT:-$WOODY/pmb_data/synthetic}
MODELS=${MODELS:-"mapanything mast3r pi3 da3 vggt pi3x"}
IMAGE_SIZE=${IMAGE_SIZE:-448x336}   # WIDTHxHEIGHT shared by all models, or "native":
                                    # each model at upstream's benchmark resolution (WAI only)
EXTRA_ARGS=${EXTRA_ARGS-"--export-ply"}   # set EXTRA_ARGS="" to export nothing
OUT=$WOODY/pmb_runs/${SLURM_JOB_ID}

source $WOODY/miniconda3/etc/profile.d/conda.sh
conda activate pmb-gpu

# Weights were pre-downloaded on the login node; don't hit the network from here.
export HF_HOME=$WOODY/hf_cache
export HF_HUB_OFFLINE=1
export TORCH_HOME=$WOODY/torch_cache
export MAPANYTHING_ROOT=$HOME/map-anything

# Print progress as it happens; a block-buffered log looks hung for minutes.
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"

DATASET=${DATASET:-synthetic}
if [ "$DATASET" != synthetic ]; then
    # Any map-anything WAI test dataset (eth3d, scannetpp, tav2_wb), with the
    # subset fetched by scripts/prepare_wai_subset.py into pmb_data/<name>_wai
    DATA_ARGS="--wai-dataset $DATASET --wai-root $WOODY/pmb_data/${DATASET}_wai
               --wai-metadata-dir $WOODY/pmb_data/wai_metadata --wai-num-views ${NUM_VIEWS:-8}"
else
    DATA_ARGS="--scenes-root $SCENES_ROOT"
fi

SIZE_ARGS="--image-size $IMAGE_SIZE"
[ "$IMAGE_SIZE" = native ] && SIZE_ARGS=""

echo "job $SLURM_JOB_ID on $(hostname), models: $MODELS, data: $DATASET, image size: $IMAGE_SIZE"

# CODE_DIR picks the checkout to run, e.g. a git worktree with a fix in it.
cd ${CODE_DIR:-$HOME/Point_map_benchmark}
echo "code: $(pwd) @ $(git rev-parse --short HEAD)"
python scripts/run_benchmark.py \
    $DATA_ARGS \
    --models $MODELS \
    $SIZE_ARGS \
    --mast3r-checkpoint-dir $WOODY/checkpoints \
    --device cuda \
    --output-dir "$OUT" \
    $EXTRA_ARGS

echo "results in $OUT"
