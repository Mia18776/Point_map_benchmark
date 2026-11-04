#!/bin/bash
# One-time setup on the login node: GPU env, model extras, weights, synthetic data.
set -euo pipefail
source /home/woody/iwfa/iwfa137h/miniconda3/etc/profile.d/conda.sh
WOODY=/home/woody/iwfa/iwfa137h
export HF_HOME=$WOODY/hf_cache

conda env list | grep -q '^pmb-gpu ' || conda create -y -q -n pmb-gpu --clone pmb
conda activate pmb-gpu
pip install -q --force-reinstall --no-deps torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -q --index-url https://download.pytorch.org/whl/cu126 torch torchvision   # pull matching nvidia-* deps
pip install -q -e "$HOME/map-anything[depth-anything-3,mast3r,dust3r]"
# croco/dust3r/DA3 pull in opencv-python 5.x, which shadows the headless 4.x
# build and cannot read EXR (WAI depth maps). Keep only the headless 4.x one.
pip uninstall -y -q opencv-python
pip install -q --force-reinstall --no-deps opencv-python-headless==4.10.0.84
# Not covered by the extras: DA3's wrapper needs addict, Pi3-X needs the pi3 package.
pip install -q --no-deps addict "pi3 @ git+https://github.com/yyfz/Pi3.git@main"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# MASt3R metric checkpoint
CK=$WOODY/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
[ -s "$CK" ] || wget -q -O "$CK" https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth

# HuggingFace weights
python - <<'PY'
from huggingface_hub import snapshot_download
for r in ["facebook/map-anything", "facebook/map-anything-apache", "yyfz233/Pi3", "facebook/VGGT-1B"]:
    print("downloading", r); snapshot_download(r)
PY

# Synthetic scene with exact GT
cd $HOME/Point_map_benchmark
python scripts/make_synthetic_scene.py --output-dir $WOODY/pmb_data/synthetic/scene0 --num-views 6
echo SETUP DONE
