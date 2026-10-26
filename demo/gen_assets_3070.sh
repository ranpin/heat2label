#!/bin/bash
# Regenerate all heat2label demo assets (DAAM + Grad-CAM) on the RTX 3070 box.
#
# This is the exact recipe used on 2026-10-20 to produce the images shown on
# https://ranpin.github.io/heat2label/ . Paths below are specific to the 3070
# layout (/data/tmp/chenrunbin/...) — adjust PY / cd targets for another box.
#
# Prereqs on the box:
#   - daam_venv: a torch 2.6 venv with DAAM + Grad-CAM deps (reuses homebot-wm torch)
#   - daam2label/  (this repo's DAAM arm, with examples/*.txt)
#   - gradcam_demo/make_gradcam_assets.py  (Grad-CAM arm)
#   - HF_ENDPOINT=hf-mirror.com (huggingface.co unreachable from the box)
#
# Individual commands are also documented in daam2label/README.md; this script
# is kept as the end-to-end orchestration (venv + mirror + sample list).
PY=/data/tmp/chenrunbin/daam_venv/bin/python
export HF_ENDPOINT=https://hf-mirror.com
cd /data/tmp/chenrunbin/daam2label || exit 1

echo "=== [1/3] DAAM motorcyclist (2 seeds) ==="
$PY demo_daam.py img2img --prompt "motorcyclist on road" --keyword motorcyclist \
    --image-list examples/motorcyclist.txt --num-outputs 2 --side 512 \
    --save-dir output/attentions/motorcyclist

echo "=== [2/3] DAAM truck (2 seeds) ==="
$PY demo_daam.py img2img --prompt "truck on road" --keyword truck \
    --image-list examples/truck.txt --num-outputs 2 --side 512 \
    --save-dir output/attentions/truck

echo "=== [3/3] Grad-CAM (cat bear airplane train) ==="
cd /data/tmp/chenrunbin/gradcam_demo || exit 1
$PY make_gradcam_assets.py --out /data/tmp/chenrunbin/gradcam_assets \
    --samples cat bear airplane train

echo "=== ALL_ASSETS_DONE ==="
echo "DAAM pngs: $(find /data/tmp/chenrunbin/daam2label/output/attentions -name '*.png' | wc -l)"
echo "GradCAM pngs: $(find /data/tmp/chenrunbin/gradcam_assets -name '*.png' 2>/dev/null | wc -l)"
