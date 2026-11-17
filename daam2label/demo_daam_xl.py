"""SDXL text2img + DAAM auto-labeling (diffusers >= 0.19, run in daam_xl_venv).

SDXL gives markedly better image quality and prompt adherence than SD 1.4/1.5.
DAAM cross-attention is captured via daam.trace_xl (a custom AttnProcessor),
then box_extract turns the word heat map into pseudo boxes. Runs on an 8 GB GPU
with enable_model_cpu_offload. Saves img/heat/box/label.json/heatmap.npy per seed
in the same layout as demo_daam.py, so assets are interchangeable.

Example (daam_xl_venv, HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1):
  python demo_daam_xl.py --prompt "three cars driving on a road" --keyword cars \
      --seed 8236 --save-dir output/xl --save-heatmap
"""
import argparse
import json
import os
import random
import time
from copy import deepcopy

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline

from daam.trace_xl import trace_xl


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt", required=True)
    p.add_argument("--keyword", required=True)
    p.add_argument("--seed", type=int, default=41)
    p.add_argument("--num-outputs", type=int, default=1)
    p.add_argument("--save-dir", default="output/xl")
    p.add_argument("--side", type=int, default=512)
    p.add_argument("--steps", type=int, default=25)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--model-id", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--min-area-frac", type=float, default=None)
    p.add_argument("--peak-min-dist-frac", type=float, default=None)
    p.add_argument("--save-heatmap", action="store_true")
    args = p.parse_args()

    box_kwargs = {}
    for n in ("threshold", "min_area_frac", "peak_min_dist_frac"):
        v = getattr(args, n)
        if v is not None:
            box_kwargs[n] = v

    t = time.perf_counter()
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    print(f"[xl] model load+offload: {time.perf_counter()-t:.1f}s (one-time)", flush=True)

    seeds = [args.seed] if args.num_outputs <= 1 else random.sample(range(10000), args.num_outputs)
    for seed in seeds:
        save_path = os.path.join(args.save_dir, args.keyword, str(seed))
        os.makedirs(save_path, exist_ok=True)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        t = time.perf_counter()
        with trace_xl(pipe) as tc:
            out = pipe(args.prompt, num_inference_steps=args.steps, generator=gen,
                       height=args.side, width=args.side, guidance_scale=args.guidance)
            hm = tc.compute_global_heat_map().compute_word_heat_map(args.keyword)
            out_img, out_heat, out_bbox, out_scores = hm.plot_overlay(
                out.images[0], src_size=(args.side, args.side), box_kwargs=box_kwargs)
        dt = time.perf_counter() - t

        out_heat_vis = cv2.applyColorMap(np.uint8(255 * out_heat), cv2.COLORMAP_JET)
        show = deepcopy(out_img)
        valid = []
        for bbox, score in zip(out_bbox, out_scores):
            bbox = [int(b) for b in bbox]
            valid.append({"bbox": bbox, "score": round(float(score), 3)})
            cv2.rectangle(show, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 3)
        cv2.imwrite(os.path.join(save_path, "img.png"), out_img)
        cv2.imwrite(os.path.join(save_path, "heat.png"), out_heat_vis)
        cv2.imwrite(os.path.join(save_path, "box.png"), show)
        with open(os.path.join(save_path, "label.json"), "w") as f:
            json.dump({args.keyword: valid}, f)
        if args.save_heatmap:
            np.save(os.path.join(save_path, "heatmap.npy"), out_heat)
        print(f"[xl] {args.keyword} seed={seed} -> {save_path} ({len(valid)} boxes) {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
