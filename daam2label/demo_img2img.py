"""Plain SD1.5 img2img: add objects onto an EXISTING image, then DAAM-label.

Per user direction: use an existing cat/dog (or other) image as the base and let
the img2img prompt generate additional objects on it — no inpainting model, no
extra processing. Uses the base StableDiffusionImg2ImgPipeline (SD1.5). Run in
daam_venv (diffusers 0.9.0).

Example:
  python demo_img2img.py --image /path/cat.png \
      --prompt "two cats sitting together, realistic photo" --keyword cats \
      --strength 0.7 --seed 5 --save-dir output/img2img --save-heatmap
"""
import argparse
import json
import os
import time
from copy import deepcopy

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionImg2ImgPipeline

from daam import set_seed, trace


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, help="Existing base image to add objects onto.")
    p.add_argument("--prompt", required=True)
    p.add_argument("--keyword", required=True)
    p.add_argument("--save-dir", default="output/img2img")
    p.add_argument("--negative-prompt",
                   default="painting, cartoon, illustration, artistic, drawing, anime, sketch, 3d render, cgi, stylized")
    p.add_argument("--strength", type=float, default=0.7)
    p.add_argument("--seed", type=int, default=5)
    p.add_argument("--side", type=int, default=512)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--model-id", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
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
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model_id, torch_dtype=torch.float16, requires_safety_checker=False)
    pipe = pipe.to("cuda")
    pipe.enable_attention_slicing()
    pipe.set_progress_bar_config(disable=True)
    print(f"[img2img] model load: {time.perf_counter()-t:.1f}s", flush=True)

    base = Image.open(args.image).convert("RGB").resize((args.side, args.side))
    save_path = os.path.join(args.save_dir, args.keyword, str(args.seed))
    os.makedirs(save_path, exist_ok=True)
    gen = set_seed(args.seed)
    t = time.perf_counter()
    with trace(pipe) as tc:
        out = pipe(prompt=args.prompt, init_image=base, strength=args.strength,
                   negative_prompt=args.negative_prompt,
                   num_inference_steps=args.steps, generator=gen,
                   guidance_scale=args.guidance)
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
    print(f"[img2img] {args.keyword} seed={args.seed} -> {save_path} ({len(valid)} boxes) {dt:.1f}s", flush=True)


if __name__ == "__main__":
    main()
