"""daam2label — DAAM-guided auto-labeling demo.

Pipeline
--------
1. Read a list of real / synthetic road images.
2. For each image, run Stable Diffusion img2img N times with different seeds,
   using a text prompt that mentions a target keyword (e.g. "motorcyclist").
3. Extract per-word cross-attention heat map via the (forked) ``daam`` library.
4. Threshold the heat map, run BFS island extraction to obtain pseudo
   bounding boxes, and dump ``img.png`` / ``heat.png`` / ``box.png`` /
   ``label.json`` per seed.

This file is a cleaned, argparse-driven version of the original
``demo_daam.py``; the vendored ``daam/`` package next to it is a modified
fork of LordOfSunis/daam v0.0.12 (Apache-2.0) — see NOTICE.
"""

import argparse
import json
import os
import random
from copy import deepcopy

import cv2
import numpy as np
import torch
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionPipeline

from daam import set_seed, trace


# ---------------------------------------------------------------------------
# Self-vehicle hood occlusion polygons (original image resolution 1920x1080).
# Triggered only when the file name contains "camera4" (rear) or "camera5"
# (front). Images without either tag skip masking entirely.
# ---------------------------------------------------------------------------
FRONT_NODE = [
    [0, 1079], [0, 1044], [254, 978], [365, 965],
    [462, 943], [679, 796], [691, 780], [721, 760], [756, 761], [846, 736], [949, 731],
    [948, 714], [929, 704], [949, 693],
    [961, 692], [980, 701], [961, 717], [960, 732],
    [1036, 732], [1150, 762], [1185, 766],
    [1234, 801], [1428, 937], [1768, 1009], [1919, 1046],
    [1919, 1079],
]
BACK_NODE = [
    [0, 1079], [0, 1042], [115, 1021], [342, 985], [563, 958],
    [811, 944], [965, 941], [1178, 944], [1425, 963], [1662, 997],
    [1919, 1042], [1919, 1079],
]


def with_mask(img, position):
    """Blank out the vehicle hood region in ``img`` and return a ``keep`` mask
    marking that region so the original pixels can be pasted back later.

    Returns (masked_img, keep_mask). If position is unknown, the image is
    returned unchanged with an all-white keep mask.
    """
    keep = np.zeros(img.shape[:2], dtype=np.uint8)
    shadow = img.copy()

    if position == "camera5":
        poly = np.array(FRONT_NODE, dtype=np.int32)
    elif position == "camera4":
        poly = np.array(BACK_NODE, dtype=np.int32)
    else:
        # No hood to remove — pass through.
        keep[:] = 255
        return img, keep

    cv2.fillConvexPoly(shadow, poly, 0)
    cv2.fillConvexPoly(keep, poly, 255)
    img = cv2.bitwise_and(img, shadow)
    return img, keep


def _load_image_list(list_file):
    with open(list_file, "r") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def text2img(args):
    """Text2img labeling: SD generates a driving scene straight from the prompt
    (no init image), then the per-word DAAM heat map is thresholded via BFS into
    pseudo bounding boxes. Saves img.png / heat.png / box.png / label.json per
    seed — the text-to-image counterpart of ``img2img``.
    """
    load_dtype = torch.float16 if args.device != "cpu" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_id, torch_dtype=load_dtype
    ).to(args.device)
    if args.device != "cpu":
        pipe.enable_attention_slicing()

    seeds = [args.seed] if args.num_outputs <= 1 else random.sample(range(10000), args.num_outputs)
    for seed in seeds:
        save_path = os.path.join(args.save_dir, args.keyword, str(seed))
        os.makedirs(save_path, exist_ok=True)
        gen = set_seed(seed)
        with torch.cuda.amp.autocast(
            dtype=torch.float16, enabled=(args.device != "cpu")
        ), torch.no_grad():
            with trace(pipe) as tc:
                out = pipe(
                    args.prompt,
                    num_inference_steps=30,
                    generator=gen,
                    height=args.side,
                    width=args.side,
                )
                hm = tc.compute_global_heat_map().compute_word_heat_map(args.keyword)
                out_img, out_heat, out_bbox = hm.plot_overlay(
                    out.images[0], src_size=out.images[0].size
                )
                out_heat_vis = cv2.applyColorMap(
                    np.uint8(255 * out_heat), cv2.COLORMAP_JET
                )
                show = deepcopy(out_img)
                valid = []
                for bbox in out_bbox:
                    bbox = [int(b) for b in bbox]
                    valid.append(bbox)
                    cv2.rectangle(show, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 4)
                cv2.imwrite(os.path.join(save_path, "img.png"), out_img)
                cv2.imwrite(os.path.join(save_path, "heat.png"), out_heat_vis)
                cv2.imwrite(os.path.join(save_path, "box.png"), show)
                with open(os.path.join(save_path, "label.json"), "w") as f:
                    json.dump({args.keyword: valid}, f)
        print(f"[text2img] {args.keyword} seed={seed} -> {save_path} ({len(valid)} boxes)")


def img2img(args):
    """Main labeling loop. See module docstring."""
    # fp16 weights + attention slicing: fit SD v1.4 on an 8 GB GPU (see README).
    load_dtype = torch.float16 if args.device != "cpu" else torch.float32
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        args.model_id, torch_dtype=load_dtype
    ).to(args.device)
    if args.device != "cpu":
        pipe.enable_attention_slicing()
    image_paths = _load_image_list(args.image_list)

    for img_path in image_paths:
        print("=" * 60)
        print(img_path)
        print("=" * 60)

        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f"[skip] cannot read {img_path}")
            continue

        image_name = os.path.basename(img_path)
        if "camera5" in image_name:
            new_image, mask_keep = with_mask(image_bgr, "camera5")
        elif "camera4" in image_name:
            new_image, mask_keep = with_mask(image_bgr, "camera4")
        else:
            new_image, mask_keep = image_bgr, np.ones(image_bgr.shape[:2], dtype=np.uint8) * 255

        # Overlay region that should be preserved (vehicle hood) on top of the
        # generated output so the model never "hallucinates" over the hood.
        image_mask_keep = cv2.add(
            image_bgr, np.zeros(np.shape(image_bgr), dtype=np.uint8), mask=mask_keep
        )

        pil_init = Image.fromarray(new_image)
        image_h, image_w = pil_init.height, pil_init.width
        pil_init = pil_init.resize((args.side, args.side))

        seeds = random.sample(range(10000), args.num_outputs)
        with torch.cuda.amp.autocast(
            dtype=torch.float16, enabled=(args.device != "cpu")
        ), torch.no_grad():
            for seed in seeds:
                save_path = os.path.join(args.save_dir, image_name, str(seed))
                os.makedirs(save_path, exist_ok=True)

                gen = set_seed(seed)
                with trace(pipe) as tc:
                    out = pipe(
                        prompt=args.prompt,
                        init_image=pil_init,
                        strength=args.strength,
                        guidance_scale=7.5,
                        generator=gen,
                    )
                    hm = tc.compute_global_heat_map().compute_word_heat_map(args.keyword)
                    out_img, out_heat, out_bbox = hm.plot_overlay(
                        out.images[0], src_size=(image_w, image_h)
                    )

                    out_img = cv2.addWeighted(out_img, 1, image_mask_keep, 1, 0)
                    out_heat_vis = cv2.applyColorMap(
                        np.uint8(255 * out_heat), cv2.COLORMAP_JET
                    )

                    show = deepcopy(out_img)
                    valid = []
                    for bbox in out_bbox:
                        if bbox[1] >= image_h - 200:  # drop ground-plane FP
                            continue
                        bbox = [int(b) for b in bbox]
                        valid.append(bbox)
                        cv2.rectangle(show, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 255), 4)

                    cv2.imwrite(os.path.join(save_path, "img.png"), out_img)
                    cv2.imwrite(os.path.join(save_path, "heat.png"), out_heat_vis)
                    cv2.imwrite(os.path.join(save_path, "box.png"), show)
                    with open(os.path.join(save_path, "label.json"), "w") as f:
                        json.dump({args.keyword: valid}, f)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    # ---- img2img labeling (main) ----
    a = sub.add_parser("img2img", help="Generate augmented images + pseudo bbox labels.")
    a.add_argument("--prompt", required=True, help='SD prompt, e.g. "motorcyclist on road".')
    a.add_argument("--keyword", required=True, help='Token whose heat map becomes bboxes, e.g. "motorcyclist".')
    a.add_argument("--image-list", default="examples/motorcyclist.txt",
                   help="Text file with one image path per line.")
    a.add_argument("--save-dir", default="output/attentions", help="Root directory for outputs.")
    a.add_argument("--num-outputs", type=int, default=4, help="Seeds per image.")
    a.add_argument("--strength", type=float, default=0.45, help="img2img denoising strength.")
    a.add_argument("--side", type=int, default=768, help="Square resize side fed to SD.")
    a.add_argument("--model-id", default="CompVis/stable-diffusion-v1-4")
    a.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a.set_defaults(func=img2img)

    # ---- text2img labeling (generate scene + pseudo bbox) ----
    b = sub.add_parser("text2img", help="Generate driving scene from prompt + pseudo bbox labels (no init image).")
    b.add_argument("--prompt", required=True, help='SD prompt, e.g. "a motorcyclist riding on a road, dashcam view".')
    b.add_argument("--keyword", required=True, help='Token whose heat map becomes bboxes, e.g. "motorcyclist".')
    b.add_argument("--seed", type=int, default=41)
    b.add_argument("--num-outputs", type=int, default=1, help="If >1, sample this many random seeds.")
    b.add_argument("--save-dir", default="output/text2img", help="Root directory for outputs.")
    b.add_argument("--side", type=int, default=512, help="Square generation side (512 fits 8GB).")
    b.add_argument("--model-id", default="CompVis/stable-diffusion-v1-4")
    b.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    b.set_defaults(func=text2img)

    return p


if __name__ == "__main__":
    ns = build_parser().parse_args()
    ns.func(ns)
