#!/usr/bin/env python3
"""Offline parameter tuning for DAAM box_extract on REAL heat maps.

The unit tests (tests/test_box_extract.py) only cover synthetic Gaussians. Real
DAAM heat maps are more diffuse, so threshold / min_area_frac / peak_min_dist_frac
need eyeballing on real outputs. This script does that WITHOUT re-running Stable
Diffusion: it loads the heat maps saved by `demo_daam.py ... --save-heatmap`
(heatmap.npy + img.png per output dir), sweeps the box_extract params, and writes
a contact sheet (one tile per param combo, boxes drawn + combo label + box count)
plus summary.csv.

NOTE on the threshold control: extract_boxes does `thr = max(otsu, thr_floor)`, so
on a strong+weak heat map Otsu already returns a high threshold (~0.26-0.30) that
cuts the weak object — sweeping thr_floor below Otsu has NO effect. The real control
is the explicit `threshold` (overrides Otsu). This script therefore sweeps
`--threshold` with "otsu" (production adaptive path, thr_floor=0.15) and explicit
floats (override Otsu, thr_floor=0 so the value is the clean control).

Pure numpy + cv2; no GPU / torch / diffusers needed (generation already ran).

Workflow on the 3070 (daam_venv), from daam2label/:
  # 1) generate once on real images, saving heat maps (needs GPU + SD):
  python demo_daam.py img2img --prompt "..." --keyword "..." \
      --image-list examples/motorcyclist.txt --save-dir output/tune --save-heatmap
  # 2) sweep params offline (cheap, iterate freely):
  python tools/tune_box_extract.py --tuning-dir output/tune --out-dir output/tune_sweep
"""

import argparse
import csv
import importlib.util
import itertools
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np


def load_box_extract():
    """Load daam/box_extract.py by file path: daam/__init__.py pulls in transformers/
    torch, which a sweep-only box does not need. box_extract itself is pure numpy."""
    here = Path(__file__).resolve().parent
    mod_path = here.parent / "daam" / "box_extract.py"
    spec = importlib.util.spec_from_file_location("box_extract", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def draw_tile(img_bgr, boxes, scores, tile_w, label):
    """Resize img to tile_w, draw scaled boxes + per-box score, prepend a label bar."""
    h, w = img_bgr.shape[:2]
    scale = tile_w / float(w)
    tile_h = max(1, int(round(h * scale)))
    tile = cv2.resize(img_bgr, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    for b, s in zip(boxes, scores):
        x1, y1, x2, y2 = [int(round(v * scale)) for v in b]
        cv2.rectangle(tile, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(tile, f"{s:.2f}", (x1, max(14, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    bar_h = 24
    bar = np.full((bar_h, tile_w, 3), 28, dtype=np.uint8)
    cv2.putText(bar, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, tile])


def contact_sheet(tiles, cols, pad=4, bg=60):
    """Grid-montage uniformly sized tiles into one contact sheet."""
    if not tiles:
        return None
    th, tw = tiles[0].shape[:2]
    rows = math.ceil(len(tiles) / cols)
    sheet = np.full((rows * th + (rows + 1) * pad, cols * tw + (cols + 1) * pad, 3),
                    bg, dtype=np.uint8)
    for idx, t in enumerate(tiles):
        r, c = divmod(idx, cols)
        y = pad + r * (th + pad)
        x = pad + c * (tw + pad)
        sheet[y:y + th, x:x + tw] = t[:th, :tw]
    return sheet


def parse_floats(s):
    return [float(x) for x in s.split(",") if x.strip()]


def parse_thresholds(s):
    """Parse a comma list of mask thresholds: "otsu" -> None (adaptive, production
    path), explicit floats -> override Otsu. Returns a list of None/float."""
    out = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        out.append(None if x.lower() == "otsu" else float(x))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tuning-dir", required=True,
                   help="Dir passed to demo_daam.py --save-dir (searched recursively for heatmap.npy).")
    p.add_argument("--out-dir", default="output/tune_sweep", help="Where to write contact sheets + summary.csv.")
    p.add_argument("--threshold", default="otsu,0.15,0.25",
                   help='Comma list of mask thresholds: "otsu" (adaptive, production) or explicit floats (override Otsu).')
    p.add_argument("--min-area-frac", default="0.001,0.003", help="Comma list of min_area_frac values.")
    p.add_argument("--peak-min-dist-frac", default="0.05,0.08", help="Comma list of peak_min_dist_frac values.")
    p.add_argument("--tile-w", type=int, default=360, help="Tile width (px) in the contact sheet.")
    p.add_argument("--cols", type=int, default=0, help="Contact-sheet columns (0 = ceil(sqrt(N))).")
    args = p.parse_args()

    box_extract = load_box_extract()
    thresholds = parse_thresholds(args.threshold)
    min_areas = parse_floats(args.min_area_frac)
    peak_dists = parse_floats(args.peak_min_dist_frac)
    combos = list(itertools.product(thresholds, min_areas, peak_dists))
    thr_repr = [("otsu" if t is None else t) for t in thresholds]
    print(f"[sweep] {len(combos)} param combos: threshold={thr_repr} "
          f"min_area_frac={min_areas} peak_min_dist_frac={peak_dists}")

    tuning_dir = Path(args.tuning_dir)
    heatmaps = sorted(tuning_dir.rglob("heatmap.npy"))
    if not heatmaps:
        sys.exit(f"[sweep] no heatmap.npy under {tuning_dir}; "
                 f"run demo_daam.py ... --save-heatmap first")
    print(f"[sweep] found {len(heatmaps)} heat map(s)")

    os.makedirs(args.out_dir, exist_ok=True)
    summary_path = os.path.join(args.out_dir, "summary.csv")
    cols = args.cols or math.ceil(math.sqrt(len(combos)))

    with open(summary_path, "w", newline="") as cf:
        writer = csv.writer(cf)
        writer.writerow(["case", "keyword", "threshold", "min_area_frac",
                         "peak_min_dist_frac", "n_boxes", "boxes", "scores"])
        for hm_path in heatmaps:
            d = hm_path.parent
            case = str(d.relative_to(tuning_dir)) if d != tuning_dir else d.name
            heat = np.load(hm_path)
            img_path = d / "img.png"
            if not img_path.exists():
                print(f"[skip] {case}: no img.png beside heatmap.npy")
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[skip] {case}: cannot read img.png")
                continue
            if img.shape[:2] != heat.shape[:2]:
                print(f"[warn] {case}: img {img.shape[:2]} != heat {heat.shape[:2]}; "
                      f"scaling boxes by img/heat ratio")
            kw = ""
            lj = d / "label.json"
            if lj.exists():
                try:
                    kw = next(iter(json.loads(lj.read_text())), "")
                except Exception:
                    pass

            sx = img.shape[1] / float(heat.shape[1])
            sy = img.shape[0] / float(heat.shape[0])
            tiles = []
            for (thr, ma, pd_) in combos:
                # "otsu" (thr=None) -> production adaptive path (thr_floor=0.15 floors Otsu);
                # explicit float -> override Otsu, thr_floor=0 so the value is the clean control.
                boxes, scores = box_extract.extract_boxes(
                    heat, threshold=thr, thr_floor=(0.15 if thr is None else 0.0),
                    min_area_frac=ma, peak_min_dist_frac=pd_)
                # boxes are in heat-map coords; scale to img coords if they differ
                iboxes = [[b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy] for b in boxes]
                thr_lbl = "otsu" if thr is None else f"{thr}"
                label = f"thr={thr_lbl} min_area={ma} peak_dist={pd_}  ->  {len(boxes)} box"
                tiles.append(draw_tile(img, iboxes, scores, args.tile_w, label))
                writer.writerow([case, kw, thr_lbl, ma, pd_, len(boxes),
                                 json.dumps([[round(v, 1) for v in b] for b in iboxes]),
                                 json.dumps([round(s, 3) for s in scores])])
            sheet = contact_sheet(tiles, cols)
            out_name = case.replace(os.sep, "_") + "_sweep.png"
            cv2.imwrite(os.path.join(args.out_dir, out_name), sheet)
            print(f"[sweep] {case}: {len(combos)} combos -> {out_name}")

    print(f"[sweep] done. contact sheets + {summary_path}")


if __name__ == "__main__":
    main()
