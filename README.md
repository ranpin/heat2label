# heat2label

Two experiments on the same idea: **turn a model's internal heat map into
pseudo bounding-box labels**, so images can be (pre-)annotated without
hand-drawing every box. Both arms threshold an activation map, extract
regions, and emit boxes — they differ only in *where the heat map comes
from* and *what domain it serves*.

This repo is the merger of two sibling projects created on the same day
(2023-03-30): [`daam2label`](daam2label/) and [`cam2label`](cam2label/).
Each arm is self-contained — `cd` into it, install its own
`requirements.txt`, and run.

| | [`daam2label/`](daam2label/) | [`cam2label/`](cam2label/) |
|---|---|---|
| Heat-map source | Stable-Diffusion **cross-attention** (DAAM), text-prompt driven | **Grad-CAM++** on an ImageNet ResNet-101, prediction driven |
| Generates images? | Yes — SD v1.4 img2img augmentation, then labels the outputs | No — labels images that already exist |
| Domain | Driving-scene long-tail classes (`motorcyclist`, `truck`) | COCO zero-shot splits (65/15, 48/17) |
| Box type | Axis-aligned bbox → `label.json` (COCO-flavoured) | Rotated box (`cv2.minAreaRect`) drawn on the image |
| Compute | CUDA GPU ≥ 8 GB recommended | CPU is fine |
| Entry point | `demo_daam.py` (argparse CLI) | `gradcam/cam.py`, `gradcam/test.py` |

**Shared pipeline**

```
image ──► model forward ──► heat map ──► threshold ──► regions ──► boxes
                 │                                            │
        (SD cross-attn | Grad-CAM++)        (BFS islands | contours + minAreaRect)
```

The two arms share no code — they are independent implementations of the
same trick, kept side by side for comparison.

## Status

2023 research demos, maintained for reference. They are **pre-labeling /
data-augmentation helpers, not production annotators** — every box is a
pseudo-label that still needs human review.

## Layout

```
heat2label/
├── daam2label/     # arm 1 — SD cross-attention (DAAM) → driving-scene boxes
├── cam2label/      # arm 2 — Grad-CAM++ → COCO zero-shot rotated boxes
├── LICENSE         # MIT (covers both arms; the DAAM fork inherits upstream MIT)
└── README.md
```

## License

MIT — see [`LICENSE`](LICENSE). The `daam2label/` arm bundles a modified
copy of [`castorini/daam`](https://github.com/castorini/daam) (MIT); see
[`daam2label/NOTICE`](daam2label/NOTICE).
