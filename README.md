# heat2label

**🔗 Live demo — https://ranpin.github.io/heat2label/** · pick the Grad-CAM++
or DAAM arm (or the side-by-side compare view) and walk the full heat-map →
pseudo-box pipeline (precomputed; both models are too heavy to run in-browser).

Two experiments on the same idea: **turn a model's internal heat map into
pseudo bounding-box labels**, so images can be (pre-)annotated without
hand-drawing every box. Both arms threshold an activation map, extract
regions, and emit boxes — they differ only in *where the heat map comes
from* and *what domain it serves*.

**Pipeline at a glance** — the `cat` category through both arms (input → heat map → pseudo-box):

<table>
  <tr>
    <th></th>
    <th>Input</th>
    <th>Heat map</th>
    <th>Pseudo-box</th>
  </tr>
  <tr>
    <td><b>Grad-CAM++</b><br><sub>ResNet-101 CAM · prediction-driven</sub></td>
    <td><img src="https://ranpin.github.io/heat2label/assets/gradcam/65_15/cat/1/input.png" width="200" alt="dalle-mini generated input"></td>
    <td><img src="https://ranpin.github.io/heat2label/assets/gradcam/65_15/cat/1/heat.png" width="200" alt="Grad-CAM++ heat map"></td>
    <td><img src="https://ranpin.github.io/heat2label/assets/gradcam/65_15/cat/1/box.png" width="200" alt="rotated pseudo-box"></td>
  </tr>
  <tr>
    <td><b>DAAM</b><br><sub>SD v1.4 cross-attention · prompt-driven</sub></td>
    <td><img src="https://ranpin.github.io/heat2label/assets/daam/65_15/cat/1/img.png" width="200" alt="SD v1.4 generated input"></td>
    <td><img src="https://ranpin.github.io/heat2label/assets/daam/65_15/cat/1/heat.png" width="200" alt="DAAM heat map"></td>
    <td><img src="https://ranpin.github.io/heat2label/assets/daam/65_15/cat/1/box.png" width="200" alt="axis-aligned pseudo-box"></td>
  </tr>
</table>

<sub>Inputs: dalle-mini generation (Grad-CAM arm, 2023) vs SD v1.4 text2img (DAAM arm). Every box is a pseudo-label — see [Status & positioning](#status--positioning).</sub>

This repo is the merger of two sibling projects created on the same day
(2023-03-30): [`daam2label`](daam2label/) and [`cam2label`](cam2label/).
Each arm is self-contained — `cd` into it, install its own
`requirements.txt`, and run.

| | [`daam2label/`](daam2label/) | [`cam2label/`](cam2label/) |
|---|---|---|
| Heat-map source | Stable-Diffusion **cross-attention** (DAAM), text-prompt driven | **Grad-CAM++** on an ImageNet ResNet-101, prediction driven |
| Generates images? | Yes — SD v1.4 text2img generation + labeling in one script | No — the arm only labels; its images are **dalle-mini generations** (author's 2023-02 wandb experiment), not COCO photos |
| Domain | The same COCO zero-shot class set as cam2label (side-by-side comparison) + driving long-tail (`motorcyclist`, `truck`) | COCO zero-shot novel classes (65/15, 48/17 splits) — all images dalle-mini-generated |
| Box type | Axis-aligned bbox → `label.json` (COCO-flavoured) | Rotated box (`cv2.minAreaRect`) → `label.json` (corners + angle), drawn on the image |
| Compute | CUDA GPU ≥ 8 GB (measured ~4.7 s/image) | CPU is fine (measured ~0.3 s/image) |
| Entry point | `demo_daam.py` (argparse CLI, `--manifest` batch mode) | `gradcam/cam.py`, `gradcam/test.py`, `gradcam/demo_assets.py` |

**Shared pipeline**

```
image ──► model forward ──► heat map ──► threshold ──► regions ──► boxes
                 │                                            │
        (SD cross-attn | Grad-CAM++)        (BFS islands | contours + minAreaRect)
```

The two arms share no code — they are independent implementations of the
same trick, kept side by side for comparison. Both are *generation →
annotation* pipelines from different eras: cam2label labels dalle-mini
generations (2023), daam2label generates with SD v1.4 and labels in one
pass (2023 method, reproduced 2026).

## Status & positioning

2023 research demos, maintained for reference — **pre-labeling /
data-augmentation helpers, not production annotators**; every box is a
pseudo-label that still needs human review.

**As an auto-labeler this approach has been superseded by open-set
detection.** Measured on an RTX 3070 (SD v1.4 fp16, 512×512, 30 steps), the
DAAM arm takes ~4.7 s per image — and it can only label images it *generates
itself*. Modern open-set detectors label *your existing* images far faster
and cleaner: YOLO-World runs at 52 FPS on a V100 (35.4 LVIS AP), and
Grounding DINO 1.5 Edge exceeds 10 FPS on a Jetson Orin NX (36.2 LVIS AP).
Box stability is this approach's weak point too — across the 568-seed,
31-keyword generation campaign behind the demo, only ~26% of generations
yielded a single clean box, and the rate is strongly category-dependent
(75% for `scissors` vs ~5% for `cat`/`couch` even after 64 seeds each with
prompt variants; threshold-sensitive, fragmented heat maps).

**What still has reference value today:**

- **Interpretability** — DAAM exposes token-level attention provenance:
  which pixels the prompt token attended to at each denoising step.
  Detectors give boxes without the "why".
- **Generation–annotation consistency** — when you are already running SD
  for long-tail data augmentation, the pseudo-labels are nearly free
  (+~0.1 s per image, ~2%: attention is accumulated as a by-product of
  generation; heat-map + box extraction measured at 0.01 s + ≤0.2 s), giving
  a built-in check that the prompted concept was actually generated — where,
  and how many times.

## 2026 retrospective

The repo was re-audited in 2026 (merge, reproduction, demo page). What the
audit found:

- **Provenance** — every image in `cam2label/` (`imgs65_15/`, `imgs48_17/`,
  `outputs/`) is a **dalle-mini generation** from the author's 2023-02 wandb
  experiment, not a COCO photo; "65/15" and "48/17" are COCO zero-shot
  *class-split* names. Decisive evidence: the commented-out wandb download
  code at the top of the original `cam.py`.
- **Dependencies** — `daam2label/`'s original pin (`diffusers>=0.19,<0.25`)
  contradicted the vendored DAAM fork, which monkey-patches pre-0.15
  diffusers APIs: the arm only runs on `diffusers==0.9.0`, and
  `demo_daam.py`'s `image=` call had to revert to `init_image=`. The 2023
  pin was unrunnable as written; fixed.
- **Reproducibility drift** — re-running the same image through the same
  algorithm in 2026 yields boxes differing from the 2023 outputs by ~14% of
  raw pixels (torchvision changed its `resize` antialias default). The demo
  therefore re-runs each pipeline end-to-end in one pass instead of mixing
  2023 boxes with 2026 heat maps.
- **Bugs & gaps** — `cam.py` crashed on OpenCV 4.x (`findContours` 3-value
  unpack); the Grad-CAM arm drew rotated boxes but never exported
  coordinates (`label.json` export added in 2026).

What still holds: the core trick (heat map → threshold → regions → boxes)
reproduces cleanly on both arms, and the positioning above — superseded as
an auto-labeler, still interesting for interpretability and
generation–annotation consistency.

## Layout

```
heat2label/
├── daam2label/     # SD cross-attention (DAAM) → text2img generation + boxes
├── cam2label/      # Grad-CAM++ → rotated boxes on dalle-mini-generated COCO novel-class images
├── demo/           # Demo page source (deployed via gh-pages → ranpin.github.io/heat2label/)
├── LICENSE         # MIT (covers both arms; the DAAM fork inherits upstream MIT)
└── README.md
```

## License

MIT — see [`LICENSE`](LICENSE). The `daam2label/` arm bundles a modified
copy of [`castorini/daam`](https://github.com/castorini/daam) (MIT); see
[`daam2label/NOTICE`](daam2label/NOTICE).
