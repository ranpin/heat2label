# daam2label

> The **DAAM arm** of [`heat2label`](../) — a merged repo holding two
> heat-map → pseudo-label experiments. Run every command below from this
> `daam2label/` directory.

Turn Stable-Diffusion cross-attention heat maps into **pseudo bounding boxes**
for long-tail classes in driving scenes. Given a real or synthetic init image
and a keyword such as `motorcyclist`, the tool runs img2img N times with
different seeds, extracts the per-word DAAM heat map for each output, and
emits `img.png` / `heat.png` / `box.png` / `label.json` under a COCO-flavoured
bbox schema. It is intended as a **data-augmentation and pre-labeling**
helper, not a substitute for human annotation.

```
prompt="motorcyclist on road", keyword="motorcyclist"
    │
    ▼
SD v1.4 img2img (N seeds, strength≈0.45)
    │
    ▼
DAAM cross-attention heat map  ──►  BFS island extraction  ──►  pseudo bboxes
    │                                                              │
    ▼                                                              ▼
 heat.png overlay                                          label.json
```

## Repository layout

```
daam2label/                  # arm of heat2label; the MIT LICENSE lives at the repo root
├── demo_daam.py             # CLI entry point (argparse)
├── daam/                    # Vendored fork of castorini/daam v0.0.12 (see NOTICE)
├── examples/                # Two synthetic placeholder images + image lists
├── docs/                    # Extra notes (modifications, citation, etc.)
├── requirements.txt
├── NOTICE                   # What we changed vs. upstream and why
└── tools/make_placeholder.ps1   # Regenerate the placeholder images
```

## Install

```bash
conda create -n daam2label python=3.9 -y
conda activate daam2label
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

Hardware: single CUDA GPU with ≥ 8 GB VRAM recommended (SD v1.4 fp16 + `xformers`).
CPU / Apple-Silicon will run but is very slow and untested against this fork.

## Quick start

The two files under `examples/` are **procedurally drawn placeholders** —
they are not real dashcam frames — but they let you smoke-test the full
pipeline without any external download.

```bash
# motorcyclist example
python demo_daam.py img2img \
    --prompt "motorcyclist on road" \
    --keyword motorcyclist \
    --image-list examples/motorcyclist.txt \
    --num-outputs 2 \
    --save-dir output/attentions/motorcyclist

# truck example
python demo_daam.py img2img \
    --prompt "truck on road" \
    --keyword truck \
    --image-list examples/truck.txt \
    --num-outputs 2 \
    --save-dir output/attentions/truck
```

Each seed produces a directory containing:

| File         | Content                                                            |
|--------------|--------------------------------------------------------------------|
| `img.png`    | The img2img output, with the vehicle-hood region pasted back on top |
| `heat.png`   | JET-colormapped DAAM heat map for the keyword                      |
| `box.png`    | `img.png` with the extracted pseudo bboxes drawn in cyan           |
| `label.json` | `{"<keyword>": [[x1, y1, x2, y2], ...]}`                           |

Or skip the init image entirely: `text2img` generates a driving scene straight
from the prompt and labels it (same `img/heat/box/label.json` outputs). This is
the mode the [live demo](https://ranpin.github.io/heat2label/) showcases:

```bash
python demo_daam.py text2img \
    --prompt "a motorcyclist riding on a city road, dashcam view" \
    --keyword motorcyclist
```

## Providing your own images

`--image-list` is a plain text file with one image path per line. Paths are
resolved relative to the current working directory. Any resolution works,
but the tool internally resizes to a square `--side` (default 768) before
feeding SD, then rescales the resulting heat map back to the original
`(width, height)` for bbox output.

### Vehicle-hood occlusion (optional)

The script contains two hardcoded polygons (`FRONT_NODE`, `BACK_NODE`) that
mask the vehicle hood when the file name contains `camera5` (front) or
`camera4` (rear) — those polygons are authored for a specific 1920×1080 rig
and will do nothing for images without either tag. Rename your files, or
edit the polygons in `demo_daam.py` to match your own rig.

## Reproducing the placeholder images

The two `examples/*.png` files were drawn by a small PowerShell script; run
it again to regenerate them without any network access:

```powershell
powershell -File tools/make_placeholder.ps1
```

## Credits & citation

- **DAAM library** — vendored under `daam/`, modified; upstream is
  [`castorini/daam`](https://github.com/castorini/daam) (MIT, © 2022
  Castorini, author Raphael Tang). See [`NOTICE`](NOTICE) and
  [`docs/modifications.md`](docs/modifications.md).
- **DAAM paper** — Tang et al., *"What the DAAM: Interpreting Stable
  Diffusion Using Cross Attention"*, ACL 2023.

  ```bibtex
  @inproceedings{tang2023daam,
    title     = {What the {DAAM}: Interpreting Stable Diffusion Using Cross Attention},
    author    = {Tang, Raphael and Lin, Qiaochu and Gong, Tianyi and Sun, Mt and Chen, Xinyi and Cohen, Andrew and Fysse, Elias and Ji, Heng},
    booktitle = {Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
    year      = {2023}
  }
  ```

- **Stable Diffusion v1.4** — `CompVis/stable-diffusion-v1-4` on
  Hugging Face, released under the CreativeML OpenRAIL-M license. Download
  of the weights is triggered automatically by `diffusers` on first run.

## License

MIT. The bundled `daam/` fork inherits the upstream MIT License; the
copyright lines in [`LICENSE`](../LICENSE) (at the `heat2label` repo root)
cover both the upstream package and the modifications in this repository.
