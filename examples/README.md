# examples/

The two PNGs in this folder are **procedurally drawn placeholders**, not
real dashcam frames and not Stable-Diffusion outputs. They exist so a
fresh clone can run the end-to-end pipeline (`demo_daam.py img2img …`)
without downloading any dataset.

- `motorcyclist_0001.png` — 1792×1024; sky, road, a solid orange rectangle
  labelled "motorcyclist", and a black vehicle-hood silhouette at the
  bottom edge.
- `truck_0001.png` — same scene, larger steel-blue rectangle labelled
  "truck".

Regenerate them any time:

```powershell
powershell -File tools/make_placeholder.ps1
```

The `.txt` files next to the images are image lists understood by
`--image-list`: one path per line, resolved relative to the repository
root (i.e. run `python demo_daam.py …` from `daam2label/`).

For real work, replace these with your own driving images and update the
corresponding `.txt` files; the tool does not require the placeholder
aesthetic and will happily accept photos of any resolution.
