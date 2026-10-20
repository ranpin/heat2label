# Modifications vs. upstream `castorini/daam` v0.0.12

The `daam/` folder is a **fork**, not a pip-installed copy. Everything not
listed below is functionally identical to the upstream release.

## `daam/heatmap.py`

| Element | Status | Reason |
|---|---|---|
| `crop_heat_map_bbox(heat_map, threshold=0.5, min_size=1024)` | **added** | BFS-connected-component extraction on the thresholded heat map; returns `[x1, y1, x2, y2]` per island. This is the whole point of the fork: convert an attention blob into a pseudo bbox. |
| `return_overlay_heat_map(im, heat_map, src_size, ...)` | **added** | Same body as the upstream `plot_overlay_heat_map` but returns `(image_np, resized_heat_map_np, bbox_list)` instead of only drawing. |
| `WordHeatMap.plot_overlay(self, image, src_size, ...)` | **changed signature** | Upstream had `plot_overlay(self, image, out_file=None, ...)`. We require `src_size` (original pre-resize W×H) so the returned bboxes are in original pixel coordinates, and we return the tuple rather than plotting. |
| `visualize_tool(...)` | **added** | Small matplotlib helper for one-off debugging; not used by the CLI. |
| `plot_overlay_heat_map_old(...)` | **kept for reference** | Dead code from upstream; retained so the diff against upstream stays small. |

## `daam/trace.py`

| Element | Status | Reason |
|---|---|---|
| `search_closest_factor(x)` | **added** | Returns two closest integer factors of `x`. |
| `UNetCrossAttentionHooker._unravel_attn` | **changed** | Upstream assumed a square latent grid (`h = w = sqrt(x.size(1))`), which breaks when the input image is not 512×512. We call `search_closest_factor` so non-square latents (e.g. 96×96 for 768-px inputs) unravel correctly. |
| `DiffusionHeatMapHooker.__init__` | **changed** | `latent_hw` is chosen between `4096` (512-px pipeline) and `9216` (768-px pipeline) based on `pipeline.unet.config.sample_size * pipeline.vae_scale_factor`. |

## Files unchanged vs. upstream

`__init__.py`, `_version.py`, `utils.py`, `hook.py`, `evaluate.py`,
`experiment.py`, and the three `run/*.py` example scripts are byte-for-byte
the upstream v0.0.12 code.

## How to reproduce the fork

```bash
git clone https://github.com/castorini/daam.git
git -C daam checkout v0.0.12   # or the matching commit
# then apply the modifications summarised above (or copy daam/ from here)
```

## Future work

The correct long-term fix is upstreaming `crop_heat_map_bbox` and the
`src_size` parameter to `castorini/daam`; this repository would then drop
the vendored copy in favour of a plain `pip install daam`. Patches welcome.
