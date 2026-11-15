"""Multi-object pseudo-box extraction from DAAM heat maps.

The original ``crop_heat_map_bbox`` did a single fixed-threshold (0.5) BFS pass
with a fixed 1024-px area cutoff. That is too naive for multi-object scenes:
weaker objects fall below the global threshold and vanish, two nearby objects
whose attention blobs touch merge into one island (one box for both), and the
absolute area cutoff does not scale with resolution.

This module keeps the connected-region (BFS) idea but makes it multi-object
robust:

  1. adaptive threshold (Otsu, with a floor) instead of a fixed 0.5;
  2. 8-connected component labeling (cv2 fast path, pure-numpy BFS fallback so
     the logic is unit-testable without cv2/torch/GPU);
  3. per-island attention-peak detection + multi-source BFS geodesic splitting,
     so touching objects that merge into one island are separated along the
     saddle between their peaks;
  4. resolution-relative minimum-area filter instead of a fixed pixel count;
  5. attention-mass confidence per box, ranking, and IoU NMS.

Boxes are ``[x1, y1, x2, y2]`` with x=column, y=row, matching the original
return shape so existing callers are unaffected.
"""

from collections import deque

import numpy as np

try:
    from numpy.lib.stride_tricks import sliding_window_view
except ImportError:  # pragma: no cover - very old numpy
    sliding_window_view = None


def otsu_threshold(values, bins=256):
    """Otsu's between-class variance method on a [0,1] array; returns threshold in [0,1]."""
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    centers = (edges[:-1] + edges[1:]) / 2.0
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    sum_total = float((centers * hist).sum())
    sum_bg = 0.0
    w_bg = 0.0
    best_var = -1.0
    best_thr = float(centers[0])
    for i in range(bins):
        w_bg += hist[i]
        if w_bg == 0:
            continue
        w_fg = total - w_bg
        if w_fg == 0:
            break
        sum_bg += centers[i] * hist[i]
        mean_bg = sum_bg / w_bg
        mean_fg = (sum_total - sum_bg) / w_fg
        var = w_bg * w_fg * (mean_bg - mean_fg) ** 2
        if var > best_var:
            best_var = var
            best_thr = float(centers[i])
    return best_thr


def connected_components(mask):
    """8-connected labeling. Returns (num_components, labels) with 0=background.

    Uses cv2 when importable (fast, production); otherwise a pure-numpy BFS so
    the pipeline stays testable on machines without cv2.
    """
    mask_u8 = mask.astype(np.uint8)
    try:
        import cv2
        num, labels = cv2.connectedComponents(mask_u8, connectivity=8)
        return num - 1, labels.astype(np.int32)
    except Exception:
        pass

    H, W = mask_u8.shape
    labels = np.zeros((H, W), dtype=np.int32)
    current = 0
    for i in range(H):
        for j in range(W):
            if mask_u8[i, j] and labels[i, j] == 0:
                current += 1
                labels[i, j] = current
                q = deque([(i, j)])
                while q:
                    ci, cj = q.popleft()
                    for di in (-1, 0, 1):
                        ni = ci + di
                        if ni < 0 or ni >= H:
                            continue
                        for dj in (-1, 0, 1):
                            nj = cj + dj
                            if nj < 0 or nj >= W:
                                continue
                            if mask_u8[ni, nj] and labels[ni, nj] == 0:
                                labels[ni, nj] = current
                                q.append((ni, nj))
    return current, labels


def _local_max_mask(values, valid, radius):
    """Boolean mask of local maxima within `valid` over a (2*radius+1) window."""
    sub = np.where(valid, values, -np.inf)
    if radius < 1 or sliding_window_view is None:
        return valid & (sub > 0)
    padded = np.pad(sub, radius, mode="constant", constant_values=-np.inf)
    win = sliding_window_view(padded, (2 * radius + 1, 2 * radius + 1))
    lmax = win.max(axis=(2, 3))
    return valid & (sub >= lmax - 1e-9) & (sub > 0)


def _merge_close_peaks(values, coords, min_dist):
    """Greedy non-maximum peak selection: keep strongest, drop any within min_dist."""
    if not coords:
        return []
    vals = np.array([values[i, j] for i, j in coords], dtype=np.float64)
    order = np.argsort(vals)[::-1]
    kept = []
    md2 = float(min_dist) ** 2
    for idx in order:
        i, j = coords[idx]
        if all((i - ki) ** 2 + (j - kj) ** 2 >= md2 for ki, kj in kept):
            kept.append((int(i), int(j)))
    return kept


def _split_by_peaks(component_mask, peaks):
    """Multi-source BFS (geodesic Voronoi) from `peaks` within `component_mask`.

    Each pixel is claimed by whichever peak's BFS front reaches it first, which
    splits one connected island into one region per object along the saddle
    between attention peaks. Returns an int region map (-1 = unreached).
    """
    H, W = component_mask.shape
    region = np.full((H, W), -1, dtype=np.int32)
    q = deque()
    for idx, (pi, pj) in enumerate(peaks):
        region[pi, pj] = idx
        q.append((pi, pj, idx))
    while q:
        ci, cj, idx = q.popleft()
        for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ni, nj = ci + di, cj + dj
            if 0 <= ni < H and 0 <= nj < W and component_mask[ni, nj] and region[ni, nj] == -1:
                region[ni, nj] = idx
                q.append((ni, nj, idx))
    return region


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms(boxes, scores, iou_thr):
    """Greedy IoU NMS; returns indices to keep (highest score first)."""
    order = list(np.argsort(scores)[::-1])
    keep = []
    while order:
        i = order.pop(0)
        keep.append(i)
        order = [j for j in order if _iou(boxes[i], boxes[j]) <= iou_thr]
    return keep


def extract_boxes(heat_map, threshold=None, thr_floor=0.15, min_area_frac=0.005,
                  peak_local_radius=3, peak_min_dist_frac=0.15, nms_iou=0.3,
                  top_k=None):
    """Extract per-object boxes from a heat map.

    Args:
        heat_map: 2D array; re-normalized to [0,1] internally (defensive).
        threshold: absolute threshold in [0,1]; None = adaptive (Otsu).
        thr_floor: lower bound applied to the adaptive threshold so a diffuse
            map does not threshold near zero and turn background into objects.
        min_area_frac: drop boxes smaller than this fraction of image area
            (default 0.005, tuned on real diffuse DAAM heat maps).
        peak_local_radius: window radius for local-maximum (peak) detection.
        peak_min_dist_frac: two peaks must be at least this fraction of
            min(H, W) apart to count as separate objects (controls splitting;
            default 0.15 — real diffuse DAAM heat maps carry many noise peaks,
            a smaller value over-segments one object into tens of boxes).
        nms_iou: IoU threshold for dedup; None disables NMS.
        top_k: keep only the k highest-confidence boxes; None keeps all.

    Returns:
        (boxes, scores): boxes are [x1, y1, x2, y2] (x=col, y=row) sorted by
        score desc; scores are attention-mass confidence normalized to [0, 1].
    """
    heat = np.asarray(heat_map, dtype=np.float64)
    while heat.ndim > 2:
        heat = heat.squeeze()
    if heat.ndim != 2:
        return [], []
    H, W = heat.shape
    lo, hi = float(heat.min()), float(heat.max())
    if hi <= lo:
        return [], []
    heat = (heat - lo) / (hi - lo)

    thr = otsu_threshold(heat) if threshold is None else float(threshold)
    thr = max(thr, float(thr_floor))
    mask = heat >= thr
    if not mask.any():
        return [], []

    num, labels = connected_components(mask)
    min_area = max(1.0, float(min_area_frac) * H * W)
    peak_min_dist = max(peak_local_radius + 1, int(round(peak_min_dist_frac * min(H, W))))

    raw_boxes, raw_scores = [], []
    for lab in range(1, num + 1):
        comp = labels == lab
        if int(comp.sum()) < min_area:
            continue
        peak_coords = [tuple(c) for c in np.argwhere(_local_max_mask(heat, comp, peak_local_radius))]
        peaks = _merge_close_peaks(heat, peak_coords, peak_min_dist)
        if len(peaks) <= 1:
            ys, xs = np.where(comp)
            raw_boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
            raw_scores.append(float(heat[comp].sum()))
        else:
            region = _split_by_peaks(comp, peaks)
            for idx in range(len(peaks)):
                sub = region == idx
                if int(sub.sum()) < min_area:
                    continue
                ys, xs = np.where(sub)
                raw_boxes.append([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())])
                raw_scores.append(float(heat[sub].sum()))

    if not raw_boxes:
        return [], []

    if nms_iou is not None:
        keep = nms(raw_boxes, raw_scores, nms_iou)
        raw_boxes = [raw_boxes[i] for i in keep]
        raw_scores = [raw_scores[i] for i in keep]

    order = np.argsort(raw_scores)[::-1]
    boxes = [raw_boxes[i] for i in order]
    scores = [raw_scores[i] for i in order]
    if top_k is not None:
        boxes, scores = boxes[:top_k], scores[:top_k]
    smax = max(scores) if scores else 0.0
    scores = [s / smax for s in scores] if smax > 0 else scores
    return boxes, scores
