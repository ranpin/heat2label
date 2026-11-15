"""Standalone validation for daam.box_extract on synthetic multi-object heat maps.

Runs without cv2/torch/GPU (pure numpy) so the multi-object extraction logic can
be verified on any machine. Run:  python3 test_box_extract.py
"""

import importlib.util
import os
import sys

import numpy as np

# Load box_extract directly by file path: daam/__init__.py pulls in transformers/
# torch, which are absent on a CPU-only test box. box_extract itself is pure numpy.
_MOD_PATH = os.path.join(os.path.dirname(__file__), "..", "daam", "box_extract.py")
_spec = importlib.util.spec_from_file_location("box_extract", _MOD_PATH)
_box = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_box)
extract_boxes = _box.extract_boxes
otsu_threshold = _box.otsu_threshold


def gauss(H, W, ci, cj, sigma, amp):
    y, x = np.mgrid[0:H, 0:W]
    return amp * np.exp(-((y - ci) ** 2 + (x - cj) ** 2) / (2.0 * sigma ** 2))


def show(name, heat, boxes, scores, expect):
    thr = otsu_threshold((heat - heat.min()) / (heat.max() - heat.min() + 1e-12))
    status = "PASS" if len(boxes) == expect else "FAIL"
    print(f"[{status}] {name}: got {len(boxes)} box(es), expect {expect}  (otsu={thr:.3f})")
    for b, s in zip(boxes, scores):
        print(f"        box={b}  conf={s:.3f}")
    return len(boxes) == expect


def main():
    H = W = 128
    results = []

    # T1: single blob -> 1 box
    h = gauss(H, W, 64, 64, 12, 1.0)
    results.append(show("T1 single blob", h, *extract_boxes(h), expect=1))

    # T2: two separated equal blobs -> 2 boxes
    h = gauss(H, W, 40, 40, 10, 1.0) + gauss(H, W, 90, 90, 10, 1.0)
    results.append(show("T2 two separated equal", h, *extract_boxes(h), expect=2))

    # T3: strong + weak blob (weak peak 0.35 < old fixed 0.5) -> 2 boxes
    h = gauss(H, W, 40, 40, 10, 1.0) + gauss(H, W, 90, 90, 10, 0.35)
    results.append(show("T3 strong + weak", h, *extract_boxes(h), expect=2))

    # T4: two touching blobs (one merged island) -> 2 boxes via peak split
    h = gauss(H, W, 64, 48, 12, 1.0) + gauss(H, W, 64, 80, 12, 1.0)
    results.append(show("T4 touching -> split", h, *extract_boxes(h), expect=2))

    # T5: blob + isolated single-pixel noise -> 1 box (noise filtered by area)
    h = gauss(H, W, 64, 64, 12, 1.0)
    h[10, 10] = 0.9
    h[120, 120] = 0.85
    h[5, 100] = 0.8
    results.append(show("T5 blob + noise", h, *extract_boxes(h), expect=1))

    # T6: three objects, mixed strengths -> 3 boxes
    h = (gauss(H, W, 30, 30, 8, 1.0)
         + gauss(H, W, 30, 96, 8, 0.6)
         + gauss(H, W, 96, 64, 10, 0.8))
    results.append(show("T6 three mixed", h, *extract_boxes(h), expect=3))

    print("\n%d/%d passed" % (sum(results), len(results)))
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
