"""Generate Grad-CAM demo assets (input / heat / box) for selected samples.

Reproduces the fixed cam.py pipeline (GradCAMPlusPlus on ResNet101, threshold
170, minAreaRect rotated box) but also saves the intermediate JET heat map so
the demo page can show the full input -> heat -> box flow.

Run from cam2label/gradcam/ (needs imgs65_15/):
    python ../../demo/make_gradcam_assets.py --out ../../demo/assets/gradcam
"""
import argparse
import os

import cv2
import numpy as np
import torchvision
from pytorch_grad_cam import GradCAMPlusPlus
from torchvision.models import resnet101

DEFAULT_SAMPLES = ["cat", "dog", "airplane", "bear", "elephant", "train"]


def tensor2img(tensor):
    a = tensor.detach().numpy()
    if a.max() > 1 or a.min() < 0:
        a = a - a.min()
        a = a / a.max()
    if a.shape[0] == 1:
        a = np.concatenate([a, a, a], axis=0)
    return a.transpose((1, 2, 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-dir", default="imgs65_15")
    ap.add_argument("--out", required=True, help="output dir for demo assets")
    ap.add_argument("--samples", nargs="+", default=DEFAULT_SAMPLES)
    args = ap.parse_args()

    model = resnet101(weights="IMAGENET1K_V1")
    model.eval()
    cam = GradCAMPlusPlus(model=model, target_layers=[model.layer4[-1]])

    for name in args.samples:
        src = os.path.join(args.input_dir, name + ".png")
        if not os.path.exists(src):
            print(f"[skip] {src}")
            continue
        dst = os.path.join(args.out, name)
        os.makedirs(dst, exist_ok=True)

        bin_data = torchvision.io.read_file(src)
        img = torchvision.io.decode_image(bin_data) / 255
        img = img.unsqueeze(0)
        input_tensor = torchvision.transforms.functional.resize(img, [224, 224])

        grayscale_cams = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cams[0]
        rgb_img = tensor2img(input_tensor[0])  # 224x224 model input

        # input.png: what the model sees (224x224)
        cv2.imwrite(
            os.path.join(dst, "input.png"),
            cv2.cvtColor(np.uint8(255 * rgb_img), cv2.COLOR_RGB2BGR),
        )
        # heat.png: JET heat map (224x224)
        cv2.imwrite(
            os.path.join(dst, "heat.png"),
            cv2.applyColorMap(np.uint8(255 * grayscale_cam), cv2.COLORMAP_JET),
        )
        # box.png: rotated box on the 224x224 image (threshold 170 + minAreaRect)
        colored = cv2.applyColorMap(np.uint8(255 * grayscale_cam), colormap=2)
        _, thresh = cv2.threshold(
            cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY), 170, 255, 0
        )
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        box_img = np.uint8(255 * cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
        if contours:
            rect = cv2.minAreaRect(contours[0])
            box = np.int64(cv2.boxPoints(rect))
            box_img = cv2.drawContours(box_img, [box], 0, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(dst, "box.png"), box_img)
        print(f"[ok] {name}")


if __name__ == "__main__":
    main()
