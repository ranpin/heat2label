"""demo_assets.py —— 为 heat2label demo 页生成 Grad-CAM 臂资产。

与 cam.py 同一管线（resize 224×224 → GradCAMPlusPlus layer4[-1]，
targets=None 用预测类 → JET 热力图 → 阈值 170 → 第一个轮廓 →
minAreaRect 绿色旋转框），但每张图产出 demo 资产套件：
input.png / heat.png / box.png + label.json（旋转框坐标，minAreaRect
四角点 + 角度）+ timing.json（分阶段墙钟，不含文件 I/O）+ meta.json
（ResNet-101 预测类别）。兼容 OpenCV 3.x/4.x。

用法（先 cd 到本目录）：
    python demo_assets.py --input-dir imgs65_15 --out-dir demo_assets/65_15
    python demo_assets.py --input-dir imgs48_17 --out-dir demo_assets/48_17
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import torch
import torchvision
from pytorch_grad_cam import GradCAMPlusPlus
from torchvision.models import resnet101, ResNet101_Weights


def main():
    parser = argparse.ArgumentParser(
        description="生成 Grad-CAM 臂 demo 资产（input/heat/box + timing + 预测类别）")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    weights = ResNet101_Weights.IMAGENET1K_V1
    categories = weights.meta["categories"]

    t = time.perf_counter()
    model = resnet101(weights=weights)
    model.eval()
    load_s = time.perf_counter() - t
    print(f"[demo_assets] model load: {load_s:.1f}s (one-time)")

    names = sorted(n for n in os.listdir(args.input_dir)
                   if n.lower().endswith((".png", ".jpg", ".jpeg"))
                   and not n.startswith("."))  # 跳过 macOS AppleDouble ._* 伪图

    with GradCAMPlusPlus(model=model, target_layers=[model.layer4[-1]]) as cam:
        for name in names:
            stem = os.path.splitext(name)[0]
            out_dir = os.path.join(args.out_dir, stem)
            os.makedirs(out_dir, exist_ok=True)

            t = time.perf_counter()
            bin_data = torchvision.io.read_file(os.path.join(args.input_dir, name))
            img = torchvision.io.decode_image(bin_data) / 255
            if img.shape[0] == 4:  # RGBA → RGB（cam.py 无此保护，遇到 RGBA 会崩）
                img = img[:3]
            input_tensor = torchvision.transforms.functional.resize(img.unsqueeze(0), [224, 224])
            prep_s = time.perf_counter() - t

            t = time.perf_counter()
            with torch.no_grad():
                pred_idx = int(model(input_tensor).argmax(1).item())
            grayscale_cams = cam(input_tensor=input_tensor, targets=None)
            cam_s = time.perf_counter() - t
            grayscale_cam = grayscale_cams[0]

            rgb = input_tensor[0].detach().numpy().transpose((1, 2, 0))
            bgr = cv2.cvtColor(np.uint8(255 * rgb), cv2.COLOR_RGB2BGR)

            t = time.perf_counter()
            heat = cv2.applyColorMap(np.uint8(255 * grayscale_cam), colormap=cv2.COLORMAP_JET)
            ret, thresh = cv2.threshold(cv2.cvtColor(heat, cv2.COLOR_BGR2GRAY), 170, 255, 0)
            found = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = found[0] if len(found) == 2 else found[1]  # cv2 4.x 两值 / 3.x 三值
            cnt = contours[0]
            rect = cv2.minAreaRect(cnt)
            box = np.int64(cv2.boxPoints(rect))
            boxed = cv2.drawContours(bgr.copy(), [box], 0, (0, 255, 0), 2)
            box_s = time.perf_counter() - t

            cv2.imwrite(os.path.join(out_dir, "input.png"), bgr)
            cv2.imwrite(os.path.join(out_dir, "heat.png"), heat)
            cv2.imwrite(os.path.join(out_dir, "box.png"), boxed)
            timing = {"prep_s": round(prep_s, 3), "cam_s": round(cam_s, 3),
                      "box_s": round(box_s, 3),
                      "total_s": round(prep_s + cam_s + box_s, 3),
                      "model_load_s": round(load_s, 1), "device": "cpu",
                      "model": "resnet101 IMAGENET1K_V1", "side": 224}
            with open(os.path.join(out_dir, "timing.json"), "w") as f:
                json.dump(timing, f)
            meta = {"image": name, "pred_idx": pred_idx, "pred_class": categories[pred_idx]}
            with open(os.path.join(out_dir, "meta.json"), "w") as f:
                json.dump(meta, f)
            label = {categories[pred_idx]: {
                "corners": box.tolist(),
                "angle_deg": round(float(rect[2]), 1),
            }}
            with open(os.path.join(out_dir, "label.json"), "w") as f:
                json.dump(label, f)
            print(f"[demo_assets] {stem} -> {out_dir} pred={categories[pred_idx]} "
                  f"prep={prep_s:.2f}s cam={cam_s:.2f}s box={box_s:.2f}s")


if __name__ == "__main__":
    main()
