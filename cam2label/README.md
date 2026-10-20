# cam2label

> [`heat2label`](../) 的 **Grad-CAM 臂**——合并仓库里两条「热力图→伪框标注」实验之一。
> 下面所有命令都在 `cam2label/gradcam/` 目录里运行。

用 Grad-CAM++ 热力图自动给图像生成**旋转包围框（bounding box）标注**的一组实验脚本。核心思路：在 ImageNet 预训练的 ResNet101 上跑 Grad-CAM++，把得到的类激活热力图二值化后取轮廓，再用最小外接旋转矩形（`cv2.minAreaRect`）画回原图，从而把「CAM」转成「框标注」——即 cam → label。

仓库里同时存放了一批文本生成图像（从代码注释看来自 dalle-mini / wandb 实验）及其生成结果，脚本被用来给这些生成图批量产框。

## 它做什么

- `cam.py`：遍历 `imgs65_15/`（或 `imgs48_17/`）里的图片，逐张 resize 到 224×224，用 `GradCAMPlusPlus`（目标层 `model.layer4[-1]`，`use_cuda=False`）算热力图，再经 `drawBounding()` 画出绿色旋转框，存到 `results65_15/`。
- `test.py`：逻辑与 `cam.py` 相同，但用 `os.walk` 递归遍历 `outputs/` 下的生成图，结果写到每类目录下的 `results/` 子目录。
- `fastRCNN.py`：一个独立的演示脚本，改用 Faster R-CNN（`fasterrcnn_resnet50_fpn`）+ `EigenCAM` 对当前目录下的 `both.png` 做检测可视化，并包含框内 CAM 归一化函数。它只在内存里生成 PIL 图像，不写盘。
- `COCO_splite.py`：只定义了两个 COCO 零样本（zero-shot）划分的类别名元组——`CLASSES_65_15` 和 `CLASSES_48_17`，注释标明 `#` 之后是新增（novel）类。

## 目录结构

```
gradcam/
├── cam.py            # 主脚本：对 imgs* 目录批量生成框标注
├── test.py           # 对 outputs/ 生成图批量生成框标注
├── fastRCNN.py       # Faster R-CNN + EigenCAM 演示（读 both.png）
├── COCO_splite.py    # COCO 零样本两种划分的类别名
├── imgs48_17/        # 48/17 划分的输入图（34 张）
├── imgs65_15/        # 65/15 划分的输入图（30 张）
├── results48_17/     # 48/17 的框标注结果（34 张）
├── results65_15/     # 65/15 的框标注结果（30 张）
├── outputs/          # 文本生成图像，共 2048 张
│   └── {48_17_256, 65_15_256}/{100step, 200step}/<类别>/samples/NNNN.png
├── testImgs/         # 少量手工测试图（5 张）
└── testResults/      # 对应的测试输出
```

`outputs/` 按「划分_分辨率 / 扩散步数 / 类别」组织，每类约 16 张采样图；部分类别目录名是生成时的 prompt（如 `frisbee;flying disc;sports`）。

## 环境依赖

见 [`gradcam/requirements.txt`](gradcam/requirements.txt)（2026 合并时补上）。CPU 即可运行，无需 CUDA：

```bash
cd gradcam
pip install -r requirements.txt
```

## 运行

路径均为相对路径，**先 `cd gradcam/` 再运行**。输入/输出目录用命令行参数指定（有默认值）：

```bash
cd gradcam
python cam.py                                        # 默认 imgs65_15 -> results65_15
python cam.py --input-dir imgs48_17 --output-dir results48_17
python test.py                                       # 默认递归处理 outputs/
python fastRCNN.py --image testImgs/both.png         # 默认读当前目录 both.png
```

- 换数据集（`imgs48_17` ↔ `imgs65_15`）用 `--input-dir` / `--output-dir` 即可，无需改源码。
- `fastRCNN.py` 用 `--image` 指定输入图（仓库里在 `testImgs/both.png`）。

## 实现要点

- 热力图 → 框的流程：`applyColorMap` → 阈值二值化（`drawBounding` 中阈值 170，可调）→ `findContours` → 取 `contours[0]` → `minAreaRect` 得旋转框 → `drawContours` 画绿框。
- `targets=None`，即自动用预测概率最大的类别来生成热力图；也可用 `ClassifierOutputTarget` 指定类别（代码里有注释示例）。

## 已知限制

- `drawBounding` 只取第一个轮廓（`contours[0]`），默认每张图只有一个目标。
- `test.py` 用 `os.walk` 递归处理输入目录下的**所有**图片——重跑会把上次写出的 `results/` 结果也再处理一遍（在已画框的图上再画框）。

> 2026 合并进 heat2label 时做过一轮修正：Windows 反斜杠路径改 `os.path.join`、加 argparse CLI、
> ResNet101 只加载一次、依赖改用现代 torchvision `weights=` 与 grad-cam 自动设备检测。
