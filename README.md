# cam2label

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

仓库**没有** `requirements.txt` 等依赖清单，依赖需从 import 推断：

- `torch`、`torchvision`（用到 `resnet101`、`fasterrcnn_resnet50_fpn` 预训练权重，首次运行会自动下载）
- [`grad-cam`](https://github.com/jacobgil/pytorch-grad-cam)（`pip install grad-cam`，提供 `pytorch_grad_cam`）
- `opencv-python`、`matplotlib`、`numpy`、`requests`、`Pillow`

参考安装：

```bash
pip install torch torchvision grad-cam opencv-python matplotlib numpy requests pillow
```

## 运行

脚本里的路径都是相对路径，**必须先 `cd gradcam/` 再运行**：

```bash
cd gradcam
python cam.py      # 处理 imgs65_15/，结果写到 results65_15/
python test.py     # 递归处理 outputs/，结果写到各类目录下的 results/
```

- 要换数据集（`imgs48_17` ↔ `imgs65_15`），需要**改源码**：`cam.py` 里切换 `path_list` / `pathA` 两行，并同步改 `myimshows`、`drawBounding` 里写死的输出目录。
- `fastRCNN.py` 需要当前目录下存在 `both.png`（仓库里在 `testImgs/both.png`）。

## 实现要点

- 热力图 → 框的流程：`applyColorMap` → 阈值二值化（`drawBounding` 中阈值 170，可调）→ `findContours` → 取 `contours[0]` → `minAreaRect` 得旋转框 → `drawContours` 画绿框。
- `targets=None`，即自动用预测概率最大的类别来生成热力图；也可用 `ClassifierOutputTarget` 指定类别（代码里有注释示例）。

## 已知限制

- 输出路径用了 Windows 反斜杠（如 `results65_15\\`），脚本按 Windows 环境编写；在 macOS/Linux 上直接跑会写出带 `\` 的异常文件名，需要先改成 `os.path.join`。
- `cam.py` / `test.py` 在循环里每张图都重新加载一次 ResNet101，效率较低。
- `drawBounding` 只取第一个轮廓（`contours[0]`），默认每张图只有一个目标。
- 脚本没有命令行参数，数据集、输入/输出目录均硬编码。
