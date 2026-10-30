# heat2label

> 两条「热力图 → 伪框标注」实验：把模型内部的热力图阈值化、提取连通区域、输出包围框，从而不用手动画框就能给图像打（预）标注。

> 🎬 **在线 Demo**：<https://ranpin.github.io/heat2label/> —— Grad-CAM++ 臂 / DAAM 臂 / 两臂对比三种视图，全部示例预计算（两个模型都太重，无法在浏览器内运行）。

## 核心结果

两臂共享同一条管线，只在「热力图从哪来」和「服务什么域」上不同。下图是 `cat` 类别分别过两条臂（输入 → 热力图 → 伪框）：

<table>
  <tr>
    <th></th>
    <th>输入</th>
    <th>热力图</th>
    <th>伪框</th>
  </tr>
  <tr>
    <td><b>Grad-CAM++</b><br><sub>ResNet-101 CAM · 预测类驱动</sub></td>
    <td><img src="https://ranpin.github.io/heat2label/assets/gradcam/65_15/cat/1/input.png" width="200" alt="dalle-mini 生成输入图"></td>
    <td><img src="https://ranpin.github.io/heat2label/assets/gradcam/65_15/cat/1/heat.png" width="200" alt="Grad-CAM++ 热力图"></td>
    <td><img src="https://ranpin.github.io/heat2label/assets/gradcam/65_15/cat/1/box.png" width="200" alt="旋转伪框"></td>
  </tr>
  <tr>
    <td><b>DAAM</b><br><sub>SD v1.4 交叉注意力 · prompt 驱动</sub></td>
    <td><img src="https://ranpin.github.io/heat2label/assets/daam/65_15/cat/1/img.png" width="200" alt="SD v1.4 生成输入图"></td>
    <td><img src="https://ranpin.github.io/heat2label/assets/daam/65_15/cat/1/heat.png" width="200" alt="DAAM 热力图"></td>
    <td><img src="https://ranpin.github.io/heat2label/assets/daam/65_15/cat/1/box.png" width="200" alt="轴对齐伪框"></td>
  </tr>
</table>

<sub>输入来源：Grad-CAM++ 臂用 dalle-mini 生成图（2023），DAAM 臂用 SD v1.4 text2img 即时生成。<b>所有框都是伪标签、需人工复核</b>；定位与诚实边界见 <a href="docs/RETROSPECTIVE.md">docs/RETROSPECTIVE.md</a>。</sub>

| | [`daam2label/`](daam2label/)（DAAM 臂） | [`cam2label/`](cam2label/)（Grad-CAM++ 臂） |
|---|---|---|
| 热力图来源 | Stable-Diffusion **交叉注意力**（DAAM），文本 prompt 驱动 | ImageNet ResNet-101 上的 **Grad-CAM++**，预测类驱动 |
| 是否生成图像 | 是——SD v1.4 text2img 生成 + 标注一个脚本完成 | 否——本臂只标注；图像是 **dalle-mini 生成图**（作者 2023-02 的 wandb 实验），并非 COCO 原图 |
| 域 | 与 Grad-CAM 臂同一套 COCO 零样本类别（并排对比）+ 驾驶长尾（`motorcyclist`、`truck`） | COCO 零样本 novel 类（65/15、48/17 两个划分）——图像均为 dalle-mini 生成 |
| 框类型 | 轴对齐框 → `label.json`（类名 → `[x1,y1,x2,y2]` 列表） | 旋转框（`cv2.minAreaRect`）→ `label.json`（四角点 + 角度），画回图上 |
| 算力 | CUDA GPU ≥ 8 GB（实测 ~4.7 s/图） | CPU 即可（实测 ~0.3 s/图） |
| 入口 | `demo_daam.py`（argparse CLI，`--manifest` 批量模式） | `gradcam/cam.py`、`gradcam/test.py`、`gradcam/demo_assets.py` |

共享管线：

```
图像 ──► 模型前向 ──► 热力图 ──► 阈值化 ──► 连通区域 ──► 框
```

两臂只在两处不同：**热力图来源**（SD 交叉注意力 | Grad-CAM++）与**区域提取方式**（BFS 岛 | 轮廓 + `minAreaRect`）。两条臂不共享代码——是同一思路的两个独立实现，并排放置以便对比；都属于「生成 → 标注」管线：cam2label 标注 dalle-mini 生成图（2023），daam2label 用 SD v1.4 一边生成一边标注（2023 的方法，2026 年复现）。本仓是 2023-03-30 同日创建的两个兄弟项目的合并；每臂自包含——`cd` 进去、装它自己的 `requirements.txt` 即可跑。项目定位、诚实边界与 2026 重审记录见 [docs/RETROSPECTIVE.md](docs/RETROSPECTIVE.md)。

## 快速开始

两臂各自独立安装运行（细节见各臂 README）：

**Grad-CAM++ 臂（cam2label，CPU 即可，~0.3 s/图）：**

```bash
cd cam2label/gradcam
pip install -r requirements.txt
python cam.py                    # 默认处理 imgs65_15/ → results65_15/（旋转框标注）
```

**DAAM 臂（daam2label，需 GPU ≥8GB，~4.7 s/图）：**

```bash
cd daam2label
pip install -r requirements.txt          # 钉死 diffusers==0.9.0，需 Python ≤3.11
python demo_daam.py --prompt "motorcyclist on road" --keyword "motorcyclist"
```

⚠️ DAAM 臂依赖苛刻：vendored 的 DAAM fork monkey-patch 了 diffusers 0.15 之前的旧注意力 API，**只能在 `diffusers==0.9.0` 上跑**（原 2023 的 `requirements.txt` pin `>=0.19,<0.25` 自相矛盾、跑不通，已修）；又因 diffusers 0.9.0 import 了 Python 3.12 已移除的 `distutils`，故需 **Python ≤3.11**。详见 [docs/RETROSPECTIVE.md](docs/RETROSPECTIVE.md)。

## 仓库结构与文档导航

```
heat2label/
├── daam2label/            # DAAM 臂：SD 交叉注意力 → text2img 生成 + 标注（详见其 README.md）
├── cam2label/             # Grad-CAM++ 臂：在 dalle-mini 生成图上产旋转框（详见其 README.md）
├── demo/                  # demo 页源码（经 gh-pages 部署 → ranpin.github.io/heat2label/）
├── docs/
│   └── RETROSPECTIVE.md   # 定位与诚实边界 + 2026 重审（来源考古 / 依赖矛盾 / 复现漂移 / bug）
├── LICENSE                # MIT（覆盖两臂；DAAM fork 继承上游 MIT）
└── README.md
```

## 许可与致谢

MIT——见 [`LICENSE`](LICENSE)。`daam2label/` 臂内含 [`castorini/daam`](https://github.com/castorini/daam)（MIT）的修改副本，见 [`daam2label/NOTICE`](daam2label/NOTICE)。仓库内所有图像均为 dalle-mini 与 Stable Diffusion v1.4 的文本生成图，不含 COCO 原图（「65/15」「48/17」只是 COCO 零样本的类别划分名）。
