# heat2label 2026 重审记录

2026 年对本仓做了一次重审（合并 daam2label + cam2label、复现两臂、制作 demo 页）。本文记录项目定位结论、审计发现的问题与处置，供复现两臂时参考。

## 定位与诚实边界

2023 年的研究 demo，留作参考——**是预标注 / 数据增强的辅助工具，不是生产级标注器**；所有框都是伪标签，仍需人工复核。

**作为自动标注器，这条路线已被开集检测取代。** DAAM 臂在 RTX 3070 上实测 ~4.7 s/图（SD v1.4 fp16，512×512，30 步），而且只能标注它**自己生成**的图；现代开集检测器能更快、更干净地标注**你已有的**图：YOLO-World 在 V100 上 52 FPS（35.4 LVIS AP），Grounding DINO 1.5 Edge 在 Jetson Orin NX 上 >10 FPS（36.2 LVIS AP）。框的稳定性也是本路线的弱项——demo 背后那次 568 种子 × 31 关键词的生成里，只有 ~26% 的生成图能给出单个干净的框，且强烈依赖类别（`scissors` 75% vs `cat`/`couch` ~5%，即便各跑 64 种子 + prompt 变体；对阈值敏感、热力图碎裂）。

**今天仍有参考价值的两点：**

- **可解释性**——DAAM 暴露 token 级注意力来源：每个 prompt token 在每一步去噪时关注了哪些像素。检测器只给框，不给「为什么」。
- **生成–标注一致性**——如果你本来就在跑 SD 做长尾数据增强，伪标签几乎是顺带的（+~0.1 s/图，约 2%：注意力是生成过程中累积的副产品；热力图 + 框提取实测 0.01 s + ≤0.2 s），并且内置了一道校验：prompt 里的概念到底有没有被生成出来、生成在哪、生成了几个。

## 来源考古（provenance）

`cam2label/` 里的**所有图像**（`imgs65_15/`、`imgs48_17/`、`outputs/`）都是 **dalle-mini 生成图**（作者 2023-02 的 wandb 实验 `ranpin/dalle-mini-tables-colab`），**不是 COCO 原图**；「65/15」「48/17」只是 COCO 零样本的**类别划分名**（标明哪些是 novel 类）。决定性证据：原始 `cam.py` 顶部被注释掉的 wandb 下载代码。

- `imgs65_15/`、`imgs48_17/`：每类 2 张的精选批（按类重命名）。
- `outputs/`：全量批——1024 张生成图 + `test.py` 产的 1024 张历史框图。

## 依赖矛盾

`daam2label/` 原始 pin（`diffusers>=0.19,<0.25`）与 vendored 的 DAAM fork **自相矛盾**：该 fork monkey-patch 的是 diffusers 0.15 之前的旧注意力 API（`CrossAttention._attention` / `_sliced_attention`、`StableDiffusionPipeline._encode_prompt`），这些在 diffusers 0.15 的 AttnProcessor 重构里已全部移除。因此本臂**只能在 `diffusers==0.9.0`**（castorini/daam v0.0.12 的目标版本）上跑，且 `demo_daam.py` 的 `image=` 调用须回退成 `init_image=`（0.9.0 签名）。2023 年的原始 pin 照写是跑不通的；已修（commit `aea4100`：requirements 钉 `diffusers==0.9.0` + `transformers==4.24.0`，demo_daam.py 改 `init_image=` + fp16 + `enable_attention_slicing`）。又因 diffusers 0.9.0 import 了 Python 3.12 已移除的 `distutils`，故需 **Python ≤3.11**。

## 复现漂移

2026 年用同一算法重跑同一张图，得到的框与 2023 年产物有 **~14% 原始像素差异**——根因是 torchvision 改了 `resize` 的 antialias 默认值。因此 demo 页**不混用** 2023 历史框与 2026 热力图，而是把每条管线端到端一次性重跑。

## Bug 与缺口

- `cam.py` 在 OpenCV 4.x 上崩溃（`findContours` 返回值从 3 个变 2 个，旧代码按 3 值解包报错）；已修。
- Grad-CAM 臂只画旋转框、**从不导出坐标**；`label.json` 导出为 2026 年补上。

## 仍然成立的结论

核心思路（热力图 → 阈值化 → 连通区域 → 框）在两臂上都能干净复现；定位也成立——作为自动标注器已被开集检测取代，但对可解释性与生成–标注一致性仍有参考价值（详见上方「定位与诚实边界」）。
