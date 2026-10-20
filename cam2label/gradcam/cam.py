"""cam.py —— Grad-CAM++ 热力图自动生成旋转包围框标注。

遍历输入目录里的图片，逐张 resize 到 224×224，用 GradCAMPlusPlus
（目标层 model.layer4[-1]）算热力图，再经 drawBounding() 画出绿色旋转框，
存到输出目录。

用法（先 cd 到本目录）：
    python cam.py                                          # 默认 imgs65_15 -> results65_15
    python cam.py --input-dir imgs48_17 --output-dir results48_17
"""
import argparse
import os

import cv2
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget  # noqa: F401  供下方注释示例使用
from torchvision.models import resnet101


def myimshows(imgs, output_dir, titles=False, fname=" ", size=6):
    print(fname)
    lens = len(imgs)

    # #存热力图
    plt.figure(figsize=(size, size))
    if titles == False:
        titles = "0123456789"
    for i in range(1, lens + 1):
        plt.xticks(())
        plt.yticks(())
        plt.subplot(111)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap=plt.cm.jet)
            plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight')

    # 存总体对比图
    plt.figure(figsize=(size * lens, size))
    if titles == False:
        titles = "0123456789"
    for i in range(1, lens + 1):
        cols = 100 + lens * 10 + i
        plt.xticks(())
        plt.yticks(())
        plt.subplot(cols)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap=plt.cm.jet)
        else:
            plt.imshow(imgs[i - 1])
        plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    fname = fname + '1'
    plt.savefig(os.path.join(output_dir, fname), bbox_inches='tight')


def drawBounding(img, img1, output_dir, fname=" "):
    img = cv2.applyColorMap(np.uint8(255 * img), colormap=2)
    ret, thresh = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 170, 255, 0)  # 修改阈值可以微调包围框
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 得到轮廓信息
    cnt = contours[0]
    x, y, w, h = cv2.boundingRect(cnt)
    # Straight Bounding Rectangle
    # imgnew = cv2.rectangle(np.uint8(255 * img1), (x, y), (x + w, y + h), (0, 255, 0), 2)  # np.uint8() 处理仅在 img1 为原图时需要
    # Rotate Rectangle
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int64(box)
    imgnew = cv2.drawContours(np.uint8(255 * img1), [box], 0, (0, 255, 0), 2)
    cv2.imwrite(os.path.join(output_dir, fname), imgnew)


def tensor2img(tensor, heatmap=False, shape=(224, 224)):
    np_arr = tensor.detach().numpy()  # [0]
    # 对数据进行归一化
    if np_arr.max() > 1 or np_arr.min() < 0:
        np_arr = np_arr - np_arr.min()
        np_arr = np_arr / np_arr.max()
    # np_arr = (np_arr * 255).astype(np.uint8)
    if np_arr.shape[0] == 1:
        np_arr = np.concatenate([np_arr, np_arr, np_arr], axis=0)
    np_arr = np_arr.transpose((1, 2, 0))
    return np_arr


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM++ 热力图自动生成旋转框标注")
    parser.add_argument("--input-dir", default="imgs65_15", help="输入图片目录（默认 imgs65_15）")
    parser.add_argument("--output-dir", default="results65_15", help="框标注输出目录（默认 results65_15）")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    path_list = os.listdir(args.input_dir)
    path_list.sort()  # 保证读取顺序
    print(path_list)

    # 模型与 CAM 只初始化一次，循环内复用（原实现每张图都重新加载 ResNet101）
    model = resnet101(weights="IMAGENET1K_V1")
    model.eval()
    target_layers = [model.layer4[-1]]  # 如果传入多个 layer，cam 输出结果将会取均值

    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        for pathB in path_list:
            path = os.path.join(args.input_dir, pathB)
            bin_data = torchvision.io.read_file(path)  # 加载二进制数据
            img = torchvision.io.decode_image(bin_data) / 255  # 解码成 CHW 的图片
            img = img.unsqueeze(0)  # 变成 BCHW 的数据，B==1; squeeze
            input_tensor = torchvision.transforms.functional.resize(img, [224, 224])

            # 对图像进行水平翻转，得到两个数据
            # input_tensor = torch.cat([input_tensor, input_tensor.flip(dims=(3,))], axis=0)

            # targets = [ClassifierOutputTarget(38)]  # 指定查看 class_num 为 38 的热力图
            # aug_smooth=True, eigen_smooth=True 使用图像增强使热力图变得更加平滑
            grayscale_cams = cam(input_tensor=input_tensor, targets=None)  # targets=None 自动调用概率最大的类别显示
            for grayscale_cam, tensor in zip(grayscale_cams, input_tensor):
                # 将热力图结果与原图进行融合
                rgb_img = tensor2img(tensor)
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                # 绘制热力图和对比图
                # myimshows([rgb_img, grayscale_cam, visualization], args.output_dir, ["image", "gradcam++", "overlap"], fname=pathB.split(".")[0] + '_cam')
                drawBounding(grayscale_cam, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR), args.output_dir, pathB)  # 在原图上画框
                # drawBounding(grayscale_cam, cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB), args.output_dir, pathB)  # 在叠加图上画框

    # 注：rgb_img, grayscale_cam, visualization 的格式、数值范围等不一样，在读取以及处理时要求也不太一样，极易出错，务必仔细！


if __name__ == "__main__":
    main()
