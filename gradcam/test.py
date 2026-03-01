from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet101
import torchvision
import torch
from matplotlib import pyplot as plt
import numpy as np
import os
import cv2

# #如果要用dalle-mini,须额外使用以下代码
# #直接下载wandb中的文件夹及图片，剩重命名问题待解决
# import wandb
# run = wandb.init()
# artifact = run.use_artifact('ranpin/dalle-mini-tables-colab/run-0epdru0b-GeneratedImages:v0', type='run_table')
# artifact_dir = artifact.download()
# gen_table.add_data(prompt, *[wandb.Image(img,caption=prompt) for img in tmp_imgs])

#或者尝试导出csv文件，读取处理
# data = np.loadtxt(open("wandb_export_2023-02-09T17_40_02.757+08_00.csv","rb"),dtype=str,delimiter=",",skiprows=1) 
# data = data[:,1:]
# print(data)
# print(type(data))
# print(data[0][0])
# data = np.asarray_chkfinite
# cv2.imshow('hhh',img)


def myimshows(imgs, titles=False, fname=" ", size=6):
    print(fname)
    lens = len(imgs)

    # #存热力图
    plt.figure(figsize=(size,size))    
    if titles == False:
        titles="0123456789"
    for i in range(1, lens + 1):
        plt.xticks(())
        plt.yticks(())
        plt.subplot(111)
        if len(imgs[i - 1].shape) == 2:
            plt.imshow(imgs[i - 1], cmap=plt.cm.jet)
            plt.title(titles[i - 1])
    plt.xticks(())
    plt.yticks(())
    # plt.savefig(os.path.join('results48_17\\',fname), bbox_inches='tight')
    plt.savefig(os.path.join('results65_15\\',fname), bbox_inches='tight')
    # plt.show()
    

    #存总体对比图
    plt.figure(figsize=(size* lens,size))    
    if titles == False:
        titles="0123456789"
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
    fname = fname+'1'
    # plt.savefig(os.path.join('results48_17\\',fname), bbox_inches='tight')
    plt.savefig(os.path.join('results65_15\\',fname), bbox_inches='tight')
    # plt.show()

def drawBounding(img,img1,root=" ",file=" "):
    img = cv2.applyColorMap(np.uint8(255 * img), colormap=2)
    ret,thresh = cv2.threshold(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),170,255,0)#修改阈值可以微调包围框
    _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)#得到轮廓信息
    cnt=contours[0]
    x,y,w,h=cv2.boundingRect(cnt)
    #Straight Bounding Rectangle
    # imgnew = cv2.rectangle(np.uint8(255 * img1),(x,y),(x+w,y+h),(0,255,0),2)#np.uint8()处理仅在img1为原图时需要
    #Rotate Rectangle
    rect=cv2.minAreaRect(cnt)
    box=cv2.boxPoints(rect)
    box=np.int64(box)
    imgnew=cv2.drawContours(np.uint8(255 * img1),[box],0,(0,255,0),2)

    # imgnew = cv2.cvtColor(imgnew, cv2.COLOR_BGR2RGB)
    # cv2.imwrite("results48_17\\"+fname,imgnew)
    print("当前处理图片为："+root+"\\results\\"+file)
    path = root+"\\results\\"
    if not os.path.exists(path):
        os.makedirs(path)
    cv2.imwrite(path+file,imgnew)

def tensor2img(tensor,heatmap=False,shape=(224,224)):
    np_arr=tensor.detach().numpy()#[0]
    #对数据进行归一化
    if np_arr.max()>1 or np_arr.min()<0:
        np_arr=np_arr-np_arr.min()
        np_arr=np_arr/np_arr.max()
    # np_arr=(np_arr*255).astype(np.uint8)
    if np_arr.shape[0]==1:
        np_arr=np.concatenate([np_arr,np_arr,np_arr],axis=0)
    np_arr=np_arr.transpose((1,2,0))
    return np_arr

path = "outputs"
for root, dirs, files in os.walk(path):
    # print(f"当前目录：{root}")
    for file in files:
        # print(f"文件：{file}")
        bin_data=torchvision.io.read_file(os.path.join(root,file))#加载二进制数据
        img=torchvision.io.decode_image(bin_data)/255#解码成CHW的图片
        img=img.unsqueeze(0)#变成BCHW的数据，B==1; squeeze
        input_tensor=torchvision.transforms.functional.resize(img,[224, 224])
    
        #对图像进行水平翻转，得到两个数据
        # input_tensor=torch.cat([input_tensor, input_tensor.flip(dims=(3,))],axis=0)
        
        model = resnet101(pretrained=True)
        target_layers = [model.layer4[-1]]#如果传入多个layer，cam输出结果将会取均值
        
        cam = GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False)
        with GradCAMPlusPlus(model=model, target_layers=target_layers, use_cuda=False) as cam:
            # targets = [ClassifierOutputTarget(38),ClassifierOutputTarget(38)] #指定查看class_num为38的热力图
            # targets = [ClassifierOutputTarget(900)] #指定查看class_num为38的热力图
            # aug_smooth=True, eigen_smooth=True 使用图像增强是热力图变得更加平滑
            grayscale_cams = cam(input_tensor=input_tensor,targets=None)#targets=None 自动调用概率最大的类别显示
            for grayscale_cam,tensor in zip(grayscale_cams,input_tensor):
                #将热力图结果与原图进行融合
                rgb_img=tensor2img(tensor)

                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                # 绘制热力图和对比图
                # myimshows([rgb_img, grayscale_cam, visualization],["image","gradcam++","overlap"],fname=pathB.split(".")[0]+'_cam')

                drawBounding(grayscale_cam,cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR),os.path.dirname(root),file)#在原图上画框
                # drawBounding(grayscale_cam,cv2.cvtColor(visualization, cv2.COLOR_BGR2RGB),pathB)  #在叠加图上画框
                # heatmap = cv2.applyColorMap(np.uint8(255 * grayscale_cam), colormap=2)
                # heatmap = drawBounding(grayscale_cam)
                # grayscale_cam = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                # # grayscale_cam = np.float32(grayscale_cam)
                # img = Image.fromarray(grayscale_cam)
                # img.show()

# 注：rgb_img, grayscale_cam, visualization的格式、数值范围等不一样，在读取以及处理时要求也不太一样，极易出错，务必仔细！       

    # for directory in dirs:
    #     print(f"子目录：{directory}")
