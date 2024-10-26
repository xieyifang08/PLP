from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms
from torchvision.models import *
from PIL import Image
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from fast_adv_imagenet.utils.model_utils import load_model

wrn101 = wide_resnet101_2(pretrained=True)
wrn101_jpeg_ddn = load_model("wide_resnet101_2_dnn_jpeg")
rn152 = resnet152(pretrained=True)
in3 = inception_v3(pretrained=True)
an = alexnet(pretrained=True)
v19 = vgg19(pretrained=True)
# model_dnn = load_model("wide_resnet101_2_dnn_jpeg")
target_layers = [wrn101.layer4[-1]]
img = Image.open("../data/images/0.png")  # 读取图片
target_category = 3

image = transforms.ToTensor()(img).unsqueeze(0)  # PILImage->tensor

cam = GradCAM(model=wrn101, target_layers=target_layers, use_cuda=True)
# cam_dnn = GradCAM(model=model_dnn, target_layers= target_layers, use_cuda=True)

cam_rn152 = GradCAM(model=rn152, target_layers= [rn152.layer4[-1]], use_cuda=True)
# cam_in3 = GradCAM(model=in3, target_layers= [in3.Mixed_7c.branch3x3_2b[-1]], use_cuda=True)
cam_an = GradCAM(model=an, target_layers= [an.features[-1]], use_cuda=True)
cam_v19 = GradCAM(model=v19, target_layers= [v19.features[-1]], use_cuda=True)
cam_wrn101_jpeg_ddn = GradCAM(model=wrn101_jpeg_ddn, target_layers=[wrn101_jpeg_ddn.model.layer4[-1]])


# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
grayscale_cam = cam(input_tensor=image, target_category=target_category)
print(image.shape, type(image))
# grayscale_dnn = cam_dnn(input_tensor=image, target_category=target_category)
grayscale_rn152 = cam_rn152(input_tensor=image, target_category=target_category)
grayscale_v19 = cam_v19(input_tensor=image, target_category=target_category)
grayscale_an = cam_an(input_tensor=image, target_category=target_category)
grayscale_cam_wrn101_jpeg_ddn = cam_wrn101_jpeg_ddn(input_tensor=image, target_category=target_category)


print(grayscale_cam.mean(), grayscale_rn152.mean(), grayscale_v19.mean(), grayscale_an.mean())
np.savetxt('grayscale_an.txt',grayscale_an[0,:]>grayscale_an.mean())
grayscale_cam = grayscale_cam[0, :]
img = np.array(img, dtype=np.float32)
visualization = show_cam_on_image(img/255.0, grayscale_cam, use_rgb=True)

# visualization_dnn = show_cam_on_image(img/255.0, grayscale_dnn[0, :], use_rgb=True)
visualization_rn152 = show_cam_on_image(img/255.0, grayscale_rn152[0, :], use_rgb=True)
visualization_v19 = show_cam_on_image(img/255.0, grayscale_v19[0, :], use_rgb=True)
visualization_an = show_cam_on_image(img/255.0, grayscale_an[0, :], use_rgb=True)
visualization_wrn101_jpeg_ddn = show_cam_on_image(img/255.0, grayscale_cam_wrn101_jpeg_ddn[0,:], use_rgb=True)


plt.subplot(1, 2, 1)
plt.imshow(visualization/255.0)
plt.title("wrn101", y=-0.2)
plt.xticks([])
plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(visualization_wrn101_jpeg_ddn/255.0)
plt.title("wrn101_jpeg_ddn", y=-0.2)

plt.xticks([])
plt.yticks([])

plt.show()
