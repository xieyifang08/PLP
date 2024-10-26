# from PIL import Image
# from torchvision import transforms
#
# from fast_adv.models.wsdan.wsdan import WSDAN
# import torch
# from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
# import cv2
# import numpy as np
#
# image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
# image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
#
# DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
# DEVICE= 'cpu'
# print(DEVICE)
#
# def returnCAM(feature_conv, weight_softmax, class_idx):
#     # generate the class activation maps upsample to 256x256
#
#     size_upsample = (32, 32)
#     bz, nc, h, w = feature_conv.shape
#     output_cam = []
#     output_cam2 = []
#     for idx in class_idx:
#         one=np.ones_like(weight_softmax[idx])
#
#         cam2 = one.dot(feature_conv.reshape((nc, h * w)))
#         cam2 = cam2.reshape(h, w)
#         cam_img2 = (cam2 - cam2.min()) / (cam2.max() - cam2.min())  # normalize
#         cam_img2 = np.uint8(255 * cam_img2)
#         output_cam2.append(cv2.resize(cam_img2, size_upsample))
#
#         cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
#         cam = cam.reshape(h, w)
#
#         cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
#         cam_img = np.uint8(255 * cam_img)
#         output_cam.append(cv2.resize(cam_img, size_upsample))
#     return output_cam,output_cam2
#
#
# m = WSDAN(num_classes=10, M=32, net='wide_resnet', pretrained=True)
# model =NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE).eval()  # keep images in the [0, 1] range
# model_file = './weights/best/2AT_cifar10_ep_13_val_acc0.8770.pth'
# weight_wsdan3 = './weights/cifar10_wsgan_3/cifar10_valacc0.pth'
# weight_wsdan_final= './weights/cifar10_wsdan_final/cifar10_valacc0.871999979019165.pth'
# weight_wsdan_best = './weights/cifar10_WSDAN_best/cifar10_valacc0.8784999758005142.pth'
# weight_wsdan_best = "../defenses/weights/cifar10_WSDAN_best/cifar10_0.87_low.pth"
# model_dict = torch.load(weight_wsdan_best)
# model.load_state_dict(model_dict)
#
# img = Image.open("../data/cifar10/org/3/org_25.png")
# img = Image.open("../data/cifar10/org/0/org_309.png")
# X = transforms.ToTensor()(img).unsqueeze(0)
# X= X.to(DEVICE)
# print(X.shape)
#
# y_pred_raw, feature_matrix, attention_map = model(X)
# print(feature_matrix.shape, attention_map.shape)
# attention_map = attention_map.detach().numpy()
# params = list(model.parameters())
# # model_dic = model.state_dict()
# # print(model.state_dict())
# # for k in model_dic.keys():
# #     print(k, type(model_dic[k]), model_dic[k].shape)
# weight_softmax = np.squeeze(params[80].data.detach().numpy())
# print(weight_softmax.shape)
# CAMs, Feature = returnCAM(attention_map, weight_softmax, [1])
# # render the CAM and output
# clean_path = '../data/cifar10/org/3/org_25.png'
# clean_path = '../data/cifar10/org/7/org_1205.png'
# clean_path = '../data/cifar10/jpeg/test/1/9_44.png'
# # clean_path ='../data/cifar10/org/8/org_1146.png'
#
# img = cv2.imread(clean_path)
# # #img=cv2.imread(img_path)
# height, width, _ = img.shape
# # print("img.shape", img.shape)
#
# heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
# heatmap2 = cv2.applyColorMap(cv2.resize(Feature[0], (width, height)), cv2.COLORMAP_JET)
# # print("s",heatmap, "\n\n\n\n\n\n",img)
# result = heatmap * 0.8 + img * 0.5
# result2 = heatmap2 * 0.8 + img * 0.5
# # name = 'CAM' + str(weight) + str(target_class) + '.png'
# print("heatmapheatmapheatmap", type(heatmap), heatmap.shape)
# CAM_PATH = './0_2_CAM.jpg'
# Feature_PATH = './0_2_Feature.jpg'
# # path='CAM' + str(weight) + str(target_class) + '.png'
#
# cv2.imwrite(CAM_PATH, result)
# cv2.imwrite(Feature_PATH, result2)


import torch.nn as nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # 卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(  # 图片的维度：(1, 28, 28)
                in_channels=1,  # 图片的高度
                out_channels=16,  # 输出的高度：filter的个数
                kernel_size=5,  # filter的像素点是5×5
                stride=1,  # 每次扫描跳的范围
                padding=2  # 补全边缘像素点
            ),  # 图片的维度：(16，28，28）
            nn.ReLU(),  # 图片的维度：(16,28,28)
            nn.MaxPool2d(kernel_size=2, ),  # 图片的维度：(16,14,14)
        )
        # 卷积层
        self.conv2 = nn.Sequential(  # 图片的维度：(16,14,14)
            nn.Conv2d(16, 32, 5, 1, 2),  # 图片的维度：(32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2)  # 图片的维度：（32，7，7）
        )
        # 线性层
        self.out = nn.Linear(32 * 7 * 7, 10)

    # 展平操作
    def forward(self, x):
        print(x.size())  # 查看模型的输入，tensorboardX input_to_model
        x = self.conv1(x)
        print(x.size())
        x = self.conv2(x)  # 图片的维度：（batch,32,7,7）
        print(x.size())
        # 展平操作, -1表示自适应
        x = x.view(x.size(0), -1)  # 图片的维度：（batch,32*7*7）
        print(x.size())
        output = self.out(x)
        return output


cnn = CNN()
print(cnn)

for name, parameter in cnn.named_parameters():
    print(name, ":", parameter.size())

params = list(cnn.parameters())
print(len(params))

model_dic = cnn.state_dict()
print(len(model_dic))
i = 0
for k in model_dic.keys():
    i+=1
    print(i, k, model_dic[k].shape)