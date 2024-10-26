
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.stats as st
from torch.autograd import Variable


class GaussianBlurConv(nn.Module):
    def __init__(self, channels=3, kernlen=5, nsig=1.5):
        super(GaussianBlurConv, self).__init__()
        self.channels = channels
        kernel = self.get_kernel(kernlen=kernlen, nsig=nsig)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel = torch.repeat_interleave(kernel, 3, axis=0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def __call__(self, x):

        x = F.conv2d(x, self.weight, padding=2, groups=self.channels)
        return x

    def get_kernel(self, kernlen=16, nsig=3):  # nsig 标准差 ，kernlen=16核尺寸
        interval = (2 * nsig + 1.) / kernlen  # 计算间隔
        x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)  # 在前两者之间均匀产生数据

        # 高斯函数其实就是正态分布的密度函数
        kern1d = np.diff(st.norm.cdf(x))  # 先积分在求导是为啥？得到一个维度上的高斯函数值
        '''st.norm.cdf(x):计算正态分布累计分布函数指定点的函数值
            累计分布函数：概率分布函数的积分'''
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))  # np.outer计算外积，再开平方，从1维高斯参数到2维高斯参数
        kernel = kernel_raw / kernel_raw.sum()  # 确保均值为1
        return kernel

import cv2
if __name__ == '__main__':

    input_x = cv2.imread("../data/images/0.png")
    cv2.imshow("input_x", input_x)
    input_x = Variable(torch.from_numpy(input_x.astype(np.float32))).permute(2, 0, 1)
    gaussian_conv = GaussianBlurConv()
    out_x = gaussian_conv(input_x)
    out_x = out_x.squeeze(0).permute(1, 2, 0).data.numpy().astype(np.uint8)
    cv2.imshow("out_x", out_x)
    cv2.waitKey(0)
