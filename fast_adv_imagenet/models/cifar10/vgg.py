import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

'''
创建VGG块
参数分别为输入通道数，输出通道数，卷积层个数，是否做最大池化
'''
def make_vgg_block(in_channel, out_channel, convs, pool=True):
    net = []

    # 不改变图片尺寸卷积
    net.append(nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1))
    net.append(nn.BatchNorm2d(out_channel))
    net.append(nn.ReLU(inplace=True))

    for i in range(convs - 1):
        # 不改变图片尺寸卷积
        net.append(nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1))
        net.append(nn.BatchNorm2d(out_channel))
        net.append(nn.ReLU(inplace=True))

    if pool:
        # 2*2最大池化，图片变为w/2 * h/2
        net.append(nn.MaxPool2d(2))

    return nn.Sequential(*net)


# 定义网络模型
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()

        net = []

        # 输入32*32，输出16*16
        net.append(make_vgg_block(3, 64, 2))

        # 输出8*8
        net.append(make_vgg_block(64, 128, 2))

        # 输出4*4
        net.append(make_vgg_block(128, 256, 4))

        # 输出2*2
        net.append(make_vgg_block(256, 512, 4))

        # 无池化层，输出保持2*2
        net.append(make_vgg_block(512, 512, 4, False))

        self.cnn = nn.Sequential(*net)

        self.fc = nn.Sequential(
            # 512个feature，每个feature 2*2
            nn.Linear(512*2*2, 256),
            nn.ReLU(),

            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.cnn(x)

        # x.size()[0]: batch size
        x = x.view(x.size()[0], -1)
        x = self.fc(x)

        return x