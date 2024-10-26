"""PyTorch implementation of Wide-ResNet taken from https://github.com/xternalz/WideResNet-pytorch"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        #self.fc2 = nn.Linear(2*nChannels[3], num_classes)
        self.nChannels = nChannels[3]
        self.se = SELayer()
        self.attention_spatial = AttentionModule_stage2_cifar(nChannels[3], nChannels[3])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                #print(m)
                m.bias.data.zero_()

    #######输出图像特征
    def forward_feature(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        #print(out.size()),64,640,8,8
        #增加了一个空间attention
        out=self.attention_spatial(out)

        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return out#self.fc(out)

    ######根据单图像特征，直接分类，用来生成对抗样本
    def forward(self,x):
        out= self.forward_feature(x)
        out = F.normalize(out, p=2, dim=1)
        #print(features.size())
        #print('class#######')
        return self.fc(out)#,out

    #####组合多图像特征，学习权重，再分类，作为loss，和预测
    def forward_attention(self,x,y):

        feature1 = self.forward_feature(x)
        feature2 = self.forward_feature(y)
        b, c = feature1.size()
        # 32*640
        out = torch.zeros_like(feature1)

        #print('合并方案，保持特征数不变')

        for i in range(b):
            #1*2*640*1,  b,c,w,h
            c = torch.cat((feature1[i].unsqueeze(0), feature2[i].unsqueeze(0)), 0).unsqueeze(0).unsqueeze(-1)
            c = self.se(c)
            #2*640
            c=torch.squeeze(c)
            # 640
            out[i] = torch.sum(c, 0, False)  # torch.sum(input, dim, keepdim=False, out=None) → Tensor
        #按照某个维度计算范数,默认根据通道,
        #目的，使adv中单张图与多图累加的特征都是同一个标准大小的特征
        out = F.normalize(out, p=2, dim=1)
        #out = self.fc(out)

        return self.fc(out)#,out

    def feature_map(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        # print(out.size()),64,640,8,8
        # 增加了一个空间attention
        #out = self.attention_spatial(out)
        return out#self.fc(out)
    def feature_map2(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        # print(out.size()),64,640,8,8
        # 增加了一个空间attention
        out = self.attention_spatial(out)
        return out#self.fc(out)








def wide_resnet(**kwargs):
    """
    Constructs a Wide Residual Networks.
    """
    model = WideResNet(**kwargs)
    return model

class SELayer(nn.Module):
    def __init__(self, channel=2, reduction=1):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c,w,_ = x.size()
        #print('x的size：se::',x.size())
        #按理来说，这里pool成3个数，然后3个数----fc（3，3），relu，fc（3，3），sigmoid-----得到3个权重*3通道特征-------fc(3*原特征,class) or pool--fc(原特征数，class)

        #input维度： （batch_size,channels,height,width）
        y = self.avg_pool(x).view(b, c)
        #print('y::',y.size())
        y = self.fc(y).view(b, c, 1, 1)
        #print(y)

        return x * y.expand_as(x)

'''
class Attention_integration(nn.Module):
    def __init__(self,num_classes):
        super(Attention_integration,self).__init__()
        self.num_classes=num_classes
        self.se = SELayer()

        self.basemodel = WideResNet(num_classes=self.num_classes, depth=28, widen_factor=10, dropRate=0.0)


        self.fc = nn.Linear(64, num_classes)

    def forward(self, x,y):
        batch, channels, height, width = x.size()
        feature1=self.basemodel(x)
        feature2 = self.basemodel(y)
        #feature3 = self.basemodel(z)

        c=torch.cat((feature1,feature2),0)
        print('x:',x.size(), y.size(),'c',c.size())
        c=self.se(c)
        print('c2',c.size())
        out = F.avg_pool2d(c, 8)
        print('pool',out.size())
        out = out.view(-1, self.nChannels)
        return self.fc(out)
'''
class AttentionModule_stage2_cifar(nn.Module):
    # input size is 8*8
    def __init__(self, in_channels, out_channels, size=(8, 8)):
        super(AttentionModule_stage2_cifar, self).__init__()

        #给所有通道共同的空间attention图
        self.point_conv=nn.Sequential(
            nn.Conv2d(in_channels,1,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        self.first_residual_blocks = BasicBlock(in_channels, out_channels)

        self.mpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 4*4

        self.middle_2r_blocks = nn.Sequential(
            BasicBlock(in_channels, out_channels),
            BasicBlock(in_channels, out_channels)
        )
        self.one_2r_blocks = nn.Sequential(
            BasicBlock(1, 1),
            BasicBlock(1, 1)
        )

        self.interpolation1 = nn.UpsamplingBilinear2d(size=size)  # 8*8

        self.conv1_1_blocks = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, bias = False),
            nn.Sigmoid()
        )

        self.last_blocks = BasicBlock(in_channels, out_channels)
        self.bn1 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        #x = self.first_residual_blocks(x)
        y=self.point_conv(x)#变成1通道

        out_trunk =x#self.trunk_branches(x)
        out_mpool1 = self.mpool1(y)
        #out_middle_2r_blocks = self.middle_2r_blocks(out_mpool1)
        out_middle_2r_blocks = self.one_2r_blocks(out_mpool1)
        out_interp = self.interpolation1(out_middle_2r_blocks)# + out_trunk
        #out_conv1_1_blocks = self.conv1_1_blocks(out_interp)
        #out = (1 + out_conv1_1_blocks) * out_trunk
        out_interp=out_interp.expand_as(x)
        #print(out_interp.size())
        out=(1+out_interp)*out_trunk

        out=self.bn1(out)#归一化一下

        #out_last = self.last_blocks(out)
        return out

