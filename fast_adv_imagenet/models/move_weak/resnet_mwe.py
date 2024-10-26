import logging
import time



import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision.transforms import Resize

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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
        b, c, w, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class DFPModule(nn.Module):
    def __init__(self, in_channels, out_channels, padding, dilation):
        super(DFPModule, self).__init__()
        self.asppConv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 3, 3, padding=padding[0], stride=2, dilation=dilation[0],
                      bias=False),
            nn.BatchNorm2d(out_channels // 3),
            nn.ReLU()
        )
        self.asppConv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 3, 3, padding=padding[1], stride=2, dilation=dilation[1],
                      bias=False),
            nn.BatchNorm2d(out_channels // 3),
            nn.ReLU()
        )
        self.asppConv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 3, 3, padding=padding[2], stride=2, dilation=dilation[2],
                      bias=False),
            nn.BatchNorm2d(out_channels // 3),
            nn.ReLU()
        )

        # 3x3卷积融合特征
        self.smooth2 = nn.Conv2d(out_channels // 3, out_channels // 3, 3, 1, 1)
        self.smooth1 = nn.Conv2d(out_channels // 3, out_channels // 3 + out_channels % 3, 3, 1, 1)

        self.se = SELayer(out_channels)

    def forward(self, x):
        x1 = self.asppConv1(x)  # torch.Size([2, 21, 112, 112])
        x2 = self.asppConv2(x)  # torch.Size([2, 21, 112, 112])
        x3 = self.asppConv3(x)  # torch.Size([2, 21, 112, 112])
        p3 = x3
        p2 = x2 + p3
        p1 = x1 + p2
        p2 = self.smooth2(p2)  # torch.Size([2, 21, 112, 112])
        p1 = self.smooth1(p1)  # torch.Size([2, 22, 112, 112])
        p_out = torch.cat((p1, p2, p3), dim=1)  # torch.Size([2, 64, 112, 112])
        p_out = self.se(p_out)  # torch.Size([2, 64, 112, 112])
        return p_out


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            downsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 64,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            window: int = 3,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        # self.dfp = DFPModule(3, 20, [3, 15, 21], [1, 5, 7])
        self.dfp = DFPModule(3, 64, [1, 5, 7], [1, 5, 7])
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.window = window
        self.query_conv = nn.Conv2d(in_channels=512 * 4, out_channels=512 * 4 // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=512 * 4, out_channels=512 * 4 // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=512 * 4, out_channels=512 * 4, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def drawCam(self, features_blobs, pa, pre_class):
        params = list(self.named_parameters())
        i = 0
        for name, par in self.named_parameters():
            print(i, name, par.shape)
            i+=1

        weight_softmax = params[pa][1].data
        weight_softmax = torch.mean(weight_softmax, dim=-1)
        weight_softmax = torch.mean(weight_softmax, dim=-1)

        CAMs = returnCAM(features_blobs, weight_softmax, pre_class)
        return CAMs

    def forward_non_local(self, x):
        m_batchsize, C, imgH, imgW = x.size()
        # 前期直接用输入x计算关注度矩阵
        proj_query = self.query_conv(x).view(m_batchsize, -1, imgW * imgH).permute(0, 2, 1)
        assert torch.isnan(proj_query).sum() == 0 and torch.isinf(proj_query).sum() == 0, ('output of query_conv layer is nan or infinit', proj_query.std())  # out 是你本层的输出 out.std()输出标准差

        proj_key = self.key_conv(x).view(m_batchsize, -1, imgW * imgH)
        assert torch.isnan(proj_key).sum() == 0 and torch.isinf(proj_key).sum() == 0, ('output of key_conv layer is nan or infinit', proj_key.std())  # out 是你本层的输出 out.std()输出标准差

        energy = torch.bmm(proj_query, proj_key)
        assert torch.isnan(energy).sum() == 0 and torch.isinf(energy).sum() == 0, ('output of bmm layer is nan or infinit', energy.std())  # out 是你本层的输出 out.std()输出标准差

        attention = self.softmax(energy)
        assert torch.isnan(attention).sum() == 0 and torch.isinf(attention).sum() == 0, ('output of self_attention softmax layer is nan or infinit', attention.std())  # out 是你本层的输出 out.std()输出标准差

        proj_value = self.value_conv(x).view(m_batchsize, -1, imgW * imgH)  # 输入图像

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        assert torch.isnan(out).sum() == 0 and torch.isinf(out).sum() == 0, ('output of bmm layer is nan or infinit', out.std())  # out 是你本层的输出 out.std()输出标准差

        out = out.view(m_batchsize, C, imgW, imgH)

        return self.gamma * out + x

    def forward_non_local_with_label(self, x, labels):
        cams = self.drawCam(x, 326, labels)  # list(batch) tensor(1,8,8)
        crops, crop_idxs = attention_crop(cams, x, self.window)
        crops = torch.stack(crops, dim=0)
        m_batchsize, C, imgH, imgW = x.size()
        _, _, cropH, cropW = crops.size()
        # 用attention crops 计算关注度矩阵
        proj_query = self.query_conv(crops).view(m_batchsize, -1, cropW * cropH).permute(0, 2, 1)
        proj_key = self.key_conv(crops).view(m_batchsize, -1, cropW * cropH)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        proj_value = self.value_conv(crops).view(m_batchsize, -1, cropW * cropH)  # 输入图像
        crops_out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        crops_out = self.gamma * crops_out.view(m_batchsize, C, cropW, cropH)

        out = []
        for batch in range(m_batchsize):
            left, up = crop_idxs[batch]
            right, down = imgW - self.window - left, imgH - self.window - up
            out.append(F.pad(crops_out[batch], [left, right, up, down]))
        out = torch.stack(out, dim=0)
        return out + x

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.dfp(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.forward_non_local(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def forward_with_label(self, x, labels):
        x = self.dfp(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.forward_non_local_with_label(x, labels)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def returnCAM(feature_conv, weight_softmax, class_idx):
    bzs, nc, h, w = feature_conv.shape
    output_cam = []
    for bz in range(bzs):
        fe = feature_conv[bz].reshape((nc, h * w))
        we = torch.unsqueeze(weight_softmax[class_idx[bz]], 0)
        cam = torch.mm(we, fe)
        cam = cam.reshape(h, w)
        cam_img = (cam - cam.min()) / (cam.max() - cam.min())  # normalize
        cam_img = torch.unsqueeze(cam_img, 0)
        output_cam.append(Resize([h, w])(cam_img))
    return output_cam

"""
二维滑动窗口最大值，算法待优化
"""


def attention_crop(cams, imgs, size):
    batchs, _, imgH, imgW = imgs.shape
    crops, crop_idxs = [], []
    for batch in range(batchs):
        cam = cams[batch]
        maxSum = 0
        idx, idy = 0, 0
        for x in range(imgH - size):
            for y in range(imgW - size):
                sum = torch.sum(cam[:, x:x + size, y:y + size])
                if sum > maxSum:
                    maxSum = sum
                    idx, idy = x, y
        crops.append(imgs[batch, :, idx:idx + size, idy:idy + size])
        crop_idxs.append([idx, idy])
    return crops, crop_idxs

def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool,
        progress: bool,
        **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    print("pretrained = ", pretrained)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_.

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)

from torchsummary import summary

if __name__ == '__main__':
    DEVICE = torch.device('cuda:0')
    m = wide_resnet101_2(num_classes=100).to(DEVICE)
    i = 0
    for name, param in m.named_parameters():
        print(i, name, param.shape)
        i += 1
    summary(m, input_size=(3, 32, 32))
    print(m)