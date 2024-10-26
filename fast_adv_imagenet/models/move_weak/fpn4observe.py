import torch
import torch.nn as nn


class DFPModule(nn.Module):
    def __init__(self, in_channels, out_channels, padding, dilation):
        super(DFPModule, self).__init__()
        self.asppConv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 11, padding=padding[0], stride=4, dilation=dilation[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.asppConv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 11, padding=padding[1], stride=4, dilation=dilation[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.asppConv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 11, padding=padding[2], stride=4, dilation=dilation[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        # 3x3卷积融合特征
        self.smooth2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)

        # self.se = SELayer(out_channels * 3)

    def forward(self, x):
        x1 = self.asppConv1(x)  # torch.Size([2, 20, 55, 55])
        x2 = self.asppConv2(x)
        x3 = self.asppConv3(x)
        p3 = x3
        p2 = x2 + p3
        p1 = x1 + p2
        p2 = self.smooth2(p2)  # torch.Size([2, 16, 32, 32])
        p1 = self.smooth1(p1)  # torch.Size([2, 16, 32, 32])
        return p1, p2, p3


class Fpn4observe(nn.Module):

    def __init__(self, num_classes: int = 100) -> None:
        super(Fpn4observe, self).__init__()
        self.dfp = DFPModule(3, 64, [2, 22, 32], [1, 5, 7])
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.features1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.features3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Linear(768 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.features1(x)  # torch.Size([2, 256, 27, 27])
        # x = self.avgpool(x)  #  # torch.Size([2, 32, 6, 6])
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)

        p1, p2, p3 = self.dfp(x)  # torch.Size([2, 64, 55, 55])

        x1 = self.features1(p1)  # torch.Size([2, 256, 55, 55])
        x2 = self.features2(p2)  # torch.Size([2, 256, 55, 55])
        x3 = self.features3(p3)  # torch.Size([2, 256, 55, 55])
        x = torch.cat((x1, x2, x3), dim=1)  # torch.Size([2, 768, 55, 55])

        x = self.avgpool(x)  # torch.Size([2, 768, 6, 6])
        # x = self.features4(x)
        x = torch.flatten(x, 1)  # torch.Size([2, 6480])
        x = self.classifier(x)
        return x

# from torchsummary import summary
#
# if __name__ == '__main__':
#     DEVICE = torch.device('cuda:0')
#     # dfp = DFPModule(in_channels=3, out_channels=16, dilation=[1, 3, 7]).to(DEVICE)
#     # summary(dfp, input_size=(3, 32, 32))
#     # print(dfp)
#
#     m = Fpn4observe()
#     summary(m, input_size=(3, 224, 224), device="cpu")
#     print(m)