from torch import nn
from fast_adv_imagenet.models.non_local import NONLocalBlock2D
from fast_adv_imagenet.models.non_local.wide_resnet import wide_resnet
from fast_adv_imagenet.utils.model_utils import load_model

class NonLocalNetwork(nn.Module):
    def __init__(self, backbone="resnet"):
        super(NonLocalNetwork, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.nl_1 = NONLocalBlock2D(in_channels=32)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )

        self.nl_2 = NONLocalBlock2D(in_channels=64)
        self.conv_3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=128*3*3, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(in_features=256, out_features=10)
        )
        self.backbone = load_model(backbone)

    def forward(self, x):

        feature_1 = self.conv_1(x)
        nl_feature_1 = self.nl_1(feature_1)

        feature_2 = self.conv_2(nl_feature_1)
        nl_feature_2 = self.nl_2(feature_2)
        output = self.conv_3(nl_feature_2)
        # print("output:", output.shape)
        # output = output.view(batch_size, -1)
        #
        # output = self.fc(output)
        output = self.backbone(output)

        return output