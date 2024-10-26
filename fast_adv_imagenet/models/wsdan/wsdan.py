import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import fast_adv.models.wsdan.vgg as vgg
import fast_adv.models.wsdan.resnet as resnet
from fast_adv.models.wsdan.wide_resnet import wide_resnet
from fast_adv.models.wsdan.inception import inception_v3, BasicConv2d

__all__ = ['WSDAN']
EPSILON = 1e-12


# Bilinear Attention Pooling
class BAP(nn.Module):
    def __init__(self, pool='GAP'):
        super(BAP, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # print("features.size: ",features.size()," attentions.size: ", attentions.size())
        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # feature_matrix: (B, M, C) -> (B, M * C)
        if self.pool is None:
            feature_matrix = (torch.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            feature_matrix = torch.cat(feature_matrix, dim=1)

        # sign-sqrt
        feature_matrix = torch.sign(feature_matrix) * torch.sqrt(torch.abs(feature_matrix) + EPSILON)

        # l2 normalization along dimension M and C
        feature_matrix = F.normalize(feature_matrix, dim=-1)
        return feature_matrix


# WS-DAN: Weakly Supervised Data Augmentation Network for FGVC
class WSDAN(nn.Module):
    def __init__(self, num_classes, M=32, net='inception_mixed_6e', pretrained=False):
        super(WSDAN, self).__init__()
        self.num_classes = num_classes
        self.M = M
        self.net = net

        # Network Initialization
        if 'inception' in net:
            if net == 'inception_mixed_6e':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_6e()
                self.num_features = 768
            elif net == 'inception_mixed_7c':
                self.features = inception_v3(pretrained=pretrained).get_features_mixed_7c()
                self.num_features = 2048
            else:
                raise ValueError('Unsupported net: %s' % net)
        elif 'vgg' in net:
            # print("getattr(vgg, net) : ", getattr(vgg, net))
            self.features = getattr(vgg, net)(pretrained=pretrained).get_features()
            self.num_features = 512
        elif 'wide_resnet' in net:
            # print("wdr")
            m = wide_resnet(num_classes=10, depth=28, widen_factor=10, dropRate=0.03)
            # 在
           # self.backbone = m
            self.features = m.get_features()
            self.num_features = 640
        elif 'resnet' in net:

            self.features = getattr(resnet, net)(pretrained=pretrained).get_features()
            self.num_features = 512 * self.features[-1][-1].expansion

        else:
            raise ValueError('Unsupported net: %s' % net)
        # Attention Maps
        self.attentions = BasicConv2d(self.num_features, self.M, kernel_size=1)

        # Bilinear Attention Pooling
        self.bap = BAP(pool='GAP')

        # Classification Layer
        self.fc = nn.Linear(self.M * self.num_features, self.num_classes, bias=False)

        logging.info('WSDAN: using {} as feature extractor, num_classes: {}, num_attentions: {}'.format(net, self.num_classes, self.M))

    def forward(self, x):
        batch_size = x.size(0)

        # Feature Maps, Attention Maps and Feature Matrix
        feature_maps = self.features(x)
        # print("forward  x: ", x.shape, "feature_maps: ", feature_maps.shape)
        if self.net != 'inception_mixed_7c':
            attention_maps = self.attentions(feature_maps)
        else:
            attention_maps = feature_maps[:, :self.M, ...]
        feature_matrix = self.bap(feature_maps, attention_maps)

        # Classification
        p = self.fc(feature_matrix * 100.)

        # Generate Attention Map
        # if self.training:
        #     # Randomly choose one of attention maps Ak
        #     attention_map = []
        #     for i in range(batch_size):
        #         attention_weights = torch.sqrt(attention_maps[i].sum(dim=(1, 2)).detach() + EPSILON)
        #         attention_weights = F.normalize(attention_weights, p=1, dim=0)
        #         k_index = np.random.choice(self.M, 2, p=attention_weights.cpu().numpy())
        #         attention_map.append(attention_maps[i, k_index, ...])
        #     attention_map = torch.stack(attention_map)  # (B, 2, H, W) - one for cropping, the other for dropping
        # else:
        #     # Object Localization Am = mean(Ak)
        #     attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        attention_map = torch.mean(attention_maps, dim=1, keepdim=True)  # (B, 1, H, W)

        # p: (B, self.num_classes)
        # feature_matrix: (B, M * C)
        # attention_map: (B, 2, H, W) in training, (B, 1, H, W) in val/testing
        return p, feature_matrix, attention_map

    def forward_attention(self,x,y):
        return self.backbone.forward_attention(x,y)

    def feature_map(self, x):
        return self.features(x)


    def load_state_dict(self, state_dict, strict=True):
        model_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items()
                           if k in model_dict and model_dict[k].size() == v.size()}

        if len(pretrained_dict) == len(state_dict):
            logging.info('%s: All params loaded' % type(self).__name__)
        else:
            logging.info('%s: Some params were not loaded:' % type(self).__name__)
            not_loaded_keys = [k for k in state_dict.keys() if k not in pretrained_dict.keys()]
            logging.info(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

        model_dict.update(pretrained_dict)
        super(WSDAN, self).load_state_dict(model_dict)

from torchsummary import summary  # 使用 pip install torchsummary
if __name__ == '__main__':
    DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
    m = WSDAN(num_classes=10, M=32, net='wide_resnet', pretrained=True).to(DEVICE)
    summary(m, input_size=(3, 64, 64))
