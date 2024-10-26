import glob
import os
import argparse

import scipy
import tqdm
import numpy as np
from copy import deepcopy
from fast_adv.attacks import DDN
import torch
from torch.utils import data

from torchvision import transforms
from torchvision.datasets import CIFAR10
from fast_adv.attacks.attacks import Attacker, PGD_L2, DDN
# from fast_adv.models.cifar10 import wide_resnet
from fast_adv.models.cifar10.wide_resnet import wide_resnet
# from fast_adv.models.cifar10.model_mixed_attention import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks.shape_precess import shape
import foolbox
import warnings

warnings.filterwarnings("ignore")

import logging
logging.basicConfig(level = logging.INFO,format = '%(levelname)s - %(message)s')
loggger = logging.getLogger()

from foolbox.adversarial import Adversarial
from foolbox.criteria import Misclassification
from foolbox.distances import MeanAbsoluteDistance, Linfinity
from foolbox.attacks import FGSM, DeepFoolL2Attack, PGD, LocalSearchAttack, GaussianBlurAttack, \
    BinarizationRefinementAttack, ContrastReductionAttack, SaltAndPepperNoiseAttack, \
    SpatialAttack, CarliniWagnerL2Attack

parser = argparse.ArgumentParser(description='Extend sample')
parser.add_argument('--max', type=float, default=2, help='max norm for the adversarial perturbations')
parser.add_argument('--imgsize', default=32, type=int, help='pic size')
parser.add_argument('--attack_name', '--at', default='DDN',
                    help='name for saving the final state dict')
parser.add_argument('--batch-size', '-b', default=16, type=int, help='mini-batch size')
parser.add_argument("--shape", type=int, default=None)
parser.add_argument("--ddn", type=int, default=False)
parser.add_argument("--PGD", type=int, default=True)
# PGD-specific
parser.add_argument('--random-start', default=True, type=bool)

# DDN-specific
parser.add_argument('--init-norm-DDN', default=1, type=float)
parser.add_argument('--gamma-DDN', default=0.05, type=float)
# parser.add_argument('--max-norm', type=float,default=1, help='max norm for the adversarial perturbations')

args = parser.parse_args()
print(args)
input = '../defenses/data/cifar10'
path_PGD = '../data/cifar10/grey/2PGD'
path_DDN = '../data/cifar10/grey/DDN'
path2_PGD = '../data/cifar10/white/2.55_AT_PGD'
path2_DDN = '../data/cifar10/white/0.5_AT_DDN'
path3_PGD = '../data/cifar10/white/0.5_MIX_PGD'
path3_DDN = '../data/cifar10/white/0.5_MIX_DDN'

image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
test_transform = transforms.Compose([
    transforms.ToTensor(),
])
# train_set = data.Subset(CIFAR10(input, train=True, transform=test_transform, download=True),list(range(0,30000)))
val_set = data.Subset(CIFAR10(input, train=True, transform=test_transform, download=True),
                      list(range(48000, 50000)))  # 91.9
# val_set = data.Subset(CIFAR10(input, train=False, transform=test_transform, download=True), list(range(0, 9999)))#91.9

val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
m = wide_resnet(num_classes=10, depth=28, widen_factor=10, dropRate=0.3)
model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
weight_norm = '../defenses/weights/best/2Norm_cifar10_ep_184_val_acc0.9515.pth'
weight_AT = '../defenses/weights/cifar10_AT/cifar10acc0.8709999859333039_45.pth'
weight_conv_mixatten = '../defenses/weights/cifar10_mixed_Attention/cifar10acc0.8709999829530716_20.pth'
weight_attention = '../defenses/weights/cifar10_Attention/cifar10acc0.8729999780654907_120.pth'
weight_smooth = '../defenses/weights/best/2random_smooth_cifar10_ep_120_val_acc0.8510.pth'
weight_025conv_mixatten = '../defenses/weights/shape_0.5_cifar10_mixed_Attention/cifar10acc0.8624999821186066_70.pth'

weight_cifar10_base = "/home/frankfeng/projects/researchData/AI_security/code/PLP/fast_adv/defenses/weights/cifar10_base/cifar10_valacc0.8339999794960022.pth"

model_dict = torch.load(weight_cifar10_base)
model.load_state_dict(model_dict)

for i, (images, labels) in enumerate(tqdm.tqdm(val_loader, ncols=80)):
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    # print(images.size())

    # attacker = DDN(max_norm=1,steps=100, device=DEVICE)
    # attacker2 = DeepFool(device=DEVICE)
    # ddn = attacker.attack(model, images, labels=labels, targeted=False)
    # deepfool= attacker2.attack(model, images, labels=labels, targeted=False)
    if args.shape is True:
        # print('23')
        image_shape = torch.zeros_like(images)
        for t in range(args.batch_size):
            # print('2323')
            new = images[t]
            new = new.cpu()
            new = new.numpy()  # * 255
            new = shape(new)

            # new = new.transpose((2, 0, 1))
            new = torch.from_numpy(new)

            new_shape = torch.renorm(new - images[t], p=2, dim=0, maxnorm=1) + images[t]

            image_shape = np.transpose(new_shape.cpu().numpy(), (1, 2, 0))
            # print('2323')
            name = '/shape_' + str(i) + str(t) + '.png'
            out_path = os.path.join(path_DDN, str(labels[t].cpu().numpy()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = out_path + name

            scipy.misc.imsave(out, image_shape)

            print('2')

    if args.ddn is True:
        attacker = DDN(steps=10, device=DEVICE, max_norm=args.max,
                       init_norm=args.init_norm_DDN, gamma=args.gamma_DDN)
        # ddn = attacker.attack(model, images, labels)
        ddn = attacker.attack(model, inputs=images, labels=labels)
        # ddn = DDN.attack(model, images, labels)

        for t in range(args.batch_size):
            ddn2 = np.transpose(ddn[t].detach().cpu().numpy(), (1, 2, 0))
            # deepfool2 = np.transpose(deepfool[t].cpu().numpy(), (1, 2, 0))
            name = '/ddn_' + str(i) + str(t) + '.png'
            out_path = os.path.join(path2_DDN, str(labels[t].cpu().numpy()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            # print(out_path)
            out = out_path + name
            scipy.misc.imsave(out, ddn2)

        # continue
    if args.PGD is True:
        attacker = PGD_L2(steps=100, device=DEVICE, max_norm=args.max)
        # print('1')
        # ddn = attacker.attack(model, images, labels)

        ddn = attacker.attack(model, inputs=images, labels=labels)

        # ddn = torch.renorm(ddn - images, p=2, dim=0, maxnorm=args.max) + images
        # print(ddn-images)

        for t in range(args.batch_size):
            ddn2 = np.transpose(ddn[t].detach().cpu().numpy(), (1, 2, 0))
            # deepfool2 = np.transpose(deepfool[t].cpu().numpy(), (1, 2, 0))
            name = '/ddn_' + str(i) + str(t) + '.png'
            out_path = os.path.join(path_PGD, str(labels[t].cpu().numpy()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            # print(out_path)
            out = out_path + name
            # print(ddn2.shape)
            scipy.misc.imsave(out, ddn2)

    '''
    print('foolbox')
    deepfool = attack(images, labels,'C&W')

    if deepfool is None:
        continue

    for t in range(args.batch_size):
        #ddn2 = np.transpose(ddn[t].cpu().numpy(), (1, 2, 0))
        deepfool2 = np.transpose(deepfool[t].cpu().numpy(), (1, 2, 0))
        name='/C&W_'+str(i)+str(t)+'.png'
        out_path=os.path.join(pathcw,str(labels[t].cpu().numpy()))
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        #print(out_path)
        out=out_path+name
        scipy.misc.imsave(out,deepfool2)
        '''
