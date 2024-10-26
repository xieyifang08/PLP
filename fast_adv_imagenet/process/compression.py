import glob
import os
import argparse

import scipy
import tqdm
import numpy as np
from copy import deepcopy
import torch
from torch.utils import data

from torchvision import transforms
from torchvision.datasets import CIFAR10
import sys

BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
from fast_adv.utils.messageUtil import send_email

from fast_adv.attacks import DDN

from fast_adv.models.cifar10.model_attention import wide_resnet
# from fast_adv.models.cifar10.model_mixed_attention import wide_resnet
# from fast_adv.models.cifar10 import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks.shape_precess import shape
import foolbox
import warnings

warnings.filterwarnings("ignore")
from foolbox.adversarial import Adversarial
from foolbox.criteria import Misclassification
from foolbox.distances import MeanAbsoluteDistance, Linfinity
from foolbox.attacks import FGSM, DeepFoolL2Attack, PGD, LocalSearchAttack, GaussianBlurAttack, \
    BinarizationRefinementAttack, ContrastReductionAttack, SaltAndPepperNoiseAttack, \
    SpatialAttack, CarliniWagnerL2Attack

import logging
logging.basicConfig(level = logging.ERROR,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
loggger = logging.getLogger()

parser = argparse.ArgumentParser(description='Extend sample')
parser.add_argument('--max-norm', type=float, default=10, help='max norm for the adversarial perturbations')
parser.add_argument('--imgsize', default=32, type=int, help='pic size')
parser.add_argument('--attack_name', '--at', default='DDN',
                    help='name for saving the final state dict')
parser.add_argument('--batch-size', '-b', default=128, type=int, help='mini-batch size')
parser.add_argument("--shape", type=int, default=None)
parser.add_argument("--ddn", type=int, default=None)
parser.add_argument("--fgsm", type=bool, default=True)
parser.add_argument("--deepfool", type=bool, default=True)
parser.add_argument("--pgd", type=bool, default=True)
parser.add_argument("--data-loader", type=str, default='train')
# parser.add_argument('--max-norm', type=float,default=1, help='max norm for the adversarial perturbations')

args = parser.parse_args()
print(args)
path = "/home/frankfeng/projects/attack"
input = '../defenses/data/cifar10'

attackers = {'FGSM': FGSM,
             'C&W': CarliniWagnerL2Attack,  # 距离无限制
             'DeepFoolAttack': DeepFoolL2Attack,  # 源码上没有限制
             'PGD': PGD,  # clip——epsilon=0.3
             'DDN': DDN,
             'LocalSearchAttack': LocalSearchAttack,
             'GaussianBlurAttack': GaussianBlurAttack,
             'BinarizationRefinementAttack': BinarizationRefinementAttack,
             'ContrastReductionAttack': ContrastReductionAttack,
             'SaltAndPepperNoiseAttack': SaltAndPepperNoiseAttack,
             'SpatialAttack': SpatialAttack}
image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
DEVICE = torch.device('cpu')
transform = transforms.Compose([
    transforms.ToTensor(),
])
# train_set = data.Subset(CIFAR10(input, train=True, transform=test_transform, download=True),list(range(0,30000)))
train_set = data.Subset(CIFAR10(input, train=True, transform=transform, download=True), list(range(50000)))  # 91.9

train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_set = CIFAR10(input, train=False, transform=transform, download=True)  # 91.9

test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
m = wide_resnet(num_classes=10, depth=28, widen_factor=10, dropRate=0.3)
model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
weight_AT = '../defenses/weights/best/2AT_cifar10_ep_13_val_acc0.8770.pth'
weight_norm = '../defenses/weights/best/2Norm_cifar10_ep_184_val_acc0.9515.pth'
model_dict = torch.load(weight_AT)
model.load_state_dict(model_dict)
'''
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
model2=NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
model_dict2 = torch.load('../defenses/weights/AT_cifar10_clean0.879_adv.pth')
model2.load_state_dict(model_dict2)
model3=NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
model_dict3 = torch.load('../defenses/weights/best/ALP_cifar10_ep_39_val_acc0.8592.pth')
model3.load_state_dict(model_dict3)
model4=NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
model_dict4 = torch.load('../defenses/weights/best/PLP1_cifar10_ep_29_val_acc0.8636.pth')
model4.load_state_dict(model_dict4)
'''


def attack(image, label, attack_name):
    fmodel = foolbox.models.PyTorchModel(model.eval().cuda(), bounds=(0, 1),
                                         num_classes=10)  # , preprocessing=(mean, std)
    criterion1 = Misclassification()
    distance = Linfinity  # MeanAbsoluteDistance
    attacker = attackers[attack_name](fmodel, criterion=criterion1, distance=distance)

    image = image.cpu().numpy()
    label = label.cpu().numpy()

    adversarials = image.copy()
    advs = attacker(image, label)  # , unpack=True, steps=self.max_iter, subsample=self.subsample)
    for i in tqdm.tqdm(range(len(advs)), ncols=80):
        if advs is not None:
            adv = torch.renorm(torch.from_numpy(advs[i] - image[i]), p=2, dim=0, maxnorm=100).numpy() + image[i]

            adversarials[i] = adv
    adversarials = torch.from_numpy(adversarials).to(DEVICE)

    return adversarials


class TVMcompression(object):



    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        print('type(img) ', type(img))
        print('img.shape() ', img.size)
        # index = random.random()
        # img.save('/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/process/img/'+str(index)+'_org.png')
        # print("JpegCompression img type", type(img))
        # print("JpegCompression img shape", img.size)
        # FGSM = attack(images, labels, 'FGSM')
        # img = numpy.array(img)
        #
        # try:
        #     with tf.Session() as sess:
        #
        #         img_data = tf.convert_to_tensor(img)
        #         # print("img_data type", type(img_data))
        #         # print("img_data", img_data.shape)
        #         # print(img_data.shape)
        #
        #         img2 = slq(img_data)
        #         # print('压缩完成', type(img2.eval()))
        #         img = Image.fromarray(img2.eval().astype('uint8')).convert('RGB')
        #         # img.save('/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/process/img/'+str(index)+'_jpeg.png')
        #         return img
        # except:
        return img

# class JpegCompression(object):
#
#
#
#     def __call__(self, img):
#         """
#         Args:
#             img (PIL Image): PIL Image
#         Returns:
#             PIL Image: PIL image.
#         """
#         # index = random.random()
#         # img.save('/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/process/img/'+str(index)+'_org.png')
#         # print("JpegCompression img type", type(img))
#         # print("JpegCompression img shape", img.size)
#         FGSM = attack(images, labels, 'FGSM')
#         img = numpy.array(img)
#
#         try:
#             with tf.Session() as sess:
#
#                 img_data = tf.convert_to_tensor(img)
#                 # print("img_data type", type(img_data))
#                 # print("img_data", img_data.shape)
#                 # print(img_data.shape)
#
#                 img2 = slq(img_data)
#                 # print('压缩完成', type(img2.eval()))
#                 img = Image.fromarray(img2.eval().astype('uint8')).convert('RGB')
#                 # img.save('/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/process/img/'+str(index)+'_jpeg.png')
#                 return img
#         except:
#             return img
#
#
#
#
#
#
#
#
#
# data_loader = loader[args.data_loader]
# print("data_loader :" , len(data_loader))
# for i, (images, labels) in enumerate(tqdm.tqdm(data_loader, ncols=80)):
#     print("\nepoch " +str(i)+'\n')
#     images, labels = images.to(DEVICE), labels.to(DEVICE)
#
#
#     if args.fgsm is True:
#         FGSM = attack(images, labels, 'FGSM')
#         for t in range(args.batch_size):
#             ddn2 = np.transpose(FGSM[t].detach().cpu().numpy(), (1, 2, 0))
#             name = '/FGSM_' + str(i) + str(t) + '.png'
#             out_path = os.path.join(path_FGSM, str(labels[t].cpu().numpy()))
#             if not os.path.exists(out_path):
#                 os.makedirs(out_path)
#             # print(out_path)
#             out = out_path + name
#             scipy.misc.imsave(out, ddn2)
#
#     if args.deepfool is True:
#         deepfool = attack(images, labels, 'DeepFoolAttack')
#         for t in range(args.batch_size):
#             ddn2 = np.transpose(deepfool[t].detach().cpu().numpy(), (1, 2, 0))
#             name = '/deepfool_' + str(i) + str(t) + '.png'
#             out_path = os.path.join(path_deepfool, str(labels[t].cpu().numpy()))
#             if not os.path.exists(out_path):
#                 os.makedirs(out_path)
#             # print(out_path)
#             out = out_path + name
#             scipy.misc.imsave(out, ddn2)
#
#     if args.pgd is True:
#         PGD = attack(images, labels, 'PGD')
#         for t in range(args.batch_size):
#             ddn2 = np.transpose(PGD[t].detach().cpu().numpy(), (1, 2, 0))
#             name = '/PGD_' + str(i) + str(t) + '.png'
#             out_path = os.path.join(path_pgd, str(labels[t].cpu().numpy()))
#             if not os.path.exists(out_path):
#                 os.makedirs(out_path)
#             # print(out_path)
#             out = out_path + name
#             scipy.misc.imsave(out, ddn2)
#
# send_email("FSGM, deepfoll, PGD 对抗样本生成", title="对抗样本生成完毕")
