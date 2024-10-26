import glob
import os
import argparse

import imageio
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import scipy
import tqdm
import numpy as np
from copy import deepcopy
import torch
from torch.utils import data
# from torchvision.transforms import InterpolationMode
import pandas as pd
from torchvision import transforms
import sys

BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
from fast_adv.utils.messageUtil import send_email
from fast_adv_imagenet.utils import model_utils

from fast_adv.attacks import DDN
import warnings

warnings.filterwarnings("ignore")
import foolbox
from foolbox.criteria import Misclassification
from foolbox.distances import MeanAbsoluteDistance, Linfinity
from foolbox.attacks import FGSM, DeepFoolL2Attack, PGD, LocalSearchAttack, GaussianBlurAttack, \
    BinarizationRefinementAttack, ContrastReductionAttack, SaltAndPepperNoiseAttack, BoundaryAttack, \
    CarliniWagnerL2Attack

parser = argparse.ArgumentParser(description='Extend sample')
parser.add_argument('--max-norm', type=float, default=10, help='max norm for the adversarial perturbations')
parser.add_argument('--img_size', default=224, type=int, help='pic size')
parser.add_argument('--data', default='../data/imagenet_train_10000', help='path to dataset')
parser.add_argument('--attack_name', '--at', default='DDN',
                    help='name for saving the final state dict')
parser.add_argument('--batch-size', '-b', default=32, type=int, help='mini-batch size')
parser.add_argument("--shape", type=int, default=None)
parser.add_argument("--ddn", type=int, default=None)
parser.add_argument("--fgsm", type=bool, default=True)
parser.add_argument("--deepfool", type=bool, default=True)
parser.add_argument("--pgd", type=bool, default=True)
parser.add_argument("--data-loader", type=str, default='train')
# parser.add_argument('--max-norm', type=float,default=1, help='max norm for the adversarial perturbations')

args = parser.parse_args()
print(args)
path = "./advs/"

attackers = {'FGSM': FGSM,
             # 'C&W': CarliniWagnerL2Attack,  # 距离无限制
             'DeepFoolAttack': DeepFoolL2Attack,  # 源码上没有限制
             'PGD': PGD,  # clip——epsilon=0.3
             'DDN': DDN,
             'LocalSearchAttack': LocalSearchAttack,
             'GaussianBlurAttack': GaussianBlurAttack,
             'BinarizationRefinementAttack': BinarizationRefinementAttack,
             'ContrastReductionAttack': ContrastReductionAttack,
             'SaltAndPepperNoiseAttack': SaltAndPepperNoiseAttack,
             'BoundaryAttack': BoundaryAttack,
             'CW': CarliniWagnerL2Attack,
             # D:\anaconda3\envs\pytorch4\Lib\site-packages\foolbox\attacks\boundary_attack.py
             # CW, DeepFool, PGD, DDN, BoundaryAttack五种攻击算法
             # 'SpatialAttack': SpatialAttack
             }
DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')


def load_data_for_defense(csv, input_dir, img_size=args.img_size, batch_size=args.batch_size):
    jir = pd.read_csv(csv)
    all_imgs = [os.path.join(input_dir, str(i)) for i in jir['ImageId'].tolist()]
    all_labels = jir['TrueLabel'].tolist()
    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})

    transformer = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                     std=[0.5, 0.5, 0.5]),
    ])
    datasets = {
        'dev_data': ImageSet(dev_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       # num_workers=8,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders


class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        # image = self.transformer(Image.open(image_path))  # .convert('RGB'))
        # b = Image.fromarray(scipy.misc.imread(image_path))
        # b = Image.fromarray(imageio.imread(image_path))
        b = Image.open(image_path).convert('RGB')  # 确保图像是RGB格式
        image = self.transformer(b)
        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label_idx': label_idx,
            'filename': os.path.basename(image_path)
        }
        return sample


test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), os.path.join(args.data, 'train'))['dev_data']

print(len(test_loader))

# 修改分类网络模型AlexNet、VGG19、ResNet50、DenseNet161、WideResNet101（后文均以 WRN 代称）、SqueezeNet 、MobileNetV2
model = model_utils.load_model("vgg19").to(DEVICE)


def attack(image, label, attack_name):
    fmodel = foolbox.models.PyTorchModel(model.eval().cuda(), bounds=(0, 1),
                                         num_classes=1000)  # , preprocessing=(mean, std)
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


print("data_loader :", len(test_loader))


# for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
#     if args.batch_size*i < 4990:
#         continue
#     images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
#     print("\nepoch " +str(i)+'\n')
#     images, labels = images.to(DEVICE), labels.to(DEVICE)
#
#
#     if args.fgsm is True:
#         FGSM = attack(images, labels, 'FGSM')
#         for t in range(images.shape[0]):
#             ddn2 = np.transpose(FGSM[t].detach().cpu().numpy(), (1, 2, 0))
#             name = str(args.batch_size * i + t) + '.jpg'
#             out_path = os.path.join(path, "fgsm")
#             if not os.path.exists(out_path):
#                 os.makedirs(out_path)
#             # print(out_path)
#             out = os.path.join(out_path, name)
#             # scipy.misc.imsave(out, ddn2)
#             imageio.imwrite(out, ddn2)
#
#     if args.deepfool is True:
#         deepfool = attack(images, labels, 'DeepFoolAttack')
#         for t in range(images.shape[0]):
#             ddn2 = np.transpose(deepfool[t].detach().cpu().numpy(), (1, 2, 0))
#             name = str(args.batch_size * i + t) + '.jpg'
#             out_path = os.path.join(path, "deepfool")
#             if not os.path.exists(out_path):
#                 os.makedirs(out_path)
#             # print(out_path)
#             out = os.path.join(out_path, name)
#             # scipy.misc.imsave(out, ddn2)
#             imageio.imwrite(out, ddn2)
#
#     if args.pgd is True:
#         PGD = attack(images, labels, 'PGD')
#         for t in range(images.shape[0]):
#             ddn2 = np.transpose(PGD[t].detach().cpu().numpy(), (1, 2, 0))
#             name = str(args.batch_size*i + t) + '.jpg'
#             out_path = os.path.join(path,"pgd")
#             if not os.path.exists(out_path):
#                 os.makedirs(out_path)
#             out = os.path.join(out_path, name)
#             # scipy.misc.imsave(out, ddn2)
#             imageio.imwrite(out, ddn2)
def save_image(image_tensor, filename):
    image_numpy = image_tensor.detach().cpu().numpy()
    image_numpy = np.transpose(image_numpy, (1, 2, 0))  # 转换为 HWC 格式
    image_numpy = np.clip(image_numpy * 255, 0, 255).astype(np.uint8)  # 确保像素值在 [0, 255] 范围内
    imageio.imwrite(filename, image_numpy)


for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
    if args.batch_size * i < 4990:
        continue
    images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
    print("\nepoch " + str(i) + '\n')
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    if args.fgsm:
        FGSM = attack(images, labels, 'FGSM')
        for t in range(images.shape[0]):
            name = str(args.batch_size * i + t) + '.jpg'
            out_path = os.path.join(path, "fgsm")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, name)
            save_image(FGSM[t], out)

    if args.deepfool:
        deepfool = attack(images, labels, 'DeepFoolAttack')
        for t in range(images.shape[0]):
            name = str(args.batch_size * i + t) + '.jpg'
            out_path = os.path.join(path, "deepfool")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, name)
            save_image(deepfool[t], out)

    if args.pgd:
        PGD = attack(images, labels, 'PGD')
        for t in range(images.shape[0]):
            name = str(args.batch_size * i + t) + '.jpg'
            out_path = os.path.join(path, "pgd")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, name)
            save_image(PGD[t], out)

    if args.cw:
        CW = attack(images, labels, 'CarliniWagnerL2Attack')
        for t in range(images.shape[0]):
            name = str(args.batch_size * i + t) + '.jpg'
            out_path = os.path.join(path, "cw")
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, name)
            save_image(CW[t], out)

send_email("FSGM, deepfoll, PGD 对抗样本生成", title="对抗样本生成完毕")

# import glob
# import os
# import argparse
# from torch.utils.data import DataLoader, Dataset
# from PIL import Image
# import scipy
# import tqdm
# import numpy as np
# from copy import deepcopy
# import torch
# from torch.utils import data
# # from torchvision.transforms import InterpolationMode
# import pandas as pd
# from torchvision import transforms
# import sys
# import imageio
#
# BASE_DIR = os.path.dirname(os.path.abspath("../"))
# sys.path.append(BASE_DIR)
# from fast_adv.utils.messageUtil import send_email
# from fast_adv_imagenet.utils import model_utils
#
# from fast_adv.attacks import DDN
# import warnings
#
# warnings.filterwarnings("ignore")
# import foolbox
# from foolbox.criteria import Misclassification
# from foolbox.distances import MeanAbsoluteDistance, Linfinity
# from foolbox.attacks import FGSM, DeepFoolL2Attack, PGD, LocalSearchAttack, GaussianBlurAttack, \
#     BinarizationRefinementAttack, ContrastReductionAttack, SaltAndPepperNoiseAttack
#
# parser = argparse.ArgumentParser(description='Extend sample')
# parser.add_argument('--max-norm', type=float, default=10, help='max norm for the adversarial perturbations')
# parser.add_argument('--img_size', default=224, type=int, help='pic size')
# parser.add_argument('--data', default='../data/imagenet_train_10000', help='path to dataset')
# parser.add_argument('--attack_name', '--at', default='DDN',
#                     help='name for saving the final state dict')
# parser.add_argument('--batch-size', '-b', default=32, type=int, help='mini-batch size')
# parser.add_argument("--shape", type=int, default=None)
# parser.add_argument("--ddn", type=int, default=None)
# parser.add_argument("--fgsm", type=bool, default=True)
# parser.add_argument("--deepfool", type=bool, default=True)
# parser.add_argument("--pgd", type=bool, default=True)
# parser.add_argument("--data-loader", type=str, default='train')
# # parser.add_argument('--max-norm', type=float,default=1, help='max norm for the adversarial perturbations')
#
# args = parser.parse_args()
# print(args)
# path = "./advs/"
#
# attackers = {'FGSM': FGSM,
#              # 'C&W': CarliniWagnerL2Attack,  # 距离无限制
#              'DeepFoolAttack': DeepFoolL2Attack,  # 源码上没有限制
#              'PGD': PGD,  # clip——epsilon=0.3
#              'DDN': DDN,
#              'LocalSearchAttack': LocalSearchAttack,
#              'GaussianBlurAttack': GaussianBlurAttack,
#              'BinarizationRefinementAttack': BinarizationRefinementAttack,
#              'ContrastReductionAttack': ContrastReductionAttack,
#              'SaltAndPepperNoiseAttack': SaltAndPepperNoiseAttack,
#
#              # 'SpatialAttack': SpatialAttack
#              }
# DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
#
#
# def load_data_for_defense(csv, input_dir, img_size=args.img_size, batch_size=args.batch_size):
#     jir = pd.read_csv(csv)
#     all_imgs = [os.path.join(input_dir, str(i)) for i in jir['ImageId'].tolist()]
#     all_labels = jir['TrueLabel'].tolist()
#     dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
#
#     transformer = transforms.Compose([
#         transforms.Resize((img_size, img_size), interpolation=Image.BILINEAR),
#         # --------------------------------------------
#         transforms.Grayscale(num_output_channels=3),  # Ensure RGB
#         transforms.ToTensor(),
#         # transforms.Normalize(mean=[0.5, 0.5, 0.5],
#         #                     std=[0.5, 0.5, 0.5]),
#     ])
#     datasets = {
#         'dev_data': ImageSet(dev_data, transformer)
#     }
#     dataloaders = {
#         ds: DataLoader(datasets[ds],
#                        batch_size=batch_size,
#                        # num_workers=8,
#                        shuffle=False) for ds in datasets.keys()
#     }
#     return dataloaders
#
#
# # class ImageSet(Dataset):
# #     def __init__(self, df, transformer):
# #         self.df = df
# #         self.transformer = transformer
# #
# #     def __len__(self):
# #         return len(self.df)
# #
# #     def __getitem__(self, item):
# #         image_path = self.df.iloc[item]['image_path']
# #         # image = self.transformer(Image.open(image_path))  # .convert('RGB'))
# #         # b = Image.fromarray(scipy.misc.imread(image_path))
# #         b = Image.fromarray(imageio.imread(image_path))
# #         image = self.transformer(b)
# #         label_idx = self.df.iloc[item]['label_idx']
# #         sample = {
# #             'dataset_idx': item,
# #             'image': image,
# #             'label_idx': label_idx,
# #             'filename': os.path.basename(image_path)
# #         }
# #         return sample
#
# class ImageSet(Dataset):
#     def __init__(self, df, transformer):
#         self.df = df
#         self.transformer = transformer
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, item):
#         image_path = self.df.iloc[item]['image_path']
#         b = Image.open(image_path).convert('RGB')  # Ensure RGB
#         image = self.transformer(b)
#         label_idx = self.df.iloc[item]['label_idx']
#         sample = {
#             'dataset_idx': item,
#             'image': image,
#             'label_idx': label_idx,
#             'filename': os.path.basename(image_path)
#         }
#         return sample
#
#
# test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), os.path.join(args.data, 'train'))['dev_data']
#
# print(len(test_loader))
#
# model = model_utils.load_model("resnet152").to(DEVICE)
#
#
# def attack(image, label, attack_name):
#     fmodel = foolbox.models.PyTorchModel(model.eval().cuda(), bounds=(0, 1),
#                                          num_classes=1000)  # , preprocessing=(mean, std)
#     criterion1 = Misclassification()
#     distance = Linfinity  # MeanAbsoluteDistance
#     attacker = attackers[attack_name](fmodel, criterion=criterion1, distance=distance)
#
#     image = image.cpu().numpy()
#     label = label.cpu().numpy()
#
#     adversarials = image.copy()
#     advs = attacker(image, label)  # , unpack=True, steps=self.max_iter, subsample=self.subsample)
#     for i in tqdm.tqdm(range(len(advs)), ncols=80):
#         if advs is not None:
#             adv = torch.renorm(torch.from_numpy(advs[i] - image[i]), p=2, dim=0, maxnorm=100).numpy() + image[i]
#
#             adversarials[i] = adv
#     adversarials = torch.from_numpy(adversarials).to(DEVICE)
#
#     return adversarials
#
#
# print("data_loader :", len(test_loader))
# for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
#     if args.batch_size * i < 4990:
#         continue
#     images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
#     print("\nepoch " + str(i) + '\n')
#     images, labels = images.to(DEVICE), labels.to(DEVICE)
#     if args.fgsm is True:
#         FGSM = attack(images, labels, 'FGSM')
#         for t in range(images.shape[0]):
#             ddn2 = np.transpose(FGSM[t].detach().cpu().numpy(), (1, 2, 0))
#             ddn2 = (np.clip(ddn2 * 255, 0, 255)).astype(np.uint8)  # Ensure RGB and uint8
#             name = str(args.batch_size * i + t) + '.jpg'
#             out_path = os.path.join(path, "fgsm")
#             if not os.path.exists(out_path):
#                 os.makedirs(out_path)
#             out = os.path.join(out_path, name)
#             imageio.imwrite(out, ddn2)
#
#     if args.deepfool is True:
#         deepfool = attack(images, labels, 'DeepFoolAttack')
#         for t in range(images.shape[0]):
#             ddn2 = np.transpose(deepfool[t].detach().cpu().numpy(), (1, 2, 0))
#             ddn2 = (np.clip(ddn2 * 255, 0, 255)).astype(np.uint8)  # Ensure RGB and uint8
#             name = str(args.batch_size * i + t) + '.jpg'
#             out_path = os.path.join(path, "deepfool")
#             if not os.path.exists(out_path):
#                 os.makedirs(out_path)
#             out = os.path.join(out_path, name)
#             imageio.imwrite(out, ddn2)
#
#     if args.pgd is True:
#         PGD = attack(images, labels, 'PGD')
#         for t in range(images.shape[0]):
#             ddn2 = np.transpose(PGD[t].detach().cpu().numpy(), (1, 2, 0))
#             ddn2 = (np.clip(ddn2 * 255, 0, 255)).astype(np.uint8)  # Ensure RGB and uint8
#             name = str(args.batch_size * i + t) + '.jpg'
#             out_path = os.path.join(path, "pgd")
#             if not os.path.exists(out_path):
#                 os.makedirs(out_path)
#             out = os.path.join(out_path, name)
#             imageio.imwrite(out, ddn2)
#
#     # if args.fgsm is True:
#     #     FGSM = attack(images, labels, 'FGSM')
#     #     for t in range(images.shape[0]):
#     #         ddn2 = np.transpose(FGSM[t].detach().cpu().numpy(), (1, 2, 0))
#     #         name = str(args.batch_size * i + t) + '.jpg'
#     #         out_path = os.path.join(path, "fgsm")
#     #         if not os.path.exists(out_path):
#     #             os.makedirs(out_path)
#     #         # print(out_path)
#     #         ddn2_uint8 = np.clip(ddn2 * 255, 0, 255).astype(np.uint8)
#     #         out = os.path.join(out_path, name)
#     #         # scipy.misc.imsave(out, ddn2)
#     #         imageio.imwrite(out, ddn2_uint8)
#     # if args.deepfool is True:
#     #     deepfool = attack(images, labels, 'DeepFoolAttack')
#     #     for t in range(images.shape[0]):
#     #         ddn2 = np.transpose(deepfool[t].detach().cpu().numpy(), (1, 2, 0))
#     #         name = str(args.batch_size * i + t) + '.jpg'
#     #         out_path = os.path.join(path, "deepfool")
#     #         if not os.path.exists(out_path):
#     #             os.makedirs(out_path)
#     #         # print(out_path)
#     #         out = os.path.join(out_path, name)
#     #         scipy.misc.imsave(out, ddn2)
#     #
#     # if args.pgd is True:
#     #     PGD = attack(images, labels, 'PGD')
#     #     for t in range(images.shape[0]):
#     #         ddn2 = np.transpose(PGD[t].detach().cpu().numpy(), (1, 2, 0))
#     #         name = str(args.batch_size * i + t) + '.jpg'
#     #         out_path = os.path.join(path, "pgd")
#     #         if not os.path.exists(out_path):
#     #             os.makedirs(out_path)
#     #         out = os.path.join(out_path, name)
#     #         scipy.misc.imsave(out, ddn2)
#
# send_email("FSGM, deepfoll, PGD 对抗样本生成", title="对抗样本生成完毕")
