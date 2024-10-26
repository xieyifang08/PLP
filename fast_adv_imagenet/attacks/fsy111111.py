import argparse
import os
import shutil
import glob

import PIL
import torchvision.models as models
import eagerpy as ep
import tqdm
import torch
from progressbar import *
import pandas as pd
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils import data
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn
import scipy
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from foolbox import PyTorchModel, accuracy, samples,utils,attacks
from foolbox.plot import images
from foolbox.attacks import LinfPGD, LinfDeepFoolAttack, DDNAttack,BoundaryAttack,SpatialAttack
from torchvision.datasets import CIFAR10

from advdigitalmark import advDigitalMark
import warnings
import csv
import pandas as pd
import imageio

# from fast_adv.process.jpeg_compression import JpegCompression
from fast_adv.utils import NormalizedModel
from nsgaii import model_name,pop_size,max_gen,train_input_dir
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
from fast_adv.models.cifar10.resnet import ResNet101
from fast_adv.models.cifar10.vgg import VGG19
from fast_adv.models.cifar10.alexnet import AlexNet
DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() ) else 'cpu')


def prepareData(input_dir, img_size=224, batch_size=10):  # img_size
    all_imgs = glob.glob(os.path.join(input_dir, './*/*.jpg'))
    print("total img count = ",len(all_imgs))
    all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]
    train = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    train_data, val_data = train_test_split(train,
                                            stratify=train['label_idx'].values, train_size=0.9, test_size=0.1)
    transformer_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomResizedCrop(img_size, (0.7, 1), interpolation=PIL.Image.BILINEAR),
        transforms.Resize([img_size, img_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transformer = transforms.Compose([
        transforms.Resize([img_size, img_size], interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                     std=[0.5, 0.5, 0.5]),
    ])
    datasets = {
        'train_data': ImageSet(train_data, transformer_train),
        'val_data': ImageSet(val_data, transformer)
    }

    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=True) for ds in datasets.keys()
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
        b = Image.fromarray(imageio.imread(image_path))
        image = self.transformer(b)
        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label_idx': label_idx,
            'filename': os.path.basename(image_path)
        }
        return sample


print("prepareData(args.data)", prepareData(train_input_dir))
train_loader = prepareData(train_input_dir)[
    'train_data']
def main() -> None:
    print(DEVICE)
    # instantiate a model (could also be a TensorFlow or JAX model)
    model = model_name(pretrained=True).eval()
    # preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    # image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
    # image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
    preprocessing = dict(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    total_success = 0

    # headers = ['img', 'advIndex', 'alpha ', 'angle', 'class', 'l2']
    # with open('result_inception_v3_gen4_30.csv', 'w')as f:
    #     f_csv = csv.writer(f)
    #     f_csv.writerow(headers)
    attack = advDigitalMark()
    # SpatialAttack()
    epoch = 0
    widgets = ['train :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    length = len(train_loader)
    print("length train loader: ", length)
    for batch_data in pbar(train_loader):
        if epoch == 20:
            break
        images, labels =batch_data['image'].to(DEVICE),batch_data['label_idx'].to(DEVICE)

        # boundaryattack = BoundaryAttack()
        # acc = boundaryattack.run(fmodel, images, labels)
        # print(acc)
        clean_acc = accuracy(fmodel, images, labels)
        print(f"\nepoch "+str(epoch)+":\nclean accuracy: ", clean_acc)
        # print(index, labels)
        # apply the attack
        # attack = LinfPGD()
        # attack = LinfDeepFoolAttack()
        # attack = DDNAttack()

        count = attack.run(fmodel, images, labels)
        total_success += count
        epoch += 1
        print("current success attack is ",str(total_success))

    print(str(model_name)+"_"+str(max_gen)+"_"+str(max_gen)+": total attack success img is ", str(total_success))

def exp_otherAttack():
    model = model_name(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    total_success = 0

    attack = SpatialAttack()

    epoch = 0
    widgets = ['train :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    length = len(train_loader)
    print("length train loader: ", length)
    for batch_data in pbar(train_loader):
        if epoch == 25:
            break
        images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
        clean_acc = accuracy(fmodel, images, labels)
        print(f"\nepoch " + str(epoch) + ":\nclean accuracy: ", clean_acc)
        epsilons = [
            1
        ]
        _, _, isSuceess = attack(fmodel, images, labels, epsilons=epsilons)
        isSuceess = isSuceess.cpu().numpy()
        total_success += np.sum(isSuceess == True)
        print("current success : ", total_success)

        # total_success += int(adv_acc*10)
        epoch += 1
    print("success attack is ", str(total_success))


def exp_balckbox_attack():
    model = model_name(pretrained=True).eval()
    preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)

    total_success = 0

    widgets = ['train :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    length = len(train_loader)
    print("length train loader: ", length)
    total_imgs = 0
    for batch_data in pbar(train_loader):
        images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
        acc = accuracy(fmodel, images, labels)
        print(images.shape[0])
        total_imgs += images.shape[0]
        total_success += int(acc * images.shape[0])
    print("success attack is ", str(total_imgs - total_success),"/"+str(total_imgs))
    print("watermark attack rate is: ",str(1- total_success/total_imgs))



def exp():
    m = model_name(pretrained=True).eval()
    # fmodel = PyTorchModel(model, bounds=(0, 1), preprocessing=preprocessing)
    image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    image_std = torch.tensor( [0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
    trans = transforms.Compose([
        transforms.ToTensor(),
    ])
    trans_randomRotation = transforms.Compose([
        transforms.RandomRotation(360, resample=False, expand=False,
                                  center=None),
        transforms.ToTensor(),
    ])
    trans_centerCrop = transforms.Compose([
        transforms.CenterCrop([200,200]),
        transforms.ToTensor(),
    ])
    trans_jpegCompression = transforms.Compose([
        JpegCompression(),
        transforms.ToTensor(),
    ])

    trans_pad = transforms.Compose([
        transforms.Pad(10, fill=(119,136,153), padding_mode='constant'),
        transforms.Resize([224,224]),
        transforms.ToTensor(),
    ])
    img1 = Image.open('/home/frankfeng/桌面/resnet101/blackboxattack/1608968874997_2_0_111.png')
    # image = trans(img1).to(DEVICE)
    # img1 = Image.fromarray((image.cpu().numpy().transpose([1, 2, 0])* 255).astype('uint8')).convert('RGB')
    # img1.save('/home/frankfeng/桌面/exp/clean.png')
    # label = torch.Tensor([111]).to(DEVICE).unsqueeze(0)
    # image = image.unsqueeze(0)
    # logist = model(image)
    # print("clean img ", logist.argmax(axis=-1))
    #
    img = Image.open('/home/frankfeng/桌面/resnet101/blackboxattack/1608968874997_2_29_alpha18.22076667633076_angle151.36959921485058_logist948_l26.7666764.png')
    image = trans(img).to(DEVICE)
    # img3 = Image.fromarray((image.cpu().numpy().transpose([1, 2, 0])* 255).astype('uint8')).convert('RGB')
    # img3.save('/home/frankfeng/桌面/exp/adv.png')
    # label = torch.Tensor([519]).to(DEVICE).unsqueeze(0)
    # image = image.unsqueeze(0)
    # logist = model(image)
    # print("adv img ", logist.argmax(-1))
    found = False
    while not found:
        image1 = trans_randomRotation(img).to(DEVICE)
        image = image1.unsqueeze(0)
        logist = model(image)
        print("trans_randomRotation img ",logist.argmax(axis=-1).cpu().numpy()[0])
        if int(logist.argmax(axis=-1).cpu().numpy()) == 948:
            found = True
    img4 = Image.fromarray((image1.cpu().numpy().transpose([1, 2, 0]) * 255).astype('uint8')).convert('RGB')
    img4.save('/home/frankfeng/桌面/exp/rotation.png')

    # image = trans_centerCrop(img).to(DEVICE)
    # img5 = Image.fromarray((image.cpu().numpy().transpose([1, 2, 0])* 255).astype('uint8')).convert('RGB')
    # img5.save('/home/frankfeng/桌面/exp/crop.png')
    # image = image.unsqueeze(0)
    # logist = model(image)
    # print("trans_centerCrop img ", logist.argmax(axis=-1))
    #
    # image = trans_jpegCompression(img).to(DEVICE)
    # img6 = Image.fromarray((image.cpu().numpy().transpose([1, 2, 0])* 255).astype('uint8')).convert('RGB')
    # img6.save('/home/frankfeng/桌面/exp/compression.png')
    # image = image.unsqueeze(0)
    # logist = model(image)
    # print("trans_jpegCompression ", logist.argmax(axis=-1))
    #
    # image = trans_pad(img).to(DEVICE)
    # img7 = Image.fromarray((image.cpu().numpy().transpose([1, 2, 0])* 255).astype('uint8')).convert('RGB')
    # img7.save('/home/frankfeng/桌面/exp/pad.png')
    # image = image.unsqueeze(0)
    # logist = model(image)
    # print("trans_pad ", logist.argmax(axis=-1))

def exp1():
    print(DEVICE)
    parser = argparse.ArgumentParser(description='CIFAR10 Training against DDN Attack')

    parser.add_argument('--data', default='../defenses/data/cifar10', help='path to dataset')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
    parser.add_argument('--cpu', action='store_true', help='force training on cpu')
    parser.add_argument('--save-folder', '--sf', default='weights/cifar10_inception_v3/',
                        help='folder to save state dicts')
    parser.add_argument('--save-freq', '--sfr', default=10, type=int, help='save frequency')
    parser.add_argument('--save-name', '--sn', default='cifar10', help='name for saving the final state dict')

    parser.add_argument('--batch-size', '-b', default=20, type=int, help='mini-batch size')
    parser.add_argument('--epochs', '-e', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--lr-decay', '--lrd', default=0.2, type=float, help='decay for learning rate')
    parser.add_argument('--lr-step', '--lrs', default=10, type=int, help='step size for learning rate decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float, help='weight decay')
    parser.add_argument('--drop', default=0.3, type=float, help='dropout rate of the classifier')

    parser.add_argument('--adv', type=int, default=None, help='epoch to start training with adversarial images')
    parser.add_argument('--max-norm', type=float, default=1, help='max norm for the adversarial perturbations')
    parser.add_argument('--steps', default=10, type=int, help='number of steps for the attack')

    parser.add_argument('--visdom-port', '--vp', type=int, default=8097,
                        help='For visualization, which port visdom is running.')
    parser.add_argument('--print-freq', '--pf', default=10, type=int, help='print frequency')

    args = parser.parse_args()
    print(args)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = data.Subset(CIFAR10(args.data, train=True, transform=train_transform, download=True),
                            list(range(30000,50000)))
    val_set = data.Subset(CIFAR10(args.data, train=True, transform=test_transform, download=True),
                          list(range(48000, 50000)))
    test_set = CIFAR10(args.data, train=False, transform=test_transform, download=True)

    train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                                   drop_last=True, pin_memory=True)
    val_loader = data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=args.workers, pin_memory=True)
    image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
    image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)

    m = AlexNet().eval()
    model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
    model_file = '/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/defenses/weights/cifar10_AlexNet/cifar10_valacc0.8019999772310257.pth'

    model_dict = torch.load(model_file)
    model.load_state_dict(model_dict)
    # preprocessing = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)
    # image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
    # image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
    preprocessing = dict(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262], axis=-3)
    fmodel = PyTorchModel(model, bounds=(0, 1))

    total_success = 0

    # headers = ['img', 'advIndex', 'alpha ', 'angle', 'class', 'l2']
    # with open('result_inception_v3_gen4_30.csv', 'w')as f:
    #     f_csv = csv.writer(f)
    #     f_csv.writerow(headers)
    attack = advDigitalMark()
    # SpatialAttack()
    epoch = 0
    widgets = ['train :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    length = len(train_loader)
    print("length train loader: ", length)
    for epoch in range(args.epochs):
        if epoch == 25:
            break
        for i, (images, labels) in enumerate(tqdm.tqdm(train_loader, ncols=80)):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            clean_acc = accuracy(fmodel, images, labels)
            print(f"\nepoch " + str(epoch) + ":\nclean accuracy: ", clean_acc)

            count = attack.run(fmodel, images, labels)
            total_success += count
        epoch += 1
        print("current success attack is ", str(total_success))

    print(str(model_name) + "_" + str(max_gen) + "_" + str(max_gen) + ": total attack success img is ",
              str(total_success))
if __name__ == "__main__":
    main()
    # exp()
    # exp_balckbox_attack()
    # exp()
    # 根据需要选择需要读的行
    # f = open('/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/test.csv')
    # df = pd.read_table(f, sep=',', skiprows=[])  # 跳过不想读的行
    # for i in tqdm.tqdm(range(len(df))):
    #     ph = "/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/imagenet/"+df["label"][i]
    #     if not os.path.exists(ph):
    #         os.makedirs(ph)
    #     srcfile = "/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/mini-imagenet/images/"+df["filename"][i]
    #     dstfile = ph+"/"+df["filename"][i]
    #     shutil.copyfile(srcfile, dstfile)


