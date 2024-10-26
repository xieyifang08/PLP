import os
import PIL
import glob
import argparse
import numpy as np
import pandas as pd
import random

from PIL import Image

import argparse
import tqdm
import glob
import PIL
from copy import deepcopy
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.optim import SGD, lr_scheduler,Adam
from torch.backends import cudnn
from progressbar import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
import sys
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST
from torch.utils import data

from fast_adv.process.preprocess import _TVM, TVMcompression, JPEGcompression, GridMaskCompression

BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)

from fast_adv.models.wsdan.wsdan import WSDAN
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN
from fast_adv.utils.messageUtil import send_email




DEVICE = torch.device('cuda:0' if torch.cuda.is_available()  else 'cpu')
print(DEVICE)
class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        image = self.transformer(Image.open(image_path).convert('RGB'))#
        #b=Image.fromarray(jpeg_compression(image_path))
        #image = self.transformer(b)
        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label_idx': label_idx,
            'filename': os.path.basename(image_path)
        }
        return sample

def load_data_for_defense(input_data, com_index, img_size=32,batch_size=32):
    compression= {
        'tvm': TVMcompression(),
        'jpeg': JPEGcompression(),
        'gridMask': GridMaskCompression(),
    }
    all_imgs = []
    all_labels = []
    #for input_dir in jpg_data:
    for input_dir in tqdm.tqdm(input_data):
        one_imgs = glob.glob(input_dir)  # (os.path.join(input_dir, './*/*.jpg'))
        one_labels = [int(img_path.split('/')[-2]) for img_path in one_imgs]
        all_imgs.extend(one_imgs)
        all_labels.extend(one_labels)
    print(len(all_labels), "compression[com_index]:", compression[com_index])
    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    # print(all_labels)
    transformer = transforms.Compose([
        # transforms.Resize((img_size, img_size), interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor(),
        compression[com_index]
    ])
    datasets = {
        'dev_data': ImageSet(dev_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=8,
                       shuffle=False) for ds in datasets.keys()
        }
    return dataloaders
    #
    # path=os.path.join(input_dir, '/*/*.jpg')
    # print(path)


def import_dataloader_from_integrated_dataset(com_index, input_dir='../defenses/data/cifar10',batch_size=32):
    compression= {
        'tvm': TVMcompression(),
        'jpeg': JPEGcompression(),
        'gridMask': GridMaskCompression(),
        'null': False,
    }
    if not compression[com_index]:
        print("org")
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    else:
        print("no org: ",compression[com_index])

        transform = transforms.Compose([
            transforms.ToTensor(),
            compression[com_index]
        ])
    # train_set = data.Subset(CIFAR10(input_dir, train=True, transform=transform, download=True), list(range(50000)))
    # train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_set = CIFAR10(input_dir, train=False, transform=transform, download=True)  # 91.9
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return test_loader

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='test_FGSM', help='path to dataset')
   # parser.add_argument('--input_dir', default='jpg_test_PGD',
                     #   help='Input directory with images.', type=str)
    parser.add_argument('--output_file', default='output.csv',
                        help='Output file to save labels', type=str)
    parser.add_argument('--target_model', default='densenet161',
                        help='cnn model, e.g. , densenet121, densenet161', type=str)
    parser.add_argument('--gpu_id', default=0, nargs='+',
                        help='gpu ids to use, e.g. 0 1 2 3', type=int)
    parser.add_argument('--batch_size', default=128,
                        help='batch size, e.g. 16, 32, 64...', type=int)
    return parser.parse_args()

def acc_test(weight_wsdan_best):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])


    test_set = CIFAR10('data/cifar10', train=False, transform=test_transform, download=True)

    test_loader = data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

    print(len(test_loader))

    image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
    image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)

    m = WSDAN(num_classes=10, M=32, net="wide_resnet", pretrained=True)
    model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
    model_file = './weights/best/2AT_cifar10_ep_13_val_acc0.8770.pth'
    weight_wsdan3 = './weights/cifar10_wsgan_3/cifar10_valacc0.pth'
    weight_wsdan_final = './weights/cifar10_wsdan_final/cifar10_valacc0.871999979019165.pth'
    # weight_wsdan_best = './weights/cifar10_WSDAN_best/cifar10_valacc0.8784999758005142.pth'
    model_dict = torch.load(weight_wsdan_best)
    model.load_state_dict(model_dict)


    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    acc = 0
    for i, (X, y) in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Raw Image
        y_pred_raw, _, attention_map = model(X)

        acc += (y_pred_raw.argmax(1) == y).float().sum().item()
    print(acc)
    return acc

def acc_test_ae(test_loader, weight_wsdan_best):

    image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
    image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)

    m = WSDAN(num_classes=10, M=32, net="wide_resnet", pretrained=True)
    model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range

    model_dict = torch.load(weight_wsdan_best)
    model.load_state_dict(model_dict)


    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    acc = 0
    total = 0
    widgets = ['test :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    for batch_data in pbar(test_loader):
        X, y = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
        total+=X.shape[0]
        # Raw Image
        y_pred_raw, _, attention_map = model(X)

        acc += (y_pred_raw.argmax(1) == y).float().sum().item()
    return acc,total


def acc_test_from_ce(test_loader, weight_wsdan_best):

    image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
    image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)

    m = WSDAN(num_classes=10, M=32, net="wide_resnet", pretrained=True)
    model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range

    model_dict = torch.load(weight_wsdan_best)
    model.load_state_dict(model_dict)


    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    acc = 0
    total = 0

    for i, (X, y) in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        X, y = X.to(DEVICE), y.to(DEVICE)
        total+=X.shape[0]
        # Raw Image
        y_pred_raw, _, attention_map = model(X)

        acc += (y_pred_raw.argmax(1) == y).float().sum().item()
    return acc,total
def acc_test_mnist(weight_wsdan_best):
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])


    test_set = MNIST('data/mnist', train=False, transform=test_transform, download=True)

    test_loader = data.DataLoader(test_set, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

    print(len(test_loader))

    image_mean = torch.tensor([0.1307]).view(1, 1, 1, 1)
    image_std = torch.tensor([0.3081]).view(1, 1, 1, 1)

    m = WSDAN(num_classes=10, M=32, net="wide_resnet", pretrained=True)
    model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
    model_dict = torch.load(weight_wsdan_best)
    model.load_state_dict(model_dict)


    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    acc = 0
    total = 0
    for i, (X, y) in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Raw Image
        y_pred_raw, _, attention_map = model(X)
        total += X.shape[0]
        acc += (y_pred_raw.argmax(1) == y).float().sum().item()
    print(acc, total)
    return acc


def test_3attack_ae_in_3compression():
    msg = ''
    print("round 1:\n")
    input_data = ['/home/unknown/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/ae/FGSM/test/*/*.png']
    weight_wsdan = './weights/cifar10_wsdan_tvm/cifar10_valacc0.9392914012738853.pth'
    test_loader = load_data_for_defense(input_data, 'tvm')['dev_data']
    print("FGSM ae in TVM dataloader is ready")
    acc1, total1 = acc_test_ae(test_loader, weight_wsdan)
    print("the acc of FGSM ae in TVM compression is =",acc1,'/', total1)
    msg += "\nthe acc of FGSM ae in TVM compression is ="+str(acc1)+'/'+str(total1)

    input_data = ['/home/unknown/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/ae/FGSM/test/*/*.png']
    test_loader = load_data_for_defense(input_data, 'jpeg')['dev_data']
    weight_wsdan = './weights/cifar10-wsdan-jpeg/cifar10_valacc0.9484474522292994.pth'
    acc1, total1 = acc_test_ae(test_loader, weight_wsdan)
    print("the acc of FGSM ae in JPEG compression is =", acc1, '/', total1)
    msg += "\nthe acc of FGSM ae in JPEG compression is ="+str(acc1)+'/'+str(total1)

    input_data = ['/home/unknown/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/ae/FGSM/test/*/*.png']
    test_loader = load_data_for_defense(input_data, 'gridMask')['dev_data']
    weight_wsdan = './weights/cifar10_wsdan_gridmask/cifar10_valacc0.8626592356687898.pth'
    acc1, total1 = acc_test_ae(test_loader, weight_wsdan)
    print("the acc of FGSM ae in gridMask compression is =", acc1, '/', total1)
    msg += "\nthe acc of FGSM ae in gridMask compression is ="+str(acc1)+'/'+str(total1)

    print("round 2:\n")
    input_data = [
        '/home/unknown/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/ae/deepfool/test/*/*.png']
    test_loader = load_data_for_defense(input_data, 'jpeg')['dev_data']
    weight_wsdan = './weights/cifar10-wsdan-jpeg/cifar10_valacc0.9484474522292994.pth'
    acc1, total1 = acc_test_ae(test_loader, weight_wsdan)
    print("the acc of deepfool ae in JPEG compression is =", acc1, '/', total1)
    msg += "\nthe acc of deepfool ae in JPEG compression is ="+str(acc1)+'/'+str(total1)

    input_data = [
        '/home/unknown/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/ae/deepfool/test/*/*.png']
    weight_wsdan = './weights/cifar10_wsdan_tvm/cifar10_valacc0.9392914012738853.pth'
    test_loader = load_data_for_defense(input_data, 'tvm')['dev_data']
    print("PGD ae in TVM dataloader is ready")
    acc1, total1 = acc_test_ae(test_loader, weight_wsdan)
    print("the acc of deepfool ae in TVM compression is =", acc1, '/', total1)
    msg += "\nthe acc of deepfool ae in TVM compression is ="+str(acc1)+'/'+str(total1)

    input_data = [
        '/home/unknown/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/ae/deepfool/test/*/*.png']
    test_loader = load_data_for_defense(input_data, 'gridMask')['dev_data']
    weight_wsdan = './weights/cifar10_wsdan_gridmask/cifar10_valacc0.8626592356687898.pth'
    acc1, total1 = acc_test_ae(test_loader, weight_wsdan)
    print("the acc of deepfool ae in gridMask compression is =", acc1, '/', total1)
    msg += "\nthe acc of deepfool ae in gridMask compression is ="+str(acc1)+'/'+str(total1)

    print("round 3:\n")
    input_data = [
        '/home/unknown/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/ae/PGD/test/*/*.png']
    test_loader = load_data_for_defense(input_data, 'jpeg')['dev_data']
    weight_wsdan = './weights/cifar10-wsdan-jpeg/cifar10_valacc0.9484474522292994.pth'
    acc1, total1 = acc_test_ae(test_loader, weight_wsdan)
    print("the acc of PGD ae in JPEG compression is =", acc1, '/', total1)
    msg += "\nthe acc of PGD ae in JPEG compression is ="+str(acc1)+'/'+str(total1)

    input_data = [
        '/home/unknown/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/ae/PGD/test/*/*.png']
    weight_wsdan = './weights/cifar10_wsdan_tvm/cifar10_valacc0.9392914012738853.pth'
    test_loader = load_data_for_defense(input_data, 'tvm')['dev_data']
    print("PGD ae in TVM dataloader is ready")
    acc1, total1 = acc_test_ae(test_loader, weight_wsdan)
    print("the acc of PGD ae in TVM compression is =", acc1, '/', total1)
    msg += "\nthe acc of PGD ae in TVM compression is ="+str(acc1)+'/'+str(total1)

    input_data = [
        '/home/unknown/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/ae/PGD/test/*/*.png']
    test_loader = load_data_for_defense(input_data, 'gridMask')['dev_data']
    weight_wsdan = './weights/cifar10_wsdan_gridmask/cifar10_valacc0.8626592356687898.pth'
    acc1, total1 = acc_test_ae(test_loader, weight_wsdan)
    print("the acc of PGD ae in gridMask compression is =", acc1, '/', total1)
    msg += "\nthe acc of PGD ae in gridMask compression is ="+str(acc1)+'/'+str(total1)
    send_email("3个攻击方法在3个数据增强下生成对抗样本及准确率测试", msg)

def test_3compression_from_integrated_dataset():
    msg = ''
    dataloader = import_dataloader_from_integrated_dataset('null')
    weight_wsdan = '../defenses/weights/cifar10_WSDAN_best/cifar10_0.87_low.pth'
    acc1, total1 = acc_test_from_ce(dataloader, weight_wsdan)
    print("the acc of org date in no compression is =", acc1, '/', total1)
    msg += "the acc of org date in no compression is ="+str(acc1)+'/'+str(total1)

    dataloader = import_dataloader_from_integrated_dataset('jpeg')
    weight_wsdan = './weights/cifar10-wsdan-jpeg/cifar10_valacc0.9484474522292994.pth'
    acc1, total1 = acc_test_from_ce(dataloader, weight_wsdan)
    print("the acc of jpeg ae in no compression is =", acc1, '/', total1)
    msg += "\nthe acc of jpeg date in no compression is ="+str(acc1)+'/'+str(total1)

    dataloader = import_dataloader_from_integrated_dataset('tvm')
    weight_wsdan = './weights/cifar10_wsdan_tvm/cifar10_valacc0.9392914012738853.pth'
    acc1, total1 = acc_test_from_ce(dataloader, weight_wsdan)
    print("the acc of tvm ae in no compression is =", acc1, '/', total1)
    msg += "\nthe acc of tvm date in no compression is ="+str(acc1)+'/'+str(total1)

    dataloader = import_dataloader_from_integrated_dataset('gridMask')
    weight_wsdan = './weights/cifar10_wsdan_gridmask/cifar10_valacc0.8626592356687898.pth'
    acc1, total1 = acc_test_from_ce(dataloader, weight_wsdan)
    print("the acc of gridmask ae in no compression is =", acc1, '/', total1)
    msg += "\nthe acc of gridmask date in no compression is ="+str(acc1)+'/'+str(total1)

    send_email("测试3种数据增强方法在干净样本的准确率", msg)
if __name__ == '__main__':
    test_3compression_from_integrated_dataset()
    # msg = ''
    # print("round 1:\n")
    # input_data = [
    #     '/home/unknown/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/ae/FGSM/test/*/*.png']
    # weight_wsdan = './weights/cifar10_wsdan_tvm/cifar10_valacc0.9392914012738853.pth'
    # test_loader = load_data_for_defense(input_data, 'tvm')['dev_data']
    # print("FGSM ae in TVM dataloader is ready")
    # acc1, total1 = acc_test_ae(test_loader, weight_wsdan)
    # print("the acc of FGSM ae in TVM compression is =", acc1, '/', total1)
    # weight_wsdan_best = './weights/cifar10_WSDAN_best/cifar10_0.87_low.pth'
    # print("the acc of best is ")
    # acc1 = acc_test(weight_wsdan_best)
    # msg += '干净样本，无增强，cifar10 best 的test acc:'+str(acc1)
    #
    # # acc_test_mnist('./weights/mnist_wsdan/mnist_valacc0.9945999825000763_25.pth')
    # input_data = ['/home/unknown/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/jpeg/test/*/*.png']
    # weight_wsdan_best = './weights/cifar10_WSDAN_best/cifar10_0.87_low.pth'
    # print("the acc of jpeg ae is ")
    # acc1, total1 = acc_test_ae(weight_wsdan_best,input_data)
    # print(acc1, '/', total1)
    # input_data = ['/home/unknown/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/tvm/test/*/*.png']
    # print("the acc of tvm ae is ")
    # acc1, total1 = acc_test_ae(weight_wsdan_best,input_data)
    # print(acc1, '/', total1)
    # input_data = ['/home/unknown/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/gridmask/test/*/*.png']
    # print("the acc of gridmask ae is ")
    # acc1, total1 = acc_test_ae(weight_wsdan_best,input_data)
    # print(acc1, '/', total1)

