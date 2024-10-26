import glob
import logging
import os
import platform
import sys
BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
import argparse
from pytorch_grad_cam import GradCAM as GradCAM_ori
from sklearn.model_selection import train_test_split

from fast_adv_imagenet.visualize.grad_cam import GradCAM


import tqdm
from progressbar import *
import imageio
import pandas as pd
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv_imagenet.utils import model_utils

"""

"""

parser = argparse.ArgumentParser(description='CIFAR10 Training data augmentation')

parser.add_argument('--data', default='../data', help='path to dataset')
parser.add_argument('--img_size', default=224, type=int, help='size of image')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--cpu', action='store_true', help='force training on cpu')
parser.add_argument('--save-folder', '--sf', default='weights/wide_resnet101_at/', help='folder to save state dicts')
parser.add_argument('--visdom_env', '--ve', type=str, default="wide_resnet101_at",
                    help='which env visdom is running.')

parser.add_argument('--save-freq', '--sfr', default=10, type=int, help='save frequency')
parser.add_argument('--save-name', '--sn', default='cifar10', help='name for saving the final state dict')

parser.add_argument('--batch-size', '-b', default=16, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=20, type=int, help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--lr-decay', '--lrd', default=0.9, type=float, help='decay for learning rate')
parser.add_argument('--lr-step', '--lrs', default=2, type=int, help='step size for learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, help='weight decay')
parser.add_argument('--drop', default=0.3, type=float, help='dropout rate of the classifier')

parser.add_argument('--adv', type=int, default=1, help='epoch to start training with adversarial images')
parser.add_argument('--max-norm', type=float, default=1, help='max norm for the adversarial perturbations')
parser.add_argument('--steps', default=10, type=int, help='number of steps for the attack')

parser.add_argument('--visdom-port', '--vp', type=int, default=8097,
                    help='For visualization, which port visdom is running.')
parser.add_argument('--print-freq', '--pf', default=10, type=int, help='print frequency')

parser.add_argument('--num-attentions', '--na', default=32, type=int, help='number of attention maps')
parser.add_argument('--backbone-net', '--bn', default='wide_resnet', help='feature extractor')
parser.add_argument('--beta', '--b', default=5e-2, help='param for update feature centers')

args = parser.parse_args()
print(args)
if args.lr_step is None: args.lr_step = args.epochs

DEVICE = torch.device('cpu')

if not os.path.exists(args.save_folder) and args.save_folder:
    os.makedirs(args.save_folder)

def load_data_for_defense(input_dir, img_size=args.img_size, batch_size=args.batch_size):

    all_imgs = glob.glob(os.path.join(input_dir, './*/*.JPEG'))
    system = platform.system()
    if system == "Windows":
        all_labels = [int(img_path.split('\\')[-2]) for img_path in all_imgs]
    else:
        all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]

    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})

    transformer = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
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


def prepareData(input_dir, img_size=args.img_size, batch_size=args.batch_size):  # img_size
    all_imgs = glob.glob(os.path.join(input_dir, './*/*.JPEG'))

    system = platform.system()
    if system == "Windows":
        all_labels = [int(img_path.split('\\')[-2]) for img_path in all_imgs]
    else:
        all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]

    train = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    train_data, val_data = train_test_split(train, stratify=train['label_idx'].values, train_size=0.4, test_size=0.1)
    scale = 256 / 224
    transformer_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transformer = transforms.Compose([
        transforms.Resize(int(img_size * scale)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    datasets = {
        'train_data': ImageSet(train_data, transformer_train),
        'val_data': ImageSet(val_data, transformer)
    }

    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       # num_workers=8,
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
        b = Image.fromarray(imageio.imread(image_path)).convert('RGB')
        image = self.transformer(b)
        label_idx = self.df.iloc[item]['label_idx']
        sample = {
            'dataset_idx': item,
            'image': image,
            'label_idx': label_idx,
            'filename': os.path.basename(image_path)
        }
        return sample

system = platform.system()
logging.info("platform is {}".format(system))
if system == "Windows":
    loader = prepareData("C:/datasets/mini_ILSVRC2012_img_train")
    train_loader = loader['train_data']
    val_loader = loader['val_data']
    # test_loader = load_data_for_defense("C:/datasets/ILSVRC2012_validation_ground_truth.txt", "C:/datasets/ILSVRC2012_img_val")['dev_data']
    test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), os.path.join(args.data, 'images'))[
        'dev_data']
else:
    loader = prepareData("/home/lzj/ff/imagenet100")
    # loader = prepareData("/seu_share/home/fiki/ff/database/imagenet100")
    train_loader = loader['train_data']
    val_loader = loader['val_data']
    # test_loader = load_data_for_defense(os.path.join("/mnt/u2/datasets", 'val.csv'),
    #                                     os.path.join("/mnt/u2/datasets", 'ILSVRC2012_img_val'))['dev_data']
    test_loader = val_loader
    adv_loader = load_data_for_defense("../attacks/advs/wide_resnet101_imagenet100_backbone/rfgsm")['dev_data']
    data_loader = load_data_for_defense("../attacks/advs/wide_resnet101_imagenet100_backbone/clean")['dev_data']

    # # 超算平台
    # loader = prepareData("/seu_share/home/fiki/ff/database/mini_ILSVRC2012_img_train")
    # train_loader = loader['train_data']
    # val_loader = loader['val_data']
    # test_loader = val_loader


def draw_cam(database_path, model_name, out_path):
    model = model_utils.load_model(model_name,  pretrained=True, num_classes=100).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    cam = GradCAM(model=model, target_layers=[model.model.layer4[-1]])
    cam_ori = GradCAM_ori(model=model, target_layers=[model.model.layer4[-1]])
    val_accs = AverageMeter()
    for batch_data, adv_batch_data in zip(data_loader, adv_loader):
        images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        adv_images = adv_batch_data['image'].to(DEVICE)

        logits = model(images)
        logging.info('logits:{}, labels:{}'.format(logits.argmax(1), labels))
        logits_adv = model(adv_images)
        logging.info('logits_adv:{}, labels:{}'.format(logits_adv.argmax(1), labels))
        # val_accs.append((logits.argmax(1) == labels).float().mean().item())
        # logging.info('Accs: {:.4f}'.format( val_accs.avg))
        for i in range(images.shape[0]):
            if logits.argmax(1)[i] != labels[i] or logits_adv.argmax(1)[i] == labels[i]:
                continue
            grayscale_cam = cam_ori(input_tensor=images[i].unsqueeze(0),
                                target_category=labels[i].cpu().item())  # (1, 224, 224)
            grayscale_adv_cam = cam_ori(input_tensor=adv_images[i].unsqueeze(0),
                                    target_category=labels[i].cpu().item())  # (1, 224, 224)
            img = np.transpose(images[i].detach().cpu().numpy(), (1, 2, 0))
            plt.subplot(1, 2, 1)
            plt.imshow(show_cam_on_image(img, grayscale_cam[0, :], use_rgb=True))
            plt.subplot(1, 2, 2)
            plt.imshow(show_cam_on_image(img, grayscale_adv_cam[0, :], use_rgb=True))
            plt.show()
            grayscale_cam = cam(input_tensor=images[i].unsqueeze(0),
                                target_category=labels[i].cpu().item())  # (1, 224, 224)
            grayscale_adv_cam = cam(input_tensor=adv_images[i].unsqueeze(0),
                                    target_category=labels[i].cpu().item())  # (1, 224, 224)
            for idx in range(10):
                plt.subplot(4, 5, idx+1)
                visualization = show_cam_on_image(img, grayscale_cam[idx, :], use_rgb=True)
                plt.imshow(visualization)
            for idx in range(10):
                plt.subplot(4, 5, idx+1+10)
                visualization = show_cam_on_image(img, grayscale_adv_cam[idx, :], use_rgb=True)
                plt.imshow(visualization)
            plt.show()

if __name__ == '__main__':
    draw_cam("", "wide_resnet101_imagenet100_backbone", "")