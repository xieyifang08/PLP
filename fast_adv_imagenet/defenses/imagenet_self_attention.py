import glob
import os
import platform
import sys
import argparse

import PIL
import tqdm
import imageio
import pandas as pd
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from torch.utils import data
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn
import scipy
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
from copy import deepcopy
from sklearn.model_selection import train_test_split
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, VisdomLogger
from fast_adv.attacks import DDN
from fast_adv.utils.messageUtil import send_email
from fast_adv_imagenet.utils import model_utils

"""

"""

parser = argparse.ArgumentParser(description='CIFAR10 Training data augmentation')

parser.add_argument('--data', default='../data', help='path to dataset')
parser.add_argument('--img_size', default=224, type=int, help='size of image')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--cpu', action='store_true', help='force training on cpu')
parser.add_argument('--save-folder', '--sf', default='weights/wide_resnet101_selfAtt_minidata_AT_finetune_b20/',
                    help='folder to save state dicts')
parser.add_argument('--visdom_env', '--ve', type=str, default="wide_resnet101_selfAtt_minidata_AT_finetune_b20",
                    help='which env visdom is running.')

parser.add_argument('--save-freq', '--sfr', default=5, type=int, help='save frequency')
parser.add_argument('--save-name', '--sn', default='cifar10', help='name for saving the final state dict')

parser.add_argument('--batch-size', '-b', default=20, type=int, help='mini-batch size')
parser.add_argument('--epochs', '-e', default=50, type=int, help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, help='initial learning rate')
parser.add_argument('--lr-decay', '--lrd', default=0.9, type=float, help='decay for learning rate')
parser.add_argument('--lr-step', '--lrs', default=2, type=int, help='step size for learning rate decay')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, help='weight decay')
parser.add_argument('--drop', default=0.3, type=float, help='dropout rate of the classifier')

parser.add_argument('--adv', type=int, default=0, help='epoch to start training with adversarial images')
parser.add_argument('--max-norm', type=float, default=1, help='max norm for the adversarial perturbations')
parser.add_argument('--steps', default=10, type=int, help='number of steps for the attack')

parser.add_argument('--visdom', '--vdm', type=bool, default=False)
parser.add_argument('--visdom-port', '--vp', type=int, default=8097,
                    help='For visualization, which port visdom is running.')
parser.add_argument('--print-freq', '--pf', default=10, type=int, help='print frequency')

parser.add_argument('--num-attentions', '--na', default=32, type=int, help='number of attention maps')
parser.add_argument('--backbone-net', '--bn', default='wide_resnet', help='feature extractor')
parser.add_argument('--beta', '--b', default=5e-2, help='param for update feature centers')
args = parser.parse_args()
print(args)
if args.lr_step is None: args.lr_step = args.epochs

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')
CALLBACK = VisdomLogger(env=args.visdom_env, port=args.visdom_port) if args.visdom else None

if not os.path.exists(args.save_folder) and args.save_folder:
    os.makedirs(args.save_folder)


def load_data_for_defense(csv, input_dir, img_size=args.img_size, batch_size=args.batch_size):
    jir = pd.read_csv(csv)
    all_imgs = [os.path.join(input_dir, str(i)) for i in jir['ImageId'].tolist()]
    all_labels = jir['TrueLabel'].tolist()
    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})

    transformer = transforms.Compose([
        transforms.Resize((img_size, img_size)),
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


def load_data_from_tianchi(csv, input_dir, img_size=args.img_size, batch_size=args.batch_size):
    jir = pd.read_csv(csv)
    all_imgs = [os.path.join(input_dir, str(i)) for i in jir['ImageId'].tolist()]
    all_labels = jir['TrueLabel'].tolist()
    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})

    transformer = transforms.Compose([
        transforms.Resize((img_size, img_size)),
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


def prepare_data_from_tianchi(csv, input_dir, img_size=args.img_size, batch_size=args.batch_size):  # img_size
    jir = pd.read_csv(csv)
    all_imgs = [os.path.join(input_dir, str(i)) for i in jir['ImageId'].tolist()]
    all_labels = jir['TrueLabel'].tolist()
    train = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    train_data, val_data = train_test_split(train,
                                            stratify=train['label_idx'].values, train_size=0.9)
    transformer_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(img_size, (0.7, 1)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5],
        #                     std=[0.5, 0.5, 0.5]),
    ])
    transformer = transforms.Compose([
        transforms.Resize([img_size, img_size]),
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
                       # num_workers=8,
                       shuffle=True) for ds in datasets.keys()
    }
    return dataloaders


def prepareData(input_dir, img_size=args.img_size, batch_size=args.batch_size):  # img_size
    # all_paths = glob.glob(os.path.join(input_dir, './*'))
    all_imgs = glob.glob(os.path.join(input_dir, './*/*.JPEG'))
    # all_imgs = []
    # for folder in range(1000):
    #     single_imgs = glob.glob(os.path.join(input_dir, str(folder), '*.JPEG'))
    #     all_imgs.extend(single_imgs[:200])
    all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]
    train = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    train_data, val_data = train_test_split(train, stratify=train['label_idx'].values, train_size=0.4, test_size=0.1)
    input_size = int(img_size * 1.5)
    transformer_train = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transformer = transforms.Compose([
        transforms.Resize((input_size, input_size)),
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
        # b = Image.fromarray(imageio.imread(image_path))
        b = Image.open(image_path).convert('RGB')
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
print("platform is ", system)
if system == "Windows":
    loader = prepareData("C:/datasets/ILSVRC2012_img_train")
    train_loader = loader['train_data']
    val_loader = loader['val_data']
    # test_loader = load_data_for_defense("C:/datasets/ILSVRC2012_validation_ground_truth.txt", "C:/datasets/ILSVRC2012_img_val")['dev_data']
    test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), os.path.join(args.data, 'images'))[
        'dev_data']
else:
    # loader = prepareData("/mnt/u2/datasets/ILSVRC2012_img_train")
    # train_loader = loader['train_data']
    # val_loader = loader['val_data']
    # test_loader = load_data_for_defense(os.path.join("/mnt/u2/datasets", 'val.csv'),
    #                                     os.path.join("/mnt/u2/datasets", 'ILSVRC2012_img_val'))['dev_data']
    # 超算平台
    loader = prepareData("/seu_share/home/fiki/ff/database/mini_ILSVRC2012_img_train")
    train_loader = loader['train_data']
    val_loader = loader['val_data']
    test_loader = val_loader

print(len(train_loader), len(val_loader), len(test_loader))

model = model_utils.load_model("wide_resnet101_2_self_attention", pretrained=True).to(DEVICE)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
if args.adv is not None:
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_decay)
else:
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

attacker = DDN(steps=args.steps, device=DEVICE)

max_loss = torch.log(torch.tensor(1000.)).item()  # for callback
best_acc = 0
best_epoch = 0

valacc_final = 0

for epoch in range(args.epochs):
    cudnn.benchmark = True
    model.train()
    requires_grad_(model, True)
    accs = AverageMeter()
    total_losses = AverageMeter()
    adv_losses = AverageMeter()
    clean_losses = AverageMeter()
    attack_norms = AverageMeter()

    i = 0
    length = len(train_loader)
    for batch_data in tqdm.tqdm(train_loader):
        images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
        i += 1
        # 原图loss
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        clean_losses.append(loss.item())
        if args.adv is not None and epoch >= args.adv:
            model.eval()
            requires_grad_(model, False)
            adv = attacker.attack(model, images, labels)
            l2_norms = (adv - images).view(args.batch_size, -1).norm(2, 1)
            mean_norm = l2_norms.mean()
            if args.max_norm:
                adv = torch.renorm(adv - images, p=2, dim=0, maxnorm=args.max_norm) + images
            attack_norms.append(mean_norm.item())
            requires_grad_(model, True)
            model.train()
            logits_adv = model(adv.detach())

            loss_adv = F.cross_entropy(logits_adv, labels)
            adv_losses.append(loss_adv.item())
            loss = loss + loss_adv  # + 0.5*F.mse_loss(logits_adv,logits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        accs.append((logits.argmax(1) == labels).float().mean().item())
        total_losses.append(loss.item())
        if CALLBACK and not ((i + 1) % args.print_freq):
            CALLBACK.scalar('Tr_Loss', epoch + i / length, total_losses.last_avg)
            if args.adv is not None and epoch >= args.adv:
                CALLBACK.scalar('Tr_adv_Loss', epoch + i / length, adv_losses.last_avg)
            CALLBACK.scalar('Tr_clean_Loss', epoch + i / length, clean_losses.last_avg)
            CALLBACK.scalar('Tr_Acc', epoch + i / length, accs.last_avg)
            if args.adv is not None and epoch >= args.adv:
                CALLBACK.scalar('L2', epoch + i / length, attack_norms.last_avg)

    print('Epoch {} | Training | Loss: {:.4f}, Accs: {:.4f}'.format(epoch, total_losses.avg, accs.avg))

    cudnn.benchmark = False
    model.eval()
    requires_grad_(model, False)
    val_accs = AverageMeter()
    val_losses = AverageMeter()

    with torch.no_grad():
        for batch_data in tqdm.tqdm(val_loader):
            images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)

            logits = model(images)
            loss = F.cross_entropy(logits, labels)

            val_accs.append((logits.argmax(1) == labels).float().mean().item())
            val_losses.append(loss.item())

    if CALLBACK:
        CALLBACK.scalar('Val_Loss', epoch + 1, val_losses.last_avg)
        CALLBACK.scalar('Val_Acc', epoch + 1, val_accs.last_avg)

    print('Epoch {} | Validation | Loss: {:.4f}, Accs: {:.4f}'.format(epoch, val_losses.avg, val_accs.avg))

    save_path = args.save_folder
    if val_accs.avg >= best_acc:  # args.adv is None and
        best_acc = val_accs.avg
        best_epoch = epoch
        best_dict = deepcopy(model.state_dict())
        files2remove = glob.glob(os.path.join(save_path, 'best_*'))
        for _i in files2remove:
            os.remove(_i)
        strsave = "best_AT_ep_%d_val_acc%.4f.pth" % (epoch, best_acc)
        torch.save(model.cpu().state_dict(),
                   os.path.join(save_path, strsave))
        model.to(DEVICE)

    if args.adv is None and val_accs.avg >= best_acc:
        best_acc = val_accs.avg
        best_epoch = epoch
        best_dict = deepcopy(model.state_dict())

    if not (epoch + 1) % args.save_freq:
        save_checkpoint(
            model.state_dict(),
            os.path.join(args.save_folder, args.save_name + 'acc{}_{}.pth'.format(val_accs.avg, (epoch + 1))), cpu=True)
    valacc_final = val_accs.avg
    scheduler.step()

# if args.adv is None:
#     model.load_state_dict(best_dict)

test_accs = AverageMeter()
test_losses = AverageMeter()

model.eval()
with torch.no_grad():
    for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        test_accs.append((logits.argmax(1) == labels).float().mean().item())
        test_losses.append(loss.item())

if args.adv is not None:
    print('\nTest accuracy with final model: {:.4f} with loss: {:.4f}'.format(test_accs.avg, test_losses.avg))
else:
    print('\nTest accuracy with model from epoch {}: {:.4f} with loss: {:.4f}'.format(best_epoch,
                                                                                      test_accs.avg, test_losses.avg))

print('\nSaving model...')
save_checkpoint(model.state_dict(),
                os.path.join(args.save_folder, args.save_name + '_valacc' + str(valacc_final) + '.pth'), cpu=True)
CALLBACK.save([args.visdom_env])
