import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)

import glob
import argparse
import platform
import logging
import pandas as pd
from torchvision import transforms

from torch.utils.data import DataLoader, Dataset
from PIL import Image
import tqdm
import numpy as np
import torch
from fast_adv_imagenet.utils.messageUtil import send_email
from fast_adv_imagenet.utils import model_utils, AverageMeter
import imageio
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")
from torchattacks import FGSM, PGD, RFGSM, CW, DeepFool, MultiAttack, SparseFool, AutoAttack, OnePixel, MIFGSM, DIFGSM, \
    Square, GN
from fast_adv.attacks.tifgsm import TIFGSM

parser = argparse.ArgumentParser(description='Extend sample')
parser.add_argument('--max-norm', type=float, default=10, help='max norm for the adversarial perturbations')
parser.add_argument('--img_size', default=224, type=int, help='pic size')
parser.add_argument('--data', default='../data', help='path to dataset')
parser.add_argument('--attack_name', '--at', default='DDN',
                    help='name for saving the final state dict')
parser.add_argument('--batch-size', '-b', default=32, type=int, help='mini-batch size')

parser.add_argument("--clean", type=bool, default=True)
parser.add_argument("--gn", type=bool, default=False)
parser.add_argument("--fgsm", type=bool, default=False)
parser.add_argument("--rfgsm", type=bool, default=False)
parser.add_argument("--mifgsm", type=bool, default=False)
parser.add_argument("--difgsm", type=bool, default=False)
parser.add_argument("--tifgsm", type=bool, default=False)
parser.add_argument("--pgd", type=bool, default=False)
parser.add_argument("--square", type=bool, default=False)
parser.add_argument("--cw", type=bool, default=False)
parser.add_argument("--deepfool", type=bool, default=False)
parser.add_argument("--output_path", type=str, default="./advs_new")
parser.add_argument("--model_name", type=str, default="wide_resnet101_imagenet100_backbone")
parser.add_argument('--report_msg', '--rm', type=str, default="第三章攻击实验")
args = parser.parse_args()
print(args)

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
print(DEVICE)


def load_data_for_defense(input_dir, img_size=args.img_size, batch_size=args.batch_size):
    all_imgs = glob.glob(os.path.join(input_dir, './*/*.JPEG'))

    if platform.system() == "Windows":
        all_labels = [int(img_path.split('\\')[-2]) for img_path in all_imgs]
    else:
        all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]

    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})

    transformer = transforms.Compose([
        transforms.Resize((img_size, img_size)),
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


class ImageSet(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
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


data_loader = load_data_for_defense("C:/datasets/imagenet100_test")["dev_data"]
# data_loader = load_data_for_defense("/mnt/datasets/ILSVRC2012_img_val_formart/")["dev_data"]
logging.info("data_loader: {}".format(len(data_loader)))

model = model_utils.load_model(args.model_name, pretrained=True, num_classes=100).to(DEVICE).eval()

if args.clean is True:
    test_accs = AverageMeter()
    for i, batch_data in enumerate(tqdm.tqdm(data_loader, ncols=80)):
        images, labels, filenames = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        # logits = model(images.detach())
        # test_accs.append((logits.argmax(1) == labels).float().mean().item())
        for t in range(images.shape[0]):
            out_path = os.path.join(args.output_path, args.model_name, "clean", str(labels[t].item()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, str(args.batch_size * i + t) + '.png')
            img = np.transpose(images[t].detach().cpu().numpy(), (1, 2, 0))
            img = (img * 255.0).astype('uint8')
            imageio.imsave(out, img)
    # print('\nTest accuracy of %s' %test_accs.avg)

if args.gn is True:
    logging.info("create gn noise images")
    attack = GN(model)
    for i, batch_data in enumerate(tqdm.tqdm(data_loader, ncols=80)):
        images, labels, filenames = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        advs = attack(images, labels)
        for t in range(images.shape[0]):
            adv = np.transpose(advs[t].detach().cpu().numpy(), (1, 2, 0))
            out_path = os.path.join(args.output_path, args.model_name, "gn", str(labels[t].item()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, str(args.batch_size * i + t) + '.png')
            adv = (adv * 255.0).astype('uint8')
            imageio.imsave(out, adv)

if args.fgsm is True:
    logging.info("create fgsm images")
    attack = FGSM(model, eps=25/255)
    for i, batch_data in enumerate(tqdm.tqdm(data_loader, ncols=80)):
        images, labels, filenames = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        advs = attack(images, labels)
        for t in range(images.shape[0]):
            adv = np.transpose(advs[t].detach().cpu().numpy(), (1, 2, 0))
            out_path = os.path.join(args.output_path, args.model_name, "fgsm_eps25", str(labels[t].item()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, str(args.batch_size * i + t) + '.png')
            adv = (adv * 255.0).astype('uint8')
            imageio.imsave(out, adv)
if args.rfgsm is True:
    logging.info("create rfgsm images")
    attack = RFGSM(model, steps=30, eps=25 / 255)
    for i, batch_data in enumerate(tqdm.tqdm(data_loader, ncols=80)):
        images, labels, filenames = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        advs = attack(images, labels)
        for t in range(images.shape[0]):
            adv = np.transpose(advs[t].detach().cpu().numpy(), (1, 2, 0))
            out_path = os.path.join(args.output_path, args.model_name, "rfgsm_eps25_step30", str(labels[t].item()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, str(args.batch_size * i + t) + '.png')
            adv = (adv * 255.0).astype('uint8')
            imageio.imsave(out, adv)
if args.mifgsm is True:
    logging.info("create mifgsm images eps=20 / 255, steps=10")
    attack = MIFGSM(model, eps=25 / 255, steps=30)
    for i, batch_data in enumerate(tqdm.tqdm(data_loader, ncols=80)):
        images, labels, filenames = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        advs = attack(images, labels)
        for t in range(images.shape[0]):
            adv = np.transpose(advs[t].detach().cpu().numpy(), (1, 2, 0))
            out_path = os.path.join(args.output_path, args.model_name, "mifgsm_eps25_step30", str(labels[t].item()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, str(args.batch_size * i + t) + '.png')
            adv = (adv * 255.0).astype('uint8')
            imageio.imsave(out, adv)

if args.difgsm is True:
    logging.info("create difgsm images eps=20 / 255, steps=10")

    attack = DIFGSM(model, eps=25 / 255, steps=30)
    for i, batch_data in enumerate(tqdm.tqdm(data_loader, ncols=80)):
        images, labels, filenames = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        advs = attack(images, labels)
        for t in range(images.shape[0]):
            adv = np.transpose(advs[t].detach().cpu().numpy(), (1, 2, 0))
            out_path = os.path.join(args.output_path, args.model_name, "difgsm_eps25_step30", str(labels[t].item()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, str(args.batch_size * i + t) + '.png')
            adv = (adv * 255.0).astype('uint8')
            imageio.imsave(out, adv)
if args.tifgsm is True:
    logging.info("create tifgsm images eps=20 / 255, steps=10")

    attack = TIFGSM(model, eps=25 / 255, steps=30)
    for i, batch_data in enumerate(tqdm.tqdm(data_loader, ncols=80)):
        images, labels, filenames = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        advs = attack(images, labels)
        for t in range(images.shape[0]):
            adv = np.transpose(advs[t].detach().cpu().numpy(), (1, 2, 0))
            out_path = os.path.join(args.output_path, args.model_name, "tifgsm_eps25_step30", str(labels[t].item()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, str(args.batch_size * i + t) + '.png')
            adv = (adv * 255.0).astype('uint8')
            imageio.imsave(out, adv)
if args.pgd is True:
    logging.info("create pgd images eps=20/255, steps=10")
    attack = PGD(model, eps=25 / 255, steps=30)
    for i, batch_data in enumerate(tqdm.tqdm(data_loader, ncols=80)):
        images, labels, filenames = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        advs = attack(images, labels)
        for t in range(images.shape[0]):
            adv = np.transpose(advs[t].detach().cpu().numpy(), (1, 2, 0))
            out_path = os.path.join(args.output_path, args.model_name, "pgd_eps25_step30", str(labels[t].item()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, str(args.batch_size * i + t) + '.png')
            adv = (adv * 255.0).astype('uint8')
            imageio.imsave(out, adv)

if args.square is True:
    logging.info("create square images eps=10/255")

    attack = Square(model, eps=25 / 255)
    for i, batch_data in enumerate(tqdm.tqdm(data_loader, ncols=80)):
        images, labels, filenames = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        advs = attack(images, labels)
        for t in range(images.shape[0]):
            adv = np.transpose(advs[t].detach().cpu().numpy(), (1, 2, 0))
            out_path = os.path.join(args.output_path, args.model_name, "square_eps25", str(labels[t].item()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, str(args.batch_size * i + t) + '.png')
            adv = (adv * 255.0).astype('uint8')
            imageio.imsave(out, adv)
if args.cw is True:
    logging.info("create cw images")
    attack = CW(model, c=100, steps=300)
    for i, batch_data in enumerate(tqdm.tqdm(data_loader, ncols=80)):
        images, labels, filenames = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        advs = attack(images, labels)
        for t in range(images.shape[0]):
            adv = np.transpose(advs[t].detach().cpu().numpy(), (1, 2, 0))
            out_path = os.path.join(args.output_path, args.model_name, "cw_c100_step300", str(labels[t].item()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, str(args.batch_size * i + t) + '.png')
            adv = (adv * 255.0).astype('uint8')
            imageio.imsave(out, adv)


if args.deepfool is True:
    logging.info("create deepfool images")

    attack = DeepFool(model, overshoot=1)
    for i, batch_data in enumerate(tqdm.tqdm(data_loader, ncols=80)):
        images, labels, filenames = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        advs = attack(images, labels)
        for t in range(images.shape[0]):
            adv = np.transpose(advs[t].detach().cpu().numpy(), (1, 2, 0))
            out_path = os.path.join(args.output_path, args.model_name, "deepfool_overshoot1", str(labels[t].item()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = os.path.join(out_path, str(args.batch_size * i + t) + '.png')
            adv = (adv * 255.0).astype('uint8')
            imageio.imsave(out, adv)

send_email("{}".format(args.report_msg), title="imagenet100对抗样本生成完毕")
