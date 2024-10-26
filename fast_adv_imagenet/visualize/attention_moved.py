import csv
import glob
import json
import os
import sys
import argparse

import PIL
import cv2
import tqdm
from progressbar import *
import imageio
import pandas as pd
import random
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils import data
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn
import scipy
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode

BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
from copy import deepcopy
from sklearn.model_selection import train_test_split
from fast_adv.models.cifar10.model_attention import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN
from fast_adv.utils.messageUtil import send_email
from fast_adv_imagenet.utils import model_utils
from fast_adv_imagenet.process.gaussian_filter import GaussianBlurConv

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

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() and not args.cpu) else 'cpu')

if not os.path.exists(args.save_folder) and args.save_folder:
    os.makedirs(args.save_folder)

image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def load_data_for_defense(csv, input_dir, img_size=args.img_size, batch_size=args.batch_size):
    jir = pd.read_csv(csv)
    all_imgs = [os.path.join(input_dir, str(i)) for i in jir['ImageId'].tolist()]
    all_labels = jir['TrueLabel'].tolist()
    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})

    transformer = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
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


def draw_cam(database_path, model_name, out_path):
    test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), database_path)['dev_data']

    model = model_utils.load_model(model_name).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    cam = GradCAM(model=model, target_layers=[model.model.layer1[-1]])

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    test_accs = AverageMeter()
    correct_classify = []
    with torch.no_grad():
        model.eval()
        for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
            images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
                'filename']
            logits = model(images)
            for i in range(images.shape[0]):
                if logits.argmax(1)[i] == labels[i]:
                    correct_classify.append(filename[i])
            test_accs.append((logits.argmax(1) == labels).float().mean().item())
    print(test_accs.avg)
    for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        for i in range(images.shape[0]):
            if filename[i] in correct_classify:
                grayscale_cam = cam(input_tensor=images[i].unsqueeze(0), target_category=labels[i].cpu().item())
                plt.imshow(grayscale_cam)
                plt.show()
                img = np.transpose(images[i].detach().cpu().numpy(), (1, 2, 0))
                visualization = show_cam_on_image(img, grayscale_cam[0, :], use_rgb=True)
                out_path_img = out_path + "/images"
                if not os.path.exists(out_path_img):
                    os.makedirs(out_path_img)
                out_path_npy = out_path + "/npys"
                if not os.path.exists(out_path_npy):
                    os.makedirs(out_path_npy)
                out = os.path.join(out_path_img, filename[i])
                np.save(os.path.join(out_path_npy, filename[i]), grayscale_cam)
                imageio.imsave(out, visualization)


def get_clean_classfiyTrue_list(model_name, database_path):
    test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), database_path)['dev_data']

    model = model_utils.load_model(model_name).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    test_accs = AverageMeter()
    correct_classify = []
    with torch.no_grad():
        model.eval()
        for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
            images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), \
                                       batch_data['filename']
            logits = model(images)
            for i in range(images.shape[0]):
                if logits.argmax(1)[i] == labels[i]:
                    correct_classify.append(filename[i])
            test_accs.append((logits.argmax(1) == labels).float().mean().item())
    print(test_accs.avg)
    return correct_classify


def draw_clean_classfiyTrue_cam(correct_classify, model_name, database_path, out_path):
    test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), database_path)['dev_data']

    model = model_utils.load_model(model_name).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    cam = GradCAM(model=model, target_layers=[model.model.layer4[-1]])

    for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        for i in range(images.shape[0]):
            if filename[i] in correct_classify:
                grayscale_cam = cam(input_tensor=images[i].unsqueeze(0), target_category=labels[i].cpu().item())
                img = np.transpose(images[i].detach().cpu().numpy(), (1, 2, 0))
                visualization = show_cam_on_image(img, grayscale_cam[0, :], use_rgb=True)
                out_path_img = out_path + "/images"
                if not os.path.exists(out_path_img):
                    os.makedirs(out_path_img)
                out_path_npy = out_path + "/npys"
                if not os.path.exists(out_path_npy):
                    os.makedirs(out_path_npy)
                out = os.path.join(out_path_img, filename[i])
                np.save(os.path.join(out_path_npy, filename[i]), grayscale_cam)
                imageio.imsave(out, visualization)


def static_att_moved_list(correct_classify, model_name, adv_path, var_mean, csv_path):
    adv_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), adv_path)['dev_data']

    model = model_utils.load_model(model_name).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    # todo: 读取csv为集合，每次准备讲adv的图片号写入res时，从csv集合中取出var值和均值比较再分类
    # out = open(csv_path, "r", newline='')
    data = pd.read_csv(csv_path)  # 读取文件中所有数据
    # 按列分离数据
    adv_filename_dict = data[["image name", csv_path.split("/")[-2]]]
    adv_filename_var_map = {}
    for i in range(len(adv_filename_dict["image name"])):
        adv_filename_var_map[adv_filename_dict["image name"][i]] = adv_filename_dict[csv_path.split("/")[-2]][i]
    res = {"defence_success_upper": [], "defence_success_down": [], "defence_fail_upper": [], "defence_fail_down": []}
    # print(list(adv_filename_var_map.keys()))
    for i, batch_data in enumerate(tqdm.tqdm(adv_loader, ncols=80)):
        images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        adv_logits = model(images)
        for i in range(images.shape[0]):
            if filename[i] in correct_classify and filename[i] + ".npy" in list(adv_filename_var_map.keys()):
                adv_var = adv_filename_var_map[filename[i] + ".npy"]
                if adv_logits.argmax(1)[i] == labels[i]:
                    if adv_var > var_mean:
                        res["defence_success_upper"].append(filename[i])
                    else:
                        res["defence_success_down"].append(filename[i])
                else:
                    if adv_var > var_mean:
                        res["defence_fail_upper"].append(filename[i])
                    else:
                        res["defence_fail_down"].append(filename[i])
    return res


def draw_lastClassfiyTrue_cam(database_path, model_names, out_path):
    test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), database_path)['dev_data']
    correct_classify = [[], [], []]
    models = []
    for m in range(len(model_names)):
        model = model_utils.load_model(model_names[m]).to(DEVICE)
        models.append(model)
        test_accs = AverageMeter()
        with torch.no_grad():
            model.eval()
            for _, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
                images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), \
                                           batch_data['filename']
                logits = model(images)
                for i in range(images.shape[0]):
                    if logits.argmax(1)[i] == labels[i]:
                        correct_classify[m].append(filename[i])
                test_accs.append((logits.argmax(1) == labels).float().mean().item())
        print(test_accs.avg)
    cams = []
    for model in models:
        cams.append(GradCAM(model=model, target_layers=[model.model.layer4[-1]]))

    for _, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        for i in range(images.shape[0]):
            if filename[i] in correct_classify[-1] and filename[i] not in correct_classify[0]:
                for j in range(len(cams)):
                    grayscale_cam = cams[j](input_tensor=images[i].unsqueeze(0), target_category=labels[i].cpu().item())
                    img = np.transpose(images[i].detach().cpu().numpy(), (1, 2, 0))
                    visualization = show_cam_on_image(img, grayscale_cam[0, :], use_rgb=True)
                    out_path_img = out_path + "/" + model_names[j] + "/images"
                    if not os.path.exists(out_path_img):
                        os.makedirs(out_path_img)
                    out_path_npy = out_path + "/" + model_names[j] + "/npys"
                    if not os.path.exists(out_path_npy):
                        os.makedirs(out_path_npy)
                    out = os.path.join(out_path_img, filename[i])
                    np.save(os.path.join(out_path_npy, filename[i]), grayscale_cam)
                    imageio.imsave(out, visualization)


def input_diversity(image, low=270, high=299):
    rnd = random.randint(low, high)
    rescaled = F.interpolate(image, size=[rnd, rnd], mode='bilinear')
    h_rem = high - rnd
    w_rem = high - rnd
    pad_top = random.randint(0, h_rem)
    pad_bottom = h_rem - pad_top
    pad_left = random.randint(0, w_rem)
    pad_right = w_rem - pad_left
    padded = F.pad(rescaled, [pad_top, pad_bottom, pad_left, pad_right], 'constant', 0)
    return padded


def draw_single_cam(img_path, model, true_label, out_path, filename, img_size=args.img_size):
    image = Image.open(img_path)  # 读取图片
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
    ])
    image = train_transform(image)
    image = transforms.ToTensor()(image).unsqueeze(0).to(DEVICE)

    image = input_diversity(image, 180, img_size)

    cam = GradCAM(model=model, target_layers=[model.model.layer4[-1]])
    grayscale_cam = cam(input_tensor=image, target_category=true_label)
    print(grayscale_cam.shape)
    print(grayscale_cam.mean())
    img = np.transpose(image[0].detach().cpu().numpy(), (1, 2, 0))
    visualization = show_cam_on_image(img, grayscale_cam[0, :], use_rgb=True)
    print(visualization.shape)
    plt.imshow(visualization / 255.0)
    plt.title("wrn101_" + str(img_size), y=-0.2)
    plt.xticks([])
    plt.yticks([])
    plt.show()
    return
    out_path_img = out_path + "/images"
    if not os.path.exists(out_path_img):
        os.makedirs(out_path_img)
    out_path_npy = out_path + "/" + "/npys"
    if not os.path.exists(out_path_npy):
        os.makedirs(out_path_npy)
    out = os.path.join(out_path_img, filename)
    np.save(os.path.join(out_path_npy, filename), grayscale_cam)
    imageio.imsave(out, visualization)


def statistic_cam(cam_input, csv_path):
    all_cam_npys = glob.glob(os.path.join(cam_input, './*.npy'))
    if os.path.exists(csv_path):
        os.remove(csv_path)
    out = open(csv_path, "w", newline='')
    csv_writer = csv.writer(out, dialect="excel")
    csv_writer.writerow(["image name", cam_input.split("/")[-2]])
    ans = np.float(0)
    for org_path in tqdm.tqdm(all_cam_npys):
        npy_name = org_path.split("\\.\\")[-1]
        npy_mean = np.load(org_path).mean()
        csv_writer.writerow([npy_name, npy_mean])
        # img = np.load(npy_path)
        ans += npy_mean
    return ans / len(all_cam_npys)


def draw_fail_classify_cam(model_name, image_path, true_label, out_path, filename, classify_success=True):
    model = model_utils.load_model(model_name).to(DEVICE).eval()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    cam = GradCAM(model=model, target_layers=[model.model.layer4[-1]])

    img = Image.open(image_path)  # 读取图片
    img = transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR)(img)

    image = transforms.ToTensor()(img).unsqueeze(0).to(DEVICE)  # PILImage->tensor
    grayscale_cam = cam(input_tensor=image, target_category=true_label)
    img = np.array(img, dtype=np.float32)
    logits = model(image)
    logits = torch.softmax(logits, dim=1)
    # print("[{}:{}]".format(logits.argmax(1).item(), format(logits.max().item(), '.4f')))

    h_x = logits.data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()
    for i in range(0, 20):
        print('{} : {:.3f}'.format(idx[i], probs[i]))
    print()
    if not classify_success:
        visualization = show_cam_on_image(img / 255.0, grayscale_cam[0, :], use_rgb=True)
        out_path_img = out_path + "/images"
        if not os.path.exists(out_path_img):
            os.makedirs(out_path_img)
        out_path_npy = out_path + "/npys"
        if not os.path.exists(out_path_npy):
            os.makedirs(out_path_npy)
        out = os.path.join(out_path_img, filename)
        np.save(os.path.join(out_path_npy, filename), grayscale_cam)
        imageio.imsave(out, visualization)
    return logits.argmax(1).item()


def statistic_clean_adv_cam(input_names, input_path, csv_path):
    out = open(csv_path, "a")
    csv_writer = csv.writer(out, dialect="excel")
    input_names.insert(0, "name")
    csv_writer.writerow(input_names)

    all_npys = glob.glob(os.path.join(input_path[0], './*.npy'))
    for org_path in tqdm.tqdm(all_npys):
        row = []
        npy_name = org_path.split("\\.\\")[-1]
        row.append(npy_name)
        for i in range(len(input_path)):
            row.append(np.load(input_path[i] + "/" + npy_name).mean())
        csv_writer.writerow(row)


def draw_cleanClassfiyTrue_lastClassfiyTrue_cam(database_path, model_names, out_path):
    clean_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), os.path.join(args.data, 'images'))[
        'dev_data']
    test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), database_path)['dev_data']
    correct_classify = [[], [], []]
    clean_correct_classify = [[], [], []]
    models = []
    for m in range(len(model_names)):
        model = model_utils.load_model(model_names[m]).to(DEVICE)
        models.append(model)
        test_accs = AverageMeter()
        with torch.no_grad():
            model.eval()
            for batch_data, clean_batch_data in tqdm.tqdm(zip(test_loader, clean_loader), ncols=80):
                images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), \
                                           batch_data['filename']
                clean_images = clean_batch_data['image'].to(DEVICE)
                logits = model(images)
                clean_logits = model(clean_images)
                for i in range(images.shape[0]):
                    if logits.argmax(1)[i] == labels[i]:
                        correct_classify[m].append(filename[i])
                    if clean_logits.argmax(1)[i] == labels[i]:
                        clean_correct_classify[m].append(filename[i])
                test_accs.append((logits.argmax(1) == labels).float().mean().item())
        print(test_accs.avg)
    cams = []
    for model in models:
        cams.append(GradCAM(model=model, target_layers=[model.model.layer4[-1]]))

    for batch_data, clean_batch_data in tqdm.tqdm(zip(test_loader, clean_loader), ncols=80):
        images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        clean_images = clean_batch_data['image'].to(DEVICE)
        for i in range(images.shape[0]):
            if filename[i] in clean_correct_classify[0] and filename[i] in clean_correct_classify[1] and filename[i] in \
                    clean_correct_classify[2]:
                if filename[i] in correct_classify[-1] and filename[i] not in correct_classify[0]:
                    for j in range(len(cams)):
                        clean_grayscale_cam = cams[j](input_tensor=clean_images[i].unsqueeze(0),
                                                      target_category=labels[i].cpu().item())
                        img = np.transpose(clean_images[i].detach().cpu().numpy(), (1, 2, 0))
                        clean_visualization = show_cam_on_image(img, clean_grayscale_cam[0, :], use_rgb=True)
                        out_path_img = out_path + "/" + model_names[j] + "/clean/images"
                        if not os.path.exists(out_path_img):
                            os.makedirs(out_path_img)
                        out_path_npy = out_path + "/" + model_names[j] + "/clean/npys"
                        if not os.path.exists(out_path_npy):
                            os.makedirs(out_path_npy)
                        out = os.path.join(out_path_img, filename[i])
                        np.save(os.path.join(out_path_npy, filename[i]), clean_grayscale_cam)
                        imageio.imsave(out, clean_visualization)

                    for j in range(len(cams)):
                        grayscale_cam = cams[j](input_tensor=images[i].unsqueeze(0),
                                                target_category=labels[i].cpu().item())
                        img = np.transpose(images[i].detach().cpu().numpy(), (1, 2, 0))
                        visualization = show_cam_on_image(img, grayscale_cam[0, :], use_rgb=True)
                        out_path_img = out_path + "/" + model_names[j] + "/images"
                        if not os.path.exists(out_path_img):
                            os.makedirs(out_path_img)
                        out_path_npy = out_path + "/" + model_names[j] + "/npys"
                        if not os.path.exists(out_path_npy):
                            os.makedirs(out_path_npy)
                        out = os.path.join(out_path_img, filename[i])
                        np.save(os.path.join(out_path_npy, filename[i]), grayscale_cam)
                        imageio.imsave(out, visualization)


def statistic_KL_mean(csv_path, input_names, input_org, input_paths):
    all_org_npys = glob.glob(os.path.join(input_org, './*.npy'))
    out = open(csv_path, "a", newline='')
    csv_writer = csv.writer(out, dialect="excel")
    csv_writer.writerow(input_names)

    for org_path in tqdm.tqdm(all_org_npys):
        row = []
        npy_name = org_path.split("/./")[-1]
        a = torch.from_numpy(np.load(org_path))
        row.append(npy_name)
        for i in range(len(input_paths)):
            b = torch.from_numpy(np.load(input_paths[i] + "/./" + npy_name))
            kl = F.kl_div(a.softmax(dim=-1).log(), b.softmax(dim=-1), reduction='sum')
            row.append(kl.item())
        csv_writer.writerow(row)


def statistic_classfiy(csv_path, input_org, input_paths, model_names, input_titles):
    out = open(csv_path + "0.csv", "a", newline='')
    csv_writer = csv.writer(out, dialect="excel")
    result = [[], [], [], [], [], [], [], []]
    correct_classify = []
    test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), input_org)['dev_data']
    model = model_utils.load_model(model_names[0]).to(DEVICE).eval()
    for _, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        logits = model(images)
        logits = torch.softmax(logits, dim=1)
        # row.append(logits.argmax(1).item())
        # row.append(logits.max().item())
        # if "1004.png" in filename:
        #     for i in range(len(filename)):
        #         if "1004.png" == filename[i]:
        #             print(filename[i],logits.argmax(1)[i].item(), logits[i].max().item())
        #     break

        for j in range(images.shape[0]):
            if logits.argmax(1)[j] == labels[j]:
                csv_writer.writerow([filename[j], labels[j].cpu().item()])
                correct_classify.append(filename[j])
                result[0].append(logits.argmax(1)[j].item())
                result[1].append(logits[j].max().item())
    for i in range(len(input_paths)):
        test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), input_paths[i])['dev_data']
        model = model_utils.load_model(model_names[i]).to(DEVICE).eval()
        for _, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
            images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
                'filename']
            logits = model(images)
            logits = torch.softmax(logits, dim=1)
            for j in range(images.shape[0]):
                if filename[j] in correct_classify:
                    result[(i + 1) * 2].append(logits.argmax(1)[j].item())
                    result[(i + 1) * 2 + 1].append(logits[j].max().item())
    out = open(csv_path, "a", newline='')
    csv_writer = csv.writer(out, dialect="excel")
    csv_writer.writerow(input_titles)
    for j in range(len(result[0])):
        row = []
        for i in range(len(result)):
            row.append(result[i][j])
        csv_writer.writerow(row)


def analyse_classfiy_kl(csv_path, avg_kl):
    out = open(csv_path, "r", newline='')
    csv_reader = csv.DictReader(out)
    result = {
        "model1_upkl_classfiy_true": 0, "model1_upkl_classfiy_false": 0, "model1_downkl_classfiy_true": 0,
        "model1_downkl_classfiy_false": 0,
        "model2_upkl_classfiy_true": 0, "model2_upkl_classfiy_false": 0, "model2_downkl_classfiy_true": 0,
        "model2_downkl_classfiy_false": 0,
        "model3_upkl_classfiy_true": 0, "model3_upkl_classfiy_false": 0, "model3_downkl_classfiy_true": 0,
        "model3_downkl_classfiy_false": 0,
    }
    for row in csv_reader:
        # name, model1_kl, model2_kl, model3_kl, clean_classfiy, clean_confidence, model1_classfiy, model1_confidence, model2_classfiy, model2_confidence, model3_classfiy, model3_confidence
        if float(row["model1-kl"]) > avg_kl[0]:
            if row["model1_classfiy"] == row["clean_classfiy"]:
                result["model1_upkl_classfiy_true"] += 1
            else:
                result["model1_upkl_classfiy_false"] += 1
        else:
            if row["model1_classfiy"] == row["clean_classfiy"]:
                result["model1_downkl_classfiy_true"] += 1
            else:
                result["model1_downkl_classfiy_false"] += 1

        if float(row["model2-kl"]) > avg_kl[1]:
            if row["model2_classfiy"] == row["clean_classfiy"]:
                result["model2_upkl_classfiy_true"] += 1
            else:
                result["model2_upkl_classfiy_false"] += 1
        else:
            if row["model2_classfiy"] == row["clean_classfiy"]:
                result["model2_downkl_classfiy_true"] += 1
            else:
                result["model2_downkl_classfiy_false"] += 1

        if float(row["model3-kl"]) > avg_kl[2]:
            if row["model3_classfiy"] == row["clean_classfiy"]:
                result["model3_upkl_classfiy_true"] += 1
            else:
                result["model3_upkl_classfiy_false"] += 1
        else:
            if row["model3_classfiy"] == row["clean_classfiy"]:
                result["model3_downkl_classfiy_true"] += 1
            else:
                result["model3_downkl_classfiy_false"] += 1
    print(result)


def analyse_classfiy_var(csv_path, avg_var):
    out = open(csv_path, "r", newline='')
    csv_reader = csv.DictReader(out)
    result = {
        "model1_upvar_classfiy_true": 0, "model1_upvar_classfiy_false": 0, "model1_downvar_classfiy_true": 0,
        "model1_downvar_classfiy_false": 0,
        "model2_upvar_classfiy_true": 0, "model2_upvar_classfiy_false": 0, "model2_downvar_classfiy_true": 0,
        "model2_downvar_classfiy_false": 0,
        "model3_upvar_classfiy_true": 0, "model3_upvar_classfiy_false": 0, "model3_downvar_classfiy_true": 0,
        "model3_downvar_classfiy_false": 0,
    }
    msg = []
    name_model1_down_false = []
    name_model1_down_true = []
    name_model1_up_false = []
    name_model1_up_true = []
    name_model2_down_false = []
    name_model2_down_true = []
    name_model2_up_false = []
    name_model2_up_true = []

    name_downFalse2downTrue = []
    name_downFalse2upTrue = []
    name_upFalse2downTrue = []
    name_upFalse2upTrue = []

    name_downFalse2downFalse = []
    name_downFalse2upFalse = []
    name_upFalse2downFalse = []
    name_upFalse2upFalse = []

    name_downTrue2downFalse = []
    name_downTrue2upFalse = []
    name_upTrue2downFalse = []
    name_upTrue2upFalse = []

    for row in csv_reader:
        # name, model1_var, model2_var, model3_var, clean_classfiy, clean_confidence, model1_classfiy, model1_confidence, model2_classfiy, model2_confidence, model3_classfiy, model3_confidence
        if float(row["model1-var"]) > avg_var[0]:
            if row["model1_classfiy"] == row["clean_classfiy"]:
                result["model1_upvar_classfiy_true"] += 1
                name_model1_up_true.append(row["name"])
                msg.append(row["model1-var"])
            else:
                result["model1_upvar_classfiy_false"] += 1
                name_model1_up_false.append(row["name"])
        else:
            if row["model1_classfiy"] == row["clean_classfiy"]:
                result["model1_downvar_classfiy_true"] += 1
                name_model1_down_true.append(row["name"])
            else:
                result["model1_downvar_classfiy_false"] += 1
                name_model1_down_false.append(row["name"])

        if float(row["model2-var"]) > avg_var[1]:
            if row["model2_classfiy"] == row["clean_classfiy"]:
                result["model2_upvar_classfiy_true"] += 1
                name_model2_up_true.append(row["name"])
            else:
                result["model2_upvar_classfiy_false"] += 1
                name_model2_up_false.append(row["name"])
        else:
            if row["model2_classfiy"] == row["clean_classfiy"]:
                result["model2_downvar_classfiy_true"] += 1
                name_model2_down_true.append(row["name"])
            else:
                result["model2_downvar_classfiy_false"] += 1
                name_model2_down_false.append(row["name"])

        if float(row["model3-var"]) > avg_var[2]:
            if row["model3_classfiy"] == row["clean_classfiy"]:
                result["model3_upvar_classfiy_true"] += 1
            else:
                result["model3_upvar_classfiy_false"] += 1
        else:
            if row["model3_classfiy"] == row["clean_classfiy"]:
                result["model3_downvar_classfiy_true"] += 1
            else:
                result["model3_downvar_classfiy_false"] += 1

    for img_name in name_model1_down_false:
        if img_name in name_model2_down_true:
            name_downFalse2downTrue.append(img_name)
        elif img_name in name_model2_up_true:
            name_downFalse2upTrue.append(img_name)
        elif img_name in name_model2_down_false:
            name_downFalse2downFalse.append(img_name)
        elif img_name in name_model2_up_false:
            name_downFalse2upFalse.append(img_name)
    for img_name in name_model1_up_false:
        if img_name in name_model2_down_true:
            name_upFalse2downTrue.append(img_name)
        elif img_name in name_model2_up_true:
            name_upFalse2upTrue.append(img_name)
        elif img_name in name_model2_down_false:
            name_upFalse2downFalse.append(img_name)
        elif img_name in name_model2_up_false:
            name_upFalse2upFalse.append(img_name)

    for img_name in name_model1_down_true:
        if img_name in name_model2_down_false:
            name_downTrue2downFalse.append(img_name)
        elif img_name in name_model2_up_false:
            name_downTrue2upFalse.append(img_name)
    for img_name in name_model1_up_true:
        if img_name in name_model2_down_false:
            name_upTrue2downFalse.append(img_name)
        elif img_name in name_model2_up_false:
            name_upTrue2upFalse.append(img_name)

    print(
        "name_upFalse2upFalse = {}\nname_upFalse2downFalse={}\nname_downFalse2downFalse={}\nname_downFalse2upFalse={}\n"
            .format(name_upFalse2upFalse, name_upFalse2downFalse, name_downFalse2downFalse, name_downFalse2upFalse))
    print("name_upFalse2upTrue = {}\nname_upFalse2downTrue={}\nname_downFalse2downTrue={}\nname_downFalse2upTrue={}\n"
          .format(name_upFalse2upTrue, name_upFalse2downTrue, name_downFalse2downTrue, name_downFalse2upTrue))

    print("name_upTrue2upFalse = {}\nname_upTrue2downFalse={}\nname_downTrue2downFalse={}\nname_downTrue2upFalse={}\n"
          .format(name_upTrue2upFalse, name_upTrue2downFalse, name_downTrue2downFalse, name_downTrue2upFalse))

    print(
        "name_upFalse2upFalse = {}\nname_upFalse2downFalse={}\nname_downFalse2downFalse={}\nname_downFalse2upFalse={}\n"
            .format(len(name_upFalse2upFalse), len(name_upFalse2downFalse), len(name_downFalse2downFalse),
                    len(name_downFalse2upFalse)))
    print("name_upFalse2upTrue = {}\nname_upFalse2downTrue={}\nname_downFalse2downTrue={}\nname_downFalse2upTrue={}\n"
          .format(len(name_upFalse2upTrue), len(name_upFalse2downTrue), len(name_downFalse2downTrue),
                  len(name_downFalse2upTrue)))

    print("name_upTrue2upFalse = {}\nname_upTrue2downFalse={}\nname_downTrue2downFalse={}\nname_downTrue2upFalse={}\n"
          .format(len(name_upTrue2upFalse), len(name_upTrue2downFalse), len(name_downTrue2downFalse),
                  len(name_downTrue2upFalse)))


def cam_cal_single_var(npy_path1, npy_path2):
    npy1 = np.load(npy_path1)
    npy2 = np.load(npy_path2)
    print("({},{}),({},{}),({},{})".format(npy1.max(), npy1.min(), npy2.max(), npy2.min(), (npy1 - npy2).max(),
                                           (npy1 - npy2).min()))
    print((npy1 - npy2).var(), (npy1 - npy2).std())


def analyse_classfiy_std(csv_path, avg_std):
    out = open(csv_path, "r", newline='')
    csv_reader = csv.DictReader(out)
    result = {
        "model1_upstd_classfiy_true": 0, "model1_upstd_classfiy_false": 0, "model1_downstd_classfiy_true": 0,
        "model1_downstd_classfiy_false": 0,
        "model2_upstd_classfiy_true": 0, "model2_upstd_classfiy_false": 0, "model2_downstd_classfiy_true": 0,
        "model2_downstd_classfiy_false": 0,
        "model3_upstd_classfiy_true": 0, "model3_upstd_classfiy_false": 0, "model3_downstd_classfiy_true": 0,
        "model3_downstd_classfiy_false": 0,
    }
    for row in csv_reader:
        # name, model1_std, model2_std, model3_std, clean_classfiy, clean_confidence, model1_classfiy, model1_confidence, model2_classfiy, model2_confidence, model3_classfiy, model3_confidence
        if float(row["model1-std"]) > avg_std[0]:
            if row["model1_classfiy"] == row["clean_classfiy"]:
                result["model1_upstd_classfiy_true"] += 1
            else:
                result["model1_upstd_classfiy_false"] += 1
        else:
            if row["model1_classfiy"] == row["clean_classfiy"]:
                result["model1_downstd_classfiy_true"] += 1
            else:
                result["model1_downstd_classfiy_false"] += 1

        if float(row["model2-std"]) > avg_std[1]:
            if row["model2_classfiy"] == row["clean_classfiy"]:
                result["model2_upstd_classfiy_true"] += 1
            else:
                result["model2_upstd_classfiy_false"] += 1
        else:
            if row["model2_classfiy"] == row["clean_classfiy"]:
                result["model2_downstd_classfiy_true"] += 1
            else:
                result["model2_downstd_classfiy_false"] += 1

        if float(row["model3-std"]) > avg_std[2]:
            if row["model3_classfiy"] == row["clean_classfiy"]:
                result["model3_upstd_classfiy_true"] += 1
            else:
                result["model3_upstd_classfiy_false"] += 1
        else:
            if row["model3_classfiy"] == row["clean_classfiy"]:
                result["model3_downstd_classfiy_true"] += 1
            else:
                result["model3_downstd_classfiy_false"] += 1
    print(result)


def cam_cal_classfiy(model_name, img_path):
    test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), img_path)['dev_data']
    model = model_utils.load_model(model_name).to(DEVICE).eval()
    val_accs = AverageMeter()
    for _, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        logits = model(images)
        val_accs.append((logits.argmax(1) == labels).float().mean().item())
    print(img_path, val_accs.avg)


def analyse_attention_variance(csv_path, input_names, input_org, input_paths):
    all_org_npys = glob.glob(os.path.join(input_org, './*.npy'))
    out = open(csv_path, "a", newline='')
    csv_writer = csv.writer(out, dialect="excel")
    csv_writer.writerow(input_names)

    for org_path in tqdm.tqdm(all_org_npys):
        row = []
        npy_name = org_path.split("/./")[-1]
        a = torch.from_numpy(np.load(org_path))
        row.append(npy_name)
        for i in range(len(input_paths)):
            b = torch.from_numpy(np.load(input_paths[i] + "/./" + npy_name))
            st = (a - b).var()
            if npy_name == "1821.png.npy":
                print(npy_name, st.item())
            row.append(st.item())
        csv_writer.writerow(row)


def draw_clean_classfiyTrue_cam_filter(correct_classify, model_name, database_path, out_path):
    test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), database_path)['dev_data']

    model = model_utils.load_model(model_name).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    cam = GradCAM(model=model, target_layers=[model.model.layer4[-1]])

    for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        for i in range(images.shape[0]):
            if filename[i] in correct_classify:
                grayscale_cam = cam(input_tensor=images[i].unsqueeze(0), target_category=labels[i].cpu().item())
                img = np.transpose(images[i].detach().cpu().numpy(), (1, 2, 0))
                visualization = show_cam_on_image(img, grayscale_cam[0, :], use_rgb=True)
                out_path_img = out_path + "/images"
                if not os.path.exists(out_path_img):
                    os.makedirs(out_path_img)
                out_path_npy = out_path + "/npys"
                if not os.path.exists(out_path_npy):
                    os.makedirs(out_path_npy)
                out = os.path.join(out_path_img, filename[i])
                np.save(os.path.join(out_path_npy, filename[i]), grayscale_cam)
                imageio.imsave(out, visualization)


if __name__ == '__main__':
    # ["cw", "fgsm", "high_attack_fgsm_inp_m_2attack", "pgd", "pgd_rn50_eps8_step8", "pgd_rn50_eps10_step10", "rfgsm"]
    correct_classify = get_clean_classfiyTrue_list("wide_resnet101_2", args.data + "/images")
    # draw_clean_classfiyTrue_cam(correct_classify, "wide_resnet101_2", args.data + "/images/", "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/clean/")
    # draw_clean_classfiyTrue_cam(correct_classify, "wide_resnet101_2", "../attacks/advs/cw/", "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/cw")
    # draw_clean_classfiyTrue_cam(correct_classify, "wide_resnet101_2", "../attacks/advs/fgsm/", "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/fgsm")
    # draw_clean_classfiyTrue_cam(correct_classify, "wide_resnet101_2", "../attacks/advs/pgd/", "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/pgd")
    # draw_clean_classfiyTrue_cam(correct_classify, "wide_resnet101_2", "../attacks/advs/pgd_rn50_eps10_step10/", "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/pgd_rn50_eps10_step10")
    # draw_clean_classfiyTrue_cam(correct_classify, "wide_resnet101_2", "../attacks/advs/rfgsm/", "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/rfgsm")
    # draw_clean_classfiyTrue_cam(correct_classify, "wide_resnet101_2", "../attacks/advs/deepfool/", "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/deepfool")

    fgsm_mean = statistic_cam("./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/fgsm/npys", "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/fgsm/var_mean.csv")
    pgd_mean = statistic_cam("./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/pgd/npys", "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/pgd/var_mean.csv")
    pgd_rn50_eps10_step10_mean = statistic_cam("./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/pgd_rn50_eps10_step10/npys", "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/pgd_rn50_eps10_step10/var_mean.csv")
    rfgsm_mean = statistic_cam("./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/rfgsm/npys", "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/rfgsm/var_mean.csv")
    cw_mean = statistic_cam("./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/cw/npys", "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/cw/var_mean.csv")
    deepfool_mean = statistic_cam("./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/deepfool/npys", "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/deepfool/var_mean.csv")

    fgsm_res = static_att_moved_list(correct_classify, "wide_resnet101_2", "../attacks/advs/fgsm/", fgsm_mean, "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/fgsm/var_mean.csv")
    pgd_res = static_att_moved_list(correct_classify, "wide_resnet101_2", "../attacks/advs/pgd/", pgd_mean, "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/pgd/var_mean.csv")
    pgd_rn50_eps10_step10_res = static_att_moved_list(correct_classify, "wide_resnet101_2", "../attacks/advs/pgd_rn50_eps10_step10/", pgd_rn50_eps10_step10_mean, "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/pgd_rn50_eps10_step10/var_mean.csv")
    cw_res = static_att_moved_list(correct_classify, "wide_resnet101_2", "../attacks/advs/cw/", cw_mean, "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/cw/var_mean.csv")
    rfgsm_res = static_att_moved_list(correct_classify, "wide_resnet101_2", "../attacks/advs/rfgsm/", rfgsm_mean, "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/rfgsm/var_mean.csv")
    deepfool_res = static_att_moved_list(correct_classify, "wide_resnet101_2", "../attacks/advs/deepfool/", deepfool_mean, "./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/deepfool/var_mean.csv")

    database_json = {}
    database_json["fgsm"] = fgsm_res
    database_json["pgd"] = pgd_res
    database_json["pgd_rn50_eps10_step10"] = pgd_rn50_eps10_step10_res
    database_json["cw"] = cw_res
    database_json["rfgsm"] = rfgsm_res
    database_json["deepfool"] = deepfool_res

    with open("./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/adv_database.json", "w") as write_file:
        json.dump(database_json, write_file, sort_keys=True)
    print("Done writing JSON data into a file")

    with open("./cams/allCleanClassfiyTrue/att_moved/wide_resnet101_2/database_moved_new.txt", "w") as f:
        f.write("fgsm\n")
        f.write(str(len(fgsm_res["defence_success_upper"])) + " : ")
        f.write(str(fgsm_res["defence_success_upper"]))
        f.write("\n")
        f.write(str(len(fgsm_res["defence_success_down"])) + " : ")
        f.write(str(fgsm_res["defence_success_down"]))
        f.write("\n")
        f.write(str(len(fgsm_res["defence_fail_upper"])) + " : ")
        f.write(str(fgsm_res["defence_fail_upper"]))
        f.write("\n")
        f.write(str(len(fgsm_res["defence_fail_down"])) + " : ")
        f.write(str(fgsm_res["defence_fail_down"]))
        f.write("\n")

        f.write("pgd\n")
        f.write(str(len(pgd_res["defence_success_upper"])) + " : ")
        f.write(str(pgd_res["defence_success_upper"]))
        f.write("\n")
        f.write(str(len(pgd_res["defence_success_down"])) + " : ")
        f.write(str(pgd_res["defence_success_down"]))
        f.write("\n")
        f.write(str(len(pgd_res["defence_fail_upper"])) + " : ")
        f.write(str(pgd_res["defence_fail_upper"]))
        f.write("\n")
        f.write(str(len(pgd_res["defence_fail_down"])) + " : ")
        f.write(str(pgd_res["defence_fail_down"]))
        f.write("\n")

        f.write("pgd_rn50_eps10_step10_res\n")
        f.write(str(len(pgd_rn50_eps10_step10_res["defence_success_upper"])) + " : ")
        f.write(str(pgd_rn50_eps10_step10_res["defence_success_upper"]))
        f.write("\n")
        f.write(str(len(pgd_rn50_eps10_step10_res["defence_success_down"])) + " : ")
        f.write(str(pgd_rn50_eps10_step10_res["defence_success_down"]))
        f.write("\n")
        f.write(str(len(pgd_rn50_eps10_step10_res["defence_fail_upper"])) + " : ")
        f.write(str(pgd_rn50_eps10_step10_res["defence_fail_upper"]))
        f.write("\n")
        f.write(str(len(pgd_rn50_eps10_step10_res["defence_fail_down"])) + " : ")
        f.write(str(pgd_rn50_eps10_step10_res["defence_fail_down"]))
        f.write("\n")

        f.write("cw_res\n")
        f.write(str(len(cw_res["defence_success_upper"])) + " : ")
        f.write(str(cw_res["defence_success_upper"]))
        f.write("\n")
        f.write(str(len(cw_res["defence_success_down"])) + " : ")
        f.write(str(cw_res["defence_success_down"]))
        f.write("\n")
        f.write(str(len(cw_res["defence_fail_upper"])) + " : ")
        f.write(str(cw_res["defence_fail_upper"]))
        f.write("\n")
        f.write(str(len(cw_res["defence_fail_down"])) + " : ")
        f.write(str(cw_res["defence_fail_down"]))
        f.write("\n")

        f.write("rfgsm_res\n")
        f.write(str(len(rfgsm_res["defence_success_upper"])) + " : ")
        f.write(str(rfgsm_res["defence_success_upper"]))
        f.write("\n")
        f.write(str(len(rfgsm_res["defence_success_down"])) + " : ")
        f.write(str(rfgsm_res["defence_success_down"]))
        f.write("\n")
        f.write(str(len(rfgsm_res["defence_fail_upper"])) + " : ")
        f.write(str(rfgsm_res["defence_fail_upper"]))
        f.write("\n")
        f.write(str(len(rfgsm_res["defence_fail_down"])) + " : ")
        f.write(str(rfgsm_res["defence_fail_down"]))
        f.write("\n")

        f.write("deepfool\n")
        f.write(str(len(deepfool_res["defence_success_upper"])) + " : ")
        f.write(str(deepfool_res["defence_success_upper"]))
        f.write("\n")
        f.write(str(len(deepfool_res["defence_success_down"])) + " : ")
        f.write(str(deepfool_res["defence_success_down"]))
        f.write("\n")
        f.write(str(len(deepfool_res["defence_fail_upper"])) + " : ")
        f.write(str(deepfool_res["defence_fail_upper"]))
        f.write("\n")
        f.write(str(len(deepfool_res["defence_fail_down"])) + " : ")
        f.write(str(deepfool_res["defence_fail_down"]))
        f.write("\n")
    sys.exit(0)
