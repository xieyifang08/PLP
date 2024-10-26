import csv
import glob
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
            images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data['filename']
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
                out_path_img = out_path+"/images"
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
                out_path_img = out_path+"/images"
                if not os.path.exists(out_path_img):
                    os.makedirs(out_path_img)
                out_path_npy = out_path + "/npys"
                if not os.path.exists(out_path_npy):
                    os.makedirs(out_path_npy)
                out = os.path.join(out_path_img, filename[i])
                np.save(os.path.join(out_path_npy, filename[i]), grayscale_cam)
                imageio.imsave(out, visualization)


def draw_lastClassfiyTrue_cam(database_path, model_names, out_path):
    test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), database_path)['dev_data']
    correct_classify = [[],[],[]]
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
                    out_path_img = out_path+"/"+model_names[j]+"/images"
                    if not os.path.exists(out_path_img):
                        os.makedirs(out_path_img)
                    out_path_npy = out_path+"/"+model_names[j]+ "/npys"
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

    image = input_diversity(image,180, img_size)

    cam = GradCAM(model=model, target_layers=[model.model.layer4[-1]])
    grayscale_cam = cam(input_tensor=image, target_category=true_label)
    print(grayscale_cam.shape)
    print(grayscale_cam.mean())
    img = np.transpose(image[0].detach().cpu().numpy(), (1, 2, 0))
    visualization = show_cam_on_image(img, grayscale_cam[0, :], use_rgb=True)
    print(visualization.shape)
    plt.imshow(visualization / 255.0)
    plt.title("wrn101_"+str(img_size), y=-0.2)
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


def statistic_cam(input_org, input_dnn, input_dnn_jpeg, csv_path):
    all_org_npys = glob.glob(os.path.join(input_org, './*.npy'))
    all_ddn_npys = glob.glob(os.path.join(input_dnn, './*.npy'))
    all_ddn_jpeg_npys = glob.glob(os.path.join(input_dnn_jpeg, './*.npy'))
    out = open(csv_path, "a")
    csv_writer = csv.writer(out, dialect="excel")
    csv_writer.writerow(["image name", "wide resnet101 base", "wide resnet101 base at", "wide resnet101 base at jpeg"])
    for org_path in tqdm.tqdm(all_org_npys):
        npy_name = org_path.split("\\.\\")[-1]
        if input_dnn+"\\.\\"+npy_name in all_ddn_npys and input_dnn_jpeg+"\\.\\"+npy_name in all_ddn_jpeg_npys:
            csv_writer.writerow([npy_name, np.load(org_path).mean(), np.load(input_dnn+"/./"+npy_name).mean(),np.load(input_dnn_jpeg+"/./"+npy_name).mean()])
        # img = np.load(npy_path)

def draw_fail_classify_cam(model_name, image_path, true_label,out_path, filename, classify_success = True):
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
    logits =  torch.softmax(logits,dim=1)
    # print("[{}:{}]".format(logits.argmax(1).item(), format(logits.max().item(), '.4f')))

    h_x = logits.data.squeeze()
    probs, idx = h_x.sort(0, True)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()
    for i in range(0, 20):
        print('{} : {:.3f}'.format(idx[i],probs[i]))
    print()
    if not classify_success:
        visualization = show_cam_on_image(img / 255.0, grayscale_cam[0,:], use_rgb=True)
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
            row.append(np.load(input_path[i]+"/"+npy_name).mean())
        csv_writer.writerow(row)

def draw_cleanClassfiyTrue_lastClassfiyTrue_cam(database_path, model_names, out_path):
    clean_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), os.path.join(args.data, 'images'))['dev_data']
    test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), database_path)['dev_data']
    correct_classify = [[],[],[]]
    clean_correct_classify = [[],[],[]]
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

    for batch_data,clean_batch_data in tqdm.tqdm(zip(test_loader, clean_loader), ncols=80):
        images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        clean_images = clean_batch_data['image'].to(DEVICE)
        for i in range(images.shape[0]):
            if filename[i] in clean_correct_classify[0] and filename[i] in clean_correct_classify[1] and filename[i] in clean_correct_classify[2]:
                if filename[i] in correct_classify[-1] and filename[i] not in correct_classify[0]:
                    for j in range(len(cams)):
                        clean_grayscale_cam = cams[j](input_tensor=clean_images[i].unsqueeze(0), target_category=labels[i].cpu().item())
                        img = np.transpose(clean_images[i].detach().cpu().numpy(), (1, 2, 0))
                        clean_visualization = show_cam_on_image(img, clean_grayscale_cam[0, :], use_rgb=True)
                        out_path_img = out_path+"/"+model_names[j]+"/clean/images"
                        if not os.path.exists(out_path_img):
                            os.makedirs(out_path_img)
                        out_path_npy = out_path+"/"+model_names[j]+ "/clean/npys"
                        if not os.path.exists(out_path_npy):
                            os.makedirs(out_path_npy)
                        out = os.path.join(out_path_img, filename[i])
                        np.save(os.path.join(out_path_npy, filename[i]), clean_grayscale_cam)
                        imageio.imsave(out, clean_visualization)

                    for j in range(len(cams)):
                        grayscale_cam = cams[j](input_tensor=images[i].unsqueeze(0), target_category=labels[i].cpu().item())
                        img = np.transpose(images[i].detach().cpu().numpy(), (1, 2, 0))
                        visualization = show_cam_on_image(img, grayscale_cam[0, :], use_rgb=True)
                        out_path_img = out_path+"/"+model_names[j]+"/images"
                        if not os.path.exists(out_path_img):
                            os.makedirs(out_path_img)
                        out_path_npy = out_path+"/"+model_names[j]+ "/npys"
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
        row=[]
        npy_name = org_path.split("/./")[-1]
        a = torch.from_numpy(np.load(org_path))
        row.append(npy_name)
        for i in range(len(input_paths)):
            b = torch.from_numpy(np.load(input_paths[i]+"/./"+npy_name))
            kl = F.kl_div(a.softmax(dim=-1).log(), b.softmax(dim=-1), reduction='sum')
            row.append(kl.item())
        csv_writer.writerow(row)


def statistic_classfiy(csv_path, input_org, input_paths, model_names,input_titles):
    out = open(csv_path+"0.csv", "a", newline='')
    csv_writer = csv.writer(out, dialect="excel")
    result = [[],[],[],[],[],[],[],[]]
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
                    result[(i+1)*2].append(logits.argmax(1)[j].item())
                    result[(i+1)*2+1].append(logits[j].max().item())
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
        "model1_upkl_classfiy_true":0,"model1_upkl_classfiy_false":0,"model1_downkl_classfiy_true":0,"model1_downkl_classfiy_false":0,
        "model2_upkl_classfiy_true": 0, "model2_upkl_classfiy_false": 0,"model2_downkl_classfiy_true": 0, "model2_downkl_classfiy_false": 0,
        "model3_upkl_classfiy_true": 0, "model3_upkl_classfiy_false": 0,"model3_downkl_classfiy_true": 0, "model3_downkl_classfiy_false": 0,
              }
    for row in csv_reader:
        # name, model1_kl, model2_kl, model3_kl, clean_classfiy, clean_confidence, model1_classfiy, model1_confidence, model2_classfiy, model2_confidence, model3_classfiy, model3_confidence
        if float(row["model1-kl"]) > avg_kl[0]:
            if row["model1_classfiy"] == row["clean_classfiy"]:
                result["model1_upkl_classfiy_true"] += 1
            else: result["model1_upkl_classfiy_false"] += 1
        else:
            if row["model1_classfiy"] == row["clean_classfiy"]:
                result["model1_downkl_classfiy_true"] += 1
            else: result["model1_downkl_classfiy_false"] += 1

        if float(row["model2-kl"]) > avg_kl[1] :
            if row["model2_classfiy"] == row["clean_classfiy"]:
                result["model2_upkl_classfiy_true"] += 1
            else: result["model2_upkl_classfiy_false"] += 1
        else:
            if row["model2_classfiy"] == row["clean_classfiy"]:
                result["model2_downkl_classfiy_true"] += 1
            else: result["model2_downkl_classfiy_false"] += 1

        if float(row["model3-kl"]) > avg_kl[2]:
            if row["model3_classfiy"] == row["clean_classfiy"]:
                result["model3_upkl_classfiy_true"] += 1
            else: result["model3_upkl_classfiy_false"] += 1
        else:
            if row["model3_classfiy"] == row["clean_classfiy"]:
                result["model3_downkl_classfiy_true"] += 1
            else: result["model3_downkl_classfiy_false"] += 1
    print(result)


def analyse_classfiy_var(csv_path, avg_var):
    out = open(csv_path, "r", newline='')
    csv_reader = csv.DictReader(out)
    result = {
        "model1_upvar_classfiy_true":0,"model1_upvar_classfiy_false":0,"model1_downvar_classfiy_true":0,"model1_downvar_classfiy_false":0,
        "model2_upvar_classfiy_true": 0, "model2_upvar_classfiy_false": 0,"model2_downvar_classfiy_true": 0, "model2_downvar_classfiy_false": 0,
        "model3_upvar_classfiy_true": 0, "model3_upvar_classfiy_false": 0,"model3_downvar_classfiy_true": 0, "model3_downvar_classfiy_false": 0,
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

    name_downFalse2downTrue=[]
    name_downFalse2upTrue=[]
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

        if float(row["model2-var"]) > avg_var[1] :
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
            else: result["model3_upvar_classfiy_false"] += 1
        else:
            if row["model3_classfiy"] == row["clean_classfiy"]:
                result["model3_downvar_classfiy_true"] += 1
            else: result["model3_downvar_classfiy_false"] += 1

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
            .format(len(name_upFalse2upFalse), len(name_upFalse2downFalse), len(name_downFalse2downFalse), len(name_downFalse2upFalse)))
    print("name_upFalse2upTrue = {}\nname_upFalse2downTrue={}\nname_downFalse2downTrue={}\nname_downFalse2upTrue={}\n"
          .format(len(name_upFalse2upTrue), len(name_upFalse2downTrue), len(name_downFalse2downTrue), len(name_downFalse2upTrue)))

    print("name_upTrue2upFalse = {}\nname_upTrue2downFalse={}\nname_downTrue2downFalse={}\nname_downTrue2upFalse={}\n"
          .format(len(name_upTrue2upFalse), len(name_upTrue2downFalse), len(name_downTrue2downFalse), len(name_downTrue2upFalse)))

def test_single_var(npy_path1, npy_path2):
    npy1 = np.load(npy_path1)
    npy2 = np.load(npy_path2)
    print("({},{}),({},{}),({},{})".format(npy1.max(),npy1.min(),npy2.max(),npy2.min(),(npy1-npy2).max(),(npy1-npy2).min()))
    print((npy1-npy2).var(),(npy1-npy2).std())


def analyse_classfiy_std(csv_path, avg_std):
    out = open(csv_path, "r", newline='')
    csv_reader = csv.DictReader(out)
    result = {
        "model1_upstd_classfiy_true":0,"model1_upstd_classfiy_false":0,"model1_downstd_classfiy_true":0,"model1_downstd_classfiy_false":0,
        "model2_upstd_classfiy_true": 0, "model2_upstd_classfiy_false": 0,"model2_downstd_classfiy_true": 0, "model2_downstd_classfiy_false": 0,
        "model3_upstd_classfiy_true": 0, "model3_upstd_classfiy_false": 0,"model3_downstd_classfiy_true": 0, "model3_downstd_classfiy_false": 0,
              }
    for row in csv_reader:
        # name, model1_std, model2_std, model3_std, clean_classfiy, clean_confidence, model1_classfiy, model1_confidence, model2_classfiy, model2_confidence, model3_classfiy, model3_confidence
        if float(row["model1-std"]) > avg_std[0]:
            if row["model1_classfiy"] == row["clean_classfiy"]:
                result["model1_upstd_classfiy_true"] += 1
            else: result["model1_upstd_classfiy_false"] += 1
        else:
            if row["model1_classfiy"] == row["clean_classfiy"]:
                result["model1_downstd_classfiy_true"] += 1
            else: result["model1_downstd_classfiy_false"] += 1

        if float(row["model2-std"]) > avg_std[1] :
            if row["model2_classfiy"] == row["clean_classfiy"]:
                result["model2_upstd_classfiy_true"] += 1
            else: result["model2_upstd_classfiy_false"] += 1
        else:
            if row["model2_classfiy"] == row["clean_classfiy"]:
                result["model2_downstd_classfiy_true"] += 1
            else: result["model2_downstd_classfiy_false"] += 1

        if float(row["model3-std"]) > avg_std[2]:
            if row["model3_classfiy"] == row["clean_classfiy"]:
                result["model3_upstd_classfiy_true"] += 1
            else: result["model3_upstd_classfiy_false"] += 1
        else:
            if row["model3_classfiy"] == row["clean_classfiy"]:
                result["model3_downstd_classfiy_true"] += 1
            else: result["model3_downstd_classfiy_false"] += 1
    print(result)


def test_classfiy(model_name, img_path):
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
            st = (a-b).var()
            if npy_name == "1821.png.npy":
                print(npy_name, st.item())
            row.append(st.item())
        csv_writer.writerow(row)


def test_filter_classfiy(model_name, img_path, kernlen=5, nsig=1.5):
    ga = GaussianBlurConv(kernlen=kernlen, nsig=nsig).to(DEVICE)
    test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), img_path)['dev_data']
    model = model_utils.load_model(model_name).to(DEVICE).eval()
    val_accs = AverageMeter()
    for _, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        images = ga(images)
        # for i in range(images.shape[0]):
        #     img = np.transpose(images[i].detach().cpu().numpy(), (1, 2, 0))
        #     plt.imshow(img)
        #     plt.show()
        logits = model(images)
        val_accs.append((logits.argmax(1) == labels).float().mean().item())
    return str(val_accs.avg)+" "


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
    # correct_classify = get_clean_classfiyTrue_list("wide_resnet101_2", args.data+"/images")
    # draw_clean_classfiyTrue_cam_filter(None, "wide_resnet101_2", args.data+"/images/", "./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack_or/wide_resnet101_2/clean/")
    # sys.exit(0)
    # test_classfiy("wide_resnet101_2", args.data+"/images/") # 0.8497
    msg = ""
    for nsig in [0.1, 0.2, 0.3, 0.4]:
        msg += test_filter_classfiy("wide_resnet101_2", args.data+"/images/", nsig= nsig) #0.735
        msg += test_filter_classfiy("wide_resnet101_2", "../attacks/advs/high_attack_fgsm_inp_m_2attack/", nsig= nsig) # 0.3749
        # test_classfiy("wide_resnet101_2", "../attacks/advs/high_attack_fgsm_inp_m_2attack/") #0.8225
        msg += test_filter_classfiy("wide_resnet101_2", "../attacks/advs/rfgsm/", nsig= nsig) #0.4478
        # test_classfiy("wide_resnet101_2", "../attacks/advs/rfgsm/") #0.8015
        msg += test_filter_classfiy("wide_resnet101_2", "../attacks/advs/pgd/", nsig= nsig) #0.1185
        # test_classfiy("wide_resnet101_2", "../attacks/advs/pgd/") #0.1299
        msg += " \n"
        print(nsig, msg)
    print(msg)
    sys.exit(0)

    #
    # sys.exit(0)
    # test_single_var("./cams/allCleanClassfiyTrue/pgd/wide_resnet101_2/clean/npys/2.png.npy", "./cams/allCleanClassfiyTrue/pgd/wide_resnet101_2/npys/2.png.npy")

    # test_classfiy("wide_resnet101_2", args.data+"/images/")
    # test_classfiy("wide_resnet101_2", "../attacks/advs/high_attack_fgsm_inp_m_2attack/")
    # for name in ['3461.png', '3504.png', '3533.png', '3577.png', '3591.png', '3643.png', '1168.png','616.png', '6187.png', '6269.png', '6270.png', '6336.png', '6347.png', '6456.png', '6502.png', '6572.png', '6575.png', '6705.png', '6790.png', '6957.png', '6980.png', '7112.png', '7141.png', '7206.png', '7337.png', '7433.png', '7475.png', '753.png', '7564.png', '758.png', '7708.png', '7833.png', '7899.png', '8051.png', '8200.png', '8265.png', '8466.png', '8552.png', '8731.png', '8878.png', '8989.png', '9167.png', '9198.png', '9303.png', '9313.png', '9439.png', '9480.png', '9554.png', '965.png', '97.png', '9702.png', '973.png', '977.png', '9841.png', '1646.png', '1694.png', '1703.png', '1718.png', '1722.png', '1734.png', '1783.png', '1787.png','1605.png', '1618.png', '1630.png', '1631.png', '1644.png', '1339.png', '1005.png','1010.png', '10.png','100.png', '1014.png','1053.png', '1252.png', '1329.png','4682.png', '9472.png','3324.png', '3974.png','1460.png', '1590.png','2425.png', '2694.png',]:
    #     print(name)
    #     draw_fail_classify_cam("wide_resnet101_2",args.data+"/images/"+name, -1, "","")
    #     draw_fail_classify_cam("wide_resnet101_2","../attacks/advs/high_attack_fgsm_inp_m_2attack/"+name, -1, "","")
    #     draw_fail_classify_cam("wide_resnet101_2_dnn","../attacks/advs/high_attack_fgsm_inp_m_2attack/"+name, -1, "","")
    #     print("------------------")
    #
    #
    # sys.exit(0)


    # 3.811891575, 3.378746867, 3.379058752
    # 240/34 = 7.05
    # {'model1_upattention_classfiy_true': 34, 'model1_upattention_classfiy_false': 3264,
    #  'model1_downattention_classfiy_true': 240, 'model1_downattention_classfiy_false': 4959,
    #  'model2_upattention_classfiy_true': 119, 'model2_upattention_classfiy_false': 3058,
    #  'model2_downattention_classfiy_true': 787, 'model2_downattention_classfiy_false': 4533,
    #  'model3_upattention_classfiy_true': 170, 'model3_upattention_classfiy_false': 3012,
    #  'model3_downattention_classfiy_true': 941, 'model3_downattention_classfiy_false': 4374}
    # analyse_classfiy_kl("./cams/allCleanClassfiyTrue/pgd/classfiy-kl-result.csv",
    #                     [3.811891575, 3.378746867, 3.379058752])
    # sys.exit(0)
    # 0.899245709 0.814234943 0.823843156
    # 3229/984 = 3.28
    # {'model1_upkl_classfiy_true': 984, 'model1_upkl_classfiy_false': 1750,
    # 'model1_downkl_classfiy_true': 3229,'model1_downkl_classfiy_false': 2534,
    # 'model2_upkl_classfiy_true': 2434, 'model2_upkl_classfiy_false': 314,
    #  'model2_downkl_classfiy_true': 5492, 'model2_downkl_classfiy_false': 257, 'model3_upkl_classfiy_true': 2496,
    #  'model3_upkl_classfiy_false': 311, 'model3_downkl_classfiy_true': 5464, 'model3_downkl_classfiy_false': 226}

    # 1.05373311 0.843412558 0.850578559
    # 2891/780 = 3.71
    # {'model1_upkl_classfiy_true': 780, 'model1_upkl_classfiy_false': 1963,
    #  'model1_downkl_classfiy_true': 2891, 'model1_downkl_classfiy_false': 2863,
    #  'model2_upkl_classfiy_true': 2423, 'model2_upkl_classfiy_false': 326,
    #  'model2_downkl_classfiy_true': 5447, 'model2_downkl_classfiy_false': 301,
    #  'model3_upkl_classfiy_true': 2470, 'model3_upkl_classfiy_false': 334,
    #  'model3_downkl_classfiy_true': 5457, 'model3_downkl_classfiy_false': 236}
    # 'model1_upkl_classfiy_true': 832, 'model1_upkl_classfiy_false': 2043, 'model1_downkl_classfiy_true': 2839, 'model1_downkl_classfiy_false': 2783
    # analyse_classfiy_kl("./cams/allCleanClassfiyTrue/pgd_rn50_eps10_step10/classfiy-kl-result.csv", [1.05373311, 0.843412558, 0.850578559])
    # 0.014699297, 0.012132069, 0.012344945
    # 2873/798 = 3.6
    # {'model1_upvar_classfiy_true': 798, 'model1_upvar_classfiy_false': 1984,
    #  'model1_downvar_classfiy_true': 2873, 'model1_downvar_classfiy_false': 2842,
    #  'model2_upvar_classfiy_true': 2469, 'model2_upvar_classfiy_false': 317,
    #  'model2_downvar_classfiy_true': 5401, 'model2_downvar_classfiy_false': 310,
    #  'model3_upvar_classfiy_true': 2483, 'model3_upvar_classfiy_false': 321,
    #  'model3_downvar_classfiy_true': 5444, 'model3_downvar_classfiy_false': 249}
    # analyse_classfiy_var("./cams/allCleanClassfiyTrue/pgd_rn50_eps10_step10/classfiy-kl-result.csv",
    #                     [0.014699297, 0.012132069, 0.012344945])
    # 0.108877578, 0.098878033, 0.099993466
    # 2603/1068 = 2.44
    # {'model1_upstd_classfiy_true': 1068, 'model1_upstd_classfiy_false': 2418,
    #  'model1_downstd_classfiy_true': 2603, 'model1_downstd_classfiy_false': 2408,
    #  'model2_upstd_classfiy_true': 3101, 'model2_upstd_classfiy_false': 383,
    #  'model2_downstd_classfiy_true': 4769, 'model2_downstd_classfiy_false': 244,
    #  'model3_upstd_classfiy_true': 3122, 'model3_upstd_classfiy_false': 370,
    #  'model3_downstd_classfiy_true': 4805, 'model3_downstd_classfiy_false': 200}
    # analyse_classfiy_std("./cams/allCleanClassfiyTrue/pgd_rn50_eps10_step10/classfiy-kl-result.csv", [0.108877578, 0.098878033, 0.099993466])
    # sys.exit(0)

    # 1.209573081	0.917857515	0.925666094
    # 2574/734 = 3.51
    # {'model1_upkl_classfiy_true': 734, 'model1_upkl_classfiy_false': 2053,
    #  'model1_downkl_classfiy_true': 2574, 'model1_downkl_classfiy_false': 3136,
    #  'model2_upkl_classfiy_true': 2386, 'model2_upkl_classfiy_false': 371,
    #  'model2_downkl_classfiy_true': 5401, 'model2_downkl_classfiy_false': 339,
    #  'model3_upkl_classfiy_true': 2451, 'model3_upkl_classfiy_false': 378,
    #  'model3_downkl_classfiy_true': 5400, 'model3_downkl_classfiy_false': 268}
    # analyse_classfiy_kl("./cams/allCleanClassfiyTrue/pgd_rn50_eps15_step10/classfiy-kl-result.csv", [1.209573081, 0.917857515, 0.925666094])

    # 1.116429127	0.87270473	0.879452873
    # 2705/740 = 3.66
    # {'model1_upkl_classfiy_true': 740, 'model1_upkl_classfiy_false': 2060,
    #  'model1_downkl_classfiy_true': 2705, 'model1_downkl_classfiy_false': 2992,
    #  'model2_upkl_classfiy_true': 2408, 'model2_upkl_classfiy_false': 349,
    #  'model2_downkl_classfiy_true': 5445, 'model2_downkl_classfiy_false': 295,
    #  'model3_upkl_classfiy_true': 2419, 'model3_upkl_classfiy_false': 352,
    #  'model3_downkl_classfiy_true': 5468, 'model3_downkl_classfiy_false': 258}
    # analyse_classfiy_kl("./cams/allCleanClassfiyTrue/pgd_rn50_eps12_step10/classfiy-kl-result.csv", [1.116429127, 0.87270473, 0.879452873])
    # sys.exit(0)
    # 0.05408207, 0.04936366, 0.049394384
    # 238/36 = 6.61
    # {'model1_upvar_classfiy_true': 36, 'model1_upvar_classfiy_false': 3372,
    #  'model1_downvar_classfiy_true': 238, 'model1_downvar_classfiy_false': 4851,
    #  'model2_upvar_classfiy_true': 126, 'model2_upvar_classfiy_false': 3214,
    #  'model2_downvar_classfiy_true': 780, 'model2_downvar_classfiy_false': 4377,
    #  'model3_upvar_classfiy_true': 177, 'model3_upvar_classfiy_false': 3115,
    #  'model3_downvar_classfiy_true': 934, 'model3_downvar_classfiy_false': 4271}
    # analyse_classfiy_var("./cams/allCleanClassfiyTrue/pgd/classfiy-kl-result.csv", [0.05408207, 0.04936366, 0.049394384])
    # analyse_classfiy_var("./cams/allCleanClassfiyTrue/pgd/classfiy-kl-result.csv", [0.06408207, 0.04936366, 0.049394384])
    # analyse_classfiy_var("./cams/allCleanClassfiyTrue/pgd/classfiy-kl-result.csv", [0.07408207, 0.04936366, 0.049394384])
    # analyse_classfiy_var("./cams/allCleanClassfiyTrue/pgd/classfiy-kl-result.csv", [0.08408207, 0.04936366, 0.049394384])
    # 0.019647567, 0.013804689, 0.013914755
    # {'model1_upvar_classfiy_true': 327, 'model1_upvar_classfiy_false': 2581,
    #  'model1_downvar_classfiy_true': 954,'model1_downvar_classfiy_false': 4635,
    #  'model2_upvar_classfiy_true': 2335, 'model2_upvar_classfiy_false': 497,
    #  'model2_downvar_classfiy_true': 5110, 'model2_downvar_classfiy_false': 555,
    #  'model3_upvar_classfiy_true': 2332,'model3_upvar_classfiy_false': 479,
    #  'model3_downvar_classfiy_true': 5264, 'model3_downvar_classfiy_false': 422}
    # analyse_classfiy_var("./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack/classfiy-result.csv",
    #                      [0.019647567, 0.019647567, 0.019647567])
    # sys.exit(9)
    # {'model1_upvar_classfiy_true': 327, 'model1_upvar_classfiy_false': 2581,
    #  'model1_downvar_classfiy_true': 954, 'model1_downvar_classfiy_false': 4635,
    #  'model2_upvar_classfiy_true': 1437, 'model2_upvar_classfiy_false': 347,
    #  'model2_downvar_classfiy_true': 6008, 'model2_downvar_classfiy_false': 705,
    #  'model3_upvar_classfiy_true': 1490, 'model3_upvar_classfiy_false': 355,
    #  'model3_downvar_classfiy_true': 6106, 'model3_downvar_classfiy_false': 546}
    #
    # # 0.019647567, 0.013804689, 0.013914755
    # # 368/96 = 3.83
    # {'model1_upvar_classfiy_true': 96, 'model1_upvar_classfiy_false': 2812,
    #  'model1_downvar_classfiy_true': 368,'model1_downvar_classfiy_false': 5221,
    #  'model2_upvar_classfiy_true': 2225, 'model2_upvar_classfiy_false': 607,
    #  'model2_downvar_classfiy_true': 4879, 'model2_downvar_classfiy_false': 786,
    #  'model3_upvar_classfiy_true': 2228,'model3_upvar_classfiy_false': 583,
    #  'model3_downvar_classfiy_true': 5039, 'model3_downvar_classfiy_false': 647}
    # analyse_classfiy_var("./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack_or/classfiy-result.csv",
    #                      [0.019647567, 0.013804689, 0.013914755])
    # sys.exit(0)
    # 0.217682357, 0.206318927, 0.206418279
    # 230/44 = 5.23
    # {'model1_upstd_classfiy_true': 44, 'model1_upstd_classfiy_false': 3917,
    #  'model1_downstd_classfiy_true': 230,'model1_downstd_classfiy_false': 4306,
    #  'model2_upstd_classfiy_true': 167, 'model2_upstd_classfiy_false': 3724,
    #  'model2_downstd_classfiy_true': 739, 'model2_downstd_classfiy_false': 3867,
    #  'model3_upstd_classfiy_true': 223,'model3_upstd_classfiy_false': 3624,
    #  'model3_downstd_classfiy_true': 888, 'model3_downstd_classfiy_false': 3762}
    # analyse_classfiy_std("./cams/allCleanClassfiyTrue/pgd/classfiy-kl-result.csv", [0.217682357, 0.206318927, 0.206418279])
    # sys.exit(0)
    # statistic_classfiy("./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack/classfiy-result.csv",
    #                    args.data+"/images",
    #                    ["../attacks/advs/high_attack_fgsm_inp_m_2attack/",
    #                     "../attacks/advs/high_attack_fgsm_inp_m_2attack/",
    #                     "../attacks/advs/high_attack_fgsm_inp_m_2attack/"
    #                     ],
    #                    ["wide_resnet101_2", "wide_resnet101_2_dnn", "wide_resnet101_2_dnn_jpeg"],
    #                   ["name",
    #                    "clean_classfiy", "clean_confidence", "model1_classfiy", "model1_confidence",
    #                    "model2_classfiy", "model2_confidence", "model3_classfiy", "model3_confidence"]
    # )
    # sys.exit(1)
    # correct_classify = get_clean_classfiyTrue_list("wide_resnet101_2", args.data+"/images")
    # draw_clean_classfiyTrue_cam(correct_classify, "wide_resnet101_2", args.data+"/images/", "./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack_or/wide_resnet101_2/clean/")
    # draw_clean_classfiyTrue_cam(correct_classify, "wide_resnet101_2", "../attacks/advs/pgd_rn50_eps15_step20/", "./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack_or/wide_resnet101_2")
    # draw_clean_classfiyTrue_cam(correct_classify, "wide_resnet101_2_dnn", "../attacks/advs/pgd_rn50_eps15_step20/", "./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack_or/wide_resnet101_2_dnn")
    # draw_clean_classfiyTrue_cam(correct_classify, "wide_resnet101_2_dnn_jpeg", "../attacks/advs/pgd_rn50_eps15_step20/", "./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack_or/wide_resnet101_2_dnn_jpeg")
    # statistic_KL_mean("./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack/kl-result.csv",
    #                   ["name", "model1-kl", "model2-kl", "model3-kl"],
    #                   "./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack/wide_resnet101_2/clean/npys",
    #                   ["./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack/wide_resnet101_2/npys",
    #                    "./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack/wide_resnet101_2_dnn/npys",
    #                    "./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack/wide_resnet101_2_dnn_jpeg/npys"
    #                    ]
    # )
    # sys.exit(0)
    # analyse_attention_variance("./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack_or/var-result.csv",
    #                   ["name", "model1-var", "model2-var", "model3-var"],
    #                   "./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack_or/wide_resnet101_2/clean/npys",
    #                   ["./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack_or/wide_resnet101_2/npys",
    #                    "./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack_or/wide_resnet101_2_dnn/npys",
    #                    "./cams/allCleanClassfiyTrue/high_attack_fgsm_inp_m_2attack_or/wide_resnet101_2_dnn_jpeg/npys"
    #                    ]
    # )
    # sys.exit(0)
    # statistic_KL_mean("./cams/lastClassfiyTrue/pgd/kl-result.csv",
    #                   ["name", "org-model1", "org-model2", "org-model3"],
    #                   "./cams/lastClassfiyTrue/pgd/wide_resnet101_2/clean/npys",
    #                   ["./cams/lastClassfiyTrue/pgd/wide_resnet101_2/npys",
    #                    "./cams/lastClassfiyTrue/pgd/wide_resnet101_2_dnn/npys",
    #                    "./cams/lastClassfiyTrue/pgd/wide_resnet101_2_dnn_jpeg/npys"
    #                    ]
    # )
    #
    # for i in range(10000):
    #     if not os.path.exists("./cams/lastClassfiyTrue/pgd/wide_resnet101_2_dnn_jpeg/images/"+str(i)+".png"):
    #         continue
    #     label1 = draw_fail_classify_cam("wide_resnet101_2",args.data + "/images/" + str(i) + ".png", 21, "", "")
    #     # label1 = draw_fail_classify_cam("wide_resnet101_2", "../attacks/advs/pgd/"+str(i)+".png", 21, "", "")
    #     label2 = draw_fail_classify_cam("wide_resnet101_2_dnn", "../attacks/advs/pgd/"+str(i)+".png", 21, "", "")
    #     # draw_fail_classify_cam("wide_resnet101_2_dnn_jpeg", "../attacks/advs/pgd/789.png", 21, "", "")
    #     if label1 != label2:
    #         print(label1,label2)
    model = model_utils.load_model("wide_resnet101_2").to(DEVICE)
    for i in range(10):
        draw_single_cam(args.data+"/images/109.png", model, 21, "./cams/lastClassfiyTrue/org", "108.png")
    # draw_single_cam(args.data+"/images/108.png", "wide_resnet101_2", 21, "./cams/lastClassfiyTrue/org", "108.png")
    # draw_single_cam(args.data+"/images/108.png", "wide_resnet101_2", 21, "./cams/lastClassfiyTrue/org", "108.png")

    # draw_single_cam(args.data+"/images/108.png", "wide_resnet101_2", 21, "./cams/lastClassfiyTrue/org", "108.png")
    # draw_single_cam(args.data+"/images/108.png", "wide_resnet101_2", 21, "./cams/lastClassfiyTrue/org", "108.png")
    #
    sys.exit(0)
    # statistic_clean_adv_cam(["clean_wrn","clean_wrn_at","clean_wrn_at_jpeg","pgd_wrn","pgd_wrn_at","pgd_wrn_at_jpeg"],
    #                         [
    #                             "./cams/lastClassfiyTrue/pgd/wide_resnet101_2/clean/npys",
    #                             "./cams/lastClassfiyTrue/pgd/wide_resnet101_2_dnn/clean/npys",
    #                             "./cams/lastClassfiyTrue/pgd/wide_resnet101_2_dnn_jpeg/clean/npys",
    #                             "./cams/lastClassfiyTrue/pgd/wide_resnet101_2/npys",
    #                             "./cams/lastClassfiyTrue/pgd/wide_resnet101_2_dnn/npys",
    #                             "./cams/lastClassfiyTrue/pgd/wide_resnet101_2_dnn_jpeg/npys"
    #                          ],
    #                         "./cams/lastClassfiyTrue/pgd/result.csv")

    # draw_cleanClassfiyTrue_lastClassfiyTrue_cam("../attacks/advs/pgd", ["wide_resnet101_2", "wide_resnet101_2_dnn", "wide_resnet101_2_dnn_jpeg"], "./cams/lastClassfiyTrue/pgd")


    # draw_lastClassfiyTrue_cam("../attacks/advs/pgd", ["wide_resnet101_2", "wide_resnet101_2_dnn", "wide_resnet101_2_dnn_jpeg"], "./cams/lastClassfiyTrue")


    # draw_single_cam(args.data+"/images/8773.png", "wide_resnet101_2", 754, "")
    # draw_single_cam(args.data+"/images/8773.png", "wide_resnet101_2_dnn", 754, "")
    # draw_single_cam(args.data+"/images/8773.png", "wide_resnet101_2_dnn_jpeg", 754, "")


    # draw_fail_classify_cam("wide_resnet101_2", "../attacks/advs/rfgsm/1310.png", 229, "./cams/rfgsm/wrn101/fail_classify", "1145.png")
    # draw_fail_classify_cam("wide_resnet101_2_dnn", "../attacks/advs/rfgsm/1310.png", 229, "./cams/rfgsm/wrn101/fail_classify", "1145.png")
    # draw_fail_classify_cam("wide_resnet101_2_dnn_jpeg", "../attacks/advs/rfgsm/1310.png", 229, "./cams/rfgsm/wrn101/fail_classify", "1145.png")
    # draw_fail_classify_cam("wide_resnet101_2_dnn", "../attacks/advs/pgd/9049.png", 809, "./cams/pgd/wrn101_dnn/fail_classify", "9049.png")
    # draw_fail_classify_cam("wide_resnet101_2_dnn_jpeg", "../attacks/advs/pgd/9049.png", 3, "./cams/org/wrn101/fail_classify", "8773.png")

    # statistic_cam("./cams/org/wrn101/npys", "./cams/org/wrn101_dnn/npys", "./cams/org/wrn101_dnn_jpeg/npys", "./result_org.csv")
    # statistic_cam(
    #     "./cams/allCleanClassfiyTrue/pgd/wide_resnet101_2/clean/npys",
    #     "./cams/allCleanClassfiyTrue/pgd/wide_resnet101_2_dnn/npys",
    #     "./cams/allCleanClassfiyTrue/pgd/wide_resnet101_2_dnn_jpeg/npys",
    #     "./cams/allCleanClassfiyTrue/pgd/result_pgd.csv")
    # statistic_cam("./cams/cw_step1000/wrn101/npys", "./cams/cw_step1000/wrn101_dnn/npys", "./cams/cw_step1000/wrn101_dnn_jpeg/npys", "./result_cw_step1000.csv")
    # statistic_cam("./cams/rfgsm/wrn101/npys", "./cams/rfgsm/wrn101_dnn/npys", "./cams/rfgsm/wrn101_dnn_jpeg/npys", "./result_rfgsm.csv")

    # 0.8497 0.9247 0.9307
    draw_cam(args.data+"/images", "wide_resnet101_2", "./cams/org/wrn101")
    # draw_cam(args.data+"/images", "wide_resnet101_2_dnn", "./cams/org/wrn101_dnn")
    # draw_cam(args.data+"/images", "wide_resnet101_2_dnn_jpeg", "./cams/org/wrn101_dnn_jpeg")
    #
    # draw_cam("../attacks/advs/cw", "wide_resnet101_2", "./cams/cw/wrn101")
    # draw_cam("../attacks/advs/cw", "wide_resnet101_2_dnn", "./cams/cw/wrn101_dnn")
    # draw_cam("../attacks/advs/cw", "wide_resnet101_2_dnn_jpeg", "./cams/cw/wrn101_dnn_jpeg")

    # 0.0281 0.0973 0.122
    # draw_cam("../attacks/advs/pgd", "wide_resnet101_2", "./cams/pgd/wrn101")
    # draw_cam("../attacks/advs/pgd", "wide_resnet101_2_dnn", "./cams/pgd/wrn101_dnn")
    # draw_cam("../attacks/advs/pgd", "wide_resnet101_2_dnn_jpeg", "./cams/pgd/wrn101_dnn_jpeg")
    # 0.8474 0.9243 0.9302
    # draw_cam("../attacks/advs/cw_step1000", "wide_resnet101_2", "./cams/cw_step1000/wrn101")
    # draw_cam("../attacks/advs/cw_step1000", "wide_resnet101_2_dnn", "./cams/cw_step1000/wrn101_dnn")
    # draw_cam("../attacks/advs/cw_step1000", "wide_resnet101_2_dnn_jpeg", "./cams/cw_step1000/wrn101_dnn_jpeg")

    # 0.1711 0.819 0.8367
    # draw_cam("../attacks/advs/rfgsm", "wide_resnet101_2", "./cams/rfgsm/wrn101")
    # draw_cam("../attacks/advs/rfgsm", "wide_resnet101_2_dnn", "./cams/rfgsm/wrn101_dnn")
    # draw_cam("../attacks/advs/rfgsm", "wide_resnet101_2_dnn_jpeg", "./cams/rfgsm/wrn101_dnn_jpeg")