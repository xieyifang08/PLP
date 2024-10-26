import csv
import glob
import logging
import platform
import shutil
import argparse
import os
import sys

from torchsummary import summary


BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
from fast_adv.visualize.t_sne import tsne_imagenet

import tqdm
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from progressbar import *
import imageio
import pandas as pd
import random
import torch
import torch.nn.functional as F
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode


from fast_adv_imagenet.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv_imagenet.utils import model_utils

"""

"""

parser = argparse.ArgumentParser(description='CIFAR10 Training data augmentation')

parser.add_argument('--data', default='../data', help='path to dataset')
parser.add_argument('--img_size', default=32, type=int, help='size of image')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
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


def load_data_for_defense(input_dir, batch_size=args.batch_size):
    all_img_paths = glob.glob(os.path.join(input_dir, './*/*.png'))
    if platform.system() == "Windows":
        all_labels = [int(img_path.split('\\')[-2]) for img_path in all_img_paths]
    else:
        all_labels = [int(img_path.split('/')[-2]) for img_path in all_img_paths]

    dev_data = pd.DataFrame({'image_path': all_img_paths, 'label_idx': all_labels})

    transformer = transforms.Compose([
        transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),

        transforms.ToTensor()
    ])
    datasets = {
        'dev_data': ImageSet(dev_data, transformer)
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=0,
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
        if image.shape[0] == 1:
            image = torch.cat([image, image, image], dim=0)
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

    cam = GradCAM(model=model, target_layers=[model.model.aspp.convs[1]])

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
    test_loader = load_data_for_defense(database_path)['dev_data']
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


def draw_clean_classfiyTrue_cam(model_name, database_path, out_path):
    test_loader = load_data_for_defense(database_path)['dev_data']

    model = model_utils.load_model(model_name, pretrained=True).to(DEVICE)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    cam = GradCAM(model=model, target_layers=[model.model.layer4[-1]])

    for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        for i in range(images.shape[0]):
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
        cams.append(GradCAM(model=model, target_layers=[model.model.aspp.convs[-1]]))

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




def statistic_cam(input_org, input_adv):
    all_clean_npys = glob.glob(os.path.join(input_org, './*.npy'))
    # all_adv_npys = glob.glob(os.path.join(input_adv, './*.npy'))
    # all_ddn_jpeg_npys = glob.glob(os.path.join(input_dnn_jpeg, './*.npy'))
    # csv_writer = csv.writer(out, dialect="excel")
    # csv_writer.writerow(["image name", "wide resnet101 base", "wide resnet101 base at", "wide resnet101 base at jpeg"])
    attention_dict = {}
    for img in tqdm.tqdm(all_clean_npys):
        adv_name = img.split("/./")[-1]
        adv = os.path.join(input_adv, adv_name)
        im = np.load(img)
        ad = np.load(adv)
        if abs(1 - im.max()) > 1e-4 or im.min() > 1e-4 or abs(1 - ad.max()) > 1e-4 or ad.min() > 1e-4:
            continue
        attention_dict[adv_name] = np.var(np.load(img) - np.load(adv))
        # csv_writer.writerow([npy_name, np.load(org_path).mean(), np.load(input_dnn + "/./" + npy_name).mean(),
        #                          np.load(input_dnn_jpeg + "/./" + npy_name).mean()])
    return attention_dict



def statistic_cam_area(clean_correct_list, adv_correct_list, input_org, input_adv):
    all_clean_npys = glob.glob(os.path.join(input_org, './*.npy'))
    attention_dict = {}
    for img in tqdm.tqdm(all_clean_npys):

        adv_name = img.split("/./")[-1]
        # if adv_name.replace(".npy","") in adv_correct_list:
        #     continue
        # if adv_name not in clean_correct_list:
        #     continue
        adv = os.path.join(input_adv, adv_name)
        im = np.load(img)
        ad = np.load(adv)
        if abs(1 - im.max()) > 1e-4 or im.min() > 1e-4 or abs(1 - ad.max()) > 1e-4 or ad.min() > 1e-4:
            continue
        im[im < 0.5] = 0
        im[im >= 0.5] = 1
        ad[ad < 0.5] = 0
        ad[ad >= 0.5] = 1
        area = np.sum((im + ad) == 2) / np.sum((im + ad) > 0)
        attention_dict[adv_name] = area, im
    return attention_dict

def statistic_cam_value(rate, input_org, input_adv):
    all_clean_npys = glob.glob(os.path.join(input_org, './*.npy'))
    attention_dict = {}
    for img in tqdm.tqdm(all_clean_npys):
        adv_name = img.split("/./")[-1]
        # if adv_name.replace(".npy","") in adv_correct_list:
        #     continue
        # if adv_name.replace(".npy","") not in clean_correct_list:
        #     continue
        adv = os.path.join(input_adv, adv_name)
        im = np.load(img, encoding='bytes', allow_pickle=True)
        ad = np.load(adv, encoding='bytes', allow_pickle=True)
        if abs(1 - im.max()) > 1e-4 or im.min() > 1e-4 or abs(1 - ad.max()) > 1e-4 or ad.min() > 1e-4:
            continue
        im4move, ad4move = im.copy(), ad.copy()
        im4move[im4move < 0.5] = 0
        im4move[im4move >= 0.5] = 1
        ad4move[ad4move < 0.5] = 0
        ad4move[ad4move >= 0.5] = 1

        area = np.sum((im4move + ad4move) == 2)
        total = np.sum((im4move + ad4move) >= 1)
        area_rate = area / total
        if area_rate < rate:
            continue
        im[im < 0.5] = 0
        clean_sum = np.sum(im)
        ad[ad < 0.5] = 0
        adv_sum = np.sum(ad)
        attention_dict[adv_name] = (adv_sum - clean_sum)/total
    return attention_dict

def compare_attention_value(baseline_dict, my_dict):
    money, illegal, total = 0, 0, 0
    sum_money=0.0

    for name in baseline_dict.keys():
        if name in my_dict:
            total += 1
            baseline_cam_value = baseline_dict[name]
            anl_cam_value = my_dict[name]
            if anl_cam_value < 0 and baseline_cam_value > 0:
                illegal += 1
                continue
            sum_money += anl_cam_value - baseline_cam_value
            if anl_cam_value >= baseline_cam_value:
                money += 1
    return total, illegal, money, sum_money


def compare_attention_area2(baseline_dict, my_dict):
    money, total = 0, 0
    baseline_total_list = []
    my_total_list = []
    better = 0.0
    baseline_sum = 0.0
    my_sum = 0.0
    fuck = 0
    for name in baseline_dict.keys():
        if name in my_dict:
            baseline_iou, _bs = baseline_dict[name]
            my_iou, _ms = my_dict[name]

            if baseline_iou > 0.9:
                fuck+=1
                continue
            total += 1
            baseline_sum += baseline_iou
            my_sum += my_iou
            if baseline_iou <= my_iou:
                money += 1
    print(fuck)
    print(total, money, baseline_sum, my_sum)

def compare_attention_area(baseline_dict, my_dict):
    money, total = 0, 0
    baseline_total_list = []
    my_total_list = []

    for name in baseline_dict.keys():
        if name in my_dict:
            total += 1
            baseline_total_list.append(baseline_dict[name])
            my_total_list.append(my_dict[name])
            if baseline_dict[name] <= my_dict[name]:
                money += 1
    print(len(baseline_dict.keys()), len(my_dict.keys()), total, money, np.array(baseline_total_list).sum(),
          np.array(my_total_list).sum())

# def statistic_single_cam(input_org, input_adv):
#     all_clean_npys = glob.glob(os.path.join(input_org, './*.npy'))
#     attention_dict = {}
#     for img in tqdm.tqdm(all_clean_npys):
#         adv_name = img.split("/./")[-1]
#         adv = os.path.join(input_adv, adv_name)
#         im = np.load(img)
#         ad = np.load(adv)
#         if abs(1 - im.max()) > 1e-4 or im.min() > 1e-4 or abs(1 - ad.max()) > 1e-4 or ad.min() > 1e-4:
#             continue
#         im[im < 0.5] = 0
#         im[im >= 0.5] = 1
#         ad[ad < 0.5] = 0
#         ad[ad >= 0.5] = 1
#         area = np.sum((im + ad) == 2)
#         total_area = np.sum((im + ad) >= 1)
#         attention_dict[adv_name] = area/total_area
#     return attention_dict

def softmax(f):
    f -= np.max(f) # f becomes [-666, -333, 0]
    return np.exp(f) / np.sum(np.exp(f))

def statistic_single_cam(input_org, input_adv):
    all_clean_npys = glob.glob(os.path.join(input_org, './*.npy'))
    attention_dict = {}
    att_sum = 0.0
    for img in tqdm.tqdm(all_clean_npys):
        if platform.system() == "Windows":
            adv_name = img.split("\\.\\")[-1]
        else:
            adv_name = img.split("/./")[-1]
        adv = os.path.join(input_adv, adv_name)
        im = np.load(img)
        ad = np.load(adv)
        if im.max() < 0.9 or ad.max() < 0.9:
            # print("{} is inlegal".format(adv_name))
            continue
        att = softmax(im-ad)
        value = np.var(att)
        attention_dict[adv_name] = value
        att_sum += value

    return attention_dict, att_sum/len(all_clean_npys)

def stastic_att_value_change(img_names, input_clean, input_adv):
    att_sum = [0.0, 0.0, 0.0, 0.0]
    for img in tqdm.tqdm(img_names):
        clean = os.path.join(input_clean, img)
        adv = os.path.join(input_adv, img)
        im = np.load(clean)
        ad = np.load(adv)
        i = 0
        for threshold in [0.5, 0.6, 0.7, 0.8]:
            im[im < threshold] = 0
            ad[ad < threshold] = 0
            att_sum[i] += np.sum(im-ad)/np.sum(im)
            i+=1
    print(att_sum[0], att_sum[1], att_sum[2], att_sum[3], len(img_names))

def compare_single_attention_area(clean_classify, adv_classify, baseline_dict, rate):

    correct_up, correct_down, wrong_up, wrong_down = [], [], [], []

    for name in baseline_dict.keys():
        if name.replace(".npy", "") not in clean_classify:
            continue
        if name.replace(".npy", "") in adv_classify:
            if baseline_dict[name] >= rate:
                correct_up.append(name)
            else:
                correct_down.append(name)
        else:
            if baseline_dict[name] >= rate:
                wrong_up.append(name)
            else:
                wrong_down.append(name)
    logging.info("重合面积rate: {}, total img: {}, correct_up:{}, correct_down:{}, wrong_up:{}, wrong_down:{}".format(rate, len(baseline_dict.keys()), len(correct_up), len(correct_down), len(wrong_up), len(wrong_down)))
    return wrong_up

def compare_correct_wrong_attenion_not_move_value(npys_list, input_org, input_adv):
    total, move_att = 0, 0
    for img in tqdm.tqdm(npys_list):
        clean_img = os.path.join(input_org, img)
        adv_img = os.path.join(input_adv, img)
        clean_img = np.load(clean_img)
        adv_img = np.load(adv_img)
        if abs(1 - clean_img.max()) > 1e-4 or clean_img.min() > 1e-4 or abs(1 - adv_img.max()) > 1e-4 or adv_img.min() > 1e-4:
            continue
        clean_img[clean_img < 0.5] = 0
        adv_img[adv_img < 0.5] = 0
        total += 1
        if np.sum(clean_img) > np.sum(adv_img):
            move_att += 1
    logging.info("wrong up: total: {}, clean_att > adv_att: {}".format(total, move_att))


def compare_attention_var(baseline_dict, my_dict):
    money, total = 0, 0
    baseline_total_list = []
    my_total_list = []

    for name in baseline_dict.keys():
        if name in my_dict:
            total += 1
            baseline_total_list.append(baseline_dict[name].min())
            my_total_list.append(my_dict[name].min())
            if baseline_dict[name].min() > my_dict[name].min():
                money += 1
    print(len(baseline_dict.keys()), len(my_dict.keys()), total, money, np.array(baseline_total_list).sum(),
          np.array(my_total_list).sum())

    # print("good list: {}".format(good_list))
    # print("bad list: {}".format(bad_list))


def get_clean_classfiyTrue_list(model_name, database_path):
    test_loader = load_data_for_defense(database_path)['dev_data']

    model = model_utils.load_model(model_name, pretrained=True).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    test_accs = AverageMeter()
    correct_classify = []
    confidence = {}
    with torch.no_grad():
        model.eval()
        for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
            images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), \
                                       batch_data['filename']
            logits = model(images)
            h_x = F.softmax(logits, dim=1).data.squeeze()
            probs, idx = h_x.sort(1, True)
            probs = probs.cpu().numpy()
            idx = idx.cpu().numpy()

            for i in range(images.shape[0]):
                if logits.argmax(1)[i] == labels[i]:
                    correct_classify.append(filename[i] + ".npy")

                confidence[filename[i] + ".npy"] = []
                for j in range(0, 5):
                    # confidence[filename[i] + ".npy"].append(probs[i,j])
                    confidence[filename[i] + ".npy"].append(idx[i, j])
                    # confidence[filename[i] + ".npy"].append('{:.3f}-{}'.format(probs[i,j], idx[i,j]))

            test_accs.append((logits.argmax(1) == labels).float().mean().item())
    # print(test_accs.avg)
    return correct_classify, confidence

def get_wrong_prediction_list(model_name, database_path):
    test_loader = load_data_for_defense(database_path)['dev_data']

    model = model_utils.load_model(model_name, pretrained=True).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    test_accs = AverageMeter()
    predictions = {}
    with torch.no_grad():
        model.eval()
        for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
            images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), \
                                       batch_data['filename']
            logits = model(images)
            for i in range(images.shape[0]):
                pre = logits.argmax(1)[i]
                label = labels[i]
                if pre != label:
                    predictions[filename[i] + ".npy"] = "{}:{}".format(pre, label)
            test_accs.append((logits.argmax(1) == labels).float().mean().item())
    print(test_accs.avg)
    return predictions

def draw_tsne(model_name, database_path):
    test_loader = load_data_for_defense(database_path)['dev_data']
    model = model_utils.load_model(model_name).to(DEVICE)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        image_batch, label_batch, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), \
                                   batch_data['filename']
        label_batch = label_batch.long().squeeze()
        inputs = image_batch
        feature = model.feature_map(inputs)
        b,c,w,h = feature.shape
        if i == 0:
            feature = feature.view(b,c*w*h)
            feature_bank = feature.cpu().detach().numpy()
            label_bank = label_batch.cpu().numpy()
        else:
            feature = feature.view(b, c * w * h)
            feature_bank = np.concatenate((feature_bank, feature.cpu().detach().numpy()))
            label_bank = np.concatenate((label_bank, label_batch.cpu().numpy()))
    tsne_imagenet(feature_bank, label_bank)

def stastic_clean_confidence2(confidence_dict, adv_confidence_dict):
    adv_labels = [0 for _ in range(100)]
    clean_labels = [0 for _ in range(100)]

    for img in confidence_dict.keys():
        if img in adv_confidence_dict.keys():
            adv_confidence_list = adv_confidence_dict[img]
            confidence_list = confidence_dict[img]
            adv_labels[adv_confidence_list[0]] += 1
            clean_labels[confidence_list[0]] += 1
            # print(confidence_list, adv_confidence_list)
        # for idx in range(len(confidence_list)):
        #     top_confs[idx] += confidence_list[idx]
    # print("conf: ", top_confs[0],top_confs[1],top_confs[2],len(img_names))
    print("clean", clean_labels)
    print("adv", adv_labels)

def stastic_attention_map_value(baseline_wrongs, anl_wrongs, baseline_input, anl_input):
    res = {}
    for img in tqdm.tqdm(baseline_wrongs.keys()):
        if img not in anl_wrongs.keys():
            clean = np.load(os.path.join(baseline_input, img))
            adv = np.load(os.path.join(anl_input, img))
            clean_sum = np.sum(clean)
            clean[clean<0.5] = 0
            clean_sum_att = np.sum(clean)

            adv_sum = np.sum(adv)
            adv[adv < 0.5] = 0
            adv_sum_att = np.sum(adv)

            res[img] = [baseline_wrongs[img], clean_sum, adv_sum, clean_sum_att, adv_sum_att]
    return res

if __name__ == '__main__':
    baseline_model_name = "wide_resnet101_imagenet100_backbone"
    dfp_model_name = "imagenet100_wide_resnet101_dfp_replace_conv1"
    anl_model_name = "imagenet100_wide_resnet101_anl"
    adv_name = "tifgsm_eps10_step10"
    # baseline_wrongs = get_wrong_prediction_list(baseline_model_name, "../attacks/advs_new/alexnet/"+adv_name)
    # anl_wrongs = get_wrong_prediction_list(anl_model_name, "../attacks/advs_new/alexnet/"+adv_name)
    # print(stastic_attention_map_value(baseline_wrongs, anl_wrongs, "./cams_new/" + baseline_model_name + "/"+adv_name+"/npys", "./cams_new/" + anl_model_name + "/"+adv_name+"/npys"))
    # sys.exit(0)
    # draw_clean_classfiyTrue_cam("vgg19", "../attacks/advs_imagenet_new/vgg19/clean", "./cam_imagenet_new/" + "vgg19" + "/clean")
    # draw_clean_classfiyTrue_cam("vgg19", "../attacks/advs_imagenet_new/vgg19/fgsm_eps0.1", "./cam_imagenet_new/" + "vgg19" + "/fgsm_eps0.1")


    # draw_clean_classfiyTrue_cam(baseline_model_name, "../attacks/advs_new/"+baseline_model_name+"/clean/", "./cams/" + baseline_model_name + "/clean")
    # draw_clean_classfiyTrue_cam(baseline_model_name, "../attacks/advs/"+baseline_model_name+"/fgsm_eps25", "./cams/" + baseline_model_name + "/fgsm_eps25")

    # clean_classify = get_clean_classfiyTrue_list(baseline_model_name, "../attacks/advs/"+baseline_model_name+"/clean")
    # draw_tsne(baseline_model_name, "../attacks/advs_new/" + baseline_model_name + "/clean")
    # sys.exit(0)
    # clean_classify, confidence_dict = get_clean_classfiyTrue_list(baseline_model_name,
    #                                                               "../attacks/advs_new/" + baseline_model_name + "/clean")
    # for adv in ["fgsm_eps25", "rfgsm_eps25_step30", "pgd_eps25_step30", "mifgsm_eps25_step30", "difgsm_eps25_step30",
    #             "tifgsm_eps25_step30", "cw_c100_step500", "square_eps25"]:
    #     print(adv)
    #     adv_classify, adv_confidence_dict = get_clean_classfiyTrue_list(baseline_model_name,
    #                                                                     "../attacks/advs_new/" + baseline_model_name + "/" + adv)
    #     # area_dict, att_avg = statistic_single_cam("./cams/" + baseline_model_name + "/clean/npys", "./cams/" + baseline_model_name + "/"+adv+"/npys")
    #     # print(att_avg)
    #     # up_right, down_right, up_wrong, down_wrong = 0,0,0,0
    #
    #     # adv_classify = get_clean_classfiyTrue_list(baseline_model_name, "../attacks/advs_new/"+baseline_model_name+"/"+adv)
    #     # weak_names = []
    #     # for img in area_dict.keys():
    #     #     if area_dict[img] > att_avg:
    #     #         if img in adv_classify:
    #     #             up_right += 1
    #     #         else:
    #     #             up_wrong += 1
    #     #     else:
    #     #         if img in adv_classify:
    #     #             down_right += 1
    #     #         else:
    #     #             down_wrong += 1
    #     #             weak_names.append(img)
    #     # # print(adv, up_right, up_wrong, down_right, down_wrong)
    #     stastic_clean_confidence2(confidence_dict, adv_confidence_dict)

        # stastic_att_value_change(imgs, "./cams/" + baseline_model_name + "/clean/npys", "./cams/" + baseline_model_name + "/"+adv+"/npys")

    # clean_classify = get_clean_classfiyTrue_list("vgg19", "../attacks/advs_imagenet_new/vgg19/clean")
    # adv_classify = get_clean_classfiyTrue_list("vgg19", "../attacks/advs_imagenet_new/vgg19/fgsm_eps0.1")
    #
    # area_dict = statistic_single_cam("./cam_imagenet_new/vgg19/clean/npys", "./cam_imagenet_new/vgg19/fgsm_eps0.1/npys")
    # for rate in [0.8, 0.7, 0.6]:
    #     wrong_up = compare_single_attention_area(clean_classify,adv_classify, area_dict, rate)
    #     compare_correct_wrong_attenion_not_move_value(wrong_up, "./cam_imagenet_new/vgg19/clean/npys", "./cam_imagenet_new/vgg19/fgsm_eps0.1/npys")

    # draw_clean_classfiyTrue_cam(anl_model_name, "../attacks/advs/clean/", "./cams/" + anl_model_name + "/clean")
    # adv_name = "gn_std0.2"
    # compare_attention_var(baseline, dfp)"clean",
    # for adv_name in ["clean", "fgsm", "rfgsm", "mifgsm_eps10_step10", "tifgsm_eps10_step10", "difgsm_eps10_step10", "pgd_eps10_step10", "square_eps10", "cw_c100_step500", "deepfool_overshoot1"]:
        # draw_clean_classfiyTrue_cam(baseline_model_name, "../attacks/advs_new/alexnet/"+adv_name, "./cams_new/"+baseline_model_name+"/"+adv_name)
    #     draw_clean_classfiyTrue_cam(anl_model_name, "../attacks/advs_new/alexnet/" + adv_name, "./cams_new/" + anl_model_name + "/" + adv_name)
    #     draw_clean_classfiyTrue_cam(dfp_model_name, "../attacks/advs_new/alexnet/"+adv_name, "./cams_new/fpas_renew/"+adv_name)

    # print("baseline count, dfp count, total, money, baseline var sum, dfp var sum")
    # baseline_clean_correct_list = get_clean_classfiyTrue_list(baseline_model_name, '../attacks/advs/clean')
    # dfp_clean_correct_list = get_clean_classfiyTrue_list(dfp_model_name, '../attacks/advs/clean')

    # clean_classify, _ = get_clean_classfiyTrue_list(baseline_model_name, "../attacks/advs_new/alexnet/clean")
    #  注意力拉回效果实验
    for adv_name in ["fgsm", "rfgsm", "pgd_eps10_step10", "mifgsm_eps10_step10", "difgsm_eps10_step10", "tifgsm_eps10_step10",  "deepfool_overshoot1", "cw_c100_step500", "square_eps10"]:
        baseline = statistic_cam_area([], [], './cams_new/'+baseline_model_name+'/clean/npys', './cams_new/'+baseline_model_name+'/' + adv_name + '/npys')
        # dfp = statistic_cam_area([], [], './cams_new/'+dfp_model_name+'/clean/npys', './cams_new/'+dfp_model_name+'/' + adv_name + '/npys')
        dfp = statistic_cam_area([], [], './cams_new/fpas_renew/clean/npys', './cams_new/fpas_renew/' + adv_name + '/npys')

        compare_attention_area2(baseline, dfp)

    # # 注意力增强效果实验
    # for rate in [0.9, 0.8, 0.7, 0.6]:
    #     print("{}, total, better count, better value".format(rate))
    #     res = {}
    #     for adv_name in ["gn", "fgsm", "rfgsm", "mifgsm_eps10_step10", "tifgsm_eps10_step10", "difgsm_eps10_step10", "pgd_eps10_step10", "square_eps10", "cw_c100_step500", "deepfool_overshoot1"]:
    #         baseline = statistic_cam_value(rate,'./cams_new/'+baseline_model_name+'/clean/npys', './cams_new/'+baseline_model_name+'/' + adv_name + '/npys')
    #         anl = statistic_cam_value(rate, './cams_new/'+anl_model_name+'/clean/npys', './cams_new/'+anl_model_name+'/' + adv_name + '/npys')
    #         res[adv_name] = compare_attention_value(baseline, anl)
    #     for key in res.keys():
    #         print(key, res[key])