import logging
import os
import platform
import sys

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
import argparse
import glob
import numpy as np

import matplotlib.pyplot as plt
import imageio
import pandas as pd
import torch
import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from progressbar import *
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger

from fast_adv_imagenet.utils import model_utils


parser = argparse.ArgumentParser()
parser.add_argument('--data', default='../data', help='path to dataset')
parser.add_argument('--output_file', default='output.csv',
                    help='Output file to save labels', type=str)
parser.add_argument('--target_model', default='densenet161',
                    help='cnn model, e.g. , densenet121, densenet161', type=str)
parser.add_argument('--gpu_id', default=0, nargs='+',
                    help='gpu ids to use, e.g. 0 1 2 3', type=int)
parser.add_argument('--batch_size', default=5,
                    help='batch size, e.g. 16, 32, 64...', type=int)
parser.add_argument('--img_size', default=224, type=int, help='size of image')
parser.add_argument('--batch-size', '-b', default=4, type=int, help='mini-batch size')

args = parser.parse_args()
print(args)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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


def load_data_for_defense(input_dir, img_size=args.img_size, batch_size=args.batch_size):
    # all_imgs = glob.glob(os.path.join(input_dir, './*/*.png'))
    all_imgs = []
    for i in range(2,10):
        imgs = glob.glob(os.path.join(input_dir, str(i), '*.png'))
        all_imgs.extend(imgs)
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

def draw_clean_classfiyTrue_cam(database_path, out_path):
    test_loader = load_data_for_defense(database_path)['dev_data']

    model = model_utils.load_model("alexnet_4att_move", num_classes=10).to(DEVICE)
    weight = "../defenses/weights/alexnet_for_move/best_imagenet100_ep_75_val_acc0.6252.pth"
    loaded_state_dict = torch.load(weight)
    model.load_state_dict(loaded_state_dict)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    GradCAMs = [
        GradCAM(model=model, target_layers=[model.model.features1]),
        GradCAM(model=model, target_layers=[model.model.features2]),
        GradCAM(model=model, target_layers=[model.model.features3]),
    ]

    for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels, filename = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE), batch_data[
            'filename']
        logits = model(images)
        for i in range(images.shape[0]):
            # if filename[i] in correct_classify: logits.argmax(1)[i] == labels[i]:
            grayscale_cam_1 = GradCAMs[0](input_tensor=images[i].unsqueeze(0), target_category=labels[i].cpu().item())
            grayscale_cam_2 = GradCAMs[1](input_tensor=images[i].unsqueeze(0), target_category=labels[i].cpu().item())
            grayscale_cam_3 = GradCAMs[2](input_tensor=images[i].unsqueeze(0), target_category=labels[i].cpu().item())

            img = np.transpose(images[i].detach().cpu().numpy(), (1, 2, 0))
            ori = img.copy()
            visualization_1 = show_cam_on_image(img, grayscale_cam_1[0, :], use_rgb=True)
            visualization_2 = show_cam_on_image(img, grayscale_cam_2[0, :], use_rgb=True)
            visualization_3 = show_cam_on_image(img, grayscale_cam_3[0, :], use_rgb=True)
            plt.subplot(2, 2, 1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
            plt.title("pre:{}, true:{}".format(logits.argmax(1)[i], labels[i]))
            plt.subplot(2, 2, 2)
            plt.imshow(visualization_1)
            plt.xticks([])
            plt.yticks([])
            plt.title("feature1")
            plt.subplot(2, 2, 3)
            plt.imshow(visualization_2)
            plt.xticks([])
            plt.yticks([])
            plt.title("feature2")
            plt.subplot(2, 2, 4)
            plt.imshow(visualization_3)
            plt.xticks([])
            plt.yticks([])
            plt.title("feature3")
            # plt.show()
            out_path_preview = os.path.join(out_path, "preview")
            if not os.path.exists(out_path_preview):
                os.makedirs(out_path_preview)
            plt.savefig(os.path.join(out_path_preview, filename[i]))
            out_path_imgs = os.path.join(out_path, "imgs", filename[i])
            if not os.path.exists(out_path_imgs):
                os.makedirs(out_path_imgs)

            out_path_imgs_ori = os.path.join(out_path_imgs, "ori.png")
            out_path_imgs_f1 = os.path.join(out_path_imgs, "feature1.png")
            out_path_imgs_f2 = os.path.join(out_path_imgs, "feature2.png")
            out_path_imgs_f3 = os.path.join(out_path_imgs, "feature3.png")
            imageio.imsave(out_path_imgs_ori, ori)
            imageio.imsave(out_path_imgs_f1, visualization_1.astype('uint8'))
            imageio.imsave(out_path_imgs_f2, visualization_2.astype('uint8'))
            imageio.imsave(out_path_imgs_f3, visualization_3.astype('uint8'))

            # out_path_img = out_path + "/images"
            # if not os.path.exists(out_path_img):
            #     os.makedirs(out_path_img)
            # out_path_npy = out_path + "/npys"
            # if not os.path.exists(out_path_npy):
            #     os.makedirs(out_path_npy)
            # out = os.path.join(out_path_img, filename[i])
            # np.save(os.path.join(out_path_npy, filename[i]), grayscale_cam)
            # imageio.imsave(out, visualization)

if __name__ == '__main__':
    draw_clean_classfiyTrue_cam('../attacks/advs/alexnet/gn_std0.2', "./cams/gn_std0.2")