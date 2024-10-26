import logging
import os
import platform
import sys
BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
import argparse
import glob
import json

import imageio
import pandas as pd
import torch
import tqdm
import numpy as np
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset, DataLoader
from progressbar import *
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import matplotlib.pyplot as plt
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv_imagenet.process.preprocess import GridMaskCompression

from fast_adv_imagenet.utils import model_utils

parser = argparse.ArgumentParser()
parser.add_argument('--data', default='../data/Diff_PGD_alexnet/alexnetnew', help='path to dataset')
parser.add_argument('--output_file', default='output.csv',
                    help='Output file to save labels', type=str)
parser.add_argument('--target_model', default='alexnet',
                    help='cnn model, e.g. , densenet121, densenet161', type=str)
parser.add_argument('--gpu_id', default=0, nargs='+',
                    help='gpu ids to use, e.g. 0 1 2 3', type=int)
parser.add_argument('--batch_size', default=8,
                    help='batch size, e.g. 16, 32, 64...', type=int)
parser.add_argument('--img_size', default=224, type=int, help='size of image')
parser.add_argument('--batch-size', '-b', default=16, type=int, help='mini-batch size')

args = parser.parse_args()
print(args)

input_data = [
    # '../attacks/advs_new/alexnet/clean',
    # # '../attacks/advs_new/alexnet/gn',
    # # '../attacks/advs_new/alexnet/gn_std0.2',
    # # '../attacks/advs_new/alexnet/gn_std0.3',
    # '../attacks/advs_new/alexnet/fgsm',
    # # '../attacks/advs_new/alexnet/fgsm_eps0.07',
    # # '../attacks/advs_new/alexnet/fgsm_eps0.1',
    # '../attacks/advs_new/alexnet/rfgsm',
    # # '../attacks/advs_new/alexnet/rfgsm_eps20_step50',
    # # '../attacks/advs_new/alexnet/rfgsm_eps25_step50',
    #
    # '../attacks/advs_new/alexnet/pgd_eps10_step10',
    # # '../attacks/advs_new/alexnet/pgd_eps20_step10',
    # # '../attacks/advs_new/alexnet/pgd_eps30_step20',
    #
    # '../attacks/advs_new/alexnet/mifgsm_eps10_step10',
    # # '../attacks/advs_new/alexnet/mifgsm_eps20_step10',
    # # '../attacks/advs_new/alexnet/mifgsm_eps25_step20',
    #
    # '../attacks/advs_new/alexnet/difgsm_eps10_step10',
    # # '../attacks/advs_new/alexnet/difgsm_eps20_step10',
    # # '../attacks/advs_new/alexnet/difgsm_eps25_step20',
    #
    # '../attacks/advs_new/alexnet/tifgsm_eps10_step10',
    # # '../attacks/advs_new/alexnet/tifgsm_eps20_step10',
    # # '../attacks/advs_new/alexnet/tifgsm_eps25_step20',
    #
    # # '../attacks/advs_new/alexnet/deepfool_overshoot0.5',
    # '../attacks/advs_new/alexnet/deepfool_overshoot1',
    # # '../attacks/advs_new/alexnet/deepfool_overshoot1_step100',
    #
    #
    # # '../attacks/advs_new/alexnet/cw',
    # # '../attacks/advs_new/alexnet/cw_c1',
    # # '../attacks/advs_new/alexnet/cw_c10',
    # # '../attacks/advs_new/alexnet/cw_c100',
    # '../attacks/advs_new/alexnet/cw_c100_step500',
    #
    # '../attacks/advs_new/alexnet/square_eps10',
    # # '../attacks/advs_new/alexnet/square_eps20',
    # # '../attacks/advs_new/alexnet/square_eps25',
    '../data/Diff_PGD_alexnet/alexnetnew'
]


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


def load_data_for_defense(input_dir, batch_size=args.batch_size):
    all_imgs = glob.glob(os.path.join(input_dir, './*/*.png'))
    system = platform.system()
    if system == "Windows":
        all_labels = [int(img_path.split('\\')[-2]) for img_path in all_imgs]
    else:
        all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]

    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})

    transformer = transforms.Compose([
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


def load_data_for_defense_gridmask(input_dir, batch_size=args.batch_size):
    all_imgs = glob.glob(os.path.join(input_dir, './*/*.png'))
    system = platform.system()
    if system == "Windows":
        all_labels = [int(img_path.split('\\')[-2]) for img_path in all_imgs]
    else:
        all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]

    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})

    transformer = transforms.Compose([
        transforms.ToTensor(),
        GridMaskCompression(),
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


if __name__ == '__main__':
    gpu_ids = args.gpu_id
    if isinstance(gpu_ids, int):
        gpu_ids = [gpu_ids]
    batch_size = args.batch_size
    target_model = args.target_model
    # inputDir = args.input_dir
    outputFile = args.output_file
    device = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
    print("device is :", device)
    # Loading data for ...

    model = model_utils.load_model("alexnet", pretrained=False, num_classes=1000).to(device)
    # weight = "../defenses/weights/best/best_imagenet100_wrn_clean_ep_99_val_acc0.7872.pth"  # wide_resnet101_imagenet100_backbone
    # weight = "../defenses/weights/best/best_imagenet100_wrn_at_ep_19_val_acc0.7715.pth"  # wide_resnet101_imagenet100_backbone

    # weight = "./weights/best/best_imagenet100_wrn_dfp_fp_clean_ep_49_val_acc0.7849.pth"  # imagenet100_wide_resnet101_dfp_fp
    # weight = "./weights/imagenet100_wide_resnet101_dfp_fp/best_imagenet100_ep_51_val_acc0.7791.pth"  # imagenet100_wide_resnet101_dfp_fp
    # weight = "./weights/imagenet100_wide_resnet101_dfp_fp/best_imagenet100_ep_81_val_acc0.7849.pth"
    weight = "./weights/imagenet_adv_train/alexnet/best_imagenet_ep_29_val_acc0.0010.pth"
    # weight = "../defenses/weights/best/best_imagenet100_wrn_dfp_clean_ep_145_val_acc0.8047.pth"  # imagenet100_wide_resnet101_dfp
    # weight = "../defenses/weights/best/best_imagenet100_wrn_dfp_at_ep_19_val_acc0.7971.pth"  # imagenet100_wide_resnet101_dfp
    # weight = "../defenses/weights/best/best_imagenet100_wrn_anl_clean_ep_33_val_acc0.7963.pth"  # imagenet100_wide_resnet101_anl
    # weight = "../defenses/weights/best/best_imagenet100_wrn_anl_at_ep_19_val_acc0.7420.pth"  # imagenet100_wide_resnet101_anl
    # weight = "../defenses/weights/best/best_imagenet100_wrn_mwe_clean_ep_31_val_acc0.7237.pth"  # imagenet100_wide_resnet101_mwe
    # weight = "../defenses/weights/best/best_imagenet100_wrn_mew_at_ep_16_val_acc0.7448.pth"  # imagenet100_wide_resnet101_mwe
    # weight = "../defenses/weights/best/best_imagenet100_wrn_alp_at_ep_29_val_acc0.7754.pth"  # wide_resnet101_imagenet100_backbone
    # weight = "../defenses/weights/best/best_imagenet100_wrn_aad_at_ep_29_val_acc0.7453.pth"  # wide_resnet101_imagenet100_backbone



    # weight = "../defenses/weights/best/best_imagenet100_wrn_dfp_clean_ep_5_val_acc0.8668.pth"  # imagenet100_wide_resnet101_dfp_replace_conv1
    # weight = "../defenses/weights/best/best_imagenet100_wrn_dfp_at_ep_3_val_acc0.7094.pth"  # imagenet100_wide_resnet101_dfp_replace_conv1
    # weight = "../defenses/weights/imagenet100_dfp_replace_conv1_rerun/best_imagenet100_ep_145_val_acc0.8047.pth"
    # weight = "../defenses/weights/imagenet100_dfp_replace_conv1_rerun_at/best_imagenet100_ep_19_val_acc0.7971.pth"
    # weight = "../defenses/weights/best/best_imagenet100_wrn_anl_clean_ep_33_val_acc0.7963.pth"  # imagenet100_wide_resnet101_anl
    # weight = "../defenses/weights/best/best_imagenet100_wrn_anl_at_ep_19_val_acc0.7420.pth"  # imagenet100_wide_resnet101_anl
    # weight = "../defenses/weights/imagenet100_anl_baseline_add_ft_at/imagenet100acc0.7838212025316456_10.pth"
    # weight = "../defenses/weights/imagenet100_anl_baseline_add_ft_at_2/imagenet100acc0.7420382165605095_20.pth"
    # weight = "../defenses/weights/imagenet100_anl_at_with_map/best_imagenet100_ep_4_val_acc0.7393.pth"
    # weight = "../defenses/weights/imagenet100_anl_block_rerun_12/best_imagenet100_ep_99_val_acc0.8294.pth"
    # weight = "../defenses/weights/imagenet100_anl_block_rerun_23/best_imagenet100_ep_59_val_acc0.8246.pth"
    # weight = "../defenses/weights/imagenet100_anl_block_rerun_34/best_imagenet100_ep_98_val_acc0.8190.pth"
    # weight = "../defenses/weights/best/best_imagenet100_wrn_mwe_clean_ep_31_val_acc0.7237.pth"  # imagenet100_wide_resnet101_mwe
    # weight = "../defenses/weights/imagenet100_mwe_at/best_imagenet100_ep_16_val_acc0.7448.pth"
    # weight = "../defenses/weights/imagenet100_wrn_alp/best_imagenet100_ep_29_val_acc0.7754.pth"
    # weight = "./weights/imagenet100_wrn_aad/best_imagenet100_ep_29_val_acc0.7453.pth"
    # weight = "./weights/imagenet100_anl_block12_only/best_imagenet100_ep_95_val_acc0.8165.pth"
    # weight = "./weights/imagenet100_anl_block23_only/best_imagenet100_ep_99_val_acc0.8090.pth"
    # weight ="./weights/imagenet100_anl_block23_only_rerun2/best_imagenet100_ep_91_val_acc0.7631.pth"

    # weight = "./weights/imagenet100_anl_block34_only_rerun/best_imagenet100_ep_94_val_acc0.7266.pth"
    # weight ="./weights/imagenet100_anl_block34_only_rerun2/best_imagenet100_ep_69_val_acc0.7354.pth"
    # weight = "./weights/imagenet100_anl_block4fc_only/best_imagenet100_ep_96_val_acc0.7252.pth"
    # weight = "./weights/imagenet100_dfp_replace_conv1_rerun/best_imagenet100_ep_145_val_acc0.8047.pth"
    loaded_state_dict = torch.load(weight)


    # model_gm = model_utils.load_model("wide_resnet101_imagenet100_backbone", pretrained=False, num_classes=100).to(device)
    # weight_gm = "../defenses/weights/best/best_imagenet100_wrn_at_ep_19_val_acc0.7715.pth"  # wide_resnet101_imagenet100_backbone
    # weight_gm = "../defenses/weights/best/best_imagenet100_wrn_anl_at_ep_19_val_acc0.7420.pth"  # imagenet100_wide_resnet101_anl
    # weight_gm = "./weights/imagenet100_wrn_aad/best_imagenet100_ep_29_val_acc0.7453.pth"  # wide_resnet101_imagenet100_backbone
    # weight_gm = "../defenses/weights/best/best_imagenet100_wrn_anl_at_ep_19_val_acc0.7420.pth"  # imagenet100_wide_resnet101_anl
    # model_gm.load_state_dict(torch.load(weight_gm))

    model.load_state_dict(loaded_state_dict)

    print("weight file is {}".format(weight))
    model.eval()
    # model_gm.eval()
    test_result = {}
    for i in range(len(input_data)):
        test_accs = AverageMeter()
        test_losses = AverageMeter()

        test_loader = load_data_for_defense(input_data[i])['dev_data']
        gridmask_loader = load_data_for_defense_gridmask(input_data[i])['dev_data']

        with torch.no_grad():
            for batch_data, gridmask_data in tqdm.tqdm(zip(test_loader, gridmask_loader)):
                images, labels = batch_data['image'].to(device), batch_data['label_idx'].to(device)
                gm_images = gridmask_data['image'].to(device)
                # # 将张量转换为NumPy数组
                # img_np = images[0].cpu().numpy()
                # # 显示图像
                # plt.imshow(np.transpose(img_np, (1, 2, 0)))
                # plt.show()
                logits = model(images.detach())
                test_accs.append((logits.argmax(1) == labels).float().mean().item())

        print('\nTest accuracy of %s:%s' % (input_data[i].split('/')[-1], test_accs.avg))
        test_result[input_data[i].split('/')[-1]] = test_accs.avg
    data = json.dumps(test_result, indent=4, ensure_ascii=False, sort_keys=False, separators=(',', ':'))
    dic = json.loads(data)
    print("weight file is {}".format(weight))
    for d in dic.keys():
        print(d, '\t', round(dic[d], 4))


# if __name__ == '__main__':
#     # weight = "../defenses/weights/best/best_imagenet100_wrn_clean_ep_99_val_acc0.7872.pth"  # wide_resnet101_imagenet100_backbone
#     # weight = "../defenses/weights/best/best_imagenet100_wrn_at_ep_19_val_acc0.7715.pth"  # wide_resnet101_imagenet100_backbone
#     # weight = "../defenses/weights/best/best_imagenet100_wrn_dfp_clean_ep_5_val_acc0.8668.pth"  # imagenet100_wide_resnet101_dfp_replace_conv1
#     # weight = "../defenses/weights/best/best_imagenet100_wrn_dfp_at_ep_3_val_acc0.7094.pth"  # imagenet100_wide_resnet101_dfp_replace_conv1
#     # weight = "../defenses/weights/best/best_imagenet100_wrn_anl_clean_ep_33_val_acc0.7963.pth"  # imagenet100_wide_resnet101_anl
#     # weight = "../defenses/weights/best/best_imagenet100_wrn_anl_at_ep_19_val_acc0.7420.pth"  # imagenet100_wide_resnet101_anl
#     # weight = "../defenses/weights/best/best_imagenet100_wrn_mwe_clean_ep_31_val_acc0.7237.pth"  # imagenet100_wide_resnet101_mwe
#     weights = [
#         # "../defenses/weights/best/best_imagenet100_wrn_clean_ep_99_val_acc0.7872.pth",  # wide_resnet101_imagenet100_backbone
#         # "../defenses/weights/best/best_imagenet100_wrn_at_ep_19_val_acc0.7715.pth",  # wide_resnet101_imagenet100_backbone
#
#         # "./weights/best/best_imagenet100_wrn_dfp_fp_clean_ep_49_val_acc0.7849.pth",  # imagenet100_wide_resnet101_dfp_fp
#         # "../defenses/weights/best/best_imagenet100_wrn_dfp_clean_ep_145_val_acc0.8047.pth",  # imagenet100_wide_resnet101_dfp
#         # "../defenses/weights/best/best_imagenet100_wrn_dfp_at_ep_19_val_acc0.7971.pth",  # imagenet100_wide_resnet101_dfp
#
#         # "../defenses/weights/best/best_imagenet100_wrn_anl_clean_ep_33_val_acc0.7963.pth",  # imagenet100_wide_resnet101_anl
#         # "../defenses/weights/best/best_imagenet100_wrn_anl_at_ep_19_val_acc0.7420.pth",  # imagenet100_wide_resnet101_anl
#
#         # "../defenses/weights/best/best_imagenet100_wrn_mwe_clean_ep_31_val_acc0.7237.pth",  # imagenet100_wide_resnet101_mwe
#         # "../defenses/weights/imagenet100_mwe_at/best_imagenet100_ep_16_val_acc0.7448.pth",  # imagenet100_wide_resnet101_mwe
#
#         # "../defenses/weights/imagenet100_wrn_alp/best_imagenet100_ep_29_val_acc0.7754.pth",  # wide_resnet101_imagenet100_backbone
#         # "./weights/imagenet100_wrn_aad/best_imagenet100_ep_29_val_acc0.7453.pth",  # wide_resnet101_imagenet100_backbone
#
#     ]
#     models = [
#         # "wide_resnet101_imagenet100_backbone",
#         # "wide_resnet101_imagenet100_backbone",
#         # "imagenet100_wide_resnet101_dfp_fp",
#         # "imagenet100_wide_resnet101_dfp",
#         # "imagenet100_wide_resnet101_dfp",
#         # "imagenet100_wide_resnet101_anl",
#         # "imagenet100_wide_resnet101_anl",
#         # "imagenet100_wide_resnet101_mwe",
#         # "imagenet100_wide_resnet101_mwe",
#         # "wide_resnet101_imagenet100_backbone",
#         # "wide_resnet101_imagenet100_backbone",
#     ]
#
#     for i in range(len(weights)):
#         elv_accs(models[i], weights[i])