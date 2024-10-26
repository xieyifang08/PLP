import csv
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
import foolbox

from fast_adv.attacks.difgsm import DIFGSM
from fast_adv.attacks.mifgsm import MIFGSM
from fast_adv.attacks.pgd import PGD
from fast_adv.attacks.rfgsm import RFGSM
from fast_adv.attacks.tifgsm import TIFGSM
from fast_adv.attacks.cw import CW
import argparse
import tqdm
import glob
import torch
from torchvision import transforms
import imageio
import warnings
from torchattacks import DeepFool, MultiAttack, SparseFool, AutoAttack, Square
import logging
import pandas as pd
import platform
from torch.utils.data import Dataset, DataLoader
from PIL import Image,ImageFile
from fast_adv_imagenet.utils.messageUtil import send_email
from fast_adv_imagenet.utils import model_utils, AverageMeter
from torchvision.transforms import InterpolationMode
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Extend sample')
parser.add_argument('--max-norm', type=float, default=10, help='max norm for the adversarial perturbations')
# parser.add_argument('--img_size', default=224, type=int, help='pic size')
parser.add_argument('--data', default='../defenses/data/cifar10', help='path to dataset')
parser.add_argument('--attack_name', '--at', default='DDN',
                    help='name for saving the final state dict')
parser.add_argument('--batch_size', default=16,
                    help='batch size, e.g. 16, 32, 64...', type=int)
parser.add_argument('--img_size', default=224, type=int, help='size of image')
parser.add_argument("--clean", type=bool, default=False)
parser.add_argument("--rfgsm", type=bool, default=False)
parser.add_argument("--fgsm", type=bool, default=False)
parser.add_argument("--pgd", type=bool, default=False)
parser.add_argument("--mifgsm", type=bool, default=False)
parser.add_argument("--difgsm", type=bool, default=False)
parser.add_argument("--tifgsm", type=bool, default=False)
parser.add_argument("--model_name", type=str, default="wide_resnet_mwe")
parser.add_argument("--attack_functions", type=list, default=["fgsm", "rfgsm", "pgd"])
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--save_epoch", type=int, default=10)

args = parser.parse_args()
print(args)

DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
logging.info("device: {}".format(DEVICE))


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


data_loader = load_data_for_defense("../../../imagenet100_test")['dev_data']
logging.info(len(data_loader))



def white_attack_func(model_name, attack_names, epochs, step):
    model = model_utils.load_model(model_name, pretrained=True).to(DEVICE)
    model.eval()
    attack_data = {}
    attack_accs = {}
    attack_funcs = {}
    for attack_name in attack_names:
        attack_data[attack_name] = []
        attack_accs[attack_name] = []
        if "rfgsm".__eq__(attack_name):
            attack_funcs[attack_name] = RFGSM(model, steps=epochs)
        if "pgd".__eq__(attack_name):
            attack_funcs[attack_name] = PGD(model, steps=epochs)
        if "mifgsm".__eq__(attack_name):
            attack_funcs[attack_name] = MIFGSM(model, steps=epochs)
        if "difgsm".__eq__(attack_name):
            attack_funcs[attack_name] = DIFGSM(model, steps=epochs)
        if "tifgsm".__eq__(attack_name):
            attack_funcs[attack_name] = TIFGSM(model, steps=epochs)
    for idx in range(0, epochs, step):
        for func in attack_accs.keys():
            attack_accs[func].append(AverageMeter())

    for batch_data in tqdm.tqdm(data_loader):
        images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
        for func_name in attack_names:
            advs = attack_funcs[func_name](images, labels)
            for idx in range(0, epochs, step):
                attack_accs[func_name][idx].append((model(advs[idx].detach()).argmax(1) == labels).float().mean().item())
    for idx in range(0, epochs, step):
        # logging.info("white attacks accs is\n model: {}\nepoch: {}\nrfgsm_10: {}\npgd_10: {}\nmifgsm_10: {}\ndifgsm_10: {}\ntifgsm_10: {}".format(
        #         model_name, epoch, rfgsm_10_accs.avg, pgd_10_accs.avg, mifgsm_10_accs.avg, difgsm_10_accs.avg, tifgsm_10_accs.avg))
        for func_name in attack_names:
            attack_data[func_name].append(attack_accs[func_name][idx].avg)
    return attack_data


def white_attack_func2(model_name, epochs, step):
    model = model_utils.load_model(model_name, pretrained=True).to(DEVICE)
    model.eval()
    attack_data = {}
    # attack_data["cw"] = []
    attack_data["deepfool"] = []
    # attack_data["square"] = []

    for epoch in range(5, epochs+1, step):
        # cw = CW(model, steps=epoch)
        deepfool = DeepFool(model, steps=epoch)
        # square = Square(model, n_queries=epoch, eps=10 / 255)
        # cw_accs = AverageMeter()
        deepfool_accs = AverageMeter()
        # square_accs = AverageMeter()
        for batch_data in tqdm.tqdm(data_loader):
            images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
            # cw_advs = cw(images, labels)
            deepfool_advs = deepfool(images, labels)
            # square_advs = square(images, labels)
            # cw_accs.append((model(cw_advs.detach()).argmax(1) == labels).float().mean().item())
            deepfool_accs.append((model(deepfool_advs.detach()).argmax(1) == labels).float().mean().item())
            # square_accs.append((model(square_advs.detach()).argmax(1) == labels).float().mean().item())
        # attack_data["cw"].append(cw_accs.avg)
        attack_data["deepfool"].append(deepfool_accs.avg)
        # attack_data["square"].append(square_accs.avg)
        logging.info(("white attacks accs is model: {}, epoch: {}, deepfool: {}".format(
                model_name, epoch, deepfool_accs.avg)))
    return attack_data

def export2csv(attack_data, output):
    with open(output, "w") as csv_file:
        writer = csv.writer(csv_file)
        for key, value in attack_data.items():
            tmp = [key]
            tmp.extend(value)
            writer.writerow(tmp)

if __name__ == '__main__':
    # export2csv(white_attack_func("imagenet100_wide_resnet101_dfp_replace_conv1", ["rfgsm", "pgd", "mifgsm", "difgsm", "tifgsm"], 10, 1), "./imagenet100_wide_resnet101_dfp.csv")
    # export2csv(white_attack_func("wide_resnet101_imagenet100_backbone", ["rfgsm", "pgd", "mifgsm", "difgsm", "tifgsm"], 10, 1), "./wide_resnet101_imagenet100_backbone.csv")
    export2csv(
        white_attack_func("imagenet100_wide_resnet101_anl", ["rfgsm", "pgd", "mifgsm", "difgsm", "tifgsm"], 10, 1),
        "./imagenet100_wide_resnet101_anl.csv")

    # export2csv(
    #     white_attack_func2("wide_resnet101_imagenet100_backbone", 10, 1),
    #     "./wide_resnet101_imagenet100_backbone2.csv")
    # export2csv(
    #     white_attack_func2("imagenet100_wide_resnet101_dfp",10, 1),
    #     "./imagenet100_wide_resnet101_dfp2.csv")
    # export2csv(white_attack_func("wide_resnet_weak_self_att", 10, 1), "./wide_resnet_weak_self_att3.csv")
    send_email("imagenet100 白盒测试完成", title="白盒测试实验")

