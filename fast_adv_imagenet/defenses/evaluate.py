import os,sys
BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
import argparse

from progressbar import *
import imageio
import pandas as pd
import torch
from PIL import Image

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from torchvision.transforms import InterpolationMode
import tqdm

from fast_adv_imagenet.utils import model_utils, AverageMeter


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
DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')

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



# test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), os.path.join(args.data, 'images'))['dev_data']
#
#
# print( len(test_loader))
#
# model = model_utils.load_model("resnet152").to(DEVICE)
# # model.eval()
# # print("data_loader :" , len(test_loader))
# # for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
# #     images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
# #     logits = model(images.detach())
# #     test_accs = AverageMeter()
# #     test_accs.append((logits.argmax(1) == labels).float().mean().item())
# # print('\nTest accuracy ', test_accs.avg)

def test_accuracy(model_name, data_path):
    model = model_utils.load_model(model_name).to(DEVICE)
    model.eval()
    test_loader = load_data_for_defense(os.path.join(args.data, 'dev.csv'), data_path)['dev_data']
    for i, batch_data in enumerate(tqdm.tqdm(test_loader, ncols=80)):
        images, labels = batch_data['image'].to(DEVICE), batch_data['label_idx'].to(DEVICE)
        logits = model(images.detach())
        test_accs = AverageMeter()
        test_accs.append((logits.argmax(1) == labels).float().mean().item())
    # print(test_accs)
    # test_losses.append(loss.item())
    print('model:{}, data:{}, test accuracy:{} '.format(model_name, data_path, test_accs.avg))
    return test_accs.avg


# "vgg19", "resnet50", "resnet152","wide_resnet101_2", "wide_resnet101_2_dnn_jpeg",
models = ["ens_adv_inception_resnet_v2"]
datas = ["cw", "fgsm", "high_attack_fgsm_inp_m_2attack", "pgd","pgd_rn50_eps8_step8", "pgd_rn50_eps10_step10","rfgsm"]
ans = []
for model in models:
    for data in datas:
        data_path = '/home/unknown/PLP/fast_adv_imagenet/attacks/advs/'+data
        ans.append("{},{},{}".format(model, data, test_accuracy(model, data_path)))

print(ans)