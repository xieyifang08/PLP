import foolbox
import warnings
warnings.filterwarnings("ignore")
from foolbox.adversarial import Adversarial
from foolbox.criteria import Misclassification
from foolbox.distances import MeanAbsoluteDistance, Linfinity
from foolbox.attacks import FGSM,DeepFoolAttack, PGD, LocalSearchAttack, GaussianBlurAttack, \
    BinarizationRefinementAttack,ContrastReductionAttack,SaltAndPepperNoiseAttack,\
    SpatialAttack
import scipy.misc
from torch.utils import data
from sklearn.model_selection import train_test_split
import torchvision.models as models
import numpy as np
import PIL
import os
import glob
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
import argparse
from torch.utils.data import Dataset, DataLoader
from progressbar import *
from fast_adv.models.cifar10 import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN
from torchvision import transforms
from torchvision.datasets import CIFAR10

parser = argparse.ArgumentParser(description='Extend sample')
parser.add_argument('--max-norm', type=float, default=1,help='max norm for the adversarial perturbations')
parser.add_argument('--data', default='/media/unknown/Data/PLP/fast_adv/defenses/data/cifar10', help='path to dataset')
parser.add_argument('--imgsize', default=64, type=int, help='pic size')

parser.add_argument('--attack_name', '--at', default='PGD',
                    help='name for saving the final state dict')
args = parser.parse_args()
print(args)
attackers = {'FGSM':FGSM,
            'DeepFoolAttack': DeepFoolAttack,
             'PGD': PGD,
             'LocalSearchAttack': LocalSearchAttack,
             'GaussianBlurAttack': GaussianBlurAttack,
             'BinarizationRefinementAttack': BinarizationRefinementAttack,
             'ContrastReductionAttack':ContrastReductionAttack,
             'SaltAndPepperNoiseAttack':SaltAndPepperNoiseAttack,
             'SpatialAttack': SpatialAttack}
DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')
m = wide_resnet(num_classes=10, depth=28, widen_factor=10, dropRate=0.3)
image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
model_dict = torch.load('/media/unknown/Data/PLP/fast_adv/defenses/weights/cifar10/cifar10_60_clean0.879.pth')
model.load_state_dict(model_dict)
#imagenet_data = datasets.ImageNet('imagenet/',split='train',download=True)

# image,label=foolbox.utils.imagenet_example()
# model.eval()


def attack(image, label):
    # mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    # std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    fmodel =  foolbox.models.PyTorchModel(model.eval().cuda(), bounds=(0, 1),num_classes=110)#, preprocessing=(mean, std)
    criterion1 = Misclassification()
    distance = Linfinity#MeanAbsoluteDistance
    # advs=Adversarial(fmodel, criterion1, image, label,distance=distance)
    #adversarial= attackers[args.attack_name](advs)
    attacker = attackers[args.attack_name](fmodel,criterion=criterion1,distance=distance)
    adversarial = attacker(image, label)
    #l2_norms = (adversarial - image).view(1, -1).norm(2, 1)
    #mean_norm = l2_norms.mean()
    if adversarial is not None:
        adversarial= torch.renorm(torch.from_numpy(adversarial - image), p=2, dim=0, maxnorm=args.max_norm).numpy() + image

    return adversarial


class ImageSet_preprocess(Dataset):
    def __init__(self, df, transformer):
        self.df = df
        self.transformer = transformer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        image_path = self.df.iloc[item]['image_path']
        label = self.df.iloc[item]['label_idx']
        image = Image.open(image_path).convert('RGB')
        adversial_sample = self.transformer(image)

        return adversial_sample,label,image_path


def load_data_jpeg_compression(input_dir,batch_size=256, img_size=args.imgsize):
    all_imgs = glob.glob(os.path.join(input_dir, './*/*.jpg'))

    all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]
    train_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    train_set, test_set = train_test_split(train_data,
                                           stratify=train_data['label_idx'],train_size=0.01, random_state=42)

    transformer = transforms.Compose([
        transforms.RandomHorizontalFlip(),
       transforms.Resize((img_size, img_size), interpolation=PIL.Image.BILINEAR),
        #transforms.RandomResizedCrop(img_size, (0.7, 1), interpolation=PIL.Image.BILINEAR),
        transforms.ToTensor()

    ])
    datasets = {
        'train_data': ImageSet_preprocess(train_set, transformer),
    }
    dataloaders = {
        ds: DataLoader(datasets[ds],
                       batch_size=batch_size,
                       num_workers=4,
                       shuffle=False) for ds in datasets.keys()
    }
    return dataloaders


if __name__ == '__main__':

    input = '/media/unknown/Data/PLP/fast_adv/defenses/data/cifar10'
    path = '/media/unknown/Data/PLP/fast_adv/data/ddn'
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    val_set = data.Subset(CIFAR10(input, train=False, transform=test_transform, download=True),
                          list(range(0, 2000)))

    val_loader = data.DataLoader(val_set, batch_size=100, shuffle=False, num_workers=2, pin_memory=True)

    widgets = ['jpeg :', Percentage(), ' ', Bar('#'), ' ', Timer(),
               ' ', ETA(), ' ', FileTransferSpeed()]
    pbar = ProgressBar(widgets=widgets)
    for batch_data in pbar(dataloader['train_data']):

        adversial_sample = batch_data[0]
        label = batch_data[1]

        image_path = batch_data[2]
        # print(adversial_sample,label,image_path)
        #print(label,image_path)
        for image,y,ImgPath in zip(adversial_sample, label,image_path):
            # print(image.shape,y)
            adversial_sample = attack(image.numpy(), y.numpy())
            if adversial_sample is None:
                continue
            path = ImgPath.replace('IJCAI_2019_AAAC_train_processed/', 'train_PGD/')

            _dir, _filename = os.path.split(path)

            if not os.path.exists(_dir):
                os.makedirs(_dir)

            adversial_sample = np.transpose(adversial_sample, (1, 2, 0))

            scipy.misc.imsave(path, adversial_sample)

        pass
    pass
