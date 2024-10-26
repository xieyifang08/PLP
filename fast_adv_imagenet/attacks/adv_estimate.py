import glob
import os
import argparse

import scipy
import tqdm
import numpy as np
from copy import deepcopy
from fast_adv.attacks import DDN,DeepFool
import torch
from torch.utils import data

from torchvision import transforms
from torchvision.datasets import CIFAR10

from fast_adv.models.cifar10 import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger

import foolbox
import warnings
warnings.filterwarnings("ignore")
from foolbox.adversarial import Adversarial
from foolbox.criteria import Misclassification
from foolbox.distances import MeanAbsoluteDistance, Linfinity
from foolbox.attacks import FGSM,DeepFoolL2Attack, PGD, LocalSearchAttack, GaussianBlurAttack, \
    BinarizationRefinementAttack,ContrastReductionAttack,SaltAndPepperNoiseAttack,\
    SpatialAttack
parser = argparse.ArgumentParser(description='Extend sample')
parser.add_argument('--max-norm', type=float, default=1,help='max norm for the adversarial perturbations')
parser.add_argument('--imgsize', default=32, type=int, help='pic size')
parser.add_argument('--attack_name', '--at', default='DDN',
                    help='name for saving the final state dict')
parser.add_argument('--batch-size', '-b', default=128, type=int, help='mini-batch size')
args = parser.parse_args()
print(args)
input='../defenses/data/cifar10'
path2='./DeepFool'
path='./DDN'
attackers = {'FGSM':FGSM,
            'DeepFoolAttack': DeepFoolL2Attack,
             'PGD': PGD,
             'DDN':DDN,
             'LocalSearchAttack': LocalSearchAttack,
             'GaussianBlurAttack': GaussianBlurAttack,
             'BinarizationRefinementAttack': BinarizationRefinementAttack,
             'ContrastReductionAttack':ContrastReductionAttack,
             'SaltAndPepperNoiseAttack':SaltAndPepperNoiseAttack,
             'SpatialAttack': SpatialAttack}
image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)
DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() ) else 'cpu')
test_transform = transforms.Compose([
    transforms.ToTensor(),
])
val_set = data.Subset(CIFAR10(input, train=False, transform=test_transform, download=True),list(range(0,500)))
val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
m = wide_resnet(num_classes=10, depth=28, widen_factor=10, dropRate=0.3)
model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
model_dict = torch.load('/media/unknown/Data/PLP/fast_adv/defenses/weights/cifar10/cifar10_60_clean0.879.pth')
model.load_state_dict(model_dict)
'''
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
model2=NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
model_dict2 = torch.load('../defenses/weights/AT_cifar10_clean0.879_adv.pth')
model2.load_state_dict(model_dict2)
model3=NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
model_dict3 = torch.load('../defenses/weights/best/ALP_cifar10_ep_39_val_acc0.8592.pth')
model3.load_state_dict(model_dict3)
model4=NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
model_dict4 = torch.load('../defenses/weights/best/PLP1_cifar10_ep_29_val_acc0.8636.pth')
model4.load_state_dict(model_dict4)
'''

def attack(image, label,attack_name):
    fmodel =  foolbox.models.PyTorchModel(model.eval().cuda(), bounds=(0, 1),num_classes=10)#, preprocessing=(mean, std)
    criterion1 = Misclassification()
    distance = Linfinity#MeanAbsoluteDistance
    attacker = attackers[attack_name](fmodel,criterion=criterion1,distance=distance)


    image = image.cpu().numpy()
    label = label.cpu().numpy()

    adversarials = image.copy()
    for i in tqdm.tqdm(range(args.batch_size), ncols=80):
        adv = attacker(image[i], label[i])  # , unpack=True, steps=self.max_iter, subsample=self.subsample)
        if adv is not None:
            adv = torch.renorm(torch.from_numpy(adv - image[i]), p=2, dim=0,
                               maxnorm=1).numpy() + image[i]

            adversarials[i] = adv
    adversarials = torch.from_numpy(adversarials).to(DEVICE)


    return adversarials

for i, (images, labels) in enumerate(tqdm.tqdm(val_loader, ncols=80)):
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    print(images.size())

    #attacker = DDN(max_norm=1,steps=100, device=DEVICE)
    attacker2 = DeepFool(device=DEVICE)
    #ddn = attacker.attack(model, images, labels=labels, targeted=False)
    #deepfool= attacker2.attack(model, images, labels=labels, targeted=False)

    deepfool = attack(images, labels,'DeepFoolAttack')


    if deepfool is None:
        continue

    for t in range(args.batch_size):
        #ddn2 = np.transpose(ddn[t].cpu().numpy(), (1, 2, 0))
        deepfool2 = np.transpose(deepfool[t].cpu().numpy(), (1, 2, 0))
        name='/deepfool_'+str(i)+str(t)+'.png'
        out_path=os.path.join(path2,str(labels[t].cpu().numpy()))
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out=out_path+name
        scipy.misc.imsave(out,deepfool2)


   # ddn_name='/ddn'+str
'''
#原始···················
    logits = model(images)
    logits2=model(adv)
    logits3=model(deepfool)
    # loss = F.cross_entropy(logits, labels)
   # print(logits.argmax(1))
    test_accs = AverageMeter()
    test_losses = AverageMeter()
    test_accs.append((logits.argmax(1) == labels).float().mean().item())

    test_accs2 = AverageMeter()
    test_losses2 = AverageMeter()
    test_accs2.append((logits2.argmax(1) == labels).float().mean().item())
    print(test_accs2)

    test_accs3 = AverageMeter()
    test_losses3 = AverageMeter()
    test_accs3.append((logits3.argmax(1) == labels).float().mean().item())
    '''

'''    for t in range(args.batch_size):
        old = images[t] * 255
        old = old.cpu()
        old = old.numpy()
        old = old.transpose((1, 2, 0))
        new = adv[t] * 255
        new = new.cpu()
        new = new.numpy()
        new = new.transpose((1, 2, 0))
        new2 = deepfool[t] * 255
        #new2 = new2.cpu()
       # new2 = new2.numpy()
        new2 = new2.transpose((1, 2, 0))

        name_new = "\DDN_img-" + str(i)+ "-"+str(t) + classification[labels[t]] + ".png"
        name_new2 = "\Deepfool_img-" + str(i) + "-" + str(t) + classification[labels[t]] + ".png"
        name = "\clean_img-" + str(i) + "-" + str(t) + classification[labels[t]] + ".png"
        out_pth_new=os.path.join(path,str(labels[t].cpu().numpy()))
        out_pth_new2 = os.path.join(path2, str(labels[t].cpu().numpy()))

        out_pth = os.path.join(input, str(labels[t].cpu().numpy()))
        print(out_pth)
        if not os.path.exists(out_pth):
            os.makedirs(out_pth)
        if not os.path.exists(out_pth_new):
            os.makedirs(out_pth_new)
        if not os.path.exists(out_pth_new2):
            os.makedirs(out_pth_new2)
        #out_pth = path + labels[i] + name
        out=out_pth+name
        out_new=out_pth_new+name_new
        out_new2 = out_pth_new2 + name_new2
        print(out)
       # new.save(out, "PNG")

        scipy.misc.imsave(out, old)
        scipy.misc.imsave(out_new, new)
        scipy.misc.imsave(out_new2, new2)'''


    #adversial_sample = np.transpose(adv, (1, 2, 0))

   # scipy.misc.imsave(path, adversial_sample)




'''
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
'''