import glob
import os
import argparse
import tqdm
from copy import deepcopy

import torch
import torch.nn.functional as F
from torch.utils import data
from torch.optim import SGD, lr_scheduler
from torch.backends import cudnn

from torchvision import transforms
from torchvision.datasets import CIFAR10

from fast_adv.models.cifar10.model_attention import wide_resnet
from fast_adv.utils import AverageMeter, save_checkpoint, requires_grad_, NormalizedModel, VisdomLogger
from fast_adv.attacks import DDN,DeepFool

image_mean = torch.tensor([0.491, 0.482, 0.447]).view(1, 3, 1, 1)
image_std = torch.tensor([0.247, 0.243, 0.262]).view(1, 3, 1, 1)


DEVICE = torch.device('cuda:0' if (torch.cuda.is_available() ) else 'cpu')
test_transform = transforms.Compose([
    transforms.ToTensor(),
])

input='../defenses/data/cifar10'
#path='/media/wanghao/000F5F8400087C68/CYJ-5-29/DDN/fast_adv/attacks/DeepFool'
test_set = data.Subset(CIFAR10(input, train=False, transform=test_transform, download=True),list(range(0,1000)))
#test_set =CIFAR10(input, train=False, transform=test_transform, download=True)

test_loader = data.DataLoader(test_set, batch_size=5, shuffle=True, num_workers=2, pin_memory=True)


m = wide_resnet(num_classes=10, depth=28, widen_factor=10, dropRate=0.3)
# model = NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
# model_dict = torch.load('../defenses/weights/cifar10/cifar10_80.pth')
# model.load_state_dict(model_dict)
# ########
# model2=NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
# model_dict2 = torch.load('../defenses/weights/AT_cifar10_clean0.879_adv.pth')
# model2.load_state_dict(model_dict2)
# model3=NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
# model_dict3 = torch.load('../defenses/weights/best/ALP_cifar10_ep_39_val_acc0.8592.pth')
# model3.load_state_dict(model_dict3)
# model4=NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
# model_dict4 = torch.load('../defenses/weights/best/PLP1_cifar10_ep_29_val_acc0.8636.pth')
# model4.load_state_dict(model_dict4)

weight_norm = '../defenses/weights/best/2Norm_cifar10_ep_184_val_acc0.9515.pth'
model5=NormalizedModel(model=m, mean=image_mean, std=image_std).to(DEVICE)  # keep images in the [0, 1] range
model_dict5 = torch.load(weight_norm)
model5.load_state_dict(model_dict5)

#model.eval()
#with torch.no_grad():
for i, (images, labels) in enumerate(tqdm.tqdm(test_loader, ncols=80)):
    images, labels = images.to(DEVICE), labels.to(DEVICE)
    logits = model5(images)
    # loss = F.cross_entropy(logits, labels)
    # print(logits)
    test_accs = AverageMeter()
    test_losses = AverageMeter()
    test_accs.append((logits.argmax(1) == labels).float().mean().item())

    ################ADV########################
    attacker = DDN(steps=100, device=DEVICE)
    attacker2 = DeepFool(device=DEVICE)
    adv = attacker.attack(model5, images, labels=labels, targeted=False)
    # deepfool = attacker2.attack(model5, images, labels=labels, targeted=False)
    if adv is None:
        adv = images
    # if deepfool is None:
    #     deepfool = images
    test_accs2 = AverageMeter()
    test_losses2 = AverageMeter()
    logits2 = model5(adv)
    # logits3 = model5(deepfool)
    test_accs2.append((logits2.argmax(1) == labels).float().mean().item())
    #print(test_accs2)

    # test_accs3 = AverageMeter()
    # test_losses3 = AverageMeter()
    # test_accs3.append((logits3.argmax(1) == labels).float().mean().item())
    '''

#AT```````````````````````
    AT_logits = model2(images)
    AT_logits2 = model2(adv)
    AT_logits3 = model2(deepfool)
    # loss = F.cross_entropy(logits, labels)
    # print(logits.argmax(1))
    AT_test_accs = AverageMeter()
    AT_test_losses = AverageMeter()
    AT_test_accs.append((AT_logits.argmax(1) == labels).float().mean().item())

    AT_test_accs2 = AverageMeter()
    AT_test_losses2 = AverageMeter()
    AT_test_accs2.append((AT_logits2.argmax(1) == labels).float().mean().item())

    AT_test_accs3 = AverageMeter()
    AT_test_losses3 = AverageMeter()
    AT_test_accs3.append((AT_logits3.argmax(1) == labels).float().mean().item())
#ALP```````````````
    Alp_logits = model3(images)
    Alp_logits2 = model3(adv)
    Alp_logits3 = model3(deepfool)
    # loss = F.cross_entropy(logits, labels)
    # print(logits.argmax(1))
    Alp_test_accs = AverageMeter()
    Alp_test_losses = AverageMeter()
    Alp_test_accs.append((Alp_logits.argmax(1) == labels).float().mean().item())

    Alp_test_accs2 = AverageMeter()
    Alp_test_losses2 = AverageMeter()
    Alp_test_accs2.append((Alp_logits2.argmax(1) == labels).float().mean().item())

    Alp_test_accs3 = AverageMeter()
    Alp_test_losses3 = AverageMeter()
    Alp_test_accs3.append((Alp_logits3.argmax(1) == labels).float().mean().item())

# PLP```````````````
    PLP_logits = model4(images)
    PLP_logits2 = model4(adv)
    PLP_logits3 = model4(deepfool)
    # loss = F.cross_entropy(logits, labels)
    # print(logits.argmax(1))
    PLP_test_accs = AverageMeter()
    PLP_test_losses = AverageMeter()
    PLP_test_accs.append((PLP_logits.argmax(1) == labels).float().mean().item())

    PLP_test_accs2 = AverageMeter()
    PLP_test_losses2 = AverageMeter()
    PLP_test_accs2.append((PLP_logits2.argmax(1) == labels).float().mean().item())

    PLP_test_accs3 = AverageMeter()
    PLP_test_losses3 = AverageMeter()
    PLP_test_accs3.append((PLP_logits3.argmax(1) == labels).float().mean().item())
# print(test_accs)
# test_losses.append(loss.item())
print('\nyuanshi')
print('\nTest accuracy1 ', test_accs.avg)
print('\nTest accuracy2 ', test_accs2.avg)
print('\nTest accuracy3 ', test_accs3.avg)
#AT```````
print('\nAT')
print('\nTest accuracy1 ', AT_test_accs.avg)
print('\nTest accuracy2 ', AT_test_accs2.avg)
print('\nTest accuracy3 ', AT_test_accs3.avg)

#ALP``
print('\nALP')
print('\nTest accuracy1 ', Alp_test_accs.avg)
print('\nTest accuracy2 ', Alp_test_accs2.avg)
print('\nTest accuracy3 ', Alp_test_accs3.avg)
#PLP`````
print('\nPLP')
print('\nTest accuracy1 ', PLP_test_accs.avg)
print('\nTest accuracy2 ', PLP_test_accs2.avg)
print('\nTest accuracy3 ', PLP_test_accs3.avg)
'''
# print(test_accs)
    # test_losses.append(loss.item())
print('\nyuanshi')
print('\nTest accuracy org ', test_accs.avg)
print('\nTest accuracy DDN ', test_accs2.avg)
# print('\nTest accuracy Deepfool ', test_accs3.avg)
