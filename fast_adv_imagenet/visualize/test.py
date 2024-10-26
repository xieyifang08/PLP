import matplotlib.pyplot as plt
import numpy as np
def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure()
    #plt.imshow(npimg)
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torchvision import datasets

transform_train = transforms.Compose([
                #transforms.RandomCrop(32, padding=4),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])

transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])


cifar_train = datasets.CIFAR10("data/cifar10", train=True, download=False, transform=transform_train )
cifar_test = datasets.CIFAR10("data/cifar10", train=False, download=False, transform=transform_test )

cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=1, shuffle=False)
cifar_test_loader = torch.utils.data.DataLoader(cifar_test, batch_size=1, shuffle=False)
for i, (images, labels) in enumerate(cifar_train_loader):
    imshow(torchvision.utils.make_grid(images.cpu().data, normalize=True), labels.cpu().numpy())  
    im = images[0]#.cpu().numpy()

    im = transforms.ToPILImage()(im)
    im = transforms.RandomCrop(32, padding=4)(im)
    im = transforms.RandomHorizontalFlip()(im)
    im = transforms.ToTensor()(im)
    im=transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(im)

    imshow(torchvision.utils.make_grid(im.cpu().data, normalize=True), labels.cpu().numpy())  
    break
