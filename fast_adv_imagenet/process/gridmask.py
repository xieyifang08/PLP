import tqdm
from torchvision.datasets import ImageFolder
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import torch.nn as nn
import numpy as np
from PIL import Image
import pdb
import math
from torch.utils import data
from torchvision.datasets import CIFAR10
from torch.utils.data import Dataset, DataLoader
import  os
from io import BytesIO
from torchvision.transforms import ToPILImage, ToTensor
# from tvm import reconstruct as tvm
import argparse
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device='cpu'



# 加上transforms，都变成224
transform = transforms.Compose([
    #transforms.RandomSizedCrop(224),
    # transforms.RandomResizedCrop(224, (0.7, 1), interpolation=Image.BILINEAR),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
 # dataset路径
# dataset = ImageFolder('../IJCAI_2019_AAAC_train',transform=transform)
# dataset = customData(r'E:\Users\13087\PycharmProjects\gridmask\data',transform=transform)
#


parser = argparse.ArgumentParser(description='process')

parser.add_argument('--data', default='../defenses/data/cifar10', help='path to dataset')
parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
parser.add_argument('--cpu', action='store_true', help='force training on cpu')
parser.add_argument('--save-freq', '--sfr', default=10, type=int, help='save frequency')
parser.add_argument('--save-name', '--sn', default='cifar10', help='name for saving the final state dict')

parser.add_argument('--batch-size', '-b', default=50, type=int, help='mini-batch size')

args = parser.parse_args()

# # dataloader是一个可迭代的对象，意味着我们可以像使用迭代器一样使用它 或者 or batch_datas, batch_labels in dataloader:
# dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

train_set = data.Subset(CIFAR10(args.data, train=True, transform=transform, download=True), list(range(50000)))
val_set = data.Subset(CIFAR10(args.data, train=True, transform=transform, download=True),
                      list(range(48000, 50000)))
test_set = CIFAR10(args.data, train=False, transform=transform, download=True)

train_loader = data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                               drop_last=True, pin_memory=True)
val_loader = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

class Grid(object):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.):
        self.d1 = d1
        self.d2 = d2
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = self.prob = prob
# 第二种方法是一开始概率为0，随着训练次数增加对训练图片进行gridmask增强的方法逐渐增大，最后变为p。实验结果结论是第二种方法好于第一种方法。

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * min(1, epoch / max_epoch)
# 在这里控制处理的概率
    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        h = img.size(1)
        w = img.size(2)

        # 1.5 * h, 1.5 * w works fine with the squared images
        # But with rectangular input, the mask might not be able to recover back to the input image shape
        # A square mask with edge length equal to the diagnoal of the input image
        # will be able to cover all the image spot after the rotation. This is also the minimum square.
        hh = math.ceil((math.sqrt(h * h + w * w)))

        d = np.random.randint(self.d1, self.d2)
        # d = self.d

        # maybe use ceil? but i guess no big difference
        self.l = math.ceil(d * self.ratio)

        mask = np.ones((hh, hh), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        for i in range(-1, hh // d + 1):
            s = d * i + st_h
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[s:t, :] *= 0
        for i in range(-1, hh // d + 1):
            s = d * i + st_w
            t = s + self.l
            s = max(min(s, hh), 0)
            t = max(min(t, hh), 0)
            mask[:, s:t] *= 0
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (hh - w) // 2:(hh - w) // 2 + w]

        mask = torch.from_numpy(mask).float().to(device)
        if self.mode == 1:
            mask = 1 - mask

        mask = mask.expand_as(img)
        img = img * mask

        return img


class GridMask(nn.Module):
    def __init__(self, d1, d2, rotate=1, ratio=0.5, mode=0, prob=1.):
        super(GridMask, self).__init__()
        self.rotate = rotate
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.grid = Grid(d1, d2, rotate, ratio, mode, prob)

    def set_prob(self, epoch, max_epoch):
        self.grid.set_prob(epoch, max_epoch)

    def forward(self, x):
        if not self.training:
            return x
        n, c, h, w = x.size()
        y = []
        for i in range(n):
            y.append(self.grid(x[i]))
        y = torch.cat(y).view(n, c, h, w)
        return y
def _quantize_img(im, depth=8):
    assert torch.is_tensor(im)
    N = int(math.pow(2, depth))
    im = (im * N).round()
    im = im / N
    return im


def _jpeg_compression(im):
    assert torch.is_tensor(im)
    im = ToPILImage()(im)
    savepath = BytesIO()
    im.save(savepath, 'JPEG', quality=75)
    im = Image.open(savepath)
    im = ToTensor()(im)
    return im


# def _TVM(im):
#     im = tvm(
#         im
#     )
#     return im
def _gridmask(im):
    # 创建实例
    grid = GridMask(96, 224, 1, 0.6, 1, 1)
    # 这里设置的是一定会被处理，240/240=1
    grid.set_prob(240, 240)
    im = grid(im)
    return im
# dataiter = iter(dataloader)
'''
imgs, labels = next(dataiter)
#print('1',imgs)
grid = GridMask(96, 224, 1, 0.6, 1, 1)
# 这里设置的是一定会被处理，240/240=1
grid.set_prob(240, 240)
imgs=grid(imgs)
#print('3',imgs)
'''
# img, labels = next(dataiter)
for i, (img, labels) in enumerate(tqdm.tqdm(train_loader, ncols=80)):
    # print(img.size(), labels.size())
    imgs=_gridmask(img)
    #imgs=torch.cat([imgs,img],dim=0)
    # print(imgs.size())
    for index in range(len(imgs)):
        new_name = ''+str(labels[index].item())+'/'+str(i)+'_'+str(index)+'.png'
        # print(new_name)
        _dir, _filename = os.path.split(new_name)
        _dir = '../data/cifar10/gridmask/'+_dir
        # print(_dir,_filename)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        img = imgs[index]
        img = transforms.ToPILImage()(img)
        img.save( _dir+'/'+ _filename)
#             print(index)
#         for path, label in dataset.imgs[t:t+50]:
#             new_path = path.replace('IJCAI_2019_AAAC_train/', 'gridmask_train/')
#             _dir, _filename = os.path.split(new_path)
#             if not os.path.exists(_dir):
#                 os.makedirs(_dir)
#             img = imgs[i]
#             img = transforms.ToPILImage()(img)
#             img.save(new_path)
#             i += 1
#         t+=50
#         print(t,len(dataset))
# print(imgs.size(),len(dataset))
'''
# 存图还没写
i=0
for path,label in dataset.imgs:
    new_path = path.replace('IJCAI_2019_AAAC_train1/', 'gridmask_train/')
    _dir, _filename = os.path.split(new_path)
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    img=imgs[i]
    img = transforms.ToPILImage()(img)
    img.save(new_path)
    i+=1
'''