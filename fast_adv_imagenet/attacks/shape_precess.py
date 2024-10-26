import random
import os
import argparse
import tqdm
import glob
import PIL
import cv2 as cv
import numpy as np
import torch


import scipy.misc
def blur(image):#均值模糊
    dst=cv.blur(image,(1,15))#垂直方向模糊，对随机噪声效果好
    return dst
def median_blur(image):#中值模糊,对于椒盐噪声效果好
    dst=cv.medianBlur(image,5)
    return dst
def custom_blur(image):#自定义模糊
    kernal=np.ones([5,5],np.float32)/25
    dst=cv.filter2D(image,-1,kernal)
    return dst
def gaussian_blur(image):
    dst=cv.GaussianBlur(image,(15,15),0)
    return dst


def clamp(pv):#防溢出
    if pv>255:
        pv=255
    elif pv<0:
        pv=0
    return pv

def gaussian_noise(image):
    h,w,c=image.shape
    for row in range(h):
        for col in range(w):
            s=np.random.normal(0,20,3)
            b=image[row,col,0]
            g=image[row,col,1]
            r=image[row,col,2]
            image[row,col,0]=clamp(b+s[0])
            image[row,col,1]=clamp(b+s[1])
            image[row, col, 2] = clamp(b + s[2])
    return image

def shape(image):
    calls = [blur, median_blur, gaussian_blur, gaussian_noise, custom_blur]
    choice = random.randint(0, len(calls) - 1)

    image_new = calls[choice](image)

    return image_new
if __name__=='__main__':
    input_dir = '/media/unknown/Data/PLP/fast_adv/defenses/data/train'
    all_imgs = glob.glob(os.path.join(input_dir, './*/*.jpg'))
    DEVICE = torch.device('cuda:0' if (torch.cuda.is_available()) else 'cpu')

    #b=scipy.misc.imread(all_imgs)
    print(len(all_imgs))
    for img_path in all_imgs:
        labels = int(img_path.split('/')[-2])
        b = scipy.misc.imread(img_path)
        #b = b.to(DEVICE)
        out=shape(b)

        out = torch.renorm(torch.from_numpy(out - b).to(DEVICE), p=2, dim=0,
                           maxnorm=1).numpy() + b
        out = np.transpose(out.cpu().numpy(), (1, 2, 0))

        new_path = img_path.replace('train/', 'shape_train_1/')
        _dir, _filename = os.path.split(new_path)
        if not os.path.exists(_dir):
            os.makedirs(_dir)
        scipy.misc.imsave(new_path,out)

    #cv.imwrite('/home/jgf/xuxiao/7/test.jpg',image_new)


