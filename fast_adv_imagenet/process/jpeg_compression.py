import glob
import os
import random

import imageio
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils import data
from torch.optim import SGD, lr_scheduler
import scipy
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import InterpolationMode

from fast_adv.models.wsdan.wsdan import WSDAN

import tqdm
import numpy
import numpy as np
from PIL import Image as Image
import scipy.misc
from scipy.ndimage import median_filter as _median_filter
#from skimage.restoration import denoise_tv_bregman as _denoise_tv_bregman
import tensorflow as tf
import pandas as pd
#parser.add_argument('--data', default='IJCAI_2019_AAAC_train/', help='path to dataset')

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

def load_data_for_defense(csv, input_dir, img_size=224, batch_size=16):
    jir = pd.read_csv(csv)
    all_imgs = [os.path.join(input_dir, str(i)) for i in jir['ImageId'].tolist()]
    all_labels = jir['TrueLabel'].tolist()
    dev_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})

    transformer = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
        JpegCompression(),
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


def _get_image_from_arr(img_arr):
    return Image.fromarray(
        np.asarray(img_arr, dtype='uint8'))


def median_filter(img_arr, size=3):
    return _median_filter(img_arr, size=size)


#def denoise_tv_bregman(img_arr, weight=30):
  #  denoised = _denoise_tv_bregman(img_arr, weight=weight) * 255.
  #  return np.array(denoised, dtype=img_arr.dtype)


def jpeg_compress(x, quality=75):
    return tf.image.decode_jpeg(
        tf.image.encode_jpeg(
            x, format='rgb', quality=quality),
        channels=3)


def slq(x, qualities=(40,60, 80, 20), patch_size=8):
    num_qualities = len(qualities)

    with tf.name_scope('slq'):
        one = tf.constant(1, name='one')
        zero = tf.constant(0, name='zero')

        x_shape = tf.shape(x)
        n, m = x_shape[0], x_shape[1]

        patch_n = tf.cast(n / patch_size, dtype=tf.int32) \
            + tf.cond(n % patch_size > 0, lambda: one, lambda: zero)
        patch_m = tf.cast(m / patch_size, dtype=tf.int32) \
            + tf.cond(n % patch_size > 0, lambda: one, lambda: zero)

        R = tf.tile(tf.reshape(tf.range(n), (n, 1)), [1, m])
        C = tf.reshape(tf.tile(tf.range(m), [n]), (n, m))
        Z = tf.image.resize_nearest_neighbor(
            [tf.random_uniform(
                (patch_n, patch_m, 3),
                0, num_qualities, dtype=tf.int32)],
            (patch_n * patch_size, patch_m * patch_size),
            name='random_layer_indices')[0, :, :, 0][:n, :m]
    #    if Z.shape!=R.shape:
     #       print('shibai')
      #      return x
      #  print(R.shape,Z.shape,C.shape)
        indices = tf.transpose(
            tf.stack([Z, R, C]),
            perm=[1, 2, 0],
            name='random_layer_indices')
   #     print(R.shape, Z.shape, C.shape)

        x_compressed_stack = tf.stack(
            list(map(
                lambda q: tf.image.decode_jpeg(tf.image.encode_jpeg(
                    x, format='rgb', quality=q), channels=3),
                qualities)),
            name='compressed_images')

        x_slq = tf.gather_nd(x_compressed_stack, indices, name='final_image')
        #print(R.shape, Z.shape, C.shape)

    return x_slq

'''
input_dir='/media/wanghao/000F5F8400087C68/CYJ-5-29/天池/IJCAI_2019_AAAC_train/*/*.jpg'
#path=os.path.join(input_dir, '/*/*.jpg')
#print(path)
all_imgs = glob.glob(input_dir)
print('22222')
print(all_imgs,'11')
for img_path in all_imgs:
    print(img_path)
    new_path = img_path.replace('IJCAI_2019_AAAC_train/', 'IJCAI_2019_AAAC_train_compression/')
    print('0')
    image_raw_data = tf.gfile.FastGFile(img_path, 'rb').read()
    print('1')
    try:

        with tf.Session() as sess:
            img_data = tf.image.decode_jpeg(image_raw_data)
            # print(img_data.eval())

            # plt.imshow(img_data.eval())
            #  plt.show()

            img_data = slq(img_data)

            # img_data = tf.image.convert_image_dtype(img_data,dtype = tf.float32)
            print('2')
            _dir, _filename = os.path.split(new_path)
            if not os.path.exists(_dir):
                os.makedirs(_dir)

            encoded_image = tf.image.encode_jpeg(img_data)
            try:
                numpy_image = encoded_image.eval()
            except:
                print('图图片不能压缩')
                pass
            with tf.gfile.GFile(new_path, "wb") as f:
                f.write(encoded_image.eval())
    except:
        print('图片片不能缩缩')
        pass


'''
def jpeg_compression(path_jpg):
    #path_jpg = '/media/wanghao/000F5F8400087C68/CYJ-5-29/天池/IJCAI_2019_AAAC_train/00020/04d8dae20b9b7147a5b3a4f74effdbfc.jpg'
    # path = ImgPath.replace('IJCAI_2019_AAAC_train/', 'IJCAI_2019_AAAC_train_PGD/')
    image_raw_data = tf.gfile.FastGFile(path_jpg, 'rb').read()

    #print(type(image_raw_data))
    try:
        with tf.Session() as sess:
            with tf.device('/gpu:0'):
                img_data = tf.image.decode_jpeg(image_raw_data)
                print("img_data type", type(img_data))
                print("img_data", img_data.shape)
                #print(img_data.shape)

                img2 = slq(img_data)
                print('压缩完成',path_jpg)
                return img2.eval()
    except:
        print('图片片不能缩缩',path_jpg)
        return numpy.array(Image.open(path_jpg))

def jpeg_c(img):

    try:
        with tf.Session() as sess:

            img_data = tf.convert_to_tensor(img)
            # print("img_data type", type(img_data))
            # print("img_data", img_data.shape)
            # print(img_data.shape)

            img2 = slq(img_data)
            # print('压缩完成', type(img2.eval()))
            # img = Image.fromarray(img2.eval().astype('uint8')).convert('RGB')
            # img.save('/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/process/img/'+str(index)+'_jpeg.png')
            return img2
    except:
        return img



class JpegCompression(object):



    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """
        # index = random.random()
        # img.save('/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/process/img/'+str(index)+'_org.png')
        # print("JpegCompression img type", type(img))
        # print("JpegCompression img shape", img.size)
        img = numpy.array(img)

        try:
            with tf.Session() as sess:

                img_data = tf.convert_to_tensor(img)
                # print("img_data type", type(img_data))
                # print("img_data", img_data.shape)
                # print(img_data.shape)

                img2 = slq(img_data)
                # print('压缩完成', type(img2.eval()))
                img = Image.fromarray(img2.eval().astype('uint8')).convert('RGB')
                # img.save('/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/process/img/'+str(index)+'_jpeg.png')
                return img
        except:
            return img

def jpeg_cifar10():
    args_data = '/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/defenses/data/cifar10'
    batch_size = 16
    train_transform = transforms.Compose([
        # transforms.RandomHorizontalFlip(0.5),  # 依概率p水平翻转
        # transforms.ColorJitter(brightness=0.126, saturation=0.5),  # 修改亮度、对比度和饱和度
        # JpegCompression(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
    ])

    train_set = data.Subset(CIFAR10(args_data, train=True, transform=train_transform, download=True),
                            list(range(30000,40000)))
    val_set = data.Subset(CIFAR10(args_data, train=True, transform=test_transform, download=True),
                          list(range(48000, 50000)))
    test_set = CIFAR10(args_data, train=False, transform=test_transform, download=True)

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2,
                                   drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    print(len(train_set), len(val_set), len(test_set))
    print(len(train_loader), len(val_loader), len(test_loader))

    path_jpeg = "/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/jpeg"
    for i, (images, labels) in enumerate(tqdm.tqdm(train_loader, ncols=80)):
        images, labels = images, labels
        for t in range(batch_size):

            new = images[t]
            new = new.numpy()
            new = jpeg_c(new)
            new_shape = torch.from_numpy(new)

            image_shape = np.transpose(new_shape.cpu().numpy(), (1, 2, 0))

            name = '/jpeg_' + str(i+188) + str(t) + '.png'
            out_path = os.path.join(path_jpeg,"train", str(labels[t].cpu().numpy()))
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            out = out_path + name
            scipy.misc.imsave(out,image_shape)

    # for i, (images, labels) in enumerate(tqdm.tqdm(test_loader, ncols=80)):
    #     images, labels = images, labels
    #     for t in range(batch_size):
    #
    #         new = images[t]
    #         new = new.numpy()
    #         new = jpeg_c(new)
    #         new_shape = torch.from_numpy(new)
    #
    #         image_shape = np.transpose(new_shape.cpu().numpy(), (1, 2, 0))
    #
    #         name = '/jpeg_' + str(i) + str(t) + '.png'
    #         out_path = os.path.join(path_jpeg,"test", str(labels[t].cpu().numpy()))
    #         if not os.path.exists(out_path):
    #             os.makedirs(out_path)
    #         out = out_path + name
    #         scipy.misc.imsave(out, image_shape)
    #
    # for i, (images, labels) in enumerate(tqdm.tqdm(val_loader, ncols=80)):
    #     images, labels = images, labels
    #     for t in range(batch_size):
    #
    #         new = images[t]
    #         new = new.numpy()
    #         new = jpeg_c(new)
    #         new_shape = torch.from_numpy(new)
    #
    #         image_shape = np.transpose(new_shape.cpu().numpy(), (1, 2, 0))
    #
    #         name = '/jpeg_' + str(i) + str(t) + '.png'
    #         out_path = os.path.join(path_jpeg,"val", str(labels[t].cpu().numpy()))
    #         if not os.path.exists(out_path):
    #             os.makedirs(out_path)
    #         out = out_path + name
    #         scipy.misc.imsave(out, image_shape)


# def jpeg_imagenet():
#     args_data = '../data/images'
#     batch_size = 16
#     # train_transform = transforms.Compose([
#     #     # transforms.RandomHorizontalFlip(0.5),  # 依概率p水平翻转
#     #     # transforms.ColorJitter(brightness=0.126, saturation=0.5),  # 修改亮度、对比度和饱和度
#     #     #
#     #     transforms.ToTensor(),
#     #     # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
#     # ])
#
#     train_loader = load_data_for_defense(os.path.join(args_data, 'dev.csv'), os.path.join(args_data, 'images'))['dev_data']
#
#     path_jpeg = "./jpeg"
#     for batch_data in tqdm.tqdm(train_loader):
#             images, labels, filename = batch_data['image'], batch_data['label_idx'], batch_data['filename']
#         for t in range(batch_size):
#             new = images[t]
#             new = new.numpy()
#             new = jpeg_c(new)
#             new_shape = torch.from_numpy(new)
#
#             image_shape = np.transpose(new_shape.cpu().numpy(), (1, 2, 0))
#
#             name = '/jpeg_' + str(i+188) + str(t) + '.png'
#             out_path = os.path.join(path_jpeg,"train", str(labels[t].cpu().numpy()))
#             if not os.path.exists(out_path):
#                 os.makedirs(out_path)
#             out = out_path + name
#             scipy.misc.imsave(out,image_shape)


if __name__ == '__main__':
    jpeg_cifar10()
    # path_jpg = '/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/process/img/0bc58747-9a2e-43e1-af1f-cf0a41f9f2ba.png'
    # t = jpeg_compression(path_jpg)
    # t2=Image.open(path_jpg)
    # print(type(t),type(t2))
    #
    #
    # scipy.misc.imsave('/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/process/img/attack-jpeg.png', t)





