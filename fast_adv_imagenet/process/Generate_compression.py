import glob
import os

import numpy
import numpy as np
import pandas as pd
from PIL import Image as Image
import scipy.misc
from scipy.ndimage import median_filter as _median_filter
from sklearn.model_selection import train_test_split
#from skimage.restoration import denoise_tv_bregman as _denoise_tv_bregman
import tensorflow as tf
#parser.add_argument('--data', default='IJCAI_2019_AAAC_train/', help='path to dataset')


def _get_image_from_arr(img_arr):
    return Image.fromarray(
        np.asarray(img_arr, dtype='uint8'))


def median_filter(img_arr, size=3):
    return _median_filter(img_arr, size=size)


#def denoise_tv_bregman(img_arr, weight=30):
  #  denoised = _denoise_tv_bregman(img_arr, weight=weight) * 255.
  #  return np.array(denoised, dtype=img_arr.dtype)


def jpeg_compress(x, quality=50):
    return tf.image.decode_jpeg(
        tf.image.encode_jpeg(
            x, format='rgb', quality=quality),
        channels=3)


def slq(x, qualities=(40,60, 80, 90), patch_size=8):
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


input_data=['/media/wanghao/000F5F8400087C68/CYJ-5-29/天池/test_FGSM/*/*.jpg']#['/media/wanghao/000F5F8400087C68/CYJ-5-29/天池/IJCAI_2019_AAAC_train_DeepFoolL2Attack/*/*.jpg']#['/media/wanghao/000F5F8400087C68/CYJ-5-29/天池/IJCAI_2019_AAAC_train_BinarizationRefinementAttack/*/*.jpg',
           #
#path=os.path.join(input_dir, '/*/*.jpg')
#print(path)
for input_dir in input_data:
    all_imgs = glob.glob(input_dir)
    all_labels = [int(img_path.split('/')[-2]) for img_path in all_imgs]

    train_data = pd.DataFrame({'image_path': all_imgs, 'label_idx': all_labels})
    print('train_set', train_data['image_path'].shape)

   # train_s#et, test_set = train_test_split(train_data,
                                 #         stratify=train_data['label_idx'], train_size=0.1, random_state=40)
    #print('train_set', train_set['image_path'].shape)
    print('22222')
    # print(all_imgs,'11')

    with tf.Session() as sess:
        for img_path in train_data['image_path']:
            print(img_path)
            new_path = img_path.replace('test_FGSM/', 'jpg_test_FGSM/')
            print('0')
            image_raw_data = tf.gfile.FastGFile(img_path, 'rb').read()
            print('1')
            try:
                img_data = tf.image.decode_jpeg(image_raw_data)
                # print(img_data.eval())

                # plt.imshow(img_data.eval())
                #  plt.show()

                img_data =slq(img_data)#slq(img_data)

                # img_data = tf.image.convert_image_dtype(img_data,dtype = tf.float32)
                print('2')
                _dir, _filename = os.path.split(new_path)
                if not os.path.exists(_dir):
                    os.makedirs(_dir)

                encoded_image = tf.image.encode_jpeg(img_data)

                # numpy_image = encoded_image.eval()

                with tf.gfile.GFile(new_path, "wb") as f:
                    f.write(encoded_image.eval())



            except:
                print('图片片不能缩缩')
                pass


def jpeg_compression(path_jpg):
    #path_jpg = '/media/wanghao/000F5F8400087C68/CYJ-5-29/天池/IJCAI_2019_AAAC_train/00020/04d8dae20b9b7147a5b3a4f74effdbfc.jpg'
    # path = ImgPath.replace('IJCAI_2019_AAAC_train/', 'IJCAI_2019_AAAC_train_PGD/')
    image_raw_data = tf.gfile.FastGFile(path_jpg, 'rb').read()
    try:
        with tf.Session() as sess:
            img_data = tf.image.decode_jpeg(image_raw_data)
            # print(img_data.eval())

            img2 = slq(img_data)
            #print('压缩完成',path_jpg)
            return img2.eval()
    except:
        print('图片片不能缩缩',path_jpg)
        return numpy.array(Image.open(path_jpg))






