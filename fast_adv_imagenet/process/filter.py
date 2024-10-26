# -*- coding: utf-8 -*-
import cv2
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import tqdm

def filter_high_f(fshift, radius_ratio):
    """
    过滤掉除了中心区域外的高频信息
    """
    # 1, 生成圆形过滤器, 圆内值1, 其他部分为0的过滤器, 过滤
    template = np.zeros(fshift.shape, np.uint8)
    crow, ccol = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)  # 圆心
    radius = int(radius_ratio * fshift.shape[0] / 2)
    if len(fshift.shape) == 3:
        cv2.circle(template, (crow, ccol), radius, (1, 1, 1), -1)
    else:
        cv2.circle(template, (crow, ccol), radius, 1, -1)
    # 2, 过滤掉除了中心区域外的高频信息
    return template * fshift


def filter_low_f(fshift, radius_ratio):
    """
    去除中心区域低频信息
    """
    # 1 生成圆形过滤器, 圆内值0, 其他部分为1的过滤器, 过滤
    filter_img = np.ones(fshift.shape, np.uint8)
    crow, col = int(fshift.shape[0] / 2), int(fshift.shape[1] / 2)
    radius = int(radius_ratio * fshift.shape[0] / 2)
    if len(fshift.shape) == 3:
        cv2.circle(filter_img, (crow, col), radius, (0, 0, 0), -1)
    else:
        cv2.circle(filter_img, (crow, col), radius, 0, -1)
    # 2 过滤中心低频部分的信息
    return filter_img * fshift


def ifft(fshift):
    """
    傅里叶逆变换
    """
    ishift = np.fft.ifftshift(fshift)  # 把低频部分sift回左上角
    iimg = np.fft.ifftn(ishift)  # 出来的是复数，无法显示
    iimg = np.abs(iimg)  # 返回复数的模
    return iimg


def get_low_high_f(img, radius_ratio):
    """
    获取低频和高频部分图像
    """
    # 傅里叶变换
    # np.fft.fftn
    f = np.fft.fftn(img)  # Compute the N-dimensional discrete Fourier Transform. 零频率分量位于频谱图像的左上角
    fshift = np.fft.fftshift(f)  # 零频率分量会被移到频域图像的中心位置，即低频

    # 获取低频和高频部分
    hight_parts_fshift = filter_low_f(fshift.copy(), radius_ratio=radius_ratio)  # 过滤掉中心低频
    low_parts_fshift = filter_high_f(fshift.copy(), radius_ratio=radius_ratio)
    # print(low_parts_fshift.max(), low_parts_fshift.min())
    # plt.subplot(121), plt.imshow(hight_parts_fshift, 'gray'), plt.title('hight_parts_fshift')
    # plt.axis('off')
    # plt.subplot(122), plt.imshow(low_parts_fshift, 'gray'), plt.title('low_parts_fshift')
    # plt.axis('off')
    # plt.show()
    low_parts_img = ifft(low_parts_fshift)  # 先sift回来，再反傅里叶变换
    high_parts_img = ifft(hight_parts_fshift)

    # 显示原始图像和高通滤波处理图像
    img_new_low = (low_parts_img - np.amin(low_parts_img)) / (np.amax(low_parts_img) - np.amin(low_parts_img) + 0.00001)
    img_new_high = (high_parts_img - np.amin(high_parts_img) + 0.00001) / (
                np.amax(high_parts_img) - np.amin(high_parts_img) + 0.00001)

    # uint8
    img_new_low = np.array(img_new_low * 255, np.uint8)
    img_new_high = np.array(img_new_high * 255, np.uint8)
    return img_new_low, img_new_high



if __name__ == '__main__':
    delta_fgsm, delta_pgd, delta_deepfool = [0 for _ in range(20)], [0 for _ in range(20)], [0 for _ in range(20)]

    for i in tqdm.tqdm(range(20)):
        print("epoch: "+str(i)+"/100")
        radius_ratio = i / 20
        tmp_fgsm = tmp_pgd = tmp_deepfool = 0
        for img_index in range(500):
            if (img_index+1) % 100 == 0:
                print("epoch: " + str(i) + "/10 : " + str(img_index))
            img_clean = cv.imread('../../fast_adv_imagenet/data/images/'+str(img_index)+'.png')[:, :, ::-1]
            img_clean = cv.resize(img_clean, (224,224))
            low_freq_part_img_clean, high_freq_part_img_clean = get_low_high_f(img_clean, radius_ratio=radius_ratio)  # multi channel or single

            img_adv = cv.imread('../attacks/advs/fgsm/'+str(img_index)+'.png')[:, :, ::-1]
            img_adv = cv.resize(img_adv, (224, 224))
            low_freq_part_img_adv, high_freq_part_img_adv = get_low_high_f(img_adv,
                                                                               radius_ratio=radius_ratio)  # multi channel or single
            tmp_fgsm += np.power(np.mean(np.power(high_freq_part_img_clean-high_freq_part_img_adv, 2)), 0.5) \
                        - np.power(np.mean(np.power(low_freq_part_img_clean-low_freq_part_img_adv, 2)), 0.5)

            # img_adv = cv.imread('C:/Users/frankfeng/Desktop/pgd/' + str(img_index) + '.jpg')[:, :, ::-1]
            # img_adv = cv.resize(img_adv, (224, 224))
            # low_freq_part_img_adv, high_freq_part_img_adv = get_low_high_f(img_adv,
            #                                                                radius_ratio=radius_ratio)  # multi channel or single
            # tmp_pgd += np.power(np.mean(np.power(high_freq_part_img_clean - high_freq_part_img_adv, 2)),
            #                                  0.5) - np.power(
            #     np.mean(np.power(low_freq_part_img_clean - low_freq_part_img_adv, 2)), 0.5)
            #
            # img_adv = cv.imread('C:/Users/frankfeng/Desktop/deepfool/' + str(img_index) + '.jpg')[:, :, ::-1]
            # img_adv = cv.resize(img_adv, (224, 224))
            # low_freq_part_img_adv, high_freq_part_img_adv = get_low_high_f(img_adv,
            #                                                                radius_ratio=radius_ratio)  # multi channel or single
            # tmp_deepfool += np.power(np.mean(np.power(high_freq_part_img_clean - high_freq_part_img_adv, 2)),
            #                     0.5) - np.power(
            #     np.mean(np.power(low_freq_part_img_clean - low_freq_part_img_adv, 2)), 0.5)
        plt.subplot(231), plt.imshow(img_clean), plt.title('original Image')
        plt.axis('off')
        plt.subplot(232), plt.imshow(low_freq_part_img_clean, "gray"), plt.title('low_freq_img')
        plt.axis('off')
        plt.subplot(233), plt.imshow(high_freq_part_img_clean), plt.title('high_freq_img')
        plt.axis('off')
        plt.subplot(234), plt.imshow(img_adv), plt.title('adv Image')
        plt.axis('off')
        plt.subplot(235), plt.imshow(low_freq_part_img_adv, "gray"), plt.title('low_freq_img')
        plt.axis('off')
        plt.subplot(236), plt.imshow(high_freq_part_img_adv), plt.title('high_freq_img')
        plt.axis('off')
        plt.savefig("./imgs/"+str(radius_ratio)+".svg",bbox_inches='tight')
        plt.show()
        delta_fgsm[i] = tmp_fgsm/500
        # delta_pgd[i] = tmp_pgd/500
        # delta_deepfool[i] = tmp_deepfool/500

    x = range(20)
    plt.plot(x, delta_fgsm, label='Frist line', linewidth=3, color='r', marker='o',
             markerfacecolor='blue', markersize=12)
    plt.xlabel('radio * 20')
    plt.ylabel('high-low delta')
    plt.title('FGSM high-low delta')
    plt.legend()
    plt.show()

    # plt.plot(x, delta_pgd, label='Frist line', linewidth=3, color='r', marker='o',
    #          markerfacecolor='blue', markersize=12)
    # plt.xlabel('radio * 10')
    # plt.ylabel('high-low delta')
    # plt.title('PGD high-low delta')
    # plt.legend()
    # plt.show()
    #
    # plt.plot(x, delta_deepfool, label='Frist line', linewidth=3, color='r', marker='o',
    #          markerfacecolor='blue', markersize=12)
    # plt.xlabel('radio * 10')
    # plt.ylabel('high-low delta')
    # plt.title('deepfool high-low delta')
    # plt.legend()
    # plt.show()

    print("delta_fgsm: ", delta_fgsm)
    # print("delta_pgd: ", delta_pgd)
    # print("delta_deepfool: ", delta_deepfool)

    print("delta_fgsm: (index, value )= ", np.argmax(delta_fgsm), np.max(delta_fgsm))
    # print("delta_pgd: (index, value )= ", np.argmax(delta_pgd), np.max(delta_pgd))
    # print("delta_deepfool: (index, value )= ", np.argmax(delta_deepfool), np.max(delta_deepfool))
    # print(low_freq_part_img.shape,"sum(low_freq_part_img)",low_freq_part_img.sum()/(low_freq_part_img.shape[0]* low_freq_part_img.shape[1]*low_freq_part_img.shape[2]), "sum(high_freq_part_img)", high_freq_part_img.sum()/(low_freq_part_img.shape[0]* low_freq_part_img.shape[1]*low_freq_part_img.shape[2]))
    # low_is_0 = (low_freq_part_img == 0)
    # print(low_is_0.shape, "0 has ",np.sum(low_is_0== True),"; 1 has ",np.sum(low_is_0==False))
    # print(low_freq_part_img.max(), high_freq_part_img.min(),high_freq_part_img.max(), high_freq_part_img.min())
    # # print(high_freq_part_img)
    # plt.subplot(241), plt.imshow(img), plt.title('Original Image')
    # plt.axis('off')
    # plt.subplot(242), plt.imshow(low_freq_part_img), plt.title('low_freq_img')
    # plt.axis('off')
    # plt.subplot(243), plt.imshow(high_freq_part_img), plt.title('high_freq_part_img')
    # plt.axis('off')
    # sss = (low_freq_part_img + high_freq_part_img)
    # # delta = sss - img
    # # plt.subplot(244), plt.imshow(delta), plt.title('delta')
    # # print(delta.max(),delta.min(), delta)
    # # print(sss.max(), sss.min())
    # # print(sss)
    # plt.subplot(244), plt.imshow(sss), plt.title('high_freq_img+low_freq_img')
    # plt.axis('off')
    #
    # low_clean,high_clean = low_freq_part_img,high_freq_part_img
    #
    # img = cv.imread('C:/Users/frankfeng/Desktop/fgsm0.jpg')[:, :, ::-1]
    # low_freq_part_img, high_freq_part_img = get_low_high_f(img, radius_ratio=radius_ratio)  # multi channel or single
    # print(low_freq_part_img.shape, "sum(low_freq_part_img)", low_freq_part_img.sum() / (
    #             low_freq_part_img.shape[0] * low_freq_part_img.shape[1] * low_freq_part_img.shape[2]),
    #       "sum(high_freq_part_img)", high_freq_part_img.sum() / (
    #                   low_freq_part_img.shape[0] * low_freq_part_img.shape[1] * low_freq_part_img.shape[2]))
    # low_is_0 = (low_freq_part_img == 0)
    # print(low_is_0.shape, "0 has ", np.sum(low_is_0 == True), "; 1 has ", np.sum(low_is_0 == False))
    # print(low_freq_part_img.max(), high_freq_part_img.min(), high_freq_part_img.max(), high_freq_part_img.min())
    # # print(high_freq_part_img)
    # plt.subplot(245), plt.imshow(img), plt.title('Original Image')
    # plt.axis('off')
    # plt.subplot(246), plt.imshow(low_freq_part_img), plt.title('low_freq_img')
    # plt.axis('off')
    # plt.subplot(247), plt.imshow(high_freq_part_img), plt.title('high_freq_part_img')
    # plt.axis('off')
    # sss = (low_freq_part_img + high_freq_part_img)
    # delta = sss - img
    # # plt.subplot(249), plt.imshow(delta), plt.title('delta')
    # # print(delta.max(), delta.min(), delta)
    # # print(sss.max(), sss.min())
    # # print(sss)
    # plt.subplot(248), plt.imshow(sss), plt.title('high_freq_img+low_freq_img')
    # plt.axis('off')
    #
    # print(np.power(np.mean(np.power(low_clean-low_freq_part_img, 2)), 0.5), np.power(np.mean(np.power(high_clean-high_freq_part_img, 2)), 0.5))
    # plt.show()