import csv
from typing import Union, Tuple, Optional, Any, Generator
import math
import time
import eagerpy as ep
from eagerpy import PyTorchTensor
import torch.nn.functional as F
from foolbox.models import Model
from foolbox.criteria import Misclassification, TargetedMisclassification
from foolbox.distances import linf
from foolbox.devutils import atleast_kd, flatten
from foolbox.attacks.base import MinimizationAttack
from foolbox.attacks.base import get_criterion
from foolbox.attacks.base import T
from foolbox.attacks.base import raise_if_kwargs
from foolbox.attacks.base import get_is_adversarial
from foolbox.plot import images
import torch
import torchvision
import numpy as np
import copy
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from pywt import dwt2, idwt2
from multiprocessing.dummy import Pool as ThreadPool
from foolbox.attacks.spatial_attack_transformations import rotate_and_shift
from torchvision.transforms import transforms
import os,sys
BASE_DIR = os.path.dirname(os.path.abspath("../"))
sys.path.append(BASE_DIR)
from fast_adv.attacks import nsgaii


class WaterMark:
    def __init__(self, password_wm=1, password_img=1, block_shape=(2, 2),alpha=20,  cores=None):
        self.block_shape = np.array(block_shape)
        self.password_wm, self.password_img = password_wm, password_img  # 打乱水印和打乱原图分块的随机种子
        self.d1 = 36  # d1/d2 越大鲁棒性越强,但输出图片的失真越大
        self.d2 = alpha
        # init data
        self.img, self.img_YUV = None, None  # self.img 是原图，self.img_YUV 对像素做了加白偶数化
        self.ca, self.hvd, = [np.array([])] * 3, [np.array([])] * 3  # 每个通道 dct 的结果
        self.ca_block = [np.array([])] * 3  # 每个 channel 存一个四维 array，代表四维分块后的结果
        self.ca_part = [np.array([])] * 3  # 四维分块后，有时因不整除而少一部分，self.ca_part 是少这一部分的 self.ca

        self.wm_size, self.block_num = 0, 0  # 水印的长度，原图片可插入信息的个数
        self.pool = ThreadPool(processes=cores)  # 水印插入分块多进程

    def set_alpha(self, d2):
        self.d2 = d2


    def init_block_index(self):
        self.block_num = self.ca_block_shape[0] * self.ca_block_shape[1]
        assert self.wm_size <= self.block_num, IndexError(
            '最多可嵌入{}kb信息，多于水印的{}kb信息，溢出'.format(self.block_num / 1000, self.wm_size / 1000))
        # self.part_shape 是取整后的ca二维大小,用于嵌入时忽略右边和下面对不齐的细条部分。
        self.part_shape = self.ca_block_shape[:2] * self.block_shape
        self.block_index = [(i, j) for i in range(self.ca_block_shape[0]) for j in range(self.ca_block_shape[1])]

    def read_img(self, img):
        # 读入图片->YUV化->加白边使像素变偶数->四维分块
        self.img = img

        self.img_shape = self.img.shape[:2]

        # 如果不是偶数，那么补上白边
        self.img_YUV = cv2.copyMakeBorder(cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV),
                                          0, self.img.shape[0] % 2, 0, self.img.shape[1] % 2,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))

        self.ca_shape = [(i + 1) // 2 for i in self.img_shape]

        self.ca_block_shape = (self.ca_shape[0] // self.block_shape[0], self.ca_shape[1] // self.block_shape[1],
                               self.block_shape[0], self.block_shape[1])
        strides = 4 * np.array([self.ca_shape[1] * self.block_shape[0], self.block_shape[1], self.ca_shape[1], 1])

        for channel in range(3):
            self.ca[channel], self.hvd[channel] = dwt2(self.img_YUV[:, :, channel], 'haar')
            # 转为4维度
            self.ca_block[channel] = np.lib.stride_tricks.as_strided(self.ca[channel].astype(np.float32),
                                                                     self.ca_block_shape, strides)

    def read_img_wm(self, wm):
        # 读入图片格式的水印，并转为一维 bit 格式
        self.wm = wm[:, :, 0]
        # 加密信息只用bit类，抛弃灰度级别
        self.wm_bit = self.wm.flatten() > 128

    def read_wm(self, wm_content, mode='img'):
        if mode == 'img':
            self.read_img_wm(wm=wm_content)
        self.wm_size = self.wm_bit.size
        # 水印加密:
        np.random.RandomState(self.password_wm).shuffle(self.wm_bit)

    def block_add_wm(self, arg):
        block, shuffler, i = arg
        # dct->flatten->加密->逆flatten->svd->打水印->逆svd->逆dct
        wm_1 = self.wm_bit[i % self.wm_size]
        block_dct = cv2.dct(block)

        # 加密（打乱顺序）
        block_dct_shuffled = block_dct.flatten()[shuffler].reshape(self.block_shape)
        U, s, V = np.linalg.svd(block_dct_shuffled)
        s[0] = (s[0] // self.d1 + 1 / 4 + 1 / 2 * wm_1) * self.d1
        if self.d2:
            s[1] = (s[1] // self.d2 + 1 / 4 + 1 / 2 * wm_1) * self.d2

        block_dct_flatten = np.dot(U, np.dot(np.diag(s), V)).flatten()
        block_dct_flatten[shuffler] = block_dct_flatten.copy()
        return cv2.idct(block_dct_flatten.reshape(self.block_shape))

    def embed(self):
        self.init_block_index()

        embed_ca = copy.deepcopy(self.ca)
        embed_YUV = [np.array([])] * 3
        self.idx_shuffle = np.random.RandomState(self.password_img) \
            .random(size=(self.block_num, self.block_shape[0] * self.block_shape[1])) \
            .argsort(axis=1)

        for channel in range(3):
            tmp = self.pool.map(self.block_add_wm,
                                [(self.ca_block[channel][self.block_index[i]], self.idx_shuffle[i], i)
                                 for i in range(self.block_num)])

            for i in range(self.block_num):
                self.ca_block[channel][self.block_index[i]] = tmp[i]

            # 4维分块变回2维
            self.ca_part[channel] = np.concatenate(np.concatenate(self.ca_block[channel], 1), 1)
            # 4维分块时右边和下边不能整除的长条保留，其余是主体部分，换成 embed 之后的频域的数据
            embed_ca[channel][:self.part_shape[0], :self.part_shape[1]] = self.ca_part[channel]
            # 逆变换回去
            embed_YUV[channel] = idwt2((embed_ca[channel], self.hvd[channel]), "haar")

        # 合并3通道
        embed_img_YUV = np.stack(embed_YUV, axis=2)
        # 之前如果不是2的整数，增加了白边，这里去除掉
        embed_img_YUV = embed_img_YUV[:self.img_shape[0], :self.img_shape[1]]
        embed_img = cv2.cvtColor(embed_img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)
        # print("embed success")
        # print(embed_img.shape)
        x_delta = (embed_img - self.img)/256

        # print("delta:", x_delta.shape)

        return x_delta
crossloss = torch.nn.CrossEntropyLoss()
class advDigitalMark(MinimizationAttack):
    """The Decoupled Direction and Norm L2 adversarial attack. [#Rony18]_
    Args:
        init_epsilon : Initial value for the norm/epsilon ball.
        steps : Number of steps for the optimization.
    """

    distance = linf

    def __init__(
        self, *, init_epsilon: float = 1.0, steps: int = 10 ):
        self.init_epsilon = init_epsilon
        self.steps = steps
        self.need_show_img = True
        self.waterMark = WaterMark(password_wm=1, password_img=1)


    def run(
        self,
        model: Model,
        inputs: T,
        criterion: Union[Misclassification, TargetedMisclassification, T],
        *,
        early_stop: Optional[float] = None,
        **kwargs: Any,
    ) -> T:
        raise_if_kwargs(kwargs)
        x, restore_type = ep.astensor_(inputs)
        criterion_ = get_criterion(criterion)
        # is_adversarial = get_is_adversarial(criterion, model)
        del inputs, criterion, kwargs
        N = len(x)

        if isinstance(criterion_, Misclassification):
            targeted = False
            classes = criterion_.labels
        elif isinstance(criterion_, TargetedMisclassification):
            targeted = True
            classes = criterion_.target_classes
        else:
            raise ValueError("unsupported criterion")

        if classes.shape != (N,):
            name = "target_classes" if targeted else "labels"
            raise ValueError(
                f"expected {name} to have shape ({N},), got {classes.shape}"
            )

        def loss_fn(
            inputs: ep.Tensor, labels: ep.Tensor
        ) -> Tuple[ep.Tensor, ep.Tensor]:
            logits = model(inputs)

            sign = -1.0 if targeted else 1.0
            print("lss_fn ",logits, labels)
            loss = sign * ep.crossentropy(logits, labels).sum()

            return loss, logits

        grad_and_logits = ep.value_and_grad_fn(x, loss_fn, has_aux=True)

        image = Image.open('./test.png')
        # image.show()
        # loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

        logist_clean = model(x)
        logist_clean = logist_clean.argmax(1)
        wm = []
        one_batch_attack_success = 0
        wm_numpy = torch.from_numpy(np.array(image, dtype=np.float32).transpose([2, 0, 1]))
        for k in range(N):
            wm.append(wm_numpy)
        wm_tensor = torch.stack(wm)
        # with open('result_inception_v3_gen1_40.csv', 'a+')as f:
        #     f_csv = csv.writer(f)
        for j in range(N): # foreach a batch
            if logist_clean[j] == classes[j]:
                blocks,alpha,angle = nsgaii.get_init()
                # x_j = "/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/attacks/test/0.1504072755143_org.png"
                # x_j = Image.open(x_j)
                # x_j =transforms.ToTensor()(x_j).to(device)
                # x_j = PyTorchTensor(x_j)
                # print("x_j", x[j].raw.shape, x_j.shape)
                attack_success_population = nsgaii.nsgaii(model, x[j], classes[j], wm_tensor[j], blocks,alpha,angle, self.waterMark)
                # print("attack_success_population", attack_success_population)

                # (alpha[single_population],
                # angle[single_population],
                # logist_population[single_population],
                # l2_population[single_population],
                # x_adv_population[single_population]))
                #
                if len(attack_success_population) > 0:
                    one_batch_attack_success += 1
                # plt.figure()
                if self.need_show_img:
                    if not os.path.exists(nsgaii.watermark_dir):
                        os.makedirs(nsgaii.watermark_dir)
                    timestamp = str(int(time.time()*1000))
                    for index in range(len(attack_success_population)):
                        alpha = attack_success_population[index][0]
                        angle = attack_success_population[index][1]
                        logist_population = attack_success_population[index][2]
                        l2_population = attack_success_population[index][3]
                        if l2_population <= 20:
                            xxx = attack_success_population[index][4].raw.cpu().numpy().transpose([1, 2, 0]) * 255
                            img = Image.fromarray(xxx.astype('uint8')).convert('RGB')
                            img.save(nsgaii.watermark_dir+timestamp+"_" + str(j) + "_" + str(index) + "_alpha"+str(alpha)+"_angle"+str(angle)+"_logist"+str(logist_population)+"_l2"+str(l2_population)+".png")

                        if index == 0:
                            img_org = x[j].raw.cpu().numpy().transpose([1, 2, 0]) * 255
                            img_org = Image.fromarray(img_org.astype('uint8')).convert('RGB')
                            img_org.save(nsgaii.watermark_dir+timestamp+"_" + str(j) + "_" + str(index) +"_"+str(classes[j].raw.cpu().numpy()) + ".png")
                        # f_csv.writerow([str(j),str(index),str(alpha),str(angle),str(logist_population),str(l2_population)])
            else:
                one_batch_attack_success += 1
        return one_batch_attack_success