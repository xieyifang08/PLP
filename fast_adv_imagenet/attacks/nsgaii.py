import math
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import cv2
# replace function1
import torchvision
from PIL import Image
from eagerpy import PyTorchTensor
import torch
from fast_adv.attacks import advdigitalmark
from fast_adv.attacks.advdigitalmark import WaterMark
from foolbox.distances import l2
from foolbox.devutils import flatten
import eagerpy as ep
import logging

# logging.basicConfig(level=logging.DEBUG,  # 控制台打印的日志级别
#                     filename='new_10.log',
#                     filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
#                     a是追加模式，默认如果不写的话，就是追加模式
                    # format=
                    # '%(levelname)s: %(message)s'
                    # 日志格式
                    # )


def fun1_PSNR(image1, image2):
    # image1 = np.array(image1)
    # image2 = np.array(image2)

    MSE = np.mean((image1 - image2) ** 2)
    if (MSE < 1.0e-10):
        return 100
    psnr = 30 * math.log10(1.0 / math.sqrt(MSE))
    return psnr


#
# def fun2_Loss_fn(image1, image2):
#
#     logits = model(inputs)
#     # sign = -1.0 if targeted else 1.0
#     # loss = sign * ep.crossentropy(logits, labels).sum()
#     return 0
# replace function2

# First function to optimize
def function1(x):
    value = -x ** 2
    return value


# Second function to optimize
def function2(x):
    value = -(x - 2) ** 2
    return value


# Function to find index of list
def index_of(a, list):
    for i in range(0, len(list)):
        if list[i] == a:
            return i
    return -1


# Function to sort by values
def sort_by_values(list1, values):
    sorted_list = []
    while (len(sorted_list) != len(list1)):
        if index_of(min(values), values) in list1:
            sorted_list.append(index_of(min(values), values))
        values[index_of(min(values), values)] = math.inf
    return sorted_list


# Function to carry out NSGA-II's fast non dominated sort
def fast_non_dominated_sort(values1, values2):
    S = [[] for i in range(0, len(values1))]
    front = [[]]
    n = [0 for i in range(0, len(values1))]
    rank = [0 for i in range(0, len(values1))]

    for p in range(0, len(values1)):
        S[p] = []
        n[p] = 0
        for q in range(0, len(values1)):
            if (values1[p] < values1[q] and values2[p] < values2[q]) or (
                    values1[p] <= values1[q] and values2[p] < values2[q]) or (
                    values1[p] < values1[q] and values2[p] <= values2[q]):
                if q not in S[p]:
                    S[p].append(q)
            elif (values1[q] < values1[p] and values2[q] < values2[p]) or (
                    values1[q] <= values1[p] and values2[q] < values2[p]) or (
                    values1[q] < values1[p] and values2[q] <= values2[p]):
                n[p] = n[p] + 1
        if n[p] == 0:
            rank[p] = 0
            if p not in front[0]:
                front[0].append(p)

    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front


# Function to calculate crowding distance
def crowding_distance(values1, values2, front):
    distance = [0 for i in range(0, len(front))]
    sorted1 = sort_by_values(front, values1[:])
    sorted2 = sort_by_values(front, values2[:])
    distance[0] = 4444444444444444
    distance[len(front) - 1] = 4444444444444444
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values1[sorted1[k + 1]] - values2[sorted1[k - 1]]) / (max(values1) - min(values1))
    for k in range(1, len(front) - 1):
        distance[k] = distance[k] + (values1[sorted2[k + 1]] - values2[sorted2[k - 1]]) / (max(values2) - min(values2))
    return distance


# Function to carry out the crossover
def crossover(a, b):
    r = random.random()
    if r > 0.5:
        return mutation((a + b) / 2)
    else:
        return mutation((a - b) / 2)


# Function to carry out the mutation operator
def mutation(solution):
    mutation_prob = random.random()
    if mutation_prob < 1:
        solution = min_x + (max_x - min_x) * random.random()
    return abs(solution)

import torchvision.models as models

# parameters
# train_input_dir = "/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/imagenet/"
train_input_dir = "/home/unknown/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/imagenet/"
# train_input_dir = '/home/f/Downloads/fast_adv11/imagenet/'

# train_input_dir = '/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/org'
# train_input_dir = '/home/f/Downloads/PLP/fast_adv/data/cifar10/org'
# train_input_dir = '/home/unknown/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/data/cifar10/org'

# train_input_dir = '/home/f/Downloads/PLP/fast_adv/attacks/watermark_attack/resnet101_adv'

model_name = models.vgg19
watermark_dir = './watermark_vgg19_new_gen1_30/'

pop_size = 30
max_gen = 1

# Initialization
min_x = -55
max_x = 55
solution = [min_x + (max_x - min_x) * random.random() for i in range(0, pop_size)]


def get_init():
    blocks = [random.randint(2, 2) for _ in range(pop_size)]
    angle = [random.uniform(0, 360) for _ in range(pop_size)]
    alpha = [random.uniform(0, 360) for _ in range(pop_size)]
    # print("init\n", blocks, "\n", angle, "\n", alpha)
    return blocks, alpha, angle


image_mean = [0.491, 0.482, 0.447]
image_std = [0.247, 0.243, 0.262]

device = torch.device('cuda')
crossloss = torch.nn.CrossEntropyLoss()
gol_index = 0


def nagaii_get2function(model, x_j, classes_j, wm_tensor_j, blocks, alpha, angle, waterMark):
    global gol_index
    bwm = []
    delta = []
    # print(blocks)
    for index in range(len(blocks)):
        waterMark.set_alpha(alpha[index])
        bwm.append(waterMark)
        # 读取原图
        bwm[index].read_img(x_j.numpy().transpose([1, 2, 0]))
        # do step
        trans = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomRotation((angle[index], angle[index]), resample=False, expand=False,
                                                  center=None),
            torchvision.transforms.ToTensor(),
        ])
        wm_tensor_j = trans(wm_tensor_j)
        # xxx = wm_tensor_j.cpu().numpy().transpose([1, 2, 0]) * 255
        # img = Image.fromarray(xxx.astype('uint8')).convert('RGB')
        #       print("img", img.size)
        # img.save("/home/frankfeng/test/" + str(gol_index) + "_delta"+str(angle[index])+".png")
        # 读取水印
        bwm[index].read_wm(wm_tensor_j.numpy().transpose([1, 2, 0]))
        # 打上盲水印
        # print("2 x[j]\n", x_j)

        delta.append(torch.from_numpy(bwm[index].embed().transpose([2, 0, 1])).to(device))
    # 把delta list stack
    final_delta = torch.stack(delta)

    single_x = x_j.raw.unsqueeze(dim=0).repeat(len(blocks), 1, 1, 1)
    single_x = PyTorchTensor(single_x)
    l2 = final_delta.view(final_delta.size(0), -1).norm(p=2, dim=1)
    x_adv = single_x + final_delta
    delta = 0

    #     for single in range(20):
    #         xxx= x_adv.raw[single].cpu().numpy().transpose([1, 2, 0])*255
    # #       print("xx", xxx.shape,xxx.min(),xxx.max())
    #         img = Image.fromarray(xxx.astype('uint8')).convert('RGB')
    # #       print("img", img.size)
    # #
    #         img.save("/home/frankfeng/test/"+str(gol_index)+"_"+str(single)+".png")

    gol_index += 1
    logits = model(x_adv)
    """ 实验：遗传算法的中间攻击效果，保存过程图 """
    # logits_clean = model(single_x)
    """ over """
    isAttackSuccess = False
    judges = logits.argmax(1).raw == classes_j.raw
    isAttackSuccess = False in judges

    # print(classes_j, " init softmax", logits.argmax(1).raw, isAttackSuccess)

    single_class = torch.Tensor([classes_j.item() for _ in range(len(blocks))]).cuda()
    losses = []
    psnrs = []
    for img_index in range(len(blocks)):
        losses.append(crossloss(logits.raw[img_index].unsqueeze(dim=0),
                                single_class.long()[img_index].unsqueeze(dim=0)))
        psnrs.append(fun1_PSNR(x_adv.raw[img_index].cpu().numpy(), x_j.raw.cpu().numpy()))

    return losses, psnrs, logits.argmax(1).raw.cpu().numpy(), l2.cpu().numpy(), x_adv,isAttackSuccess
    """ 实验：遗传算法的中间攻击效果，保存过程图 """
    # return losses, psnrs, logits.argmax(
    #     1).raw.cpu().numpy(), l2.cpu().numpy(), x_adv, isAttackSuccess, logits.raw.cpu().numpy(), logits_clean.raw.cpu().numpy()
    """ over """


foreach_success = []


def nsgaii(model, x_j, classes_j, wm_tensor_j, blocks, alpha, angle, waterMark):
    gen_no = 0
    while (gen_no < max_gen):
        # losses, psnrs, logist_population, l2_population, x_adv_population, isAttackSuccess, logists, logists_clean = nagaii_get2function(
        losses, psnrs, logist_population, l2_population, x_adv_population, isAttackSuccess = nagaii_get2function(
            model, x_j, classes_j,
            wm_tensor_j, blocks,
            alpha, angle, waterMark)

        print("gen_no: ",str(gen_no), " logist_population: ", logist_population)
        """
            实验：遗传算法        
        """
        # if gen_no >0:
        #     losses_format = []
        #     for loss in losses:
        #         losses_format.append(loss.cpu().numpy())
        #     function1_values_tmp = losses_format
        #     function2_values_tmp = psnrs
        #     function1_tmp = [i.item() * 1 for i in function1_values_tmp]
        #     function2_tmp = [j * 1 for j in function2_values_tmp]
        #     loss_format = []
        #     psnr_format = []
        #     psnr_index = np.argsort(function2_tmp)
        #     for i in range(len(function1_tmp)):
        #         loss_format.append(function1_tmp[psnr_index[i]])
        #         psnr_format.append(function2_tmp[psnr_index[i]])
        #     print("loss =",loss_format)
        #     print("psnes =",psnr_format)
        #     # plt.plot(loss_format, psnr_format, colors[gen_no%8], linewidth=1, markersize=5, label="round 3")
        #     print("\n")
        """  over  """
        """
            实验：遗传算法的中间攻击效果，保存过程图
        """
        # save_folder = './nsgaii_mid_img_10/'
        # if not os.path.exists(save_folder):
        #     os.makedirs(save_folder)
        # timestamp1 = str(int(time.time() * 1000))
        # function21 = [j * 1 for j in psnrs]
        # psnr_index1 = np.argsort(function21)
        #
        # attack_success_population1 = []
        #
        # for single_population in psnr_index1:
        #     attack_success_population1.append((alpha[single_population], angle[single_population],
        #                                        logist_population[single_population],
        #                                        l2_population[single_population],
        #                                        x_adv_population[single_population],
        #                                        logists[single_population],
        #                                        logists_clean[single_population])
        #                                       )
        # for index in range(len(attack_success_population1)):
        #     alpha0 = attack_success_population1[index][0]
        #     angle0 = attack_success_population1[index][1]
        #     logist_population0 = attack_success_population1[index][2]
        #     l2_population0 = attack_success_population1[index][3]
        #     logits0 = attack_success_population1[index][5]
        #     xxx0 = attack_success_population1[index][4].raw.cpu().numpy().transpose([1, 2, 0]) * 255
        #     img0 = Image.fromarray(xxx0.astype('uint8')).convert('RGB')
        #     logits0_idx = logits0.argsort()[::-1][0:5]
        #     logits0_value = sorted(logits0, reverse=True)
        #     logits0_dict = dict(zip(logits0_idx, logits0_value[0:5]))
        #     logging.info(save_folder + timestamp1 + "_gen" + str(gen_no) + "_" + str(index) + "_logist" + str(
        #         logist_population0) + "_logists " + str(logits0_dict) + ".png")
        #     img0.save(save_folder + timestamp1 + "_gen" + str(gen_no) + "_" + str(index) + "_alpha" + str(
        #         alpha0) + "_angle" + str(angle0) + "_logist" + str(logist_population0) + "_l2" + str(
        #         l2_population0) + ".png")
        #
        #     if index == 0:
        #         logits_clean0 = attack_success_population1[index][6]
        #         img_org0 = x_j.raw.cpu().numpy().transpose([1, 2, 0]) * 255
        #         img_org0 = Image.fromarray(img_org0.astype('uint8')).convert('RGB')
        #         logits_clean0_idx = logits_clean0.argsort()[::-1][0:5]
        #         logits_clean0_value = sorted(logits0, reverse=True)
        #         logits_clean0_dict = dict(zip(logits_clean0_idx, logits_clean0_value[0:5]))
        #         logging.debug(save_folder + timestamp1 + "_" + str(gen_no) + "_" + str(index) + "_" + str(
        #             classes_j.raw.cpu().numpy()) + "_logists " + str(logits_clean0_dict) + ".png")
        #         img_org0.save(save_folder + timestamp1 + "_" + str(gen_no) + "_" + str(index) + "_" + str(
        #             classes_j.raw.cpu().numpy()) + ".png")

        # if isAttackSuccess:
        #     break
        """ over """


        # if gen_no == 0:
        #     print("init losses", losses)
        #     print("init psnrs ", psnrs)
        # function1_values = [function1(solution[i])for i in range(0,pop_size)]
        # function2_values = [function2(solution[i])for i in range(0,pop_size)]
        function1_values = losses
        function2_values = psnrs
        # print(losses,psnrs)
        non_dominated_sorted_solution = fast_non_dominated_sort(function1_values[:], function2_values[:])
        # print("The best front for Generation number ",gen_no, " is")
        # for valuez in non_dominated_sorted_solution[0]:
        #     print(round(blocks[valuez],3),end=" ")
        #     print(round(alpha[valuez], 3), end=" ")
        #     print(round(angle[valuez], 3), end=" ")
        # print("\n")
        crowding_distance_values = []
        for i in range(0, len(non_dominated_sorted_solution)):
            crowding_distance_values.append(
                crowding_distance(function1_values[:], function2_values[:], non_dominated_sorted_solution[i][:]))
        blocks2 = blocks[:]
        alpha2 = alpha[:]
        angle2 = angle[:]
        # Generating offsprings
        while (len(blocks2) != 2 * pop_size):
            blocks2.append(random.randint(2, 2))
            alpha2.append(random.uniform(0, 360))
            angle2.append(random.uniform(0, 360))
        losses2, psnrs2, logist_population, l2_population, x_adv_population, _ = nagaii_get2function(model, x_j,
                                                                                                           classes_j,
                                                                                                           wm_tensor_j,
                                                                                                           blocks2,
                                                                                                           alpha2,
                                                                                                           angle2,
                                                                                                           waterMark)
        function1_values2 = losses2
        function2_values2 = psnrs2
        non_dominated_sorted_solution2 = fast_non_dominated_sort(function1_values2[:], function2_values2[:])
        crowding_distance_values2 = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            crowding_distance_values2.append(
                crowding_distance(function1_values2[:], function2_values2[:], non_dominated_sorted_solution2[i][:]))
        new_solution = []
        for i in range(0, len(non_dominated_sorted_solution2)):
            non_dominated_sorted_solution2_1 = [
                index_of(non_dominated_sorted_solution2[i][j], non_dominated_sorted_solution2[i]) for j in
                range(0, len(non_dominated_sorted_solution2[i]))]
            front22 = sort_by_values(non_dominated_sorted_solution2_1[:], crowding_distance_values2[i][:])
            front = [non_dominated_sorted_solution2[i][front22[j]] for j in
                     range(0, len(non_dominated_sorted_solution2[i]))]
            front.reverse()
            for value in front:
                new_solution.append(value)
                if (len(new_solution) == pop_size):
                    break
            if (len(new_solution) == pop_size):
                break

        blocks = [blocks2[i] for i in new_solution]
        alpha = [alpha2[i] for i in new_solution]
        angle = [angle2[i] for i in new_solution]
        gen_no = gen_no + 1

        # experiment: nsgail foreach success
        # for logist in logist_population:
        #     if logist != classes_j:
        #         foreach_success.append(True)

    if not isAttackSuccess:
        losses, psnrs, logist_population, l2_population, x_adv_population, _ = nagaii_get2function(model, x_j,
                                                                                                         classes_j,
                                                                                                         wm_tensor_j,
                                                                                                         blocks, alpha,
                                                                                                         angle,
                                                                                                         waterMark)
    function1_values = losses
    function2_values = psnrs
    function1 = [i.item() * 1 for i in function1_values]
    function2 = [j * 1 for j in function2_values]
    # print("loss: ", function1, '\npsnr: ', function2)
    psnr_index = np.argsort(function2)

    # print("new_solution",new_solution)
    # plt.xlabel('loss')
    # plt.ylabel('psnr')
    # plt.scatter(function1, function2)
    # plt.show()
    # logist_population_format = []
    # l2_population_format = []
    # for index in new_solution:
    #     logist_population_format.append(logist_population.raw[index].cpu().item())
    #     l2_population_format.append(l2_population[index])
    # print("format: ", logist_population_format)
    # print("blocks", blocks)
    # print("alpha", alpha)
    # print("angle", angle)
    print("logist_population_format: ", logist_population)
    # print("l2_population_format: ", l2_population)

    attack_success_population = []
    for single_population in psnr_index:
        if logist_population[single_population] != classes_j:
            attack_success_population.append((alpha[single_population], angle[single_population],
                                              logist_population[single_population], l2_population[single_population],
                                              x_adv_population[single_population]))
    return attack_success_population

# if __name__ == '__main__':
#     """
#
#     """
#     image = Image.open('./test2.png')
#     # image.show()
#     # loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
#
#     adv_found = ep.zeros(x, len(x)).bool()
#
#     logist_clean = model(x)
#     logist_clean = logist_clean.argmax(1)
#     wm = []
#     one_batch_attack_success = 0
#     wm_numpy = torch.from_numpy(np.array(image, dtype=np.float32).transpose([2, 0, 1]))
#     for k in range(N):
#         wm.append(wm_numpy)
#     wm_tensor = torch.stack(wm)
#
#     for j in range(N):  # foreach a batch
#         if logist_clean[j] == classes[j]:
#             blocks, alpha, angle = nsgaii.get_init()
#             # x_j = "/home/frankfeng/researchData/code/adversarial_training_code/PLP/fast_adv/attacks/test/0.1504072755143_org.png"
#             # x_j = Image.open(x_j)
#             # x_j =transforms.ToTensor()(x_j).to(device)
#             # x_j = PyTorchTensor(x_j)
#             # print("x_j", x[j].raw.shape, x_j.shape)
#             attack_success_population = nsgaii.nsgaii(model, x[j], classes[j], wm_tensor[j], blocks, alpha, angle,
#                                                       self.waterMark)
#             # print("attack_success_population", attack_success_population)
#
#             # (alpha[single_population],
#             # angle[single_population],
#             # logist_population[single_population],
#             # l2_population[single_population],
#             # x_adv_population[single_population]))
#             #
#             if len(attack_success_population) > 0:
#                 one_batch_attack_success += 1
#             # plt.figure()
#             timestamp = str(int(time.time() * 1000))
#             for index in range(len(attack_success_population)):
#                 alpha = attack_success_population[index][0]
#                 angle = attack_success_population[index][1]
#                 logist_population = attack_success_population[index][2]
#                 l2_population = attack_success_population[index][3]
#                 if l2_population <= 20:
#                     xxx = attack_success_population[index][4].raw.cpu().numpy().transpose([1, 2, 0]) * 255
#                     img = Image.fromarray(xxx.astype('uint8')).convert('RGB')
#                     img.save("./watermark_resnet50_gen3/" + timestamp + "_" + str(j) + "_" + str(
#                         index) + "_alpha" + str(alpha) + "_angle" + str(angle) + "_logist" + str(
#                         logist_population) + "_l2" + str(l2_population) + ".png")
#
#                 if index == 0:
#                     img_org = x[j].raw.cpu().numpy().transpose([1, 2, 0]) * 255
#                     img_org = Image.fromarray(img_org.astype('uint8')).convert('RGB')
#                     img_org.save(
#                         "./watermark_resnet50_gen3/" + timestamp + "_" + str(j) + "_" + str(index) + "_" + str(
#                             classes[j].raw.cpu().numpy()) + ".png")
#
#         else:
#             one_batch_attack_success += 1
#     return one_batch_attack_success
