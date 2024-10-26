import glob
import os
import shutil
import random

import tqdm


def copy():
    distDir = "C:\\projects\\PLP\\fast_adv_imagenet\\data\\train"
    name = 5000

    for label in tqdm.tqdm(range(1000)):
        srcDir = "C:\\datasets\\ILSVRC2012_img_train\\" + str(label)
        fileList = []
        for file in os.listdir(srcDir):
            fileList.append(file)

        randNums = []
        for i in range(5):
            randNums.append(random.randint(0, len(fileList) - 1))

        for i in range(5):
            shutil.copy(srcDir + "\\" + fileList[randNums[i]], distDir + "\\" + str(name) + ".png")
            name += 1


def makeMiniDataSet():
    for label in tqdm.tqdm(range(1000)):
        imgs = glob.glob(os.path.join("C:/datasets/ILSVRC2012_img_train", str(label), './*'))
        dist_path = os.path.join("C:/datasets/mini_ILSVRC2012_img_train", str(label))
        if not os.path.exists(dist_path):
            os.makedirs(dist_path)
        for i in range(200):
            shutil.copy(imgs[i], os.path.join(dist_path, os.path.basename(imgs[i])))


def imagenetLableRename():
    f = open("./labels.txt", "r")
    lines = f.readlines()
    map = {}
    for i, line in enumerate(lines):
        map[line.strip()] = i
    old_path = "/seu_share/home/fiki/ff/database/imagenet100"
    for key in map.keys():
        src = os.path.join(old_path, key)
        dst = os.path.join(old_path, str(map[key]))
        print(src, dst)
        if os.path.exists(src):
            os.rename(src, dst)


def split_test_database_from_imagenet100(imagenet100, test_path):
    for i in range(100):
        src_file_list = glob.glob(imagenet100 + '/' + str(i) + '/*')
        print(i)
        for j in range(100):
            srcfile = src_file_list[j]
            fpath, fname = os.path.split(srcfile)  # 分离文件名和路径
            dstpath = os.path.join(test_path, str(i))
            if not os.path.exists(dstpath):
                os.makedirs(dstpath)  # 创建路径
            shutil.move(srcfile, os.path.join(dstpath, fname))


def imagenetLableRename():
    f = open("./labels.txt", "r")
    lines = f.readlines()
    map = {}
    for i, line in enumerate(lines):
        map[line.strip()] = i
    old_path = "C:/datasets/ILSVRC2012_img_val"
    for key in map.keys():
        src = os.path.join(old_path, key)
        dst = os.path.join(old_path, str(map[key]))
        print(src, dst)
        if os.path.exists(src):
            os.rename(src, dst)


# if __name__ == '__main__':
#     split_test_database_from_imagenet100( "../../../imagenet100_test","../../../imagenet100")

import scipy.io
import shutil


def move_valimg(val_dir='./data/imagenet/val', devkit_dir='./data/ILSVRC2012_devkit_t12'):
    """
    move valimg to correspongding folders.
    val_id(start from 1) -> ILSVRC_ID(start from 1) -> WIND
    organize like:
    /val
       /n01440764
           images
       /n01443537
           images
        .....
    """
    # load synset, val ground truth and val images list
    synset = scipy.io.loadmat(os.path.join(devkit_dir, 'data', 'meta.mat'))

    ground_truth = open(os.path.join(devkit_dir, 'data', 'ILSVRC2012_validation_ground_truth.txt'))
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]

    root, _, filenames = next(os.walk(val_dir))
    for filename in filenames:
        # val image name -> ILSVRC ID -> WIND
        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id - 1]
        WIND = synset['synsets'][ILSVRC_ID - 1][0][1][0]
        print("val_id:%d, ILSVRC_ID:%d, WIND:%s" % (val_id, ILSVRC_ID, WIND))

        # move val images
        output_dir = os.path.join(root, WIND)
        if os.path.isdir(output_dir):
            pass
        else:
            os.mkdir(output_dir)
        shutil.move(os.path.join(root, filename), os.path.join(output_dir, filename))


import pandas as pd


def imagenet_val_rename(new_root, val_dir, csv):
    data = pd.read_csv(csv, sep=',', header='infer')  # ImageId, TrueLabel
    foldNames = {}
    for imageid, truelabel in tqdm.tqdm(zip(data["ImageId"], data["TrueLabel"])):
        foldNames[imageid] = truelabel
        new_path = os.path.join(new_root, str(truelabel))
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        shutil.copy(os.path.join(val_dir, imageid), os.path.join(new_path, imageid))


# if __name__ == '__main__':
#     imagenet_val_rename("C:\\datasets\\ILSVRC2012_img_val_formart", "C:\\datasets\\ILSVRC2012_img_val","C:\\datasets\\val.csv")
# val_dir = 'C:\\datasets\\ILSVRC2012_img_val'
# devkit_dir = 'C:\\datasets\\ILSVRC2012_devkit_t12'
# move_valimg(val_dir, devkit_dir)


class Node:
    def __init__(self, key, children=None):
        if children is None:
            children = []
        self.key = key
        self.children = children

"""
递归寻找key节点
"""
def findChildrenByKey(node, key):
    if node is None:  # 如果当前node是空，说明经没有节点了，函数返回None
        return None
    if node.key == key:  # 如果当前节点的key就是我们要找的，就返回当前节点
        return node

    # 代码运行到这里，说明当前节点node不是我们要找的key，那么就for循环当前节点node的所有孩子节点
    for child in node.children:
        # 每个孩子节点都调用依次findChildrenByKey，以孩子节点为目标，寻找key
        subTree = findChildrenByKey(child, key)
        if subTree is not None: # 如果输出subTree不为None，说明孩子节点中找到了目标key，直接返回
            return subTree

"""
    深度优先遍历当前节点到每个子节点的路径
    node是当前节点
    path是遍历到当前节点时，从根节点到当前节点的路径数据
    result用来保存最终结果
"""
def dfs(node, path, result):
    if node is None: # 如果当前节点为None，直接退出
        return
    path.append(node.key) # 否则，将当前节点放入path，保村中间结果保持
    result.append(path[:]) # 把path放入result。
    for child in node.children: #for循环当前节点的每个孩子，
        dfs(child, path, result)
    path.pop() # 递归到叶子节点了，清空中间结果


if __name__ == '__main__':
    node = Node(0, [Node(1, [Node(2, [Node(2), Node(3, [Node(5), Node(6)]), Node(4)])]), Node(7, [Node(8), Node(9)])])
    root = findChildrenByKey(node, 1)
    result = []
    dfs(root, [], result)
    print(result)