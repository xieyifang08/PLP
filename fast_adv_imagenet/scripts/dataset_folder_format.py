import pandas as pd
import os
import sys
data = pd.read_csv(r'H:\\labels.csv',sep=',',header='infer',usecols=[0,1])

array=data.values[0::,0::]  # 读取全部行，全部列
print(array[0][0])
print(array[0][1].strip())

name_dics = {}
for i in range(len(array)):
    name_dics[array[i][1].strip()] = array[i][0]

path = "H:\\datasets\\ILSVRC2012_img_train"
old_names = os.listdir(path)  # 取路径下的文件名，生成列表
for old_name in old_names:  # 遍历列表下的文件名
    new_name=old_name.replace(old_name,str(name_dics[old_name]))   # 将原来名字里的‘test’替换为‘test2’
    os.rename(os.path.join(path,old_name),os.path.join(path,new_name))  # 子文件夹重命名
    print (old_name,"has been renamed successfully! New name is: ",new_name)   # 输出
