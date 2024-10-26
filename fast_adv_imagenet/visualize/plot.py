import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from matplotlib.pyplot import MultipleLocator

# 设置距离
x1 = np.array([0, 0.25, 0.5, 1, 2, 3])

# 设置相似度
y1 = np.array([0.849, 0.76, 0.671, 0.476, 0.27, 0.161])

# 插值法之后的x轴值，表示从0到3间距为0.1的30个数
xnew1 = np.arange(0, 3, 0.1)

# 实现函数
func = interpolate.interp1d(x1, y1, kind='quadratic')

# 利用xnew和func函数生成ynew,xnew数量等于ynew数量
ynew1 = func(xnew1)

# 平滑处理后曲线
plt.plot(xnew1, ynew1, "r", ls="--", lw=1, label='012RANDOM')
#################################################
x2 = np.array([0, 0.25, 0.5, 1, 2, 3])
y2 = np.array([0.799, 0.745, 0.67, 0.499, 0.348, 0.25])
xnew2 = np.arange(0, 3, 0.1)
func = interpolate.interp1d(x2, y2, kind='quadratic')
ynew2 = func(xnew2)
plt.plot(xnew2, ynew2, "b", ls="--", lw=1, label='0.25RANDOM')
######################################################
x3 = np.array([0, 0.25, 0.5, 1, 2, 3])
y3 = np.array([0.659, 0.67, 0.634, 0.556, 0.458, 0.399])
xnew3 = np.arange(0, 3, 0.1)
func = interpolate.interp1d(x3, y3, kind='quadratic')
ynew3 = func(xnew3)
plt.plot(xnew3, ynew3, "y", ls="--", lw=1, label='0.5RANDOM')
######################################################


x4 = np.array([0, 0.25, 0.5, 1, 2, 3])
y4 = np.array([0.859, 0.78, 0.681, 0.451, 0.221, 0.124])
xnew4 = np.arange(0, 3, 0.1)
func = interpolate.interp1d(x4, y4, kind='quadratic')
ynew4 = func(xnew4)
plt.plot(xnew4, ynew4, "g", ls="-", lw=1, label='012RAT')
######################################################
x5 = np.array([0, 0.25, 0.5, 1, 2, 3])
y5 = np.array([0.859, 0.7678, 0.6638, 0.492, 0.243, 0.141])
xnew5 = np.arange(0, 3, 0.1)
func = interpolate.interp1d(x5, y5, kind='quadratic')
ynew5 = func(xnew5)
plt.plot(xnew5, ynew5, "c", ls="-", lw=1, label='05_RAT')
######################################################
x6 = np.array([0, 0.25, 0.5, 1, 2, 3])
y6 = np.array([0.83, 0.775, 0.673, 0.47, 0.233, 0.134])
xnew6 = np.arange(0, 3, 0.1)
func = interpolate.interp1d(x6, y6, kind='quadratic')
ynew6 = func(xnew6)
plt.plot(xnew6, ynew6, "m", ls="-", lw=1, label='025_RAT')
# 设置x,y轴代表意思
plt.xlabel("ℓ2 radius")
plt.ylabel("Accuracy")

x_major_locator = MultipleLocator(0.25)
# 把x轴的