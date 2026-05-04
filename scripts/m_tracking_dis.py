from symbol import parameters
from time import time
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np


# 含细胞 21
path = "./out_tracking/2023_6_6_11_44.txt"


with open(path, "r") as f:
    dis = f.readlines()
    print("Read distance file finished!...")



x = np.arange(len(dis))
y = [float(i.split("\n")[0]) for i in dis]

plt.plot(x, y)

plt.xlabel('Data Frame')
# plt.scatter(x, y, c="red")

# 添加y轴标签
plt.ylabel('Distance (pixel)')
# 添加图形标题
plt.title('Tracking Distance Data')
# 添加图例
plt.legend()
# 显示图形
plt.show()

print("end")