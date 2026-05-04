from symbol import parameters
from time import time
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np

path = "./out_spikes/spikes_left_right.h5"


file = h5py.File(path)
keys = list(file.keys())


data = file["data"]


left = data[0, :]
right = data[1, :]

x = np.arange(len(left))

plt.plot(x, left, label="left")
plt.plot(x, right, label="right")

plt.xlabel('Data Frame')
# plt.scatter(frame, y_frame, c="red")

# 添加y轴标签
plt.ylabel('Spikes Number')
# 添加图形标题
plt.title('Spikes Data')
# 添加图例
plt.legend()
# 显示图形
plt.show()

print("end")