from symbol import parameters
from time import time
import h5py
import os
import matplotlib.pyplot as plt
import numpy as np


# path = "C:/Users/tju/Desktop/BrainICROA/1667543959.179362.h5"
# time_txt = "C:/Users/tju/Desktop/BrainICROA/1667543959.179362_sti_timestamp.txt"

# 含细胞 21
# path = "C:/Users/admin/Desktop/BrainICROA/1667546325.2290661.h5"
# time_txt = "C:/Users/admin/Desktop/BrainICROA/1667546325.2290661_sti_timestamp.txt"
# ele = 24


# 含细胞 21
# path = "C:/Users/admin/Desktop/BrainICROA/1107.h5"
# time_txt = "C:/Users/admin/Desktop/BrainICROA/1107_sti_timestamp.txt"
# ele = 24

# 含细胞 21, 8s信号，刺激在前
path = "E:\\Exp_Data_MZY\\MetaData\\AI\\20240408\\cell1\\raw\\train0.h5"
# time_txt = "./1672475231.0732474_sti_timestamp.txt"
ele = 55

# root = "C:/Users/admin/Desktop/BrainICROA/1668398936.4360092"
# path = root + ".h5"
# time_txt = root + "_sti_timestamp.txt"
# ele = 24

# 71
# path = "C:/Users/admin/Desktop/BrainICROA/1667546481.6972933.h5"
# time_txt = "C:/Users/admin/Desktop/BrainICROA/1667546481.6972933_sti_timestamp.txt"
# ele = 37


file = h5py.File(path)
keys = list(file.keys())

# f = open(time_txt)
# time_stamp = f.read()
# time_stamp = time_stamp.split("\n")

data = file["data"]
para = file["parameter"]
sample = para[0]
constant = para[1]

frame_time = 1.0 / sample

# frame = []; y_frame = []
# for i in range(len(time_stamp)):
#     if time_stamp[i] != "21," and time_stamp[i] != "":
#         f = float(time_stamp[i]) / frame_time
#         # frame.append(int(f))
#         frame.append(int(f) + 282 / (frame_time * 1000))
#         y_frame.append(0.3)

test_data = data[ele - 1]
sti_data = data[69]
sti_pos = np.where(sti_data > 0)[0]

frame = []; y_frame = []
for i in range(len(sti_pos)):
    frame.append(sti_pos[i])
    y_frame.append(0.3)


print("data length:", len(test_data))

x = np.arange(len(test_data))
y = test_data * constant / 1000.0

plt.plot(x, y)

plt.xlabel('Data Frame')
plt.scatter(frame, y_frame, c="red")

# 添加y轴标签
plt.ylabel('Voltages(mV)')
# 添加图形标题
plt.title('Channel Data')
# 添加图例
plt.legend()
# 显示图形
plt.show()

print("end")