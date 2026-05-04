import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']

baseline = [55, 55, 45, 40] 
test = [55, 36, 28, 30]
# plt.title('Experiments of robots in virtual environments to avoid obstacles')  # 标题
plt.title('人工脑控制的机器人避障实验')  # 标题

# plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示汉字
# plt.rcParams['axes.unicode_minus'] =False

# plt.xlabel('Tests (5min, 5Hz-150mV-4s stimulus)',fontsize=12)  # x轴标题以及标题大小设置
# plt.ylabel('Hits',fontsize=12)  # y轴标题

plt.xlabel('训练次数（人工脑培养天数）',fontsize=12)  # x轴标题以及标题大小设置
plt.ylabel('障碍物撞击次数',fontsize=12)  # y轴标题

#刻度值字体大小设置（x轴和y轴同时设置）
plt.tick_params(labelsize=10)

plt.plot(["0(21天)", "2(21天)", "3(22天)", "4(24天)"], baseline, linestyle='-', marker='o', color='tab:orange')  # 绘制折线图
plt.plot(["0(21天)", "2(21天)", "3(22天)", "4(24天)"], test, linestyle='-', marker='o', color='tab:blue')  # 绘制折线图   # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
# plt.plot(x_swin_mlp_flops, x_swin_mlp_acc, linestyle='dotted', marker='o', color='tab:cyan')


# plt.plot(x_cycle_mlp_flops, x_cycle_mlp_acc, linestyle='dashed', marker='o', color="tab:orange")
# plt.plot(x_sparse_mlp_flops, x_sparse_mlp_acc, linestyle='dashed', marker='o', color="tab:green")
# plt.plot(x_hire_mlp_flops, x_hire_mlp_acc, linestyle='dashed', marker='o', color="tab:pink")
# plt.plot(x_res_mlp_flops, x_res_mlp_acc, linestyle='dashed', marker='o', color="tab:purple")
# plt.plot(x_wave_mlp_flops, x_wave_mlp_acc, linestyle='dashed', marker='o', color="tab:gray")

# plt.plot(x_strip_mlp_flops, x_strip_mlp_acc, linestyle='-', marker='o', color="red")

# point = ["$P_1$", "$P_2$","$P_3$","$P_4$"]
# for i in range(len(x_strip_mlp_flops)):
#     plt.text(x_strip_mlp_flops[i]-1.0, x_strip_mlp_acc[i]+0.1, point[i], fontsize=10, color="red")

plt.grid(linestyle = '--')

# 设置曲线名称
# plt.legend(['RegNetY', 'Swin', 'CycleMLP', 'Sparse-MLP', 'Hire-MLP', 'ResMLP', 'Wave-MLP', 'Strip-MLP(ours)'],loc=0)
#图例大小可选----'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'

plt.legend(["训练前测试结果", "训练后测试结果"], loc=0)

plt.savefig('./out/robot_avoid.png',bbox_inches='tight',pad_inches=0.1) #保存图片，这里增加这两个参数可以消除保存下来图像的白边节省空间，bbox_inches='tight',pad_inches=0)
plt.show()  # 显示曲线图
