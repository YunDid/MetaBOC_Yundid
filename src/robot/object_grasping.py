# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2024.01.02
# --------------------------------------------------------


from cmath import sqrt
from faulthandler import disable
import random
import math
import time
import numpy as np
from scipy.interpolate import interp1d
from collections import deque
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import os
from src.robot.task import TASK, MAP, GRASP_STATE, GRASP_CONTROL


class Grasping(object):

    def __init__(self):
        # QThread.__init__(self, parent=None)
        # self.cur_state = GRASP_STATE.Reset
        self.cur_state = GRASP_STATE.Grasping
        self.cur_control = GRASP_CONTROL.Mode  # Mode   Freedom
        self.first_arm_count = 0
        # 用于记录抓取角度 当进入抓取流程后，根据该角度进行抓取
        self.first_arm_ang = 0
        self.arm_step = [1, 2, 3, 4, 5, 6, 7, 8]
        # 1~7：7个抓取状态
        # 8：竖直状态
        self.last_signal = None
        self.time_wating = 0  # 等待的时间
        self.clamp = False
        self.cur_step_id = 0  # 当前的arm_step id
        self.angle_deque = deque(maxlen=5)
        self.control_sig = {"free": [0, 0, 0, 0, 0, 0], "cur_step": 8, "speed": 1}
        # 增加角度容忍度
        self.STABLE_THRESHOLD = 5  # 稳定阈值，连续稳定5次
        self.ANGLE_TOLERANCE = 5.0  # 角度容忍度
    #     滑动加权窗口
        self.WINDOW_SIZE = 5
        self.weights = [0.1, 0.2, 0.3, 0.4, 0.5]  # 总和为1，最新的数据权重最大
        self.angle_window = deque(maxlen=self.WINDOW_SIZE)
        # 限制每次角度变化量
        self.MAX_ANGLE_CHANGE = 10  # 每次最大角度变化量，可以根据需要调整
        self.last_command_time = 0
        # 存储抓取的调整次数 初始为0
        self.adjustment_times = 0
        # 抓取成功的次数
        self.grap_times = 0
        # 抓取平均角度误差
        self.angle_sum = 0

    def limit_angle_change(self, new_angle):
        current_angle = self.angle_window[-2] if len(self.angle_window) >= 2 else new_angle
        if abs(new_angle - current_angle) > self.MAX_ANGLE_CHANGE:
            if new_angle > current_angle:
                return current_angle + self.MAX_ANGLE_CHANGE
            else:
                return current_angle - self.MAX_ANGLE_CHANGE
        return new_angle

    def save_grabing_txt(self, save_name):
        directory = os.path.dirname(save_name)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(save_name, 'a+') as f:
            if self.grap_times != 0:
                average_adjustments = self.adjustment_times / self.grap_times
                average_angle_error = self.angle_sum / self.adjustment_times
                # Write additional values to file
                f.write(f"Grap_times: {self.grap_times}, Average Adjustments: {average_adjustments}, Average Angle Error: {average_angle_error}\n")

    def weighted_smooth_angle(self, new_angle):
        self.angle_window.append(new_angle)
        if len(self.angle_window) < self.WINDOW_SIZE:
            return new_angle  # 窗口未满时返回原值
        smoothed_angle = sum(w * angle for w, angle in zip(self.weights, self.angle_window))
        return smoothed_angle

    def update_cur_control_mode(self, state):
        """
        全自由度控制 & 离散状态控制
        """
        self.cur_control = state

    def update_cur_state(self, state):
        """
        抓取存在三个状态：初始、方向调整、抓取动作
        """
        self.cur_state = state

    def update_angle(self, angle):
        self.angle_deque.append(angle)
        smoothed_angle = np.mean(self.angle_deque)
        return smoothed_angle

    def update_grasp(self):
        if self.cur_state == GRASP_STATE.Reset:
            pass
        elif self.cur_state == GRASP_STATE.Direction_Adjust:
            pass
        elif self.cur_state == GRASP_STATE.Grasping:
            pass


    def get_control_signal(self, angle):
        """
        通过输入角度的控制信号，得到用于控制小车的标准格式信号
        """
        # print(f'发送调整角度的命令angle === {angle}')
        if self.cur_control == GRASP_CONTROL.Mode:
            if self.cur_state == GRASP_STATE.Grasping:
                # if self.time_wating == 0:  # 只做初始化
                #     self.time_wating = time.time()
                #     print("update time stamp")
                #     return self.control_sig
                # else:
                #     tm = time.time()
                #     if tm - self.time_wating < 3.5:    # 超过时间，则更新控制参数
                #         print(tm - self.time_wating)
                #         return self.control_sig    # 返回上一次参数，不一定可行，可能需要添加默认无效参数
                #     else:
                #         self.time_wating = 0    # 更新步骤时，重置时间戳
                #         print("zero time")
                self.first_arm_count+=1
                if self.first_arm_count == 1:
                    self.first_arm_ang = angle
                speed = 1
                cur_step = self.arm_step[self.cur_step_id]
                if self.first_arm_ang == 0:
                    free_6 = [0, 0, 0, 0, 0, 0]
                else:
                    # first_arm_ang为可控抓取时的角度
                    free_6 = [self.first_arm_ang/100, 2000, 0, 0, 0, 0]

                self.cur_step_id = self.cur_step_id + 1
                if self.cur_step_id == 3:
                    self.clamp = True #进入强制抓取模式，避免反复抬头
                if self.cur_step_id == 8:  # 抓取完成后，回复至原竖直状态
                    self.grap_times +=1
                    self.cur_step_id = 0
                    speed = 1
                    free_6 = [0, 0, 0, 0, 0, 0]
                    self.clamp = False
                    self.cur_state = GRASP_STATE.Reset
                    self.first_arm_count = 0
                if self.cur_step_id == 0:
                    self.clamp = False

            elif self.cur_state == GRASP_STATE.Reset:
                self.clamp = False
                speed = 1
                cur_step = 8
                free_6 = [0, 0, 0, 0, 0, 0]
                self.time_wating = 0
                self.first_arm_count = 0

            self.control_sig = {"free": free_6, "cur_step": cur_step, "speed": speed}
            self.last_signal = self.control_sig
            return self.control_sig

        elif self.cur_control == GRASP_CONTROL.Freedom:
            self.clamp = False
            self.first_arm_count = 0
            cur_step = 0  # 自由控制状态
            speed = 0
            free_6 = [angle, 2000, 0, 0, 0, 0]  # 存放6个自由度的控制数值，整型
            # 该模式需设置 2000，用于将angle参数同步到状态控制模式

            # 这里angle需要根据机械臂的设定做一些调整，正负靠神经元来控制
            # if np.mean(self.angle_deque) < 10:
            self.control_sig = {"free": free_6, "cur_step": cur_step, "speed": speed}
            self.last_signal = self.control_sig

            # self.cur_control = GRASP_CONTROL.Mode    # 初始化为Mode,并根据

            return self.control_sig