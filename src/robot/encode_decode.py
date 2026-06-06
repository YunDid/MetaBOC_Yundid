# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

import os
import numpy as np
import time
import math

from operator import index
from tkinter.messagebox import NO
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *

from src.Ui_EncodeDecode import Ui_EncodeDecodeSetting
from src.robot.task import TASK, MAP


class EncodingDecoding(QDialog, Ui_EncodeDecodeSetting):

    def __init__(self, parent=None, para=None):
        super(EncodingDecoding, self).__init__(parent)
        self.setupUi(self)
        self.setWindowModality(Qt.WindowModality.WindowModal)

        self.para = para
        self.task = TASK.Obstacle_Avoidance
        # ***************  for encoding  ***************
        if para is not None:
            en = para["para_encoding"]
            de = para["para_decoring"]
            
            self.spb_sti_max.setValue(en[0]["sti_max"])
            self.spb_sti_min.setValue(en[0]["sti_min"])
            self.spb_distance_max.setValue(en[0]["distance_max"])
            self.spb_distance_min.setValue(en[0]["distance_min"])

            self.dsp_wv_max.setValue(de[0]["wv_max"])
            self.dsp_wv_min.setValue(de[0]["wv_min"])
            self.spb_fre_max.setValue(de[0]["max_fre"])

        # *************** for decoding  ***************

        self.left_mea = []    # 用于缓存不同时刻的
        self.right_mea = []

        self.cur_left_whell = 0.5    # 记录两侧的实时轮速
        self.cur_right_whell = 0.5

        self.if_change_para = False

        self.grasping_ang_dis = 1000    # 用于记录和更新grasping任务的机械臂的角度误差绝对值

        # for button
        self.pb_ok.clicked.connect(self.ok)
        self.pb_cancel.clicked.connect(self.cancel)

        self.spb_sti_max.valueChanged.connect(self.sti_changed)
        self.spb_sti_min.valueChanged.connect(self.sti_changed)
        self.spb_distance_max.valueChanged.connect(self.distance_changed)
        self.spb_distance_min.valueChanged.connect(self.distance_changed)
        self.dsp_wv_max.valueChanged.connect(self.wheel_velocity)
        self.dsp_wv_min.valueChanged.connect(self.wheel_velocity)
        self.Wheel_speed_threshold = 1.0

    def ok(self):
        para_encoding = {"sti_max":self.spb_sti_max.value(),
            "sti_min":self.spb_sti_min.value(),
            "distance_max":self.spb_distance_max.value(),
            "distance_min":self.spb_distance_min.value()}
        para_decoring = {"wv_max":self.dsp_wv_max.value(),
            "wv_min":self.dsp_wv_min.value(),
            "max_fre":self.spb_fre_max.value()}
        
        np.savez("./encode_decode/en_de_coding.npz", para_encoding=[para_encoding], 
                para_decoring=[para_decoring])
        self.if_change_para = True
        self.close()

    def cancel(self):
        self.if_change_para = False
        self.close()

    def closeEvent(self, a0: QCloseEvent) -> None:
        if not self.if_change_para:
            path = "./encode_decode/en_de_coding.npz"
            if os.path.exists(path):
                para = np.load(path, allow_pickle=True)
                en = para["para_encoding"]
                de = para["para_decoring"]
                
                self.spb_sti_max.setValue(en[0]["sti_max"])
                self.spb_sti_min.setValue(en[0]["sti_min"])
                self.spb_distance_max.setValue(en[0]["distance_max"])
                self.spb_distance_min.setValue(en[0]["distance_min"])

                self.dsp_wv_max.setValue(de[0]["wv_max"])
                self.dsp_wv_min.setValue(de[0]["wv_min"])
                self.spb_fre_max.setValue(de[0]["max_fre"])
        self.if_change_para = False
        return super().closeEvent(a0)

    def set_task(self, task):
        self.task = task

    def get_para(self):
        para_encoding = {"sti_max":self.spb_sti_max.value(),
            "sti_min":self.spb_sti_min.value(),
            "distance_max":self.spb_distance_max.value(),
            "distance_min":self.spb_distance_min.value()}
        para_decoring = {"wv_max":self.dsp_wv_max.value(),
            "wv_min":self.dsp_wv_min.value(),
            "max_fre":self.spb_fre_max.value()}
        return para_encoding, para_decoring

    def sti_changed(self, value):
        max = self.spb_sti_max.value()
        min = self.spb_sti_min.value()
        if min > max:
            self.spb_sti_min.setValue(max)

    def distance_changed(self, value):
        max = self.spb_distance_max.value()
        min = self.spb_distance_min.value()
        if min > max:
            self.spb_distance_min.setValue(max)

    def wheel_velocity(self, value):
        max = self.dsp_wv_max.value()
        min = self.dsp_wv_min.value()
        if min > max:
            self.dsp_wv_min.setValue(max)


    def normalize_distance(self, x):
        out = 0
        if x < self.spb_distance_max.value():
            out = (x - self.spb_distance_min.value()) / (self.spb_distance_max.value() - self.spb_distance_min.value())
        else:
            out = 1.0
        return out

    # # 对MEA输出的信号解码
    # def decode(self, left, right):
    #     """
    #     返回解码结果: left right MEA输出的spikes信号,100ms内spikes个数的各电极的平均值, 
    #     output: 转化为控制小车的角速度信号
    #     """

    #     if len(self.left_mea) < 2:
    #         self.left_mea.append(left)
    #         self.right_mea.append(right)
    #     else:
    #         self.left_mea[0] = self.left_mea[1]
    #         self.left_mea[1] = left

    #         self.right_mea[0] = self.right_mea[1]
    #         self.right_mea[1] = right

    #     w_l = (left / self.spb_fre_max.value()) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
    #     w_r = (right / self.spb_fre_max.value()) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()


    #     if left >= self.spb_fre_max.value():
    #         w_l = self.dsp_wv_max.value()
    #     if right >= self.spb_fre_max.value():
    #         w_r = self.dsp_wv_max.value()

    #     return w_l, w_r


    def encode(self, left, right, ang_dis_left, ang_dis_right):
        """
        环境信息，转化为刺激信号
        """
        if self.task == TASK.Obstacle_Avoidance:
            # return self.encode_obstacle_avoidance(left, right)
            return self.encode_obstacle_avoidance_with_angleAndDis(left, right, ang_dis_left, ang_dis_right)
            # return self.encode_obstacle_avoidance_with_angle(left, right, ang_dis_left, ang_dis_right)
        elif self.task == TASK.Object_Tracking:
            return self.encode_object_tracking(left, right, ang_dis_left, ang_dis_right)
        elif self.task == TASK.Object_Grasping:
            """
            仅left、right为有效值，ang_xx为None
            """
            return self.encode_object_grasp(left, right, ang_dis_left, ang_dis_right)


    # 对MEA输出的信号解码
    def decode(self, left, right, min_dis_left, min_dis_right):
        """
        MEA输出的信号，转化为机器人的控制信号
        """
        if self.task == TASK.Obstacle_Avoidance:
            # return self.decode_obstacle_avoidance(left, right, min_dis_left, min_dis_right)    # 原方法 非
            return self.decode_obstacle_avoidance_same_side_history_gap(left, right, min_dis_left, min_dis_right)  # 与同侧历史值比较 momen
            # return self.decode_obstacle_avoidance_Different_side_history_gap(left, right, min_dis_left, min_dis_right) # 与异侧历史值比较
        elif self.task == TASK.Object_Tracking:
            # return self.decode_object_tracking(left, right, min_dis_left, min_dis_right)
            return self.decode_object_tracking_same_side_history_gap(left, right, min_dis_left, min_dis_right)
        elif self.task == TASK.Object_Grasping:
            # min_dis_left
            return self.decode_object_grasp(left, right, min_dis_left, min_dis_right)

    def encode_obstacle_avoidance_with_angle(self, left, right, ang_dis_left, ang_dis_right):
        """
        返回编码结果: left right左右轮的最近障碍物距离信息, 转化为刺激信号
        加入角度信息的编码方法
        """

        # 距离过大时，不进行刺激
        # if left > 200 and right > 200:
        #     return 0, 0

        # weight = 1.0
        # dis_left = weight * left * math.sin(ang_dis_left)
        # dis_right = weight * right * math.sin(ang_dis_right)
        # print("dis angle left:", dis_left, ang_dis_left)
        # print("dis angle right:", dis_right, ang_dis_right)

        # left = left + abs(dis_left)
        # right = right + abs(dis_right)

        # if left < 80 and right < 80:  # 距离障碍物都很近时
        #     if left > right:   # 两边距离相近，对远一点的增加距离
        #         left = left + 80
        #     else:
        #         right = right + 80 

        ang_l = abs(ang_dis_left) % (math.pi)
        ang_r = abs(ang_dis_right) % (math.pi)
        l_n = 1 - abs(ang_l) / (math.pi / 2)
        r_n = 1 - abs(ang_r) / (math.pi / 2)

        # 避障
        # l_n = 1.0 - self.normalize_distance(left)
        # r_n = 1.0 - self.normalize_distance(right)

        
        # print("angle left:", ang_l/3.14159*180)
        # print("angle right:", ang_r/3.14159*180)

        # print("distance left norm:", l_n)
        # print("distance right norm:", r_n)

        l = (self.spb_sti_max.value() - self.spb_sti_min.value()) * l_n + self.spb_sti_min.value()
        r = (self.spb_sti_max.value() - self.spb_sti_min.value()) * r_n + self.spb_sti_min.value()

        # if ang_l/3.14159*180 < 1:
        #     print()
        # print("freqency left:", l)
        # print("freqency right:", r)
        return l, r

    def encode_obstacle_avoidance_with_angleAndDis(self, left, right, ang_dis_left, ang_dis_right):
        """
        返回编码结果: left right左右轮的最近障碍物距离信息, 转化为刺激信号
        加入角度信息的编码方法
        """
        if left < 80 and right < 80:  # 距离障碍物都很近时
            if left > right:   # 两边距离相近，对远一点的增加距离
                left = left + 80
            else:
                right = right + 80

        ang_l = abs(ang_dis_left) % (math.pi)
        ang_r = abs(ang_dis_right) % (math.pi)

        print(60 * "+", ang_l, ang_r)
        
        # 避障
        if ang_l > (math.pi / 2):
            l_n = 1.0 - self.normalize_distance(left)
            r_n = 1.0 - self.normalize_distance(right)
        else:
            l_n = 1.0 - (self.normalize_distance(left)+abs(ang_l) / (math.pi / 2))/2
            r_n = 1.0 - (self.normalize_distance(right)+abs(ang_r) / (math.pi / 2))/2

        if l_n < 0 or r_n < 0:
            print(30 * "=", l_n, ang_l)
            print(30 * "=", r_n, ang_r)
        l = (self.spb_sti_max.value() - self.spb_sti_min.value()) * l_n + self.spb_sti_min.value()
        r = (self.spb_sti_max.value() - self.spb_sti_min.value()) * r_n + self.spb_sti_min.value()
        return l, r


    def encode_obstacle_avoidance(self, left, right):
        """
        返回编码结果: left right左右轮的最近障碍物距离信息, 转化为刺激信号
        """

        # if left - right < 30:
        #     left = left + 30
        # else:
        #     right = right + 30


        # 距离过大时，不进行刺激
        # if left > 200 and right > 200:
        #     return 0, 0

        # self.spb_distance_min.setValue(0)
        # self.spb_distance_max.setValue(300)


        if left < 80 and right < 80:  # 距离障碍物都很近时
            if left > right:   # 两边距离相近，对远一点的增加距离
                left = left + 80
            else:
                right = right + 80 

        # 避障
        l_n = 1.0 - self.normalize_distance(left)
        r_n = 1.0 - self.normalize_distance(right)

        
        # print("distance left:", left)
        # print("distance right:", right)

        # print("distance left norm:", l_n)
        # print("distance right norm:", r_n)

        l = (self.spb_sti_max.value() - self.spb_sti_min.value()) * l_n + self.spb_sti_min.value()
        r = (self.spb_sti_max.value() - self.spb_sti_min.value()) * r_n + self.spb_sti_min.value()

        # print("freqency left:", l)
        # print("freqency right:", r)
        return l, r

    def decode_obstacle_avoidance(self, left, right, min_dis_left, min_dis_right):
        """
        返回解码结果: left right MEA输出的spikes信号
        output: 转化为控制小车的角速度信号
        """

        if len(self.left_mea) < 2:
            self.left_mea.append(left)
            self.right_mea.append(right)
        else:
            self.left_mea[0] = self.left_mea[1]
            self.left_mea[1] = left

            self.right_mea[0] = self.right_mea[1]
            self.right_mea[1] = right

        # 计算spike差值
        gap = 0
        if left > right:
            gap = left - right
        else:
            gap = right - left
        
        print(30*"--", gap)

        # 自适应归一化
        # max_value = 20
        # if gap > 20 and gap < 50:
        #     max_value = 50
        # else:
        #     max_value = 80
        max_value = 30

        # spike差值施加在发放更大的一侧的轮速
        if left >= right:
            w_l = (gap / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
            w_r = 0 + self.dsp_wv_min.value()
        else:
            w_r = (gap / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
            w_l = 0 + self.dsp_wv_min.value()


        # # spike差值施加在距离更近的轮子
        # if min_dis_left <= min_dis_right:
        #     w_l = (gap / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
        #     w_r = 0 + self.dsp_wv_min.value()
        # else:
        #     w_r = (gap / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
        #     w_l = 0 + self.dsp_wv_min.value()


        # 直接归一化
        # max_value = 30
        # w_l= left / max_value
        # w_r = right / max_value



        if left >= self.spb_fre_max.value():
            w_l = self.dsp_wv_max.value()
        if right >= self.spb_fre_max.value():
            w_r = self.dsp_wv_max.value()

        return w_l, w_r
    
    # def decode_obstacle_avoidance_same_side_history_gap(self, left, right, min_dis_left, min_dis_right):
    #     """
    #     返回解码结果: left right MEA输出的spikes信号
    #     output: 转化为控制小车的角速度信号
    #     同一侧，与历史值进行比较
    #     """

    #     num_history_point = 3    # 当前值与历史多少个值进行比较，num_history_point >= 2
    #     if len(self.left_mea) < num_history_point:
    #         self.left_mea.append(left)
    #         self.right_mea.append(right)
    #         return 0.5, 0.5

    #     # 计算spike差值
    #     gap_left = left - np.array(self.left_mea).mean()    # 可能为负值
    #     gap_right = right - np.array(self.right_mea).mean()
        
    #     print(30*"--", "left gap", gap_left)
    #     print(30*"--", "right gap", gap_right)

    #     # 自适应归一化，max_value可能需要根据机器人运动进行修改
    #     max_value = 30
    #     # w_l = (gap_left / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
    #     # w_r = (gap_right / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
    #     w_l = (gap_left / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.cur_left_whell
    #     w_r = (gap_right / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.cur_right_whell

    #     if left >= self.spb_fre_max.value():
    #         w_l = self.dsp_wv_max.value()
    #     if right >= self.spb_fre_max.value():
    #         w_r = self.dsp_wv_max.value()

    #     # 更新当前历史值
    #     if len(self.left_mea) >= num_history_point:
    #         self.left_mea.pop(0)
    #         self.left_mea.append(left)

    #         self.right_mea.pop(0)
    #         self.right_mea.append(right)

    #     # 更新当前轮速
    #     self.cur_left_whell = w_l
    #     self.cur_right_whell = w_r

    #     return w_l, w_r

    def decode_obstacle_avoidance_same_side_history_gap(self, left, right, min_dis_left, min_dis_right):
        """
        返回解码结果: left right MEA输出的spikes信号
        output: 转化为控制小车的角速度信号
        同一侧，与历史值进行比较
        """

        num_history_point = 3  # 当前值与历史多少个值进行比较，num_history_point >= 2
        if len(self.left_mea) < num_history_point:
            self.left_mea.append(left)
            self.right_mea.append(right)
            return 0.5, 0.5

        # 计算spike差值
        gap_left = left - np.array(self.left_mea).mean()  # 可能为负值
        gap_right = right - np.array(self.right_mea).mean()
        if gap_left < -30:
            gap_left = -30
            # print(f'gapleft<0 = {gap_left}--self.left_mea = {self.left_mea}--leftnow={left}')
        if gap_right < -30:
            gap_right = -30
            # print(f'gapleft<0 = {gap_right}--self.left_mea = {self.right_mea}--rightnow = {right}')
        # 自适应归一化，max_value可能需要根据机器人运动进行修改
        max_value = 20
        # w_l = (gap_left / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
        # w_r = (gap_right / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
        w_l = (gap_left / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
        w_r = (gap_right / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()

        if left >= self.spb_fre_max.value():
            w_l = self.dsp_wv_max.value()
        if right >= self.spb_fre_max.value():
            w_r = self.dsp_wv_max.value()

        # 更新当前历史值
        if len(self.left_mea) >= num_history_point:
            self.left_mea.pop(0)
            self.left_mea.append(left)

            self.right_mea.pop(0)
            self.right_mea.append(right)
        # 每个轮子保底 wv_min（原逻辑只在两轮同时为 0 时兜底，会出现单轮停住急转 / 完全停车）。
        # 现在每轮都不低于 wv_min → 车一直有最小前进速度、不会完全停住，差速仍可转向。
        # 注意：wv_min 由 Encode/Decode 参数设定；设为 0 仍允许停车，要保证一直走就设 >0（如 0.2）。
        wv_min = self.dsp_wv_min.value()
        if w_l < wv_min:
            w_l = wv_min
        if w_r < wv_min:
            w_r = wv_min
        # self.Wheel_speed_threshold = 1.0
        if w_l > w_r:
            if w_l > self.Wheel_speed_threshold:
                w_r = self.Wheel_speed_threshold * w_r / w_l
                w_l = self.Wheel_speed_threshold
        else:
            if w_r > self.Wheel_speed_threshold:
                w_l = self.Wheel_speed_threshold * w_l / w_r
                w_r = self.Wheel_speed_threshold
        self.cur_left_whell = w_l
        self.cur_right_whell = w_r

        return w_l, w_r

    def decode_obstacle_avoidance_Different_side_history_gap(self, left, right, min_dis_left, min_dis_right):
        """
        返回解码结果: left right MEA输出的spikes信号
        output: 转化为控制小车的角速度信号
        """

        if len(self.left_mea) < 2:
            self.left_mea.append(left)
            self.right_mea.append(right)
        else:
            self.left_mea[0] = self.left_mea[1]
            self.left_mea[1] = left

            self.right_mea[0] = self.right_mea[1]
            self.right_mea[1] = right

        # 计算spike差值
        gap = 0
        if left > right:
            gap = left - right
        else:
            gap = right - left

        print(30 * "--", gap)
        max_value = 20

        # spike差值施加在发放更大的一侧的轮速
        # if left >= right:
        #     w_l = (gap / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
        #     w_r = 0 + self.dsp_wv_min.value()
        # else:
        #     w_r = (gap / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
        #     w_l = 0 + self.dsp_wv_min.value()

        # spike差值施加在距离更近的轮子
        if min_dis_left <= min_dis_right:
            w_l = (gap / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
            w_r = 0 + self.dsp_wv_min.value() + 3
        else:
            w_r = (gap / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
            w_l = 0 + self.dsp_wv_min.value() +4

        # 直接归一化
        # max_value = 30
        # w_l= left / max_value
        # w_r = right / max_value

        if left >= self.spb_fre_max.value():
            w_l = self.dsp_wv_max.value()
        if right >= self.spb_fre_max.value():
            w_r = self.dsp_wv_max.value()

        # 每个轮子保底 wv_min（原逻辑只在两轮同时为 0 时兜底，会出现单轮停住急转 / 完全停车）。
        # 现在每轮都不低于 wv_min → 车一直有最小前进速度、不会完全停住，差速仍可转向。
        # 注意：wv_min 由 Encode/Decode 参数设定；设为 0 仍允许停车，要保证一直走就设 >0（如 0.2）。
        wv_min = self.dsp_wv_min.value()
        if w_l < wv_min:
            w_l = wv_min
        if w_r < wv_min:
            w_r = wv_min
        #  设置阈值
        # 阈值
        # self.Wheel_speed_threshold = 1.0
        if w_l > w_r:
            if w_l > self.Wheel_speed_threshold:
                # 按比例缩放
                w_r = self.Wheel_speed_threshold * w_r / w_l
                w_l = self.Wheel_speed_threshold
        else:
            if w_r > self.Wheel_speed_threshold:
                w_l = self.Wheel_speed_threshold * w_l / w_r
                w_r = self.Wheel_speed_threshold

        return w_l, w_r

    def encode_object_tracking(self, left, right, ang_dis_left, ang_dis_right):
        
        if left < 80 and right < 80:  # 距离障碍物都很近时
            if left - right < 80:   # 两边距离相近，对远一点的增加距离
                left = left + 80
            else:
                right = right + 80 

        # tracking
        if left > right:
            left = 2*left
            right = 0.5 * right
        else:
            right = 2*right
            left = 0.5* left
        l_n = self.normalize_distance(left)
        r_n = self.normalize_distance(right)

        
        # print("distance left:", left)
        # print("distance right:", right)

        # print("distance left norm:", l_n)
        # print("distance right norm:", r_n)

        l = (self.spb_sti_max.value() - self.spb_sti_min.value()) * l_n + self.spb_sti_min.value()
        r = (self.spb_sti_max.value() - self.spb_sti_min.value()) * r_n + self.spb_sti_min.value()

        # print("freqency left:", l)
        # print("freqency right:", r)
        return l, r

    def decode_object_tracking(self, left, right, min_dis_left, min_dis_right):
        """
        细胞的spike信号转化为控制轮速的信号
        """

        if len(self.left_mea) < 2:
            self.left_mea.append(left)
            self.right_mea.append(right)
        else:
            self.left_mea[0] = self.left_mea[1]
            self.left_mea[1] = left

            self.right_mea[0] = self.right_mea[1]
            self.right_mea[1] = right

        # 计算spike差值
        gap = 0
        if left > right:
            gap = left - right
        else:
            gap = right - left

        print(30 * "--", gap)
        max_value = 30
        # spike差值应该给更远的轮子
        print(f'min_dis_left={min_dis_left},min_dis_right={min_dis_right}')
        if min_dis_left >= min_dis_right:
            w_l = (gap / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
            w_r = 0 + self.dsp_wv_min.value()
        else:
            w_r = (gap / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
            w_l = 0 + self.dsp_wv_min.value()
        # 每个轮子保底 wv_min（原逻辑只在两轮同时为 0 时兜底，会出现单轮停住急转 / 完全停车）。
        # 现在每轮都不低于 wv_min → 车一直有最小前进速度、不会完全停住，差速仍可转向。
        # 注意：wv_min 由 Encode/Decode 参数设定；设为 0 仍允许停车，要保证一直走就设 >0（如 0.2）。
        wv_min = self.dsp_wv_min.value()
        if w_l < wv_min:
            w_l = wv_min
        if w_r < wv_min:
            w_r = wv_min
        #  设置阈值
        # 阈值
        # self.Wheel_speed_threshold = 0.5
        if w_l > w_r:
            if w_l > self.Wheel_speed_threshold:
                # 按比例缩放
                w_r = self.Wheel_speed_threshold * w_r / w_l
                w_l = self.Wheel_speed_threshold
        else:
            if w_r > self.Wheel_speed_threshold:
                w_l = self.Wheel_speed_threshold * w_l / w_r
                w_r = self.Wheel_speed_threshold
        print(f'目标跟踪解码得到的左轮轮速===[{w_l}],右轮轮速===[{w_r}]')

        return w_l, w_r


    def decode_object_tracking_same_side_history_gap(self, left, right, min_dis_left, min_dis_right):
        """
        细胞的spike信号转化为控制轮速的信号
        """
        num_history_point = 3  # 当前值与历史多少个值进行比较，num_history_point >= 2
        if len(self.left_mea) < num_history_point:
            self.left_mea.append(left)
            self.right_mea.append(right)
            return 0.5, 0.5

        # 计算spike差值
        gap_left = left - np.array(self.left_mea).mean()  # 可能为负值
        gap_right = right - np.array(self.right_mea).mean()
        if gap_left < -30:
            gap_left = -30
            # print(f'gapleft<0 = {gap_left}--self.left_mea = {self.left_mea}--leftnow={left}')
        if gap_right < -30:
            gap_right = -30
            # print(f'gapleft<0 = {gap_right}--self.left_mea = {self.right_mea}--rightnow = {right}')
        # 自适应归一化，max_value可能需要根据机器人运动进行修改
        max_value = 20
        # w_l = (gap_left / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
        # w_r = (gap_right / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
        w_l = (gap_left / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()
        w_r = (gap_right / max_value) * (self.dsp_wv_max.value() - self.dsp_wv_min.value()) + self.dsp_wv_min.value()

        if left >= self.spb_fre_max.value():
            w_l = self.dsp_wv_max.value()
        if right >= self.spb_fre_max.value():
            w_r = self.dsp_wv_max.value()

        # 更新当前历史值
        if len(self.left_mea) >= num_history_point:
            self.left_mea.pop(0)
            self.left_mea.append(left)

            self.right_mea.pop(0)
            self.right_mea.append(right)
        # 每个轮子保底 wv_min（原逻辑只在两轮同时为 0 时兜底，会出现单轮停住急转 / 完全停车）。
        # 现在每轮都不低于 wv_min → 车一直有最小前进速度、不会完全停住，差速仍可转向。
        # 注意：wv_min 由 Encode/Decode 参数设定；设为 0 仍允许停车，要保证一直走就设 >0（如 0.2）。
        wv_min = self.dsp_wv_min.value()
        if w_l < wv_min:
            w_l = wv_min
        if w_r < wv_min:
            w_r = wv_min
        # self.Wheel_speed_threshold = 1.0
        if w_l > w_r:
            if w_l > self.Wheel_speed_threshold:
                w_r = self.Wheel_speed_threshold * w_r / w_l
                w_l = self.Wheel_speed_threshold
        else:
            if w_r > self.Wheel_speed_threshold:
                w_l = self.Wheel_speed_threshold * w_l / w_r
                w_r = self.Wheel_speed_threshold
        self.cur_left_whell = w_l
        self.cur_right_whell = w_r

        return w_l, w_r

    def encode_object_grasp(self, left, right, ang_dis_left, ang_dis_right):
        """
        将细胞的响应转化为机械臂转向的控制角度信号, left right输入有一个为0  一个非0，非0的用于控制
        单电极控制：
        双电极控制：
        """

        if left > 0:
            self.grasping_ang_dis = left
        elif right > 0:
            self.grasping_ang_dis = right

        # grasping
        l_n = self.norm_angle_arm(left)
        r_n = self.norm_angle_arm(right)
       
        # print("distance left:", left)
        # print("distance right:", right)

        # print("distance left norm:", l_n)
        # print("distance right norm:", r_n)

        # 刺激频率转化
        l = (self.spb_sti_max.value() - self.spb_sti_min.value()) * l_n + self.spb_sti_min.value()
        r = (self.spb_sti_max.value() - self.spb_sti_min.value()) * r_n + self.spb_sti_min.value()

        # print("freqency left:", l)
        # print("freqency right:", r)
        return l, r

    def norm_angle_arm(self, x):
        out = 0
        max_ = 90
        min_ = 0    # 90°和0°范围内，不考虑方向
        if x < max_:
            out = (x - min_) / (max_ - min_)
        else:
            out = 1.0
        return out


    def decode_object_grasp(self,left, right, min_dis_left, min_dis_right):
        """
        spike： left right，转化为控制机械臂转向的信号
        """

        if len(self.left_mea) < 2:
            self.left_mea.append(left)
            self.right_mea.append(right)
        else:
            self.left_mea[0] = self.left_mea[1]
            self.left_mea[1] = left

            self.right_mea[0] = self.right_mea[1]
            self.right_mea[1] = right

        max_value = 90
        min_value = 0

        max_value_free = 30    # 超参数，需实验调节

        ang_l = (left / max_value_free) * (max_value - min_value) + min_value
        ang_r = (right / max_value_free) * (max_value - min_value) + min_value

        if left >= max_value:
            ang_l = max_value
        if right >= max_value:
            ang_r = max_value

        step = 0.5    # 控制机械臂转动步长
        ang_l = self.grasping_ang_dis - ang_l * step
        ang_r = self.grasping_ang_dis - ang_r * step
        ang_l /=200
        ang_r /=200

        return ang_l, ang_r