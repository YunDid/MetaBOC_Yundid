# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------


import math, copy

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import time
from datetime import datetime
import os

class Robot(object):

    def __init__(self, axis_length=60, wheel_radius=16):  # wheel_radius = 20
        # 机器人初始位置
        # self.start_x = 150    # 50
        # self.start_y = 150    # 50

        self.start_x = 100    # 50
        self.start_y = 450    # 50

        self.type = "VIRTUAL_ROBOT"

        # 机器人当前位置(中心点)
        self.x = self.start_x
        self.y = self.start_y

        self.radius = 20  # 所画中心点
        self.center_velocity = 0

        self.hits = 0     # 小车撞击障碍物次数

        # robot参数
        self.axis_length = axis_length       # robot轴长  40
        self.wheel_radius = wheel_radius     # 车轮半径   10
        self.w_left = 0              # 角速度
        self.w_right = 0

        self.points = []        # 路径上更新点的集合
        self.angles_list = []   # 存储角度，用于小车撞击后，恢复小车位置
        self.points_his = []    # 历史轨迹
        self.tp_points_number = 0     # 用于记录距离上一次回退，走过的点

        self.angles = 0         # robot主轴与水平面的夹角，顺时针为正，单位为弧度（非角度）

        self.run_directions = 1  # 小车前进或后退，值为 1 或 -1

        # 车轮四个位点
        self.wheel_lux = self.start_x + self.wheel_radius
        self.wheel_luy = self.start_y - self.axis_length / 2
        self.wheel_ldx = self.start_x - self.wheel_radius
        self.wheel_ldy = self.start_y - self.axis_length / 2

        self.wheel_rux = self.start_x + self.wheel_radius
        self.wheel_ruy = self.start_y + self.axis_length / 2
        self.wheel_rdx = self.start_x - self.wheel_radius
        self.wheel_rdy = self.start_y + self.axis_length / 2

        self.alpha = math.atan(2 * self.wheel_radius / self.axis_length)
        self.big_length = math.sqrt(self.wheel_radius**2 + (self.axis_length*0.5)**2)  # 斜边


    def get_start_pos(self):
        return self.start_x, self.start_y

    def set_initial_pos(self, x, y):
        self.x = x
        self.y = y

    def update_position(self, new_x, new_y):
        self.x = new_x
        self.y = new_y

    def get_current_pos(self):
        return self.x, self.y

    def back_to_hit_pos(self):
        """
        小车撞击后，小车回退至撞击前的位置
        """
        length = len(self.points)
        index = 50   # 50   200
        max_num_length = self.tp_points_number
        print("max length", max_num_length)
        self.tp_points_number = 0

        if length > index:
            point_x = self.points[-index].x()
            point_y = self.points[-index].y()
            angel = self.angles_list[-index]

            # 计算两点之间的距离，以确定更新位置：
            dis = math.sqrt((point_x - self.x)**2 + (point_y - self.y)**2)

            distance = 200
            if max_num_length:   # 找两遍
                max_dis = 0; max_id = 0; dis_ = 0
                for i in range(1, max_num_length):    # 每三个点判断距离，增加判断速度
                    point_x = self.points[-i].x()
                    point_y = self.points[-i].y()
                    angel = self.angles_list[-i]
                    dis_ = math.sqrt((point_x - self.x)**2 + (point_y - self.y)**2)
                    if max_dis < dis_:
                        max_dis = dis_
                        max_id = -i 
                    if dis_ > distance:
                        break
                if dis_ < distance:    # 当遍历数据小于该距离时，更新为遍历中的最大距离
                    point_x = self.points[max_id].x()
                    point_y = self.points[max_id].y()
                    angel = self.angles_list[max_id]

        if max_num_length < 40:
        # else:    # 数据比较少时，回退到最初的点
            point_x = self.points[0].x()
            point_y = self.points[0].y()
            angel = self.angles_list[0]

        self.angles = angel
        self.change_robot_pos(point_x, point_y, angel)
    
    def change_robot_pos(self, x, y, angles):
        """
        x,y: 中心位置点
        """
        self.x = x
        self.y = y
        self.wheel_lux = self.x + math.cos(math.pi / 2.0 - angles - self.alpha) * self.big_length
        self.wheel_luy = self.y - math.sin(math.pi / 2.0 - angles - self.alpha) * self.big_length
        self.wheel_ldx = self.x + math.cos(math.pi / 2.0 - angles + self.alpha) * self.big_length
        self.wheel_ldy = self.y - math.sin(math.pi / 2.0 - angles + self.alpha) * self.big_length

        self.wheel_rux = self.x - math.cos(math.pi / 2.0 - angles + self.alpha) * self.big_length
        self.wheel_ruy = self.y + math.sin(math.pi / 2.0 - angles + self.alpha) * self.big_length
        self.wheel_rdx = self.x - math.cos(math.pi / 2.0 - angles - self.alpha) * self.big_length
        self.wheel_rdy = self.y + math.sin(math.pi / 2.0 - angles - self.alpha) * self.big_length

    def hits_once_time(self):
        self.hits = self.hits + 1

    # 设置小车运动方向
    def set_robot_direction(self, dire):
        self.run_directions = dire

    # 通过频率计算小车位置,输入为左右轮的轮速
    def update_wheels(self, left, right, up_time):

        if left is None:
            self.w_right = right
        elif right is None:
            self.w_left = left
        else:
            self.w_left = left
            self.w_right = right

        # 计算左右轮速度
        vl = self.w_left * self.wheel_radius
        vr = self.w_right * self.wheel_radius
        self.center_velocity = (vl + vr) / 2.0    # 中间速度

        print("left wheel velocity:", self.w_left)
        print("right wheel velocity:", self.w_right)

        tm = up_time * 0.001    # ms to s

        # 计算偏转角度 & 旋转半径
        if vr - vl or vl - vr:  # 向左边转 或向右边转
            w = (vl - vr) * tm / self.axis_length            # 角速度
            last_angle = self.angles
            self.angles = self.angles + w                    # 更新当前的全局角度

            r = self.center_velocity * self.axis_length / (vl - vr)            # 旋转半径

            self.x = self.x + r * (math.cos(math.pi / 2.0 - last_angle - w) - math.cos(math.pi / 2.0 - last_angle))
            self.y = self.y + r * (math.sin(math.pi / 2.0 - last_angle) -  math.sin(math.pi / 2.0 - last_angle - w))

            self.wheel_lux = self.x + math.cos(math.pi / 2.0 - self.angles - self.alpha) * self.big_length
            self.wheel_luy = self.y - math.sin(math.pi / 2.0 - self.angles - self.alpha) * self.big_length
            self.wheel_ldx = self.x + math.cos(math.pi / 2.0 - self.angles + self.alpha) * self.big_length
            self.wheel_ldy = self.y - math.sin(math.pi / 2.0 - self.angles + self.alpha) * self.big_length

            self.wheel_rux = self.x - math.cos(math.pi / 2.0 - self.angles + self.alpha) * self.big_length
            self.wheel_ruy = self.y + math.sin(math.pi / 2.0 - self.angles + self.alpha) * self.big_length
            self.wheel_rdx = self.x - math.cos(math.pi / 2.0 - self.angles - self.alpha) * self.big_length
            self.wheel_rdy = self.y + math.sin(math.pi / 2.0 - self.angles - self.alpha) * self.big_length

            points = QPoint(self.x, self.y)
            # if points in self.points:
            #     self.points.remove(points)
            self.points.append(points)
            self.angles_list.append(self.angles)
            self.tp_points_number = self.tp_points_number + 1
            # if points not in self.points:
            #     self.points.append(points)
        else:    # 走直线
            w_angle = (self.w_left + self.w_right) / 2.0
            self.x = self.x + self.wheel_radius * math.cos(self.angles) * w_angle * tm * self.run_directions
            self.y = self.y + self.wheel_radius * math.sin(self.angles) * w_angle * tm * self.run_directions
            self.wheel_lux = self.wheel_lux + self.wheel_radius * math.cos(self.angles) * w_angle * tm * self.run_directions
            self.wheel_luy = self.wheel_luy + self.wheel_radius * math.sin(self.angles) * w_angle * tm * self.run_directions
            self.wheel_ldx = self.wheel_ldx + self.wheel_radius * math.cos(self.angles) * w_angle * tm * self.run_directions
            self.wheel_ldy = self.wheel_ldy + self.wheel_radius * math.sin(self.angles) * w_angle * tm * self.run_directions

            self.wheel_rux = self.wheel_rux + self.wheel_radius * math.cos(self.angles) * w_angle * tm * self.run_directions
            self.wheel_ruy = self.wheel_ruy + self.wheel_radius * math.sin(self.angles) * w_angle * tm * self.run_directions
            self.wheel_rdx = self.wheel_rdx + self.wheel_radius * math.cos(self.angles) * w_angle * tm * self.run_directions
            self.wheel_rdy = self.wheel_rdy + self.wheel_radius * math.sin(self.angles) * w_angle * tm * self.run_directions
            points = QPoint(self.x, self.y)
            # if points in self.points:
            #     self.points.remove(points)
            self.points.append(points)
            self.angles_list.append(self.angles)
            self.tp_points_number = self.tp_points_number + 1
            # if points not in self.points:
            #     self.points.append(points)

    def save_points(self):
        # 获取当前时间戳
        timestamp = time.time()
        dt_object = datetime.fromtimestamp(timestamp)
        formatted_date = dt_object.strftime("%Y-%m-%d %H-%M-%S")
        
        # 创建保存路径
        self.save_points_path = os.path.join("./out_points", "points-" + formatted_date[:-9] + "-" + formatted_date[-8:] + ".txt")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(self.save_points_path), exist_ok=True)
        
        # 获取当前时间用于记录
        current_time = time.time()
        timestamp_str = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        # 构建输出字符串
        out_points = timestamp_str + ": "
        
        # 只保存坐标值，不保存 PyQt5.QtCore.QPoint 部分
        for point in self.points:
            out_points += f"({point.x()}, {point.y()}),"
        
        # 移除最后一个逗号并添加换行符
        if out_points.endswith(","):
            out_points = out_points[:-1]
        out_points += "\n"
        
        # 写入文件
        with open(self.save_points_path, "a") as f:
            f.write(out_points)


    def reset_state(self):
        self.x = self.start_x
        self.y = self.start_y

        self.points_his.append(copy.copy(self.points))

        self.points.clear()
        self.points = []
        self.angles_list.clear()
        self.angles_list = []


        self.angles = 0
        self.w_left = 0
        self.w_right = 0
        self.center_velocity = 0
        self.run_directions = 1

        self.wheel_lux = self.start_x + self.wheel_radius
        self.wheel_luy = self.start_y - self.axis_length / 2
        self.wheel_ldx = self.start_x - self.wheel_radius
        self.wheel_ldy = self.start_y - self.axis_length / 2

        self.wheel_rux = self.start_x + self.wheel_radius
        self.wheel_ruy = self.start_y + self.axis_length / 2
        self.wheel_rdx = self.start_x - self.wheel_radius
        self.wheel_rdy = self.start_y + self.axis_length / 2

    def get_angle(self):
        return self.angles

    def set_train_test_mode(self, mode):
        """
        重置参数
        """
        self.points.clear()
        self.angles_list.clear()
        self.angles_list = []
        self.points_his.clear()
        self.points_his = []
        self.points = []
        self.tp_points_number = 0
        self.hits = 0