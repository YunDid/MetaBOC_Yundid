# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------


from cmath import sqrt
from faulthandler import disable
import os
import random
import math
import time
import numpy as np
from scipy.interpolate import interp1d

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from src.robot.task import TASK, MAP


class Tracking(object):

    def __init__(self):
        
        self.point_list = []    # 人为添加的点
        self.path_point_list = []    # 通过插值,计算出来的路径上的点
        self.cur_position_id = 0
        self.indice_interplate = 0   # 用于记录已完成插值的位置点

        self.time_of_distance = []    # 单位时间内的跟踪距离

    def add_point(self, point):
        self.point_list.append(point)

    def clear_point(self):
        self.cur_position_id = 0

        self.point_list.clear()
        self.point_list = []

        self.path_point_list.clear()
        self.path_point_list = []

        self.time_of_distance.clear()
        self.time_of_distance = []


    def get_points_list(self):
        """
        人为添加的点
        """
        return self.point_list

    def get_paths_list(self):
        """
        插值得到的轨迹
        """
        return self.path_point_list
    
    def get_cur_pos_id(self):
        """
        当前的追踪目标物体所在的位置
        """
        return int(self.cur_position_id)

    def update_time_distance(self, dis):
        """
        每次调用距离更新函数时，自动更新距离信息
        """
        if dis > 5000:
            pass
        else:
            current_time = time.time()
            self.time_of_distance.append((current_time, dis))

    def save_tracking_distance_txt(self, save_name):
        directory = os.path.dirname(save_name)
        if directory:
            os.makedirs(directory, exist_ok=True)  # 输出目录首跑可能不存在
        with open(save_name, "a+") as f:
            for t in self.time_of_distance:
                # 手动格式化字符串
                record = f'{{"current_time": {t[0]}, "distance": {t[1]}}}'
                f.write(record + '\n')



    def get_mean_distance(self):
        if len(self.time_of_distance) > 0:
            return np.mean(np.array(self.time_of_distance))
        else:
            return -1

    def cubic_spline_interpolation(self):
        """
        https://stackoverflow.com/questions/52014197/how-to-interpolate-a-2d-curve-in-python
        """
        # interpolations_methods = ['slinear', 'quadratic', 'cubic']
        method = 'cubic'
        interpolations_methods = ['cubic']
        num_points = len(self.point_list)    # 人为添加了多少点
        alpha = np.linspace(0, 1, 25*num_points)

        points = []
        if len(self.point_list) > 3:
            pts = np.array(self.point_list)

            # pts = pts[self.indice_interplate:]
            # self.indice_interplate = len(self.point_list) - 3

            distance = np.cumsum( np.sqrt(np.sum( np.diff(pts, axis=0)**2, axis=1 )) )
            distance = np.insert(distance, 0, 0)/distance[-1]

            interpolated_points = {}
            for method in interpolations_methods:
                interpolator =  interp1d(distance, pts, kind=method, axis=0)
                interpolated_points[method] = interpolator(alpha)

            # if len(self.path_point_list) < 1:
            #     begin = 0
            # else:
            #     begin = 25 * (num_points - 1)

            for i in range(len(interpolated_points[method])):
                pt = QPoint(int(interpolated_points[method][i][0]), int(interpolated_points[method][i][1]))
                points.append(pt)

            self.path_point_list = points
    

    def update_cur_pos(self):
        """
        更新当前的位置
        """
        speed = 0.2

        if self.cur_position_id + speed < len(self.path_point_list) - 1:
            self.cur_position_id = self.cur_position_id + speed
        else:
            self.cur_position_id = len(self.path_point_list) - 1

