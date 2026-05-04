# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

import math
import time
import random

from src.robot.obstacle_map import ObstacleMap
from src.robot.task import TASK, MAP


from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class RealObstacleMap(ObstacleMap):

    def __init__(self, width, height, robot):
        ObstacleMap.__init__(self, width, height, robot)
        self.obstacle_num = 0

        self.real_obstacles = []
        self.real_obstacles_angle = []

    def reset_obstacle(self):
        for i in self.obstacles:
            i["state"] = True
        self.last_obs_index = -1
        self.real_obstacles.clear()
        self.real_obstacles = []

        self.real_obstacles_angle.clear()
        self.real_obstacles_angle = []

    def get_all_obstacle_pos(self):
        """
        接收雷达传回的障碍物信息，小车位置信息
        """
        mess = self.robot.recive_message()
        return mess
    
    def update_real_obstacle(self, obs_mess, robot_mess, scale):
        """
        根据真实环境中，机器人上的雷达返回的信息，更新虚拟环境中的障碍物信息
        obs_mess: 障碍物信息
        robot_mess: 机器人的当前位置和角度 x y angle
        """
        self.real_obstacles.clear()
        self.real_obstacles = []

        self.real_obstacles_angle.clear()
        self.real_obstacles_angle = []

        x, y, angle = robot_mess[0], robot_mess[1], robot_mess[2]


        t1 = time.time()
        for i in range(len(obs_mess)):
            dis = obs_mess[i][0]
            ang = obs_mess[i][1]

            # g_angle = angle + ang * 0.001
            ang_hudu = ang * math.pi / 180
            g_angle = angle + ang_hudu          # 全局的角度

            obs_x = (x + dis * math.cos(g_angle)* scale)
            obs_y = (y + dis * math.sin(g_angle)* scale)
            # if obs_x < 0:
            #     obs_x = 0
            # if obs_y < 0:
            #     obs_y = 0
            points = QPoint(int(obs_x), int(obs_y))    # 绝对位置坐标
            self.real_obstacles.append(points)
            self.real_obstacles_angle.append(ang)
        print("convert time:", time.time()-t1)


    def generate_map(self, mode):
        return self.obstacles

    def compute_distance(self):
        """
        用于避障计算
        x, y: 车轮前方的坐标点
        cen_x, cen_y: 车中心点
        """

        x, y, angle = self.robot.get_real_robot_mess()
        x_left, y_left = self.robot.wheel_lux, self.robot.wheel_luy
        x_right, y_right = self.robot.wheel_rux, self.robot.wheel_ruy

        min_dis = 1000000; index = -1; ang = 0
        min_dis_left = 1000000; min_dis_right = 100000
        index_left = -1; index_right = -1
        for i in range(len(self.real_obstacles)):
            obs_x = self.real_obstacles[i].x()
            obs_y = self.real_obstacles[i].y()
            dis = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            dis_left = math.sqrt((x_left - obs_x)**2 + (y_left - obs_y)**2)
            dis_right = math.sqrt((x_right - obs_x)**2 + (y_right - obs_y)**2)
            if dis < min_dis:
                min_dis = dis
                index = i
                ang = self.real_obstacles_angle[i]

            if dis_left < min_dis_left:
                min_dis_left = dis_left
                index_left = i
            if dis_right < min_dis_right:
                min_dis_right = dis_right
                index_right = i

        left_hit = False; right_hit = False
        min_distance = 50  # 5cm 即认为撞击到
        if ang < 90 and (min_dis < min_distance or min_dis_left < min_distance):
            left_hit = True

        if ang > 90 and (min_dis < min_distance or min_dis_right < min_distance):
            right_hit = True

        print("Left distance:", min_dis_left)
        print("Right distance:", min_dis_right)
        index_list = [index_left, index, index_right]

        return min_dis, index_list, left_hit, right_hit, min_dis_left, min_dis_right

    def compute_distance_tracking(self):
        """
        x, y: 车轮前方的坐标点
        cen_x, cen_y: 车中心点
        """

        x, y, angle = self.robot.get_real_robot_mess()
        x_left, y_left = self.robot.wheel_lux, self.robot.wheel_luy
        x_right, y_right = self.robot.wheel_rux, self.robot.wheel_ruy

        min_dis = 1000000; index = -1; ang = 0
        min_dis_left = 1000000; min_dis_right = 100000
        index_left = -1; index_right = -1    # 用于查找最近的两个障碍物散点
        angle_left = 0; angle_right = 0
        for i in range(len(self.real_obstacles)):
            obs_x = self.real_obstacles[i].x()
            obs_y = self.real_obstacles[i].y()
            dis = math.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            dis_left = math.sqrt((x_left - obs_x)**2 + (y_left - obs_y)**2)
            dis_right = math.sqrt((x_right - obs_x)**2 + (y_right - obs_y)**2)
            if dis < min_dis:
                min_dis = dis
                index = i
                ang = self.real_obstacles_angle[i]
                ang_hudu = ang * math.pi / 180    # 相对于车轮中心位置的，后续可能需要更新为左右两侧车轮的角度

            if dis_left < min_dis_left:
                min_dis_left = dis_left
                index_left = i
                angle_left = ang_hudu
            if dis_right < min_dis_right:
                min_dis_right = dis_right
                index_right = i
                angle_right = ang_hudu

        left_hit = False; right_hit = False
        min_distance = 50  # 5cm 即认为撞击到
        if ang < 90 and (min_dis < min_distance or min_dis_left < min_distance):
            left_hit = True

        if ang > 90 and (min_dis < min_distance or min_dis_right < min_distance):
            right_hit = True

        print("Left distance:", min_dis_left)
        print("Right distance:", min_dis_right)
        index_list = [index_left, index, index_right]

        return min_dis, index_list, left_hit, right_hit, min_dis_left, min_dis_right, angle_left, angle_right