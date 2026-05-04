# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

from cmath import sqrt
from faulthandler import disable
import random
import math

from src.robot.task import TASK, MAP


class ObstacleMap(object):

    def __init__(self, width, height, robot):
        self.width = width    # 地图的宽高
        self.height = height

        self.obstacle_num = 12
        self.obstacles = []

        self.threshod = 200    # 前进方向的区间距离:点到直线的距离

        self.robot = robot

        self.last_obs_index = -1   # 上一个被撞击的障碍物

        # random.seed(66)
    

    def update_render_mode(self, mode):
        if mode == TASK.Obstacle_Avoidance:
            pass
        elif mode == TASK.Object_Tracking:
            pass
        elif mode == TASK.Object_Grasping:
            pass
            

    # 通过障碍物数量控制生成的图
    def generate_map(self, mode):
        min_radius = 30; max_radius = 40     # 12个
        # min_radius = 10; max_radius = 15     # 12个

        # mode = 2   # 0 空地图  1  随机种子点地图, 2 人为定义   3  规律地图
        if mode == MAP.Random_Map:
            # min_radius = 50; max_radius = 70
            #随机设定地图
            for i in range(self.obstacle_num):
                r = random.randint(min_radius, max_radius)      # 半径
                x = random.randint(0 + max_radius, self.height - max_radius)
                y = random.randint(0 + max_radius, self.width - max_radius)
                self.obstacles.append({"x":x, "y":y, "r":r, "show_state":True})

        elif mode == MAP.Human_Map:
            # 人为设定固定地图
            x = [145,  342, 523, 724, 1290, 889, 1271, 751, 296, 812,1024, 380, 636, 551, 127, 784, 58, 342, 1000, 1186]
            y = [325,  112, 159, 255, 120 , 437, 638 , 38 , 597, 611, 796, 394, 466, 759, 809, 840, 596, 864, 160, 344]
            radius = [33, 31, 36, 34, 35, 32, 30, 35, 31, 33, 33, 33, 32, 30, 36, 30, 38, 32, 36, 32]
            # x = [400, 300, 784, 847, 700]
            # y = [325, 600, 111, 458, 656]
            for i in range(len(x)):
                # r = random.randint(min_radius, max_radius)      # 半径
                r = radius[i]
                x_ = x[i]
                y_ = y[i]
                self.obstacles.append({"x":x_, "y":y_, "r":r, "show_state":True})
            # print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^",self.obstacles)
        elif mode == MAP.Regular_Map:
            # 规律网格地图
            x = []; y = []
            for i in range(8):      # 行
                for j in range(7):  # 列
                    if i % 2 == 0:
                        x_ = 110 + j * 200
                    else:
                        x_ = j * 200 + 20
                    y_ = 50 + i * 150
                    x.append(x_)
                    y.append(y_)

            for i in range(len(x)):
                # r = random.randint(min_radius, max_radius)      # 半径
                r = 20
                x_ = x[i]
                y_ = y[i]
                self.obstacles.append({"x":x_, "y":y_, "r":r, "show_state":True})

        return self.obstacles
    

    def update_tracking_pos(self, pos):
        x_ = pos.x()
        y_ = pos.y()
        r = 12
        if len(self.obstacles) > 0:
            self.obstacles.clear()
            self.obstacles = []
        self.obstacles.append({"x":x_, "y":y_, "r":r, "show_state":True})


    def hide_obstacle(self, index):
        if self.last_obs_index != -1:
            self.obstacles[self.last_obs_index]["show_state"] = True
        self.last_obs_index = index
        self.obstacles[index]["show_state"] = False

    def reset_obstacle(self):
        for i in self.obstacles:
            i["show_state"] = True
        self.last_obs_index = -1

    def compute_angle(self, ob_x, ob_y, direction):
        if direction == "left":
            cen_x = self.robot.wheel_lux
            cen_y = self.robot.wheel_luy
        elif direction == "right":
            cen_x = self.robot.wheel_rux
            cen_y = self.robot.wheel_ruy

        OA = [20, 0]; OB = [ob_x - cen_x, ob_y - cen_y]
        cos_angle = 20*(ob_x - cen_x) / (20 * math.sqrt(OB[0]*OB[0] + OB[1]*OB[1]))

        angle = math.acos(cos_angle)

        if ob_y < cen_y:
            angle = -angle

        return angle

    # 计算车子的轴线，从而判断障碍物是在前进方向，还是背后
    def compute_line(self, ob_x, ob_y):
        l_po_x = self.robot.wheel_ldx
        l_po_y = self.robot.wheel_ldy

        r_po_x = self.robot.wheel_rdx
        r_po_y = self.robot.wheel_rdy

        A = r_po_y - l_po_y
        B = r_po_x - l_po_x
        C = -A * l_po_x - B * l_po_y

        cen_x = self.robot.x
        cen_y = self.robot.y

        for_cen_x = self.robot.x + 20
        for_cen_y = self.robot.y

        OA = [20, 0]; OB = [ob_x - cen_x, ob_y - cen_y]
        cos_angle = 20*(ob_x - cen_x) / (20 * math.sqrt(OB[0]*OB[0] + OB[1]*OB[1]))

        angle = math.acos(cos_angle)

        if ob_y < cen_y:
            angle = 2 * math.pi - angle

        return angle

    def compute_distance(self, x, y, cen_x, cen_y, angle):
        """
        x, y: 车轮前方的坐标点
        cen_x, cen_y: 车中心点
        """

        min_dis = 1000000; index = -1
        for i in range(len(self.obstacles)):
            if not self.obstacles[i]["show_state"]:
                continue

            cur_x = self.obstacles[i]["x"]
            cur_y = self.obstacles[i]["y"]
            r = self.obstacles[i]["r"]

            dis_car = self.curve_pos_dis(cen_x, cen_y, angle, cur_x, cur_y)
            ang = self.compute_line(cur_x, cur_y)

            va = self.robot.angles % (2.0 * math.pi)
            dis_angle = ang - va
            dis_angle2 = ang - va - math.pi * 2
            dis_angle3 = ang - va + math.pi * 2

            if dis_car <= self.threshod and (abs(dis_angle) <= math.pi / 2 or 
                    abs(dis_angle2) <=math.pi / 2 or abs(dis_angle3) <= math.pi / 2):
                dis = math.sqrt((cur_x - x)**2 + (cur_y - y)**2) - r
                if dis < min_dis:
                    min_dis = dis
                    index = i

        return min_dis, index

    def compute_angle_distance(self, nearst_id, direction):

        if len(self.obstacles) == 0:
            return 0

        # 障碍物坐标
        cur_x = self.obstacles[nearst_id]["x"]
        cur_y = self.obstacles[nearst_id]["y"]

        ang = self.compute_angle(cur_x, cur_y, direction)     # 障碍物与机器人中心位置的角度
        va = self.robot.angles % (2.0 * math.pi)  # 机器人的全局角度

        # print(direction, "angle:", (ang - va)*180/math.pi)  
        return ang - va



    # 计算某一点距离小车中直线的距离，从而判断障碍物是否在前进方向上
    def curve_pos_dis(self, cen_x, cen_y, angle, x, y):
        va = angle % (2.0 * math.pi)
        k = math.tan(va)
        b = cen_y - k * cen_x
        A = k; B = -1; C = b
        dis = abs(A*x + B*y + C) / math.sqrt(A*A + B*B)

        return dis

    def check_if_out_of_border(self, x, y):
        """
        计算Robot是否已出界
        """
        if x < 2 or x >= self.height or y < 2 or y >= self.width:
            return True
        else:
            return False
