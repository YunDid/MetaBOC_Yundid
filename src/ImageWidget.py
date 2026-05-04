# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

# TODO
# 1. 机械譬更新和控制转向的方向性待测试调整 ang_control 正数向x轴方向旋转，顺时针
# 2. 奖惩刺激的施加待补充;
# 3. 连续训练的逻辑待完善，
# 4. 训练/测试指标更新:
# 5. 编码/解码调试;
from operator import index
from tkinter.messagebox import NO
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import math
import numpy as np
import time

from src.Ui_ImageWidget import Ui_ImageWidget
from src.robot.robot import Robot
from src.robot.obstacle_map import ObstacleMap
from src.robot.communication import Communication
from src.robot.real_robot import RealRobot
from src.robot.real_obst_map import RealObstacleMap

from src.robot.task import TASK, MAP, GRASP_STATE, GRASP_CONTROL, MAP_MODE, SYSTEM_DEVICE
from src.robot.object_tracking import Tracking
from src.robot.object_grasping import Grasping

from src.Arm3DPostionWidget import Arm3DPosWidget


gray_color_table = [qRgba(255, 0, 0, 100) for i in range(256)]

gray_color_table[0] = qRgba(0, 255, 255, 100)    # nucleus_white
gray_color_table[1] = qRgba(0, 168, 208, 100)
gray_color_table[2] = qRgba(0, 119, 204, 100)     # nucleus_black
gray_color_table[3] = qRgba(0, 255, 0, 100)     # cortex_white
gray_color_table[4] = qRgba(23, 107, 8, 100)   # cortex_black
gray_color_table[5] = qRgba(255, 255, 0, 100)   # lens_white
gray_color_table[6] = qRgba(255, 153, 0, 100)    # lens_black
gray_color_table[7] = qRgba(255, 0, 255, 100)   # rectangle
gray_color_table[8] = qRgba(255, 0, 102, 100)
gray_color_table[9] = qRgba(204, 0, 153, 100)
gray_color_table[10] = qRgba(153, 255, 51, 100)
gray_color_table[11] = qRgba(102, 255, 102, 100)
gray_color_table[12] = qRgba(51, 153, 102, 100)
gray_color_table[255] = qRgba(255, 0, 255, 100)


class ImageWidget(QWidget, Ui_ImageWidget):

    left_distance = pyqtSignal(float)
    right_distance = pyqtSignal(float)
    velocity = pyqtSignal(float)
    robot_angles = pyqtSignal(float)
    current_pos = pyqtSignal(float, float)
    pos_left = pyqtSignal(float, float)
    pos_right = pyqtSignal(float, float)

    mea_control = pyqtSignal(float, float)
    outof_border = pyqtSignal(bool)
    left_wheel_hit = pyqtSignal(bool)
    right_wheel_hit = pyqtSignal(bool)
    real_robot_connect_server = pyqtSignal(bool)
    zoomChanged = pyqtSignal(float)
    cell_state = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.setupUi(self)
        self.setStyleSheet("background-color: rgb(171, 171, 0);")

        self.setMouseTracking(True)

        # 定义边界
        h = self.minimumHeight()-2; w = self.minimumWidth()

        # 机器人相关
        self.robot = Robot()
        self.obstacle = ObstacleMap(h, w, self.robot)  # x y
        self.map = self.obstacle.generate_map(MAP.Human_Map)

        # 画布相关
        self.pixmap = QPixmap()
        self._painter = QPainter()

        data = self.read("./resources/robot_ruound.png")
        self.image = QImage.fromData(data)
        self.pixmap = QPixmap.fromImage(self.image)

        self.x = self.robot.x  # 当前机器人所在的位置
        self.y = self.robot.y
        self.path_last_x = 0   # 上一个点的坐标
        self.path_last_y = 0
        self.nearest_obstacle_id_left = None  # 最近的障碍物index
        self.nearest_obstacle_id_right = None # 最近的障碍物index
        self.nearest_obstacle_id_mid = None   # 最近的障碍物index

        self.real_obstacle_list = []   # 真实机器人最近的障碍物： 左中右

        self.task = TASK.Obstacle_Avoidance    # 初始的任务
        self.mid_mouse_press_state = False
        self.tracking_point = None    # 用于目标跟踪的动态点
        self.tracking = Tracking()

        self.grasping = Grasping()

        self.arm_data_from_real_robot = None    # 机械臂的角度信息
        # self.control_mode = 0   # 0 人为控制VER，1 MEA控制VER, 2人为控制RER(真实场景)，3MEA控制真实场景
        self.control_mode = MAP_MODE.Virtual  # 虚拟的还是真实的
        self.control_count = 0

        # 移动画布
        self.last_pos = QPoint(0, 0)
        self.current_ord = QPoint(0, 0)

        # for MEA Connection and information communication (ic)
        self.mea_ic = Communication()
        self.mea_ic.set_obstacle_map(self.map)
        self.mea_ic.set_task(self.task)

        self.scale = 1.0

        self.is_use_dynamic_model_mode = False    # 是否使用动力学模型


        self.timer_back = QTimer(self)    # 机器人回退时间
        self.timer_back.timeout.connect(self.update_direction_robot)

        self.arm_3d_widget = None  # 用于可视化机械臂的窗口

        self.human_arm_angle = 0    # 人为控制机械臂的角度
        self.obr = [-1, 0]   # 障碍物的角度和距离
        # 传递机器人的方向 1为左，2为右，0为其它
        self.direction = 0
        self.repaint()

    def close(self):
        self.mea_ic.close()
        del self.mea_ic
        self.mea_ic = None

    # read image
    def read(self, imagePath):
        try:
            with open(imagePath, 'rb') as f:
                return f.read()
        except:
            return None

    def initialPainter(self, painter):
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.scale(self.scale, self.scale)
        # painter.translate(self.offsetToCenter())

    def offsetToCenter(self):
        s = self.scale
        area = super(ImageWidget, self).size()
        w, h = self.minimumHeight() * s, self.minimumWidth() * s
        aw, ah = area.width(), area.height()
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QPointF(x, y)

    def paintEvent(self, event):
        if not self.pixmap:
            return super(ImageWidget, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        self.initialPainter(p)

        p.translate(self.current_ord)
        # if self.task == TASK.Object_Grasping:
        #     p.drawPixmap(0, 0, self.pixmap)  # 画实时图像

        if self.task == TASK.Obstacle_Avoidance:
            # p.setBrush(QBrush(QColor(144, 213, 186, 128)))
            # p.setBrush(QBrush(QColor(80, 80, 80, 64)))
            p.setBrush(QBrush(QColor(140, 140, 140, 64)))
            p.drawRect(self.rect())
        elif self.task == TASK.Object_Tracking:
            # p.setBrush(QBrush(QColor(178, 242, 246, 128)))
            p.setBrush(QBrush(QColor(20, 20, 20, 128)))

            # p.setBrush(QBrush(QColor(255, 255, 255, 255)))
            # p.setBrush(QBrush(QColor(180, 180, 180, 128)))
            p.drawRect(self.rect())
        elif self.task == TASK.Object_Grasping:
            # p.setBrush(QBrush(QColor(116, 180, 251, 128)))
            # p.drawRect(self.rect())
            pass

        # draw obstacle map 障碍物图
        pen = QPen(QColor(0, 0, 0, 255))
        p.setBrush(QBrush(QColor(0, 0, 0, 128)))
        for i in range(len(self.map)):
            temp = self.map[i]
            if temp["show_state"]:
                p.drawEllipse(int(temp['x'] - temp['r']), int(temp['y'] - temp['r']), int(2*temp['r']), int(2*temp['r']))

        # 突出距离robot最近的障碍物
        if self.robot.type != "REAL_ROBOT" and self.task == TASK.Obstacle_Avoidance:    # 虚拟环境状态
            if self.nearest_obstacle_id_left is not None and self.nearest_obstacle_id_left != -1:
                pen = QPen(QColor(255, 0, 0, 255))    # 红色表示左侧
                p.setBrush(QBrush(QColor(255, 0, 0, 128)))
                p.setPen(pen)
                temp = self.map[self.nearest_obstacle_id_left]
                p.drawEllipse(int(temp['x'] - temp['r']), int(temp['y'] - temp['r']), int(2*temp['r']), int(2*temp['r']))
            if self.nearest_obstacle_id_right is not None and self.nearest_obstacle_id_right != -1:
                pen = QPen(QColor(0, 255, 0, 255))    # 绿色表示右侧
                p.setBrush(QBrush(QColor(0, 255, 0, 128)))
                p.setPen(pen)
                temp = self.map[self.nearest_obstacle_id_right]
                p.drawEllipse(int(temp['x'] - temp['r'] + 1), int(temp['y'] - temp['r'] + 1), int(2*temp['r'] - 2), int(2*temp['r'] - 2))
        elif self.robot.type == "REAL_ROBOT":    # 更新真实的物理世界
            if len(self.obstacle.real_obstacles) > 0:
                pen = QPen(QColor(255, 0, 0, 255), 5)    # 绿色表示右侧
                p.setBrush(QBrush(QColor(0, 255, 0, 128)))
                p.setPen(pen)
                p.drawPoints(QPolygon(self.obstacle.real_obstacles))

                if len(self.real_obstacle_list) > 0:   # 突出显示最近的障碍物
                    # p.setBrush(QBrush(QColor(0, 255, 0, 128)))
                    p.setPen(QPen(QColor(0, 255, 255, 255), 10))
                    p.drawPoints(QPolygon([self.obstacle.real_obstacles[self.real_obstacle_list[0]]]))
                    p.drawPoints(QPolygon([self.obstacle.real_obstacles[self.real_obstacle_list[1]]]))
                    p.setPen(QPen(QColor(255, 0, 255, 255), 10))
                    p.drawPoints(QPolygon([self.obstacle.real_obstacles[self.real_obstacle_list[2]]]))


        # 突出robot的初始位置
        # p.end()
        # p.begin(self)
        p.setPen(QPen(QColor(255, 0, 0, 255), 3))
        p.drawLine(int(self.robot.wheel_ldx), int(self.robot.wheel_ldy), int(self.robot.wheel_lux), int(self.robot.wheel_luy))
        p.setPen(QPen(QColor(0, 255, 0, 255), 3))
        p.drawLine(int(self.robot.wheel_rdx), int(self.robot.wheel_rdy), int(self.robot.wheel_rux), int(self.robot.wheel_ruy))
        p.setPen(QPen(QColor(0, 0, 0, 255), 3))
        p.drawLine(int((self.robot.wheel_ldx + self.robot.wheel_lux)/2), int((self.robot.wheel_ldy + self.robot.wheel_luy)/2), \
            int((self.robot.wheel_rdx + self.robot.wheel_rux)/2), int((self.robot.wheel_rdy + self.robot.wheel_ruy)/2))
        p.drawEllipse(int(self.robot.start_x - (self.robot.radius/2)), int(self.robot.start_y-(self.robot.radius/2)), int(self.robot.radius), int(self.robot.radius))

        # 画当前小车位置
        pen = QPen(QColor(255, 0, 255, 255))
        p.setBrush(QBrush(QColor(255, 0, 255, 128)))
        p.setPen(pen)
        p.drawEllipse(int(self.robot.x-(self.robot.radius/2)), int(self.robot.y-(self.robot.radius/2)), int(self.robot.radius), int(self.robot.radius))

        # 画小车路径
        pen_path = QPen(QColor(0, 0, 255, 255), 2)
        p.setPen(pen_path)
        p.drawPolyline(QPolygon(self.robot.points))

        # 画小车历史路径
        for i in range(len(self.robot.points_his)):
            pen_path = QPen(QColor(gray_color_table[i % 14]), 1)
            p.setPen(pen_path)
            p.drawPolyline(QPolygon(self.robot.points_his[i]))

        # 画小车视野
        # pen_car = QPen(QColor(0, 255, 255, 255), 2)
        # p.setPen(pen_car)
        # points = []
        # points.append(QPoint(self.robot.x, self.robot.y))
        # points.append(QPoint(int(self.robot.x + 100), int(self.curve(self.robot.x + 100))))
        # p.drawPolyline(QPolygon(points))

        # 更新Object Tracking的点
        if self.task == TASK.Object_Tracking and self.tracking_point is not None and self.robot.type == "VIRTUAL_ROBOT":
            pen_track_point = QPen(QColor(250, 12, 37), 1)
            p.setPen(pen_track_point)
            p.drawEllipse(int(self.tracking_point.x()-6), int(self.tracking_point.y()-6), 12, 12)

            if len(self.tracking.get_points_list()) > 3:
                paths = self.tracking.get_paths_list()
                cur_id = self.tracking.get_cur_pos_id()
                cur_pos = paths[cur_id]

                # 画人为鼠标点的路线
                pen_track_pt = QPen(QColor(0, 255, 0, 255), 2)
                p.setPen(pen_track_pt)
                p.drawPolyline(QPolygon(paths[:cur_id]))

                # 未走的部分画虚线
                pen_future = QPen(QColor(0, 0, 255, 255), 2, Qt.DashDotLine)
                # pen_future.setDashPattern(paths[id:])
                p.setPen(pen_future)
                p.drawPolyline(QPolygon(paths[cur_id:]))

                # 当前跟踪目标的位置
                pen_track_pt = QPen(QColor(251, 80, 99, 255), 2)
                p.setPen(pen_track_pt)
                p.drawEllipse(int(cur_pos.x()-8), int(cur_pos.y()-8), 16, 16)

                # 画人为添加的控制点
                for mm in self.tracking.get_points_list():
                    p.drawEllipse(int(mm[0]-8), int(mm[1]-8), 16, 16)

        p.end()

    def wheelEvent(self, event):
        delta = event.angleDelta()  #旋转角度
        v_delta = delta.y()
        if v_delta:
            self.zoomChanged.emit(v_delta)
        event.accept()


    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def curve(self, x):
        va = self.robot.angles % (2.0 * math.pi)
        k = math.tan(va)
        b = self.robot.y - k * self.robot.x
        return k * x + b



    def update_position_x(self, x):
        self.path_last_x = x
        self.x = x

    def update_position_y(self, y):
        self.path_last_y = y
        self.y = y

    def check_obstacle_id_change(self, id_left, id_right):
        left_change = False; right_change = False
        if id_left != self.nearest_obstacle_id_left:
            self.nearest_obstacle_id_left = id_left
            left_change = True
        if id_right != self.nearest_obstacle_id_right:
            self.nearest_obstacle_id_right = id_right
            right_change = True
        return left_change, right_change

    def compute_distance(self, is_human):
        if self.task == TASK.Object_Tracking:
            paths = self.tracking.get_paths_list()
            cur_id = self.tracking.get_cur_pos_id()
            cur_pos = paths[cur_id]
            self.obstacle.update_tracking_pos(cur_pos)   # 将当前要跟踪的目标位置，作为障碍物更新到障碍物图


        min_dis_left, nearest_obstacle_id_left = self.obstacle.compute_distance(self.robot.wheel_lux, self.robot.wheel_luy, self.robot.x, self.robot.y, self.robot.angles)
        min_dis_right, nearest_obstacle_id_right = self.obstacle.compute_distance(self.robot.wheel_rux, self.robot.wheel_ruy, self.robot.x, self.robot.y, self.robot.angles)
        min_dis_mid, self.nearest_obstacle_id_mid = self.obstacle.compute_distance((self.robot.wheel_rux + self.robot.wheel_lux) / 2, (self.robot.wheel_ruy + self.robot.wheel_luy) / 2, self.robot.x, self.robot.y, self.robot.angles)

        left_id_change, right_id_change = self.check_obstacle_id_change(nearest_obstacle_id_left, nearest_obstacle_id_right)

        ang_dis_left = self.obstacle.compute_angle_distance(self.nearest_obstacle_id_left, "left")
        ang_dis_right = self.obstacle.compute_angle_distance(self.nearest_obstacle_id_right, "right")


        # 目标跟踪模式，自动更新距离
        if self.task == TASK.Object_Tracking:
            self.tracking.update_time_distance((min_dis_left + min_dis_right + min_dis_mid) / 3.0)


        # 判断是否撞击
        left_hit = False; right_hit = False
        if self.nearest_obstacle_id_left != -1:
            if min_dis_left <= 3:
                left_hit = True
                # self.obstacle.hide_obstacle(self.nearest_obstacle_id_left)  # 撞击到障碍物，隐藏障碍物
                self.left_wheel_hit.emit(True)    # 状态栏显示撞击信息
        if self.nearest_obstacle_id_right != -1:
            if min_dis_right <= 3:
                right_hit = True
                # self.obstacle.hide_obstacle(self.nearest_obstacle_id_right)
                
                self.right_wheel_hit.emit(True)


        if self.nearest_obstacle_id_right != -1:
            if min_dis_mid <= 3:
                if min_dis_left < min_dis_right:
                    left_hit = True
                    # self.obstacle.hide_obstacle(self.nearest_obstacle_id_left)  # 撞击到障碍物，隐藏障碍物
                    self.left_wheel_hit.emit(True)    # 状态栏显示撞击信息
                else:
                    right_hit = True
                    # self.obstacle.hide_obstacle(self.nearest_obstacle_id_right)
                    
                    self.right_wheel_hit.emit(True)

        # 判断中间区域是否撞击，如果撞击，则以左侧或右侧撞击进行反馈
        if self.nearest_obstacle_id_mid != -1:
            if min_dis_mid <= 3:
                if min_dis_left < min_dis_right:
                    left_hit = True
                    self.left_wheel_hit.emit(True)    # 状态栏显示撞击信息
                else:
                    right_hit = True
                    self.right_wheel_hit.emit(True)


        if left_hit or right_hit:
            self.robot.hits_once_time()     # 记录撞击次数
            if self.task == TASK.Object_Tracking:
                # 碰撞后小车暂停1秒等待障碍物移动, 但是休息会使得原始信号可视化卡顿，故暂时隐藏
                time.sleep(1)
                pass
            else:
                self.robot.back_to_hit_pos()    # 撞击后，小车回退至撞击前的某个位置

        # 判断是否出界
        state_left = self.obstacle.check_if_out_of_border(self.robot.wheel_lux, self.robot.wheel_luy)
        state_right = self.obstacle.check_if_out_of_border(self.robot.wheel_rux, self.robot.wheel_ruy)
        out_border = False
        if state_left or state_right:
            out_border = True
            self.outof_border.emit(True)
            return

        if is_human == 1:
            self.mea_ic.update_distance(min_dis_left, left_hit, min_dis_right, right_hit, out_border, ang_dis_left, ang_dis_right, left_id_change, right_id_change)    # 更新小车实时检测的距离，转化为刺激输入至MEA

        self.left_distance.emit(min_dis_left)
        self.right_distance.emit(min_dis_right)

        self.velocity.emit(self.robot.center_velocity)
        self.robot_angles.emit(self.robot.angles)
        self.current_pos.emit(self.robot.x, self.robot.y)
        self.pos_left.emit(self.robot.wheel_lux, self.robot.wheel_luy)
        self.pos_right.emit(self.robot.wheel_rux, self.robot.wheel_ruy)

        return left_hit, right_hit, min_dis_left, min_dis_right

    def compute_distance_hide(self, is_human):
        """
        撞击障碍物后，障碍物消失，继续撞击下一个障碍物
        """
        min_dis_left, nearest_obstacle_id_left = self.obstacle.compute_distance(self.robot.wheel_lux, self.robot.wheel_luy, self.robot.x, self.robot.y, self.robot.angles)
        min_dis_right, nearest_obstacle_id_right = self.obstacle.compute_distance(self.robot.wheel_rux, self.robot.wheel_ruy, self.robot.x, self.robot.y, self.robot.angles)
        min_dis_mid, self.nearest_obstacle_id_mid = self.obstacle.compute_distance((self.robot.wheel_rux + self.robot.wheel_lux) / 2, (self.robot.wheel_ruy + self.robot.wheel_luy) / 2, self.robot.x, self.robot.y, self.robot.angles)

        left_id_change, right_id_change = self.check_obstacle_id_change(nearest_obstacle_id_left, nearest_obstacle_id_right)

        ang_dis_left = self.obstacle.compute_angle_distance(self.nearest_obstacle_id_left, "left")
        ang_dis_right = self.obstacle.compute_angle_distance(self.nearest_obstacle_id_right, "right")

        # 判断是否撞击
        left_hit = False; right_hit = False
        if self.nearest_obstacle_id_left != -1:
            if min_dis_left <= 3:
                left_hit = True
                self.obstacle.hide_obstacle(self.nearest_obstacle_id_left)  # 撞击到障碍物，隐藏障碍物
                self.left_wheel_hit.emit(True)    # 状态栏显示撞击信息
        if self.nearest_obstacle_id_right != -1:
            if min_dis_right <= 3:
                right_hit = True
                self.obstacle.hide_obstacle(self.nearest_obstacle_id_right)
                
                self.right_wheel_hit.emit(True)


        if self.nearest_obstacle_id_right != -1:
            if min_dis_mid <= 3:
                if min_dis_left < min_dis_right:
                    left_hit = True
                    self.obstacle.hide_obstacle(self.nearest_obstacle_id_left)  # 撞击到障碍物，隐藏障碍物
                    self.left_wheel_hit.emit(True)    # 状态栏显示撞击信息
                else:
                    right_hit = True
                    self.obstacle.hide_obstacle(self.nearest_obstacle_id_right)
                    
                    self.right_wheel_hit.emit(True) 

        # 判断中间区域是否撞击，如果撞击，则以左侧或右侧撞击进行反馈
        if self.nearest_obstacle_id_mid != -1:
            if min_dis_mid <= 3:
                if min_dis_left < min_dis_right:
                    left_hit = True
                    self.left_wheel_hit.emit(True)    # 状态栏显示撞击信息
                else:
                    right_hit = True
                    self.right_wheel_hit.emit(True)


        if left_hit or right_hit:
            self.robot.hits_once_time()     # 记录撞击次数
            # self.robot.back_to_hit_pos()    # 撞击后，小车回退至撞击前的某个位置

        # 判断是否出界
        state_left = self.obstacle.check_if_out_of_border(self.robot.wheel_lux, self.robot.wheel_luy)
        state_right = self.obstacle.check_if_out_of_border(self.robot.wheel_rux, self.robot.wheel_ruy)
        out_border = False
        if state_left or state_right:
            out_border = True
            self.obstacle.reset_obstacle()
            self.outof_border.emit(True)
            return

        if is_human == 1:
            self.mea_ic.update_distance(min_dis_left, left_hit, min_dis_right, right_hit, out_border, ang_dis_left, ang_dis_right, left_id_change, right_id_change)    # 更新小车实时检测的距离，转化为刺激输入至MEA

        self.left_distance.emit(min_dis_left)
        self.right_distance.emit(min_dis_right)

        self.velocity.emit(self.robot.center_velocity)
        self.robot_angles.emit(self.robot.angles)
        self.current_pos.emit(self.robot.x, self.robot.y)
        self.pos_left.emit(self.robot.wheel_lux, self.robot.wheel_luy)
        self.pos_right.emit(self.robot.wheel_rux, self.robot.wheel_ruy)

        return left_hit, right_hit,  min_dis_left, min_dis_right

    def compute_distance_real(self, is_human):
        """
        真实机器人避障计算
        """
        min_dis, self.real_obstacle_list, left_hit, right_hit, min_dis_left, min_dis_right = self.obstacle.compute_distance()

        # 是否撞击后的操作
        if left_hit:
            self.left_wheel_hit.emit(True)    # 状态栏显示撞击信息
        if right_hit:                
                self.right_wheel_hit.emit(True)

        if left_hit or right_hit:
            self.robot.hits_once_time()     # 记录撞击次数
            # self.robot.back_to_hit_pos()    # 撞击后，小车回退至撞击前的某个位置

        # # 判断是否出界
        # state_left = self.obstacle.check_if_out_of_border(self.robot.wheel_lux, self.robot.wheel_luy)
        # state_right = self.obstacle.check_if_out_of_border(self.robot.wheel_rux, self.robot.wheel_ruy)
        # out_border = False
        # if state_left or state_right:
        #     out_border = True
        #     self.obstacle.reset_obstacle()
        #     self.outof_border.emit(True)
        #     return

        ang_dis_left = 0; ang_dis_right = 0    # 后续需要根据实际获取的角度更新这两个值

        if is_human == 3:
            self.mea_ic.update_distance(min_dis_left, left_hit, min_dis_right, right_hit, False, ang_dis_left, ang_dis_right, False, False)    # 更新小车实时检测的距离，转化为刺激输入至MEA

        self.left_distance.emit(min_dis)
        self.right_distance.emit(min_dis)

        self.velocity.emit(self.robot.center_velocity)
        self.robot_angles.emit(self.robot.angles)
        self.current_pos.emit(self.robot.x, self.robot.y)
        self.pos_left.emit(self.robot.wheel_lux, self.robot.wheel_luy)
        self.pos_right.emit(self.robot.wheel_rux, self.robot.wheel_ruy)

        return left_hit, right_hit,  min_dis_left, min_dis_right


    def compute_distance_real_tracking(self, is_human):
        """
        真实机器人的跟踪tracking计算
        """
        min_dis, self.real_obstacle_list, left_hit, right_hit, min_dis_left, min_dis_right, ang_dis_left, ang_dis_right = self.obstacle.compute_distance_tracking()

        # # 是否撞击后的操作
        # if left_hit:
        #     self.left_wheel_hit.emit(True)    # 状态栏显示撞击信息
        # if right_hit:                
        #         self.right_wheel_hit.emit(True)

        if left_hit or right_hit:
            self.robot.hits_once_time()     # 记录撞击次数
            # self.robot.back_to_hit_pos()    # 撞击后，小车回退至撞击前的某个位置


        if is_human == 3:
            self.mea_ic.update_distance(min_dis_left, left_hit, min_dis_right, right_hit, False, ang_dis_left, ang_dis_right, False, False)    # 更新小车实时检测的距离，转化为刺激输入至MEA

        self.left_distance.emit(min_dis)
        self.right_distance.emit(min_dis)

        self.velocity.emit(self.robot.center_velocity)
        self.robot_angles.emit(self.robot.angles)
        self.current_pos.emit(self.robot.x, self.robot.y)
        self.pos_left.emit(self.robot.wheel_lux, self.robot.wheel_luy)
        self.pos_right.emit(self.robot.wheel_rux, self.robot.wheel_ruy)

        return left_hit, right_hit,  min_dis_left, min_dis_right


    # 人为控制改变 轮速
    def update_wheel_fre_left(self, left, update_time):
        self.robot.update_wheels(left, right=None, up_time=update_time)
        # self.compute_distance()
        self.update()

    def update_wheel_fre_right(self, right, update_time):
        self.robot.update_wheels(left=None, right=right, up_time=update_time)
        # self.compute_distance()
        self.update()

    # robot 在自由运动状态下，不断更新小车位置
    def update_robot(self, left_fre, right_fre, update_time, is_human):
        """
        # is_human: 0为人为控制
        # is_human: 1为mea控制虚拟环境
        # is_human: 2为人为控制真实环境
        # is_human: 3为mea控制真实环境
        """
        if is_human == 1:
            self.mea_ic.en_de_code.Wheel_speed_threshold = 5.0
        elif is_human == 3:
            self.mea_ic.en_de_code.Wheel_speed_threshold = 1.0

        if self.task == TASK.Obstacle_Avoidance:
            if is_human != 2 and is_human != 3:  # 虚拟环境
                left_hit, right_hit,  min_dis_left, min_dis_right = self.compute_distance(is_human)    # 撞击后回退
                # left_hit, right_hit = self.compute_distance_hide(is_human) # 撞击后继续撞击下一个障碍物
            

            if is_human == 0:
                self.robot.update_wheels(left=left_fre, right=right_fre, up_time=update_time)
            elif is_human == 1:
                left, right, l_spike, r_spike = self.mea_ic.get_mea_control_infor(left_hit, right_hit, min_dis_left, min_dis_right)
                # self.mea_control.emit(left, right)
                if l_spike < 1 and r_spike < 1:    # 判断是否放入细胞，如果未放入，速度置零
                    self.cell_state.emit(False)
                    left = 0; right = 0
                else:
                    self.cell_state.emit(True)

                self.robot.update_wheels(left=left, right=right, up_time=update_time)
            elif is_human == 2:
                obs_mess, robot_mess, scale = self.robot.update_wheels()
                self.obstacle.update_real_obstacle(obs_mess, robot_mess, scale)
                left_hit, right_hit,  min_dis_left, min_dis_right = self.compute_distance_real(is_human)
                self.robot.send_control_mess(left=left_fre, right=right_fre)

            elif is_human == 3:
                obs_mess, robot_mess, scale = self.robot.update_wheels()    # 获取轮速和输出轮速
                self.obstacle.update_real_obstacle(obs_mess, robot_mess, scale)    # 更新地图
                left_hit, right_hit, min_dis_left, min_dis_right = self.compute_distance_real(is_human)  # 计算障碍物距离，并传入细胞

                if left_hit or right_hit:
                    self.robot.set_robot_direction(-1)    # 改变机器人运动方向
                    self.timer_back.start(1500)    # 开启计时器，1.5s后，改变方向
                else:
                    # self.robot.set_robot_direction(1)     # 恢复机器人运动方向
                    pass

                left, right, l_spike, r_spike = self.mea_ic.get_mea_control_infor(left_hit, right_hit, min_dis_left, min_dis_right)  # 获取细胞的输出，控制信号
                # self.mea_control.emit(left, right)
                if l_spike < 1 and r_spike < 1:    # 判断是否放入细胞，如果未放入，速度置零
                    self.cell_state.emit(False)
                    left = 0; right = 0
                else:
                    self.cell_state.emit(True)
                

                self.robot.send_control_mess(left, right)

        elif self.task == TASK.Object_Tracking:
            # 虚拟环境，需先判断是否有目标，再更新机器人位置
            if self.robot.type == "VIRTUAL_ROBOT":                
                if (self.tracking_point is None) or (len(self.tracking.point_list) <= 4):
                    return

            if is_human != 2 and is_human != 3:  # 虚拟环境
                left_hit, right_hit,  min_dis_left, min_dis_right = self.compute_distance(is_human)    # 撞击后回退
            # print(f'ishuman={is_human},left={min_dis_left},right={min_dis_right}')

            if self.robot.type == "VIRTUAL_ROBOT" and is_human == 0:  # 虚拟环境，更新位置，人工控制
                self.tracking.update_cur_pos()
                self.robot.update_wheels(left=left_fre, right=right_fre, up_time=update_time)
            elif self.robot.type == "VIRTUAL_ROBOT" and is_human == 1:  # 虚拟环境，更新位置, MEA控制
                self.tracking.update_cur_pos()
                left, right, l_spike, r_spike = self.mea_ic.get_mea_control_infor(left_hit, right_hit,  min_dis_left, min_dis_right)

                if l_spike < 1 and r_spike < 1:    # 判断是否放入细胞，如果未放入，速度置零
                    self.cell_state.emit(False)
                    left = 0; right = 0
                else:
                    self.cell_state.emit(True)

                self.robot.update_wheels(left=left, right=right, up_time=update_time)


            elif self.robot.type == "REAL_ROBOT" and is_human == 2:    # 人为控制真实的机器人
                obs_mess, robot_mess, scale = self.robot.update_wheels()
                self.obstacle.update_real_obstacle(obs_mess, robot_mess, scale)
                left_hit, right_hit,  min_dis_left, min_dis_right = self.compute_distance_real_tracking(is_human)
                self.robot.send_control_mess(left=left_fre, right=right_fre)
            elif self.robot.type == "REAL_ROBOT" and is_human == 3: # MEA控制
                obs_mess, robot_mess, scale = self.robot.update_wheels()    # 获取轮速和输出轮速
                self.obstacle.update_real_obstacle(obs_mess, robot_mess, scale)    # 更新地图,待跟踪的目标
                left_hit, right_hit, min_dis_left, min_dis_right = self.compute_distance_real_tracking(is_human)  # 计算目标物体的距离和角度，并传入细胞            

                if left_hit or right_hit:    # 撞击后，自动回退
                    self.robot.set_robot_direction(-1)    # 改变机器人运动方向
                else:
                    self.robot.set_robot_direction(1)     # 恢复机器人运动方向

                left, right, l_spike, r_spike = self.mea_ic.get_mea_control_infor(left_hit, right_hit,  min_dis_left, min_dis_right)  # 获取细胞的输出，控制信号
                # self.mea_control.emit(left, right)
                if l_spike < 1 and r_spike < 1:    # 判断是否放入细胞，如果未放入，速度置零
                    self.cell_state.emit(False)
                    left = 0; right = 0
                else:
                    self.cell_state.emit(True)
                
                self.robot.send_control_mess(left, right)

        elif self.task == TASK.Object_Grasping:
            if is_human != 2 and is_human != 3:  # 虚拟环境
                left_hit, right_hit,  min_dis_left, min_dis_right = self.compute_distance(is_human)

            if self.robot.type == "VIRTUAL_ROBOT" and is_human == 0:  # 虚拟环境，更新位置，人工控制
                ang_dis = self.arm_3d_widget.compute_angle_dis()
                # 相差角度小于 10 判断抓取成功，需要进入下一次抓取流程
                if abs(ang_dis) < 10 :
                    self.grasping.grap_times += 1
                #     因为抓取成功，所以随机初始化位置
                    self.arm_3d_widget.init_glview_random()
                else:

#                     if ang_dis > 0:
#                         min_dis_left = abs(ang_dis)
#                         min_dis_right = 0

#                     min_dis_right = ang_dis
#                     min_dis_left = 0
#                 self.mea_ic.update_distance(None, None, None, None, False, min_dis_left, min_dis_right, False,
#                                             False)  # 更新角度信息，转化为刺激输入至MEA

#                 angle_conrol_left, angle_conrol_right, l_spike, r_spike = self.mea_ic.get_mea_control_infor(left_hit=None, right_hit=None,  
#                                                                                 min_dis_left=ang_dis, min_dis_right=None)

#                 if l_spike < 1 and r_spike < 1:    # 判断是否放入细胞，如果未放入，速度置零
#                     self.cell_state.emit(False)
#                     left = 0; right = 0
#                 else:
#                     self.cell_state.emit(True)

#                 TWO_ELE_CONTROL = True
#                 if TWO_ELE_CONTROL:
#                     if ang_dis < 0:
#                         ang = -angle_conrol_right
#                         self.direction = 2
#                     else:
#                         ang = angle_control_left
#                         self.direction = 1
#                     self.grasping.adjustment_times += 1
#                 else:  # 单电极，使用左侧电极控制
#                     ang = angle_control_left
#                     if limited_angle < 0:
#                         ang = -angle_control_left

                    # else:
                    #     min_dis_right = ang_dis
                    #     min_dis_left = 0

                # else:
                    # min_dis_right = ang_dis
                    # min_dis_left = 0

                    # TODO 待真实刺激 奖惩 检验
                    # self.mea_ic.update_distance(min_dis_left, None, min_dis_right, None, False, None, None, False,
                    #                             False,ang_dis )  # 更新角度信息，转化为刺激输入至MEA

                    # angle_conrol_left, angle_conrol_right, l_spike, r_spike = self.mea_ic.get_mea_control_infor(left_hit=None, right_hit=None,
                    #                                                                 min_dis_left=ang_dis, min_dis_right=None)

                    # if l_spike < 1 and r_spike < 1:    # 判断是否放入细胞，如果未放入，速度置零
                    #     self.cell_state.emit(False)
                    #     left = 0
                    #     right = 0
                    # else:
                    #     self.cell_state.emit(True)
                    # TODO angle_control_right 先假设一个值
                    # print(f'ang_dis = {ang_dis}')
                    ang_dis = 20
                    angle_conrol_right = 10
                    angle_control_left = -10
                    TWO_ELE_CONTROL = True
                    if TWO_ELE_CONTROL:
                        if ang_dis < 0:
                            ang = angle_conrol_right
                            self.direction = 1
                        else:
                            ang = angle_control_left
                            self.direction = 2
                        self.grasping.adjustment_times += 1
                    # else:  # 单电极，使用左侧电极控制
                    #     ang = angle_control_left
                    #     if limited_angle < 0:
                    #         ang = -angle_control_left
                    #     else:
                    #         ang = angle_control_left
                    #     self.grasping.adjustment_times += 1
                    # 更新机械臂运动
                    # print(f'arm_global_arm ==== {self.arm_3d_widget.global_angle_arm }')
                    # 存在问题，在调用时缺少真实的arm_data数值
                    self.arm_3d_widget.update_arm_angle_virtual(ang, self.direction)

                # 判断当前调整是否满足条件，满足则进行下一次训练；不满足则继续; 训练和调整次数的计数，保存实时的时间
                # if True:
                #     self.arm_3d_widget.init_glview_random()


            elif self.robot.type == "VIRTUAL_ROBOT" and is_human == 1:  # 虚拟环境，更新位置, MEA控制
                ang_dis = self.arm_3d_widget.compute_angle_dis()
                # 相差角度小于 10 判断抓取成功，需要进入下一次抓取流程
                if abs(ang_dis) < 10 :
                    self.grasping.grap_times += 1
                    self.grasping.angle_sum +=abs(ang_dis)
                #     因为抓取成功，所以随机初始化位置
                    self.arm_3d_widget.init_glview_random()
                else:
                    self.grasping.angle_sum +=abs(ang_dis)
                    if ang_dis > 0:
                        min_dis_left = abs(ang_dis)
                        min_dis_right = 0
                    else:
                        min_dis_right = ang_dis
                        min_dis_left = 0
                    self.mea_ic.update_distance(min_dis_left, None, min_dis_right, None, False, None, None, False,
                                                False,ang_dis )  # 更新角度信息，转化为刺激输入至MEA

                    angle_control_left, angle_conrol_right, l_spike, r_spike = self.mea_ic.get_mea_control_infor(left_hit=None, right_hit=None,
                                                                                    min_dis_left=ang_dis, min_dis_right=None)
                    angle_conrol_right=angle_conrol_right/1000
                    angle_control_left= angle_control_left/1000
                    if l_spike < 1 and r_spike < 1:    # 判断是否放入细胞，如果未放入，速度置零
                        self.cell_state.emit(False)
                        left = 0; right = 0
                    else:
                        self.cell_state.emit(True)
                    TWO_ELE_CONTROL = True
                    if TWO_ELE_CONTROL:
                        if ang_dis < 0:
                            ang = angle_conrol_right
                            self.direction = 2
                        else:
                            ang = -angle_control_left
                            self.direction = 1
                        self.grasping.adjustment_times += 1
                    else:  # 单电极，使用左侧电极控制
                        ang = angle_control_left
                        if ang_dis < 0:
                            ang = -angle_control_left
                    self.grasping.adjustment_times += 1
                    print(f'ang======{ang}')
                    self.arm_3d_widget.update_arm_angle_virtual(ang, self.direction)

            elif self.robot.type == "REAL_ROBOT" and is_human == 2:    # 人为控制真实的机器人
                self.arm_data_from_real_robot, robot_mess = self.robot.update_arm_from_real_robot()  # 暂时，接收两次信息

                print('==================================')
                arm_angle = self.get_arm_angle(self.arm_data_from_real_robot)
                if arm_angle is None:
                    print("Arm angle data is None!...")
                    return
                try:
                    if self.grasping.clamp == False:
                        obs_tar = robot_mess["obs"][0]  # 获取角度单位：角度，非弧度；目标暂时以第一个点为准, TODO 筛选目标点
                    else:
                        obs_tar =(0, 10)
                except:
                    obs_tar = None
                # 距离，角度
                if obs_tar is None:
                    self.obr = [0, 10]
                else:
                    self.obr = [obs_tar[0], obs_tar[1]]
                if  obs_tar ==(0, 10):
                    dis_angle = 170
                else:
                    dis_angle = obs_tar[1] - arm_angle
                print(f'obs_tar == {obs_tar}')


                if self.grasping.clamp != True:  # 如果还没抓取障碍物，则判断障碍物角度
                    limited_angle = dis_angle
                else:  # 已经抓取到障碍物
                    # 此时说明虽然进入了后续的抓取步骤，但实际上obs_tar还在，即没有抓到目标物
                    if self.grasping.cur_step_id > 6 and obs_tar[0] != 0:
                        limited_angle = dis_angle
                        self.grasping.cur_step_id = 0
                        self.grasping.clamp = False
                        # 设置抓取模式
                        self.grasping.update_cur_control_mode(GRASP_CONTROL.Freedom)
                    else:
                        # 抓到
                        if self.grasping.cur_step_id == 5:
                            limited_angle = dis_angle
                            # if self.grasping.clamp == True:
                                # self.grasping.grap_times += 1
                        limited_angle = 0
                if (
                        abs(limited_angle) >= 10 and abs(
                    limited_angle) < 352) and self.grasping.clamp != True:  # 如果目标物体与机械臂角度大于阈值（10°），则神经元控制调整位置
                    self.grasping.update_cur_control_mode(GRASP_CONTROL.Freedom)  # 更新抓取任务状态: 方向调整，通过机械臂转向实现
                    # self.obstacle.update_real_obstacle(obs_mess, robot_mess, scale)  # 更新目标物体的位置
                    print(f'调整模态')  # 如果目标物体与机械臂角度大于阈值（10°），则神经元控制调整位置
                    # self.grasping.free_count += 1
                    self.grasping.update_cur_control_mode(GRASP_CONTROL.Freedom)  # 更新抓取任务状态: 方向调整，通过机械臂转向实现
                    smoothed_angle = self.grasping.update_angle(limited_angle)
                    if smoothed_angle < 10:
                        free_signal = self.grasping.control_sig
                    else:
                        # self.obstacle.update_real_obstacle(obs_mess, robot_mess, scale)  # 更新目标物体的位置
                        free_signal = self.grasping.get_control_signal(self.human_arm_angle)
                    self.grasping.adjustment_times += 1
                    self.robot.tcp_send_arm_control(free_signal)  # 传输控制信号
                    return
                else:
                    self.grasping.angle_sum += limited_angle
                    # print(f'抓取模态')
                    self.grasping.clamp = True
                    cur_armang = arm_angle
                    # print(f'cur_arm_ang=={cur_armang}')
                    self.grasping.update_cur_control_mode(GRASP_CONTROL.Mode)  # 抓取状态
                curtime = time.time()
                if curtime - self.grasping.last_command_time > 5.0:

                    grasping_sig = self.grasping.get_control_signal(cur_armang)
                    print(f'send controlsig == {grasping_sig}')
                    if self.grasping.cur_step_id == 7:
                        self.robot.tcp_send_arm_control(grasping_sig)
                        time.sleep(5)
                    else:
                        self.robot.tcp_send_arm_control(grasping_sig)

                    self.control_count = self.grasping.cur_step_id
                    self.grasping.last_command_time = curtime
                else:
                    pass

            elif self.robot.type == "REAL_ROBOT" and is_human == 3:    # MEA控制真实的机器人
                self.arm_data_from_real_robot, robot_mess = self.robot.update_arm_from_real_robot()  # 暂时，接收两次信息
                print('==================================')
                arm_angle = self.get_arm_angle(self.arm_data_from_real_robot)
                if arm_angle is None:
                    print("Arm angle data is None!...")
                    return
                try:
                    if self.grasping.clamp == False:
                        obs_tar = robot_mess["obs"][0]  # 获取角度单位：角度，非弧度；目标暂时以第一个点为准, TODO 筛选目标点
                    else:
                        obs_tar = (0, 10)
                except:
                    obs_tar = None
                # 距离，角度
                if obs_tar is None:
                    self.obr = (0, 10)
                else:
                    self.obr = [obs_tar[0], obs_tar[1]]
                if obs_tar == (0, 10):
                    dis_angle = 170
                else:
                    dis_angle = obs_tar[1] - arm_angle
                # print(f'obs_tar == {obs_tar}')

                if self.grasping.clamp != True:  # 如果还没抓取障碍物，则判断障碍物角度
                    limited_angle = dis_angle
                else:  # 已经抓取到障碍物
                    # 此时说明虽然进入了后续的抓取步骤，但实际上obs_tar还在，即没有抓到目标物
                    if self.grasping.cur_step_id > 5 and obs_tar[0] != 0:
                        limited_angle = dis_angle
                        limited_angle = 0
                        self.grasping.cur_step_id = 0
                        self.grasping.clamp = False
                        # 设置抓取模式
                        self.grasping.update_cur_control_mode(GRASP_CONTROL.Freedom)
                    else:
                        # 抓到
                        if self.grasping.cur_step_id == 5:
                            limited_angle = dis_angle
                        # self.grasping.grap_times += 1
                        limited_angle = 0
                if (
                        abs(limited_angle) >= 40 and abs(
                    limited_angle) < 352) and self.grasping.clamp != True:  # 如果目标物体与机械臂角度大于阈值（10°），则神经元控制调整位置
                    self.grasping.update_cur_control_mode(GRASP_CONTROL.Freedom)  # 更新抓取任务状态: 方向调整，通过机械臂转向实现
                    # self.obstacle.update_real_obstacle(obs_mess, robot_mess, scale)  # 更新目标物体的位置
                    print(f'调整模态')
                    if limited_angle > 0:
                        min_dis_left = abs(limited_angle)
                        min_dis_right = 0
                    else:
                        min_dis_right = limited_angle
                        min_dis_left = 0
                    self.mea_ic.update_distance(min_dis_left, None, min_dis_right, None, False, None, None, False,
                                                False)  # 更新角度信息，转化为刺激输入至MEA

                    # MEA控制，通过输入angle，从MEA获得控制信号
                    angle_control_left, angle_conrol_right, l_spike, r_spike = self.mea_ic.get_mea_control_infor(None,
                                                                                                                 None,
                                                                                                                 limited_angle,
                                                                                                                 None)  # 获取细胞的输出，控制信号
                    TWO_ELE_CONTROL = True
                    if TWO_ELE_CONTROL:
                        if limited_angle < 0:
                            ang = -angle_conrol_right
                            self.direction = 2
                            print(f'limited_angle == {limited_angle}|| angle_conrol_right={angle_conrol_right}')
                        else:
                            ang = angle_control_left
                            self.direction = 1
                            print(f'limited_angle == {limited_angle}|| -angle_control_left={-angle_control_left}')
                        self.grasping.adjustment_times += 1
                    else:  # 单电极，使用左侧电极控制
                        ang = angle_control_left
                        if limited_angle < 0:
                            ang = -angle_control_left
                        else:
                            ang = angle_control_left
                        self.grasping.adjustment_times += 1
                    # 获取发送信号
                    free_signal = self.grasping.get_control_signal(ang)
                    self.robot.tcp_send_arm_control(free_signal)  # 传输控制信号
                    time.sleep(0.5)
                    return
                else:
                    self.grasping.angle_sum += limited_angle
                    # 进入抓取流程后不给刺激 这里是为了保持实时输出刺激和spike检测
                    min_dis_right = 0
                    min_dis_left = 0
                    self.mea_ic.update_distance(min_dis_left, None, min_dis_right, None, False, None, None, False,
                                                False)  # 更新角度信息，转化为刺激输入至MEA

                    # MEA控制，通过输入angle，从MEA获得控制信号
                    angle_control_left, angle_conrol_right, l_spike, r_spike = self.mea_ic.get_mea_control_infor(None,
                                                                                                                 None,
                                                                                                                 0,
                                                                                                                 None)  # 获取细胞的输出，控制信号
                    # print(f'抓取模态')
                    self.grasping.clamp = True
                    cur_armang = arm_angle
                    # print(f'cur_arm_ang=={cur_armang}')
                    self.grasping.update_cur_control_mode(GRASP_CONTROL.Mode)  # 抓取状态
                curtime = time.time()
                if curtime - self.grasping.last_command_time > 5.0:
                    grasping_sig = self.grasping.get_control_signal(cur_armang)
                    print(f'send controlsig == {grasping_sig}')
                    if self.grasping.cur_step_id == 7:
                        self.robot.tcp_send_arm_control(grasping_sig)
                        # 第七步需要time.sleep不然会执行过快放下障碍物不稳定
                        time.sleep(5)
                    else:
                        self.robot.tcp_send_arm_control(grasping_sig)
                    self.control_count = self.grasping.cur_step_id
                    self.grasping.last_command_time = curtime
                else:
                    pass
            self.update()

        # if left_hit or right_hit:
        #     if is_human == 0:
        #         time.sleep(2)    # 神经元控制，撞击后休息2s钟

    def update_direction_robot(self):
        self.timer_back.stop()
        self.robot.set_robot_direction(1)


    def get_arm_angle(self, arm_data_from_real_robot):
        if arm_data_from_real_robot is not None:
            arm_angle = arm_data_from_real_robot.split(":")
            return float(arm_angle[0][1:])
        else:
            return None

    def get_arm_data_from_real_robot(self):
        return self.arm_data_from_real_robot, self.control_count, self.obr, self.direction

    def reset_state(self):
        self.robot.reset_state()
        self.obstacle.reset_obstacle()

        self.nearest_obstacle_id_left = None
        self.nearest_obstacle_id_right = None
        self.update()
    
    def mouseMoveEvent(self, event):
        pos = event.pos()

        # 显示鼠标距离图像原点位置，默认左上角为原点位置
        self.parent().window().labelCoordinates.setText(
            'x pos: %d; y pos: %d' % (pos.x(), pos.y()) + "  ")

        if event.buttons() and Qt.MiddleButton and self.mid_mouse_press_state:
            self.current_ord = self.current_ord - self.last_pos + event.pos()
            self.last_pos = event.pos()
            event.accept()
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.last_pos = event.pos()
            self.mid_mouse_press_state = True
        elif event.button() == Qt.LeftButton:
            if self.task == TASK.Object_Tracking and self.robot.type == "VIRTUAL_ROBOT":
                self.tracking_point = event.pos()
                self.tracking.add_point((self.tracking_point.x(), self.tracking_point.y()))

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.mid_mouse_press_state = False
        elif event.button() == Qt.LeftButton:
            if self.task == TASK.Object_Tracking and self.robot.type == "VIRTUAL_ROBOT":
                # 添加点后,需要更新规划路线
                self.tracking.cubic_spline_interpolation()
                self.update()


    # for real robot
    def initial_real_robot(self, ip, port):
        # 切换为对真实机器人的控制
        print("Real robot mode start!...")

        # 定义边界
        h = self.minimumHeight()-2; w = self.minimumWidth()

        self.robot = RealRobot(ip, port)
        state = self.robot.connect_robot()
        if state:
            self.real_robot_connect_server.emit(True)
        else:
            self.real_robot_connect_server.emit(False)

        self.control_mode = MAP_MODE.Real

        self.obstacle = RealObstacleMap(h, w, self.robot)  # x 
        self.map = self.obstacle.generate_map(MAP.Empty_Map)
        self.mea_ic.set_obstacle_map(self.map)

        self.repaint()

    def initial_human(self):
        
        # 切换为对虚拟环境
        print("Virtual robot mode start!...")

        # 定义边界
        h = self.minimumHeight()-2; w = self.minimumWidth()

        if self.robot.type == "REAL_ROBOT":
            self.robot.stop_robot()    # 先停止机器人的运动，再断开连接
            self.robot.disconnect()
            self.real_robot_connect_server.emit(False)


        self.control_mode = MAP_MODE.Virtual
        self.robot = Robot()
        self.obstacle = ObstacleMap(h, w, self.robot)  # x y
        if self.task == TASK.Obstacle_Avoidance:
            self.map = self.obstacle.generate_map(MAP.Human_Map)
        elif self.task == TASK.Object_Tracking:
            self.map = self.obstacle.generate_map(MAP.Empty_Map)
        
        self.mea_ic.set_obstacle_map(self.map)

        self.repaint()

    def update_ip_port(self, ip, port):
        if self.robot.type == "REAL_ROBOT":
            self.robot.set_ip_port(ip, port)
            state = self.robot.connect_robot()
            if state:
                self.real_robot_connect_server.emit(True)
            else:
                self.real_robot_connect_server.emit(False)

    def stop_robot_move(self):
        if self.robot.type == "REAL_ROBOT":
            self.robot.stop_robot()

    def set_train_test_mode(self, mode):
        self.mea_ic.set_training_mode(mode)
        self.robot.set_train_test_mode(mode)
        if self.task == TASK.Object_Tracking:
            self.tracking.clear_point()

        if mode: # train
            if self.task == TASK.Object_Grasping:
                self.arm_3d_widget.init_glview_random()
        else:
            if self.task == TASK.Object_Grasping:
                self.arm_3d_widget.init_glview()

        self.update()

    # for task mode change
    def set_task_mode(self, mode):
        if mode == TASK.Obstacle_Avoidance:
            self.obstacle.update_render_mode(TASK.Obstacle_Avoidance)
            self.task = TASK.Obstacle_Avoidance
 
        elif mode == TASK.Object_Tracking:
            self.obstacle.update_render_mode(TASK.Object_Tracking)
            self.task = TASK.Object_Tracking

        elif mode == TASK.Object_Grasping:
            self.task = TASK.Object_Grasping
            self.obstacle.update_render_mode(TASK.Object_Grasping)

        self.mea_ic.set_task(self.task)

        self.tracking_point = None
        self.tracking.clear_point()

    # for map mode change
    def set_map_mode(self, mode):
        h = self.minimumHeight()-2; w = self.minimumWidth()

        if self.control_mode == MAP_MODE.Virtual:
            if mode == MAP.Empty_Map:
                self.obstacle = ObstacleMap(h, w, self.robot)  # x y
                self.map = self.obstacle.generate_map(MAP.Empty_Map)
                self.mea_ic.set_obstacle_map(self.map)
            elif mode == MAP.Random_Map:
                self.obstacle = ObstacleMap(h, w, self.robot)  # x y
                self.map = self.obstacle.generate_map(MAP.Random_Map)
                self.mea_ic.set_obstacle_map(self.map)
            elif mode == MAP.Human_Map:
                self.obstacle = ObstacleMap(h, w, self.robot)  # x y
                self.map = self.obstacle.generate_map(MAP.Human_Map)
                self.mea_ic.set_obstacle_map(self.map)
            elif mode == MAP.Regular_Map:
                self.obstacle = ObstacleMap(h, w, self.robot)  # x y
                self.map = self.obstacle.generate_map(MAP.Regular_Map)
                self.mea_ic.set_obstacle_map(self.map)
        elif self.control_mode == MAP_MODE.Real:
            if mode == MAP.Empty_Map:
                self.obstacle = RealObstacleMap(h, w, self.robot)  # x 
                self.map = self.obstacle.generate_map(MAP.Empty_Map)
                self.mea_ic.set_obstacle_map(self.map)
            elif mode == MAP.Random_Map:
                self.obstacle = RealObstacleMap(h, w, self.robot)  # x y
                self.map = self.obstacle.generate_map(MAP.Random_Map)
                self.mea_ic.set_obstacle_map(self.map)
            elif mode == MAP.Human_Map:
                self.obstacle = RealObstacleMap(h, w, self.robot)  # x y
                self.map = self.obstacle.generate_map(MAP.Human_Map)
                self.mea_ic.set_obstacle_map(self.map)
            elif mode == MAP.Regular_Map:
                self.obstacle = RealObstacleMap(h, w, self.robot)  # x y
                self.map = self.obstacle.generate_map(MAP.Regular_Map)
                self.mea_ic.set_obstacle_map(self.map)
        
        self.update()

    def set_dynamic_model(self, state):
        self.is_use_dynamic_model_mode = state
        self.mea_ic.set_dynamic_model_state(state)

    def update_arm_angle(self, angle):
        """
        人为控制机械臂的角度值
        """
        self.human_arm_angle = angle

    def update_system(self, system):
        from src.platform_config import MCS_AVAILABLE
        if system == SYSTEM_DEVICE.MEA2100:
            if not MCS_AVAILABLE:
                print("MCS not available on this platform; ignoring MEA2100 switch.")
                return
            self.mea_ic.update_systems(SYSTEM_DEVICE.MEA2100)
        elif system == SYSTEM_DEVICE.INTAN:
            self.mea_ic.update_systems(SYSTEM_DEVICE.INTAN)
        elif system == SYSTEM_DEVICE.MAXWELL:
            self.mea_ic.update_systems(SYSTEM_DEVICE.MAXWELL)

    def set_intan_data_path(self, data_dir):
        self.mea_ic.set_intan_data_path(data_dir)

    def set_arm_pos_3d_widget(self, wid):
        self.arm_3d_widget = wid