# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

import math
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from collections import deque
import numpy as np
import random
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import OpenGL.GL as ogl
from torch.onnx.symbolic_opset9 import prim_tolist

from src.Ui_Arm3DPosWidget import Ui_Arm3DPostionWidget

class Arm3DPosWidget(QDialog, Ui_Arm3DPostionWidget):
    close_dialog = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(Arm3DPosWidget, self).__init__(parent)
        self.setupUi(self)

        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowCloseButtonHint, True)
        self.pb_close.setStyleSheet(
                # "QPushButton{\n"
                # "color:rgb(190,190,190);\n"
                # "border-radius:25px;\n"
                # "background-color:white;\n"
                # "border-color:rgb(100, 100, 100);\n"
                # "}\n"
                "QPushButton:hover{\n"
                "color:white;\n"
                "border-radius:25px;\n"
                "background-color:gray;\n"
                "border-color:rgb(200, 200, 200);\n"
                "}\n"
                "\n"
                "QPushButton:pressed{\n"
                "color:black;\n"
                "border-radius:25px;\n"
                "background-color: rgb(180, 180, 180);\n"
                "}\n"
                "")

        self.axis_length = [13, 10, 2, 13.5]    # 单位cm
        self.gl_widget = gl.GLViewWidget(self)
        self.init_glview()
        
        self.global_angle_arm = 0    # 机械臂的全局角度,角度单位

        self.gridLayout_3d.addWidget(self.gl_widget)

        self.pb_close.clicked.connect(self.close_dialog)

        self.control_count = 0
        self.isChange = True
        # 滑动窗口，用于稳定障碍物的坐标
        self.WINDOW_SIZE = 5
        self.weights = [0.1, 0.2, 0.3, 0.4, 0.5]  # 总和为1，最新的数据权重最大
        self.coordinates_window = deque(maxlen=self.WINDOW_SIZE)

        # 加载箭头图片并创建 QLabel
        self.left_arrow = QLabel(self)
        self.right_arrow = QLabel(self)
        left_pixarrow = QPixmap("./resources/leftarrow.png")
        right_pixarrow = QPixmap("./resources/rightarrow.png")

        # 等比例缩放箭头图标大小为窗口的1/10，保持透明度
        self.left_arrow.setPixmap(left_pixarrow.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.right_arrow.setPixmap(right_pixarrow.scaled(50, 50, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 设置箭头标签的背景透明
        self.left_arrow.setAttribute(Qt.WA_TranslucentBackground)
        self.right_arrow.setAttribute(Qt.WA_TranslucentBackground)

        # 创建一个新的QWidget用于箭头
        self.arrow_widget = QWidget(self)
        self.arrow_layout = QVBoxLayout(self.arrow_widget)
        self.arrow_layout.addWidget(self.left_arrow)
        self.arrow_layout.addWidget(self.right_arrow)
        self.arrow_widget.setFixedSize(80, 50)

        # 将箭头小部件放置在右上角
        self.gridLayout_3d.addWidget(self.arrow_widget, 0, 0, alignment=Qt.AlignTop | Qt.AlignRight)

        # 初始状态设为隐藏
        self.left_arrow.setVisible(True)
        self.right_arrow.setVisible(False)
        # # 启动定时器，每隔1秒通知刷新一次数据
        # self.timer = QTimer(self)
        # self.timer.timeout.connect(self.update_data)
        # self.timer.start(100)

    def init_glview(self):        
        self.gl_widget.clear()
        self.state_vitual_train = False

        gl_item = gl.GLGridItem()
        size_axes = 100
        gl_item.setSize(x=size_axes, y=size_axes, z=size_axes)
        self.gl_widget.addItem(gl_item)

        # 坐标轴
        ax = gl.GLAxisItem(antialias=True, glOptions="opaque")
        ax.setSize(40, 40, 40)
        # ax.setText("x", "y", "z")
        self.gl_widget.addItem(ax)

        # axis = Custom3DAxis(self)
        # axis.setSize(40, 40, 40)
        # self.gl_widget.addItem(axis)
        # 创建长方体的8个顶点
        x_length = 4
        y_length = 2
        z_length = 2
        vertices = np.array([
            [0, 0, 0],
            [x_length, 0, 0],
            [x_length, y_length, 0],
            [0, y_length, 0],
            [0, 0, z_length],
            [x_length, 0, z_length],
            [x_length, y_length, z_length],
            [0, y_length, z_length]
        ])

        # 创建长方体的6个面，每个面由两个三角形组成
        faces = np.array([
            [0, 1, 2], [0, 2, 3],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [2, 3, 7], [2, 7, 6],
            [0, 3, 7], [0, 7, 4],
            [1, 2, 6], [1, 6, 5]
        ])
        self.target_color1 = np.empty((1, 4))
        self.target_color1[0] = (0.0, 1.0, 0.0, 1.0)  # 绿色，最后一位为透明度

        # 目标的可视化
        self.target_pos = np.empty((1, 3))         # 存放点的位置，为5 * 3的向量，感觉说是矩阵更合适
        self.target_size = np.empty((1))           # 存放点的大小
        self.target_size[0] = 2                  # 第一个点的大小
        mesh_data = gl.MeshData(vertexes=vertices, faces=faces)
        self.target_color = np.empty((1, 4))       # 存放点的颜色
        self.target_color[0] = (0.0, 1.0, 0.0, 1)  # 红色，最后一位为透明度
        self.target_item1 = gl.GLMeshItem(meshdata=mesh_data, smooth=False, color=self.target_color1[0], shader='shaded')
        self.target_pos[0] = (18.0, -1, 0)             # 第一个点的坐标

        # 设置长方体的位置
        self.target_item1.translate(*self.target_pos[0])
        self.gl_widget.addItem(self.target_item1)

        self.target_item = gl.GLScatterPlotItem(pos=self.target_pos, size=self.target_size, color=self.target_color, pxMode=False)  # 设置Item
        self.gl_widget.addItem(self.target_item)  # 当w使用addItem()后，才会生效显示图像

        # 机械臂关节点
        self.a_color = np.empty((5, 4))       # 存放点的颜色
        self.a_pos = np.empty((5, 3))         # 存放点的位置，为53 * 3的向量，感觉说是矩阵更合适
        self.a_size = np.empty((1))           # 存放点的大小

        # self.a_pos[0] = [0, 0, 0]
        # self.a_pos[1] = [10, 0, 10]
        # self.a_pos[2] = [10, 10, 10]
        # self.a_pos[3] = [15, 15, 5]
        # self.a_pos[4] = [20, 18, 5]

        pos1, pos2, pos3, pos4, pos5, pos6 = self.decode_pos(None)
        p1, p2, p3, p4, p5 = self.compute_pos(pos1, pos2, pos3, pos4, pos5, pos6)

        self.a_pos[0] = p1
        self.a_pos[1] = p2
        self.a_pos[2] = p3
        self.a_pos[3] = p4
        self.a_pos[4] = p5

        self.a_size[0] = 0.5                  # 第一个点的大小
        self.a_color[0] = (0.0, 1.0, 0.0, 1)  # 绿色，最后一位为透明度
        self.a_color[1] = (0.0, 1.0, 0.0, 1)  # 绿色，最后一位为透明度
        self.a_color[2] = (0.0, 1.0, 0.0, 1)  # 绿色，最后一位为透明度
        self.a_color[3] = (0.0, 1.0, 0.0, 1)  # 绿色，最后一位为透明度
        self.a_color[4] = (0.0, 1.0, 0.0, 1)  # 绿色，最后一位为透明度
        self.arm_point = gl.GLScatterPlotItem(pos=self.a_pos, size=self.a_size, color=self.a_color, pxMode=False)  # 设置Item
        self.gl_widget.addItem(self.arm_point)  

        # 机械臂轴
        self.arm_color = np.empty((5, 4))       # 机械臂颜色
        self.arm_pos = np.empty((5, 3))
        # self.arm_pos[0] = [0, 0, 0]
        # self.arm_pos[1] = [10, 0, 10]
        # self.arm_pos[2] = [10, 10, 10]
        # self.arm_pos[3] = [15, 15, 5]
        # self.arm_pos[4] = [20, 18, 5]

        self.arm_pos[0] = p1
        self.arm_pos[1] = p2
        self.arm_pos[2] = p3
        self.arm_pos[3] = p4
        self.arm_pos[4] = p5

        self.arm_color[0] = (0.0, 1.0, 1.0, 1)
        self.arm_color[1] = (0.0, 1.0, 1.0, 1)
        self.arm_color[2] = (0.0, 1.0, 1.0, 1)
        self.arm_color[3] = (0.0, 1.0, 1.0, 1)
        self.arm_color[4] = (0.0, 1.0, 1.0, 1)  # 浅蓝色，最后一位为透明度
        self.arm_line = gl.GLLinePlotItem(pos=self.arm_pos, color=self.arm_color, width=3)
        self.gl_widget.addItem(self.arm_line)

    def update_arrow_position(self):
        # 将左箭头图标位置设置到右上角
        self.left_arrow_item.setPos(self.width() - self.left_arrow_item.pixmap().width() - 10, 10)
        # 将右箭头图标位置设置到右上角
        self.right_arrow_item.setPos(self.width() - self.right_arrow_item.pixmap().width() - 10, 10)

    def update_visual(self, p1, p2, p3, p4, p5,direction):
        self.a_pos[0] = p1
        self.a_pos[1] = p2
        self.a_pos[2] = p3
        self.a_pos[3] = p4
        self.a_pos[4] = p5

        self.arm_pos[0] = p1
        self.arm_pos[1] = p2
        self.arm_pos[2] = p3
        self.arm_pos[3] = p4
        self.arm_pos[4] = p5

        self.arm_point.setData(pos=self.a_pos, size=self.a_size, color=self.a_color, pxMode=False)
        self.arm_line.setData(pos=self.arm_pos, color=self.arm_color, width=3)
        # self.arm_line = gl.GLLinePlotItem(pos=self.arm_pos, color=self.arm_color, width=3)
        # self.gl_widget.addItem(self.arm_line)
        # if self.isChange == True:

        self.target_item1.resetTransform()
        self.target_item1.translate(*self.target_pos[0])
        # direction = 1
        # self.target_item.setData(pos=self.target_pos, size=self.target_size, color=self.target_color, pxMode=False)
        # 根据 direction 设置箭头的可见性
        if direction == 1:
            self.left_arrow.setVisible(True)
            self.right_arrow.setVisible(False)
        elif direction == 2:
            self.left_arrow.setVisible(False)
            self.right_arrow.setVisible(True)
        else:
            self.left_arrow.setVisible(False)
            self.right_arrow.setVisible(False)

        self.gl_widget.update()

    def closeEvent(self, a0: QCloseEvent) -> None:
        # self.close_dialog.emit(True)
        return super().closeEvent(a0)

    def close_dialog(self):
        self.close()

    def update_arm_pos(self, pos, control_count, obr, direction):
        # 需解析数据，转化为角度
        if self.state_vitual_train:
            # 虚拟环境的训练阶段，不通过该函数更新
            return

        pos1, pos2, pos3, pos4, pos5, pos6 = self.decode_pos(pos)
        p1, p2, p3, p4, p5 = self.compute_pos(pos1, pos2, pos3, pos4, pos5, pos6)

        # if obr[0] == -1:
        #     self.target_pos[0] = p5
        # else:
        if 0 < control_count < 7:
            posx, posy = self.decode_obr_pos(obr)
            self.coordinates_window.append((posx, -1, 0))
            coordinates = np.mean(np.array(self.coordinates_window), axis=0)
            self.target_pos[0] = coordinates
        # 更新目标物的位置
        if 3 < control_count < 8:
            # 如果control_count > 3 则认可进入抓取流程，将target_item挂到arm_line线的末端
            if control_count == 7:
                pass
            else:
                self.target_pos[0] = p5
        if control_count == 2:
            # self.isChange = True
            pass
        self.control_count = control_count

        self.update_visual(p1, p2, p3, p4, p5,direction)

    def decode_obr_pos(self, obr):
        distance, angle_degrees = obr[0], obr[1]
        # angle_degrees =

        # 计算 x 和 y 坐标
        x = distance * math.cos(angle_degrees)
        y = distance * math.sin(angle_degrees)
        x /= 5.0
        # y /= 10.0
        # x = -x
        # 保持坐标不在正负轴移动
        if x < 0:
            x = -x
        if x > 20:
            x = 20.0
        if x < 15:
            x = 15.0
        print(f'x: {x}, y: {y}')

        return x, y
    def decode_pos_virtual(self,pos):
        if pos is not None:
            p1 = pos[0]   # 第一个水平角度
            p2 = pos[1][0]
            p3 = pos[2][0]
            p4 = pos[3][0]
            p5 = pos[4][0]
            p6 = 0.0

            return p1, p2, p3, p4, p5, p6
        else:
            return 0.0, 0.1, 1.3, 1.57, 0.0, 0.0

    def decode_pos(self, pos):
        if pos is not None:
            tp = pos.split(":")
            p1 = int(tp[0][1:]) / 100.0    # 第一个水平角度
            p2 = int(tp[1]) / 100.0
            p3 = int(tp[2]) / 100.0
            p4 = int(tp[3]) / 100.0
            p5 = int(tp[4]) / 100.0
            p6 = int(tp[5][:-1]) / 100.0

            return p1, p2, p3, p4, p5, p6
        else:
            return 0.0, 0.1, 1.3, 1.57, 0.0, 0.0

    def compute_pos(self, pos1, pos2, pos3, pos4, pos5, pos6):
        """
        pos5: 为机械爪与前一个轴的角度，一般为 0
        pos6: 机械臂的张开和闭合
        """

        pos1 = -pos1

        # 需计算位置信息
        p1 = [0, 0, 0]    # 第一个点，不会变化

        # 第二个点
        z2 = self.axis_length[0] * math.cos(-pos2)
        b2 = self.axis_length[0] * math.sin(-pos2)
        x2 = b2 * math.cos(pos1)
        y2 = b2 * math.sin(pos1)
        p2 = [x2, y2, z2]

        # 第三个点
        q0 = abs(pos2 - pos3)
        q00 = pos2 - pos3
        z3 = self.axis_length[1] * math.cos(q0)
        b3 = self.axis_length[1] * math.sin(q0)
        x3 = b3 * math.cos(pos1)
        y3 = b3 * math.sin(pos1)

        if q00 > 0:
            x3 = -x3
            y3 = -y3
        p3 = [x3 + x2, y3 + y2, z3 + z2]

        # 第四个点
        q1 = abs(q0) + pos4
        z4 = self.axis_length[2] * math.cos(q1)
        b4 = self.axis_length[2] * math.sin(q1)
        x4 = b4 * math.cos(pos1)
        y4 = b4 * math.sin(pos1)
        if q1 < 0:
            x4 = -x4
            y4 = -y4

        x4_ = x2 + x3 + x4
        y4_ = y2 + y3 + y4
        if math.pi/2 - (q00 + pos4) < 0:
            z4 = -z4
        p4 = [x4_, y4_, z2 + z3 + z4]

        # 第五个点
        z5 = (self.axis_length[2] + self.axis_length[3]) * math.cos(q1)
        b5 = (self.axis_length[2] + self.axis_length[3]) * math.sin(q1)
        x5 = b5 * math.cos(pos1)
        y5 = b5 * math.sin(pos1)
        
        if q1 < 0:
            x5 = -x5
            y5 = -y5

        x5_ = x2 + x3 + x5
        y5_ = y2 + y3 + y5


        if math.pi/2 - (q00 + pos4) < 0:
            z5 = -z5
        p5 = [x5_, y5_, z2 + z3 + z5]
        # p5 = [x3 + x4 * 1.3, y4 * 1.3 + y3, z3 + z4 * 1.3]

        return p1, p2, p3, p4, p5
    
    # ================================ for virtual env training =========================================
    def get_random_xy(self):
        import random
        x = random.random()
        x = int(x*36) - 18
        y = math.sqrt(18*18 - x*x)
        x = abs(x)
        return x, y

    def init_glview_random(self):        
        """
        随机初始化目标小球的位置，用于虚拟环境的训练过程
        """
        self.gl_widget.clear()
        self.state_vitual_train = True

        gl_item = gl.GLGridItem()
        size_axes = 100
        gl_item.setSize(x=size_axes, y=size_axes, z=size_axes)
        self.gl_widget.addItem(gl_item)

        # 坐标轴
        ax = gl.GLAxisItem(antialias=True, glOptions="opaque")
        ax.setSize(40, 40, 40)
        # ax.setText("x", "y", "z")
        self.gl_widget.addItem(ax)

        self.target_color1 = np.empty((1, 4))
        self.target_color1[0] = (0.0, 1.0, 0.0, 1.0)  # 绿色，最后一位为透明度

        # 目标的可视化
        self.target_pos = np.empty((1, 3))         # 存放点的位置，为5 * 3的向量，感觉说是矩阵更合适
        self.target_size = np.empty((1))           # 存放点的大小
        self.target_size[0] = 2                  # 第一个点的大小
        # mesh_data = gl.MeshData(vertexes=vertices, faces=faces)
        self.target_color = np.empty((1, 4))       # 存放点的颜色
        self.target_color[0] = (0.0, 1.0, 0.0, 1)  # 红色，最后一位为透明度
        # self.target_item1 = gl.GLMeshItem(meshdata=mesh_data, smooth=False, color=self.target_color1[0], shader='shaded')

        x, y = self.get_random_xy()
        self.target_pos[0] = (x, y, 0)             # 第一个点的坐标  x y z


        self.target_item = gl.GLScatterPlotItem(pos=self.target_pos, size=self.target_size, color=self.target_color, pxMode=False)  # 设置Item
        self.gl_widget.addItem(self.target_item)  # 当w使用addItem()后，才会生效显示图像

        # 机械臂关节点
        self.a_color = np.empty((5, 4))       # 存放点的颜色
        self.a_pos = np.empty((5, 3))         # 存放点的位置，为53 * 3的向量，感觉说是矩阵更合适
        self.a_size = np.empty((1))           # 存放点的大小

        # self.a_pos[0] = [0, 0, 0]
        # self.a_pos[1] = [10, 0, 10]
        # self.a_pos[2] = [10, 10, 10]
        # self.a_pos[3] = [15, 15, 5]
        # self.a_pos[4] = [20, 18, 5]
    
        pos1, pos2, pos3, pos4, pos5, pos6 = self.decode_pos(None)
        # pos1 = -math.pi / 2
        pos1=random.choice([0, -math.pi / 2])
        p1, p2, p3, p4, p5 = self.compute_pos(pos1, pos2, pos3, pos4, pos5, pos6)

        self.a_pos[0] = p1
        self.a_pos[1] = p2
        self.a_pos[2] = p3
        self.a_pos[3] = p4
        self.a_pos[4] = p5

        self.a_size[0] = 0.5                  # 第一个点的大小
        self.a_color[0] = (0.0, 1.0, 0.0, 1)  # 绿色，最后一位为透明度
        self.a_color[1] = (0.0, 1.0, 0.0, 1)  # 绿色，最后一位为透明度
        self.a_color[2] = (0.0, 1.0, 0.0, 1)  # 绿色，最后一位为透明度
        self.a_color[3] = (0.0, 1.0, 0.0, 1)  # 绿色，最后一位为透明度
        self.a_color[4] = (0.0, 1.0, 0.0, 1)  # 绿色，最后一位为透明度
        self.arm_point = gl.GLScatterPlotItem(pos=self.a_pos, size=self.a_size, color=self.a_color, pxMode=False)  # 设置Item
        self.gl_widget.addItem(self.arm_point)  

        # 机械臂轴
        self.arm_color = np.empty((5, 4))       # 机械臂颜色
        self.arm_pos = np.empty((5, 3))
        # self.arm_pos[0] = [0, 0, 0]
        # self.arm_pos[1] = [10, 0, 10]
        # self.arm_pos[2] = [10, 10, 10]
        # self.arm_pos[3] = [15, 15, 5]
        # self.arm_pos[4] = [20, 18, 5]

        self.arm_pos[0] = p1
        self.arm_pos[1] = p2
        self.arm_pos[2] = p3
        self.arm_pos[3] = p4
        self.arm_pos[4] = p5

        self.arm_color[0] = (0.0, 1.0, 1.0, 1)
        self.arm_color[1] = (0.0, 1.0, 1.0, 1)
        self.arm_color[2] = (0.0, 1.0, 1.0, 1)
        self.arm_color[3] = (0.0, 1.0, 1.0, 1)
        self.arm_color[4] = (0.0, 1.0, 1.0, 1)  # 浅蓝色，最后一位为透明度
        self.arm_line = gl.GLLinePlotItem(pos=self.arm_pos, color=self.arm_color, width=3)
        self.gl_widget.addItem(self.arm_line)

    def compute_angle_dis(self):
        """
        计算与y轴的角度差值
        """

        arm_pos = self.a_pos[4]
        ob_pos = self.target_pos[0]

        tan_ang = arm_pos[1] / (arm_pos[0] + 0.00001)
        ang_arm = math.atan(tan_ang) / math.pi * 180
        self.global_angle_arm = -abs(ang_arm)

        tan_ob = ob_pos[1] / (ob_pos[0] + 0.00001)
        ang_ob = abs(math.atan(tan_ob)/math.pi * 180)

        print("arm angle:", ang_arm)
        print("object angle:", ang_ob)

        ang_dis = ang_ob - ang_arm    # 机械臂从y轴向x方向运动为正方向，即顺时针为正方向
        print("angle dis", ang_dis) 
        return ang_dis


    def update_arm_angle_virtual(self, ang, direction):
        """
        用于训练，仅在虚拟环境中调用
        """
        pos1, pos2, pos3, pos4, pos5, pos6 = self.decode_pos(None)
        # pos1 = (self.global_angle_arm + ang) / 180.0
        # ang =  math.degrees(ang)
        print(f'ang========================={ang}')

        pos1 = (self.global_angle_arm +ang) / 180.0 * math.pi
        p1, p2, p3, p4, p5 = self.compute_pos(pos1, pos2, pos3, pos4, pos5, pos6)

        self.update_visual_train_virtual(p1, p2, p3, p4, p5, direction)

    def update_visual_train_virtual(self, p1, p2, p3, p4, p5,direction):
        self.a_pos[0] = p1
        self.a_pos[1] = p2
        self.a_pos[2] = p3
        self.a_pos[3] = p4
        self.a_pos[4] = p5

        self.arm_pos[0] = p1
        self.arm_pos[1] = p2
        self.arm_pos[2] = p3
        self.arm_pos[3] = p4
        self.arm_pos[4] = p5

        self.arm_point.setData(pos=self.a_pos, size=self.a_size, color=self.a_color, pxMode=False)
        self.arm_line.setData(pos=self.arm_pos, color=self.arm_color, width=3)

        # self.target_item1.resetTransform()
        # self.target_item1.translate(*self.target_pos[0])
        # # direction = 1
        # # self.target_item.setData(pos=self.target_pos, size=self.target_size, color=self.target_color, pxMode=False)
        # print(f'p5== {p5},target_pos[0]== {self.target_pos[0]}')
        # 根据 direction 设置箭头的可见性
        if direction == 1:
            self.left_arrow.setVisible(True)
            self.right_arrow.setVisible(False)
        elif direction == 2:
            self.left_arrow.setVisible(False)
            self.right_arrow.setVisible(True)
        else:
            self.left_arrow.setVisible(False)
            self.right_arrow.setVisible(False)

        self.gl_widget.update()

# class CustomTextItem(gl.GLGraphicsItem.GLGraphicsItem):
#     def __init__(self, X, Y, Z, text):
#         gl.GLGraphicsItem.GLGraphicsItem.__init__(self)
#         self.text = text
#         self.X = X
#         self.Y = Y
#         self.Z = Z

#     def setGLViewWidget(self, GLViewWidget):
#         self.GLViewWidget = GLViewWidget

#     def setText(self, text):
#         self.text = text
#         self.update()

#     def setX(self, X):
#         self.X = X
#         self.update()

#     def setY(self, Y):
#         self.Y = Y
#         self.update()

#     def setZ(self, Z):
#         self.Z = Z
#         self.update()

#     def paint(self):
#         # self.GLViewWidget.qglColor(QtCore.Qt.black)
#         self.GLViewWidget.renderText(self.X, self.Y, self.Z, self.text)


# class Custom3DAxis(gl.GLAxisItem):
#     """Class defined to extend 'gl.GLAxisItem'."""
#     def __init__(self, parent, color=(0,0,0,.6)):
#         gl.GLAxisItem.__init__(self)
#         self.parent = parent
#         self.c = color

#     def draw_labels(self):
#         x,y,z = self.size()
#         #X label
#         self.xLabel = CustomTextItem(X=x/2, Y=-y/20, Z=-z/20, text="X")
#         self.xLabel.setGLViewWidget(self.parent)
#         self.parent.addItem(self.xLabel)
#         #Y label
#         self.yLabel = CustomTextItem(X=-x/20, Y=y/2, Z=-z/20, text="Y")
#         self.yLabel.setGLViewWidget(self.parent)
#         self.parent.addItem(self.yLabel)
#         #Z label
#         self.zLabel = CustomTextItem(X=-x/20, Y=-y/20, Z=z/2, text="Z")
#         self.zLabel.setGLViewWidget(self.parent)
#         self.parent.addItem(self.zLabel)

#     def paint(self):
#         self.setupGLState()
#         if self.antialias:
#             ogl.glEnable(ogl.GL_LINE_SMOOTH)
#             ogl.glHint(ogl.GL_LINE_SMOOTH_HINT, ogl.GL_NICEST)
#         ogl.glBegin(ogl.GL_LINES)

#         x,y,z = self.size()
#         #Draw Z
#         ogl.glColor4f(self.c[0], self.c[1], self.c[2], self.c[3])
#         ogl.glVertex3f(0, 0, 0)
#         ogl.glVertex3f(0, 0, z)
#         #Draw Y
#         ogl.glColor4f(self.c[0], self.c[1], self.c[2], self.c[3])
#         ogl.glVertex3f(0, 0, 0)
#         ogl.glVertex3f(0, y, 0)
#         #Draw X
#         ogl.glColor4f(self.c[0], self.c[1], self.c[2], self.c[3])
#         ogl.glVertex3f(0, 0, 0)
#         ogl.glVertex3f(x, 0, 0)
#         #Draw labels
#         self.draw_labels()
#         ogl.glEnd()