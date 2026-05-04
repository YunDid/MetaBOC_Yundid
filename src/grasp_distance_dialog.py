# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

import sys

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import numpy as np

import pyqtgraph as pg

from src.Ui_GraspDistanceDialog import Ui_GraspDistanceDialog

class GraspDistanceDialog(QDialog, Ui_GraspDistanceDialog):
    close_dialog = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(GraspDistanceDialog, self).__init__(parent)
        self.setupUi(self)

        self.pb_close.setStyleSheet(
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

        self.init_graphics_view()

        self.pb_close.clicked.connect(self.close_dialog)

        # 启动定时器，每隔1秒通知刷新一次数据
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)

    def init_graphics_view(self):
        python_version = sys.version
        vr = python_version.split(".")
        if int(vr[1]) <= 6:
            colors = '008080'
        else:
            colors = 'red'
        self.graphicsView_distance.setTitle("Distance Between the Robotic Arm and the Target",
                         color=colors,  # 008080  red
                         size='12pt')
        self.graphicsView_distance.setLabel("left","Distance (cm)")
        self.graphicsView_distance.setLabel("bottom","Time")
        self.graphicsView_distance.showGrid(x=True, y=True)
        # self.pw.setBackground("w")
        self.graphicsView_distance.setYRange(min=0, max=20)
        self.curve_left_sti = self.graphicsView_distance.plot(pen=pg.mkPen('r', width=1))

    def update_data(self):
        # 从真实机器人获取距离信息，进行可视化
        return
        path = "./out_tracking/2023_7_3_11_4.txt"
        with open(path, "r") as f:
            dis = f.readlines()

        x = np.arange(len(dis)) * 100
        y = [float(i.split("\n")[0]) / 100 for i in dis]
        
        self.curve_left_sti.setData(x, y)

    def closeEvent(self, a0: QCloseEvent) -> None:
        # self.close_dialog.emit(True)
        return super().closeEvent(a0)

    def close_dialog(self):
        self.close()