# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------


import numpy as np
import sys

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *

import pyqtgraph as pg

from src.Ui_StiSpikeSignalDataDialog import Ui_StiSpikeSignalDataDialog
from src.infor_com_mea.spike_detection import SpikeDetection 

class StiSpikesSignalDataDialog(QDialog, Ui_StiSpikeSignalDataDialog):
    close_widget = pyqtSignal(bool)

    def __init__(self, parent=None, mea_ic=None):
        super(StiSpikesSignalDataDialog, self).__init__(parent)
        self.setupUi(self)
        # self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setWindowModality(Qt.WindowModality.NonModal)

        self.mea_ic = mea_ic
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
            colors = '008080'

        line_color = 'r'
        # line_color = '008080'
        colors = 'r'
        width = 2

        self.left_sti = self.graphicsView_left_sti
        self.left_sti.setTitle("Stimulus of Left",
                         color=colors,
                         size='12pt')  # 12pt
        self.left_sti.setLabel("left","Frequency (Hz)")
        self.left_sti.setLabel("bottom","Data Frame")
        self.left_sti.showGrid(x=True, y=True)
        self.left_sti.setBackground("w")
        self.left_sti.setYRange(min=0, max=20)
        self.curve_left_sti = self.left_sti.plot(pen=pg.mkPen(line_color, width=width))


        self.right_sti = self.graphicsView_right_sti
        self.right_sti.setTitle("Stimulus of Right",
                         color=colors,
                         size='12pt')
        self.right_sti.setLabel("left","Frequency (Hz)")
        self.right_sti.setLabel("bottom","Data Frame")
        self.right_sti.showGrid(x=True, y=True)
        self.right_sti.setBackground("w")
        self.right_sti.setYRange(min=0,
                        max=20)
        self.curve_right_sti = self.right_sti.plot(pen=pg.mkPen(line_color, width=width))


        self.differ_sti = self.graphicsView_sti_difference
        self.differ_sti.setTitle("Difference of Stimulus (L-R)",
                         color=colors,
                         size='12pt')
        self.differ_sti.setLabel("left","Frequency (Hz)")
        self.differ_sti.setLabel("bottom","Data Frame")
        self.differ_sti.showGrid(x=True, y=True)
        self.differ_sti.setBackground("w")
        self.differ_sti.setYRange(min=-10,
                        max=10)
        self.curve_differ_sti = self.differ_sti.plot(pen=pg.mkPen(line_color, width=width))


        self.left_spike = self.graphicsView_left_spikes
        self.left_spike.setTitle("Spikes of Left",
                         color=colors,
                         size='12pt')
        # self.left_spike.setTitle("",
        #                  color=colors,
        #                  size='12pt')
        self.left_spike.setLabel("left","Spikes")
        self.left_spike.setLabel("bottom","Data Frame")
        self.left_spike.showGrid(x=True, y=True)
        self.left_spike.setBackground("w")
        self.left_spike.setYRange(min=0,
                        max=50)
        self.left_spike_curve = self.left_spike.plot(pen=pg.mkPen(line_color, width=width))


        self.right_spike = self.graphicsView_right_spikes
        self.right_spike.setTitle("Spikes of Right",
                         color=colors,
                         size='12pt')
        # self.right_spike.setTitle("",
        #                  color=colors,
        #                  size='12pt')
        self.right_spike.setLabel("left","Spikes")
        self.right_spike.setLabel("bottom","Data Frame")
        self.right_spike.showGrid(x=True, y=True)
        self.right_spike.setBackground("w")
        self.right_spike.setYRange(min=0,
                        max=50)
        self.right_spike_curve = self.right_spike.plot(pen=pg.mkPen(line_color, width=width))

        self.spikes_differ = self.graphicsView_spikes_difference
        self.spikes_differ.setTitle("Difference of Spikes (L-R)",
                         color=colors,
                         size='12pt')
        self.spikes_differ.setLabel("left","Spikes")
        self.spikes_differ.setLabel("bottom","Data Frame")
        self.spikes_differ.showGrid(x=True, y=True)
        self.spikes_differ.setBackground("w")
        self.spikes_differ.setYRange(min=-30,
                        max=30)
        self.spikes_differ_curve = self.spikes_differ.plot(pen=pg.mkPen(line_color, width=width))

    def update_data(self):
        left_sti, right_sti, left_spikes, right_spikes = self.mea_ic.get_stifre_ctrol_signal()
        x_sti = np.arange(0, len(left_sti), 1)
        x_spikes = np.arange(0, len(left_spikes), 1)
        
        self.curve_left_sti.setData(x_sti, left_sti)
        self.curve_right_sti.setData(x_sti, right_sti)
        self.curve_differ_sti.setData(x_sti, np.array(left_sti) - np.array(right_sti))

        self.left_spike_curve.setData(x_spikes, left_spikes)
        self.right_spike_curve.setData(x_spikes, right_spikes)
        self.spikes_differ_curve.setData(x_spikes, np.float32(np.array(left_spikes)) - np.float32(np.array(right_spikes)))


    def closeEvent(self, a0: QCloseEvent) -> None:
        # self.close_dialog.emit(True)
        return super().closeEvent(a0)

    def close_dialog(self):
        self.close()