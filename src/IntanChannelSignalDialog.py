# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2024.04.01
# --------------------------------------------------------


from operator import index
from pickle import FALSE
from tkinter.messagebox import NO
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import pyqtgraph as pg

from src.Ui_IntanChannelSignalDialog import Ui_INTANChannelSignalDialog

class INTANChannelSignalDialog(QDialog, Ui_INTANChannelSignalDialog):
    close_dialog = pyqtSignal(bool)

    def __init__(self, parent=None, recording=None):
        super(INTANChannelSignalDialog, self).__init__(parent)
        self.setupUi(self)

        self.recording = recording

        self.initial_array()

        self.pw = self.graphicsView
        self.pw.setTitle("Recording Data",
                         color='008080',   # 008080  red
                         size='12pt')
        self.pw.setLabel("left","Voltage(μV)")
        self.pw.setLabel("bottom","Time")
        self.pw.showGrid(x=True, y=True)
        # self.pw.setBackground("w")

        # self.pw.setYRange(min=-1*self.spb_voltage_range.value(),
        #                 max=self.spb_voltage_range.value())
        self.pw.setYRange(-500,500)
        self.pw.setXRange(0,30000)
        

        self.curve = self.pw.plot(
            pen=pg.mkPen('r', width=1)
        )

        self.i = 0
        self.x = [] # x轴的值
        self.y = [] # y轴的值
        self.channel_index = 15

        # 启动定时器，每隔1秒通知刷新一次数据
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(100)

        self.spb_voltage_range.valueChanged.connect(self.update_y_range)

        self.show_channel_data()    # 默认显示参考通道



    def initial_array(self):
        self.frame_intan.setStyleSheet(
                "QFrame{background-color: rgb(250, 250, 250);}\n"
                "QPushButton{\n"
                "border-radius:25px;\n"
                "background-color: rgb(220, 220, 220);\n"
                "border-width:2px;\n"
                "border-style:soild;\n"
                "}\n"
                "\n"
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
                "background-color: rgb(173, 255, 172);\n"
                "}\n")

        self.frame_intan_b.setStyleSheet(
                "QFrame{background-color: rgb(250, 250, 250);}\n"
                "QPushButton{\n"
                "border-radius:25px;\n"
                "background-color: rgb(220, 220, 220);\n"
                "border-width:2px;\n"
                "border-style:soild;\n"
                "}\n"
                "\n"
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
                "background-color: rgb(173, 255, 172);\n"
                "}\n")

        self.pb_button = {                
                "A-000":self.pb_000, "A-001":self.pb_001, "A-002":self.pb_002,
                "A-003":self.pb_003, "A-004":self.pb_004, "A-005":self.pb_005,
                "A-006":self.pb_006, "A-007":self.pb_007, "A-008":self.pb_008,
                "A-009":self.pb_009, "A-010":self.pb_010, "A-011":self.pb_011,
                "A-012":self.pb_012, "A-013":self.pb_013, "A-014":self.pb_014,
                "A-015":self.pb_015, "A-016":self.pb_016, "A-017":self.pb_017,
                "A-018":self.pb_018, "A-019":self.pb_019, "A-020":self.pb_020,
                "A-021":self.pb_021, "A-022":self.pb_022, "A-023":self.pb_023,
                "A-024":self.pb_024, "A-025":self.pb_025, "A-026":self.pb_026,
                "A-027":self.pb_027, "A-028":self.pb_028, "A-029":self.pb_029,
                "A-030":self.pb_030, "A-031":self.pb_031,
                
                # for intan b
                "B-000":self.pb_000_b, "B-001":self.pb_001_b, "B-002":self.pb_002_b,
                "B-003":self.pb_003_b, "B-004":self.pb_004_b, "B-005":self.pb_005_b,
                "B-006":self.pb_006_b, "B-007":self.pb_007_b, "B-008":self.pb_008_b,
                "B-009":self.pb_009_b, "B-010":self.pb_010_b, "B-011":self.pb_011_b,
                "B-012":self.pb_012_b, "B-013":self.pb_013_b, "B-014":self.pb_014_b,
                "B-015":self.pb_015_b, "B-016":self.pb_016_b, "B-017":self.pb_017_b,
                "B-018":self.pb_018_b, "B-019":self.pb_019_b, "B-020":self.pb_020_b,
                "B-021":self.pb_021_b, "B-022":self.pb_022_b, "B-023":self.pb_023_b,
                "B-024":self.pb_024_b, "B-025":self.pb_025_b, "B-026":self.pb_026_b,
                "B-027":self.pb_027_b, "B-028":self.pb_028_b, "B-029":self.pb_029_b,
                "B-030":self.pb_030_b, "B-031":self.pb_031_b,}

        self.button_group = QButtonGroup(self.frame_intan)
        for key in self.pb_button.keys():
            self.pb_button[key].setCheckable(True)
            self.button_group.addButton(self.pb_button[key])
        
        self.button_group.buttonToggled.connect(self.array_button_clicked)

    def array_button_clicked(self, button):
        text = button.text()
        state = button.isChecked()

        self.channel_index = self.recording.channel_map[text]
        print(text)
        if state:
            button.setStyleSheet(
                "QPushButton{\n"
                "border-radius:25px;\n"
                "background-color: rgb(173, 255, 172);\n"
                "border-width:2px;\n"
                "border-style:soild;\n"
                "}\n"
                "\n"
                "QPushButton:hover{\n"
                "color:white;\n"
                "border-radius:25px;\n"
                "background-color:gray;\n"
                "border-color:rgb(200, 200, 200);\n"
                "}\n")
        else:    
            button.setStyleSheet(
                "QPushButton{\n"
                "border-radius:25px;\n"
                "background-color: rgb(220, 220, 220);\n"
                "border-width:2px;\n"
                "border-style:soild;\n"
                "}\n"
                "\n"
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
                "background-color: rgb(173, 255, 172);\n"
                "}\n"
                "")

    def show_channel_data(self):
        x, y = self.recording.get_one_second_signal(self.channel_index)
        self.curve.setData(x, y)

    def update_data(self):
        x, y = self.recording.get_one_second_signal(self.channel_index)
        self.curve.setData(x, y)

    def update_y_range(self):

        self.pw.setYRange(min=-1*self.spb_voltage_range.value(),
                        max=self.spb_voltage_range.value())

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.close_dialog.emit(True)
        return super().closeEvent(a0)