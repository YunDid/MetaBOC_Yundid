# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------


from operator import index
from pickle import FALSE
from tkinter.messagebox import NO
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import pyqtgraph as pg

from src.Ui_ChannelSignalDialog import Ui_ChannelSignalDialog

class ChannelSignalDialog(QDialog, Ui_ChannelSignalDialog):
    close_dialog = pyqtSignal(bool)

    def __init__(self, parent=None, recording=None):
        super(ChannelSignalDialog, self).__init__(parent)
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

        self.pw.setYRange(min=-1*self.spinBox.value(),
                        max=self.spinBox.value())

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

        self.spinBox.valueChanged.connect(self.update_y_range)

        self.show_channel_data()    # 默认显示参考通道


    def initial_array(self):
        self.frame_array.setStyleSheet(
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

        self.pb_button = {"21":self.pb_21,"31":self.pb_31,"41":self.pb_41,
                            "51":self.pb_51,"61":self.pb_61,"71":self.pb_71,

                "12":self.pb_12,"22":self.pb_22,"32":self.pb_32,"42":self.pb_42,
                "52":self.pb_52,"62":self.pb_62,"72":self.pb_72,"82":self.pb_82,

                "13":self.pb_13,"23":self.pb_23,"33":self.pb_33,"43":self.pb_43,
                "53":self.pb_53,"63":self.pb_63,"73":self.pb_73,"83":self.pb_83,
                
                "14":self.pb_14,"24":self.pb_24,"34":self.pb_34,"44":self.pb_44,
                "54":self.pb_54,"64":self.pb_64,"74":self.pb_74,"84":self.pb_84,
                
                "25":self.pb_25,"35":self.pb_35,"45":self.pb_45,"55":self.pb_55,
                "65":self.pb_65,"75":self.pb_75,"85":self.pb_85, "15":self.pb_15,
                
                "16":self.pb_16,"26":self.pb_26,"36":self.pb_36,"46":self.pb_46,
                "56":self.pb_56,"66":self.pb_66,"76":self.pb_76,"86":self.pb_86,
                
                "17":self.pb_17,"27":self.pb_27,"37":self.pb_37,"47":self.pb_47,
                "57":self.pb_57,"67":self.pb_67,"77":self.pb_77,"87":self.pb_87,
                
                "28":self.pb_28,"38":self.pb_38,"48":self.pb_48,
                "58":self.pb_58,"68":self.pb_68,"78":self.pb_78}
        

        self.button_group = QButtonGroup(self.frame_array)
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

        self.pw.setYRange(min=-1*self.spinBox.value(),
                        max=self.spinBox.value())

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.close_dialog.emit(True)
        return super().closeEvent(a0)