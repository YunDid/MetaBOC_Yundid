# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

import os
import numpy as np

from operator import index
from tkinter.messagebox import NO
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *

from src.robot.task import SYSTEM_DEVICE

from src.Ui_SelectStimulating import Ui_SelectStimulatingDialog

class Parameter(object):
    def __init__(self, para, sti_para, recording_list, stimulating_list) -> None:
        self.para = para
        self.sti_para = sti_para
        self.recording_list = recording_list
        self.stimulating_list = stimulating_list

    def get_para(self):
        return self.para, self.sti_para, self.recording_list, self.stimulating_list 


class SelectStimulatingPara(QDialog, Ui_SelectStimulatingDialog):
    select_sti_para = pyqtSignal(bool)
    # close_widget = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(SelectStimulatingPara, self).__init__(parent)
        self.setupUi(self)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.parameter_dict = {}        # 所有的刺激参数
        self.stimulating_data = None    # 指定选择的刺激参数
        self.frame_signal.setContentsMargins(0, 0, 0, 0)
        self.frame_signal.setStyleSheet("border: 1px solid grey")

        self.listWidget.itemClicked.connect(self.item_clicked)

        self.charView = None
        self.gridLayout_view = None
        self.series = None
        self.x_aix = None
        self.y_aix = None

        self.pb_remove.clicked.connect(self.remove_data)
        self.pb_ok.clicked.connect(self.ok)
        self.pb_cancel.clicked.connect(self.cancel)

        self.sys_device = SYSTEM_DEVICE.MEA2100

        self.initial()

    def initial(self):
        # 重新加载已经设置的刺激参数
        if self.sys_device == SYSTEM_DEVICE.MEA2100:
            path = "./out/MEA2100"
        elif self.sys_device == SYSTEM_DEVICE.INTAN:
            path = "./out/INTAN"
        elif self.sys_device == SYSTEM_DEVICE.MAXWELL:
            path = "./out/MAXWELL"
            os.makedirs(path, exist_ok=True)  # 首次切到 Maxwell 时目录可能不存在

        files = os.listdir(path)
        for i in range(len(files)):
            full_p = os.path.join(path, files[i])
            data = np.load(full_p, allow_pickle=True)   #  一个文件只能有一个sti_para，多个para
            para = data["para"]
            sti_para = data["sti_para"][0]
            recording_left_list = data["recording_left_list"]
            recording_right_list = data["recording_right_list"]
            stimulating_left_list = data["stimulating_left_list"]
            stimulating_right_list = data["stimulating_right_list"]

            recording_list = [recording_left_list, recording_right_list]
            stimulating_list = [stimulating_left_list, stimulating_right_list]
            
            self.add_list(para, sti_para, recording_list, stimulating_list)

        if self.sys_device == SYSTEM_DEVICE.INTAN:
            self.frame_signal.hide()

    def update_system(self, system):
        if system != self.sys_device:
            # 清空原列表，更新列表
            self.listWidget.clear()

            self.sys_device = system

            self.initial()

    def ok(self):
        count = self.listWidget.count()
        state = False; text = None
        for i in range(count):
            item = self.listWidget.item(i)
            st = item.checkState()
            if st:
                state = True
                name = item.text()
        if state:
            self.stimulating_data = self.parameter_dict[name]
            self.close()
        else:
            QMessageBox.information(self,'Tips','Please select one stimulating mode!',QMessageBox.Ok)

        self.select_sti_para.emit(True)


    def cancel(self):
        self.close()

    def add_list(self, para, sti_para, recording_list, stimulating_list):
        if sti_para:
            sti_name = sti_para["stimulate_name"]
            self.listWidget.addItem(sti_name)
            count = self.listWidget.count()
            self.listWidget.item(count-1).setCheckState(False)
            self.parameter_dict[sti_name] = Parameter(para, sti_para, recording_list, stimulating_list)
    
    def item_clicked(self, value):
        count = self.listWidget.count()
        for i in range(count):
            if value.text() == self.listWidget.item(i).text():
                value.setCheckState(True)
            else:
                self.listWidget.item(i).setCheckState(False)

        sti_name = value.text()

        para_class = self.parameter_dict[sti_name]
        para, sti_para, recording_list, stimulating_list = para_class.get_para()
        self.show_signal(para, sti_para, recording_list, stimulating_list)

    def show_signal(self, para, sti_para, recording_list, stimulating_list):
        # Maxwell 复用 MCS 双相脉冲参数 schema（amplitude/duration/cycles/ISI），波形图照画
        if self.sys_device in (SYSTEM_DEVICE.MEA2100, SYSTEM_DEVICE.MAXWELL):
            self.frame_signal.show()
            if self.charView is None:
                self.charView = QChartView(self.frame_signal)
                self.charView.setContentsMargins(0, 0, 0, 0)
                self.charView.chart().setMargins(QMargins(0, 0, 0, 0))

                self.gridLayout_view = QGridLayout(self.frame_signal)
                self.gridLayout_view.setObjectName("gridLayout_view")
                self.gridLayout_view.addWidget(self.charView, 0, 0, 1, 1)
                self.gridLayout_view.setContentsMargins(0, 0, 0, 0)

                self.series = QLineSeries(self)

                self.x_aix = QValueAxis(self)
                self.x_aix.setRange(0, 1000)
                self.x_aix.setLabelFormat("%d")
                self.x_aix.setTickCount(6)
                self.x_aix.setMinorTickCount(3)
                self.x_aix.setTitleText("Duration Time(μs)")

                self.y_aix = QValueAxis(self)
                self.y_aix.setRange(-1000, 1000)
                self.y_aix.setLabelFormat("%d")
                self.y_aix.setTickCount(6)
                self.y_aix.setMinorTickCount(3)
                self.y_aix.setTitleText("Amplitude (mV)")

                self.charView.chart().setAxisX(self.x_aix)
                self.charView.chart().setAxisY(self.y_aix)

                self.charView.chart().setTitle("Stimulating Signal")
                self.charView.chart().addSeries(self.series)

                self.series.attachAxis(self.x_aix)
                self.series.attachAxis(self.y_aix)
                self.charView.chart().legend().hide()
                self.charView.show()
            else:
                self.series.replace([QPointF(0,0)])

            x_global = 0; ISI = 0
            # 单元组的循环次数
            for k in range(sti_para["reapeat_times"]):
                # 一个单元组的信号
                for i in range(len(para)):
                    tp = para[i]; point_list = []
                    amp_1 = tp["amplitude_1"]
                    amp_2 = tp["amplitude_2"]
                    duration_1 = tp["duration_1"]
                    duration_2 = tp["duration_2"]
                    duration_3 = tp["duration_3"]
                    cycles = tp["cycles"]
                    ISI = tp["ISI"]
                    # 单个信号的cycle
                    for j in range(cycles):
                        point0 = QPointF(x_global, 0)
                        point1 = QPointF(x_global, amp_1)
                        x_global = x_global + duration_1
                        point2 = QPointF(x_global, amp_1)
                        point3 = QPointF(x_global, 0)
                        x_global = x_global + duration_2
                        point4 = QPointF(x_global, 0)
                        point5 = QPointF(x_global, amp_2)
                        x_global = x_global + duration_3
                        point6 = QPointF(x_global, amp_2)
                        point7 = QPointF(x_global, 0)
                        x_global = x_global + ISI * 1000    # 单个方波间隔
                        point8 = QPointF(x_global, 0)
                        point_list = [point0, point1, point2, point3, point4, point5, point6, point7, point8]
                        self.series.append(point_list)
                # x_global = x_global + ISI*1000

            self.x_aix.setRange(0, x_global)
            self.spb_total_time.setValue(x_global/1000)
        elif self.sys_device == SYSTEM_DEVICE.INTAN:
            self.frame_signal.hide()
            self.spb_total_time.setValue(sti_para["total_time"])

        self.spb_repeat_times.setValue(sti_para["reapeat_times"])
        
        # 显示选中的电极
        strings_temp = ""
        for i in range(len(recording_list[0])):
            strings_temp = strings_temp + recording_list[0][i] + ","
        self.lineEdit_recording_left.setText(strings_temp)

        strings_temp = ""
        for i in range(len(recording_list[1])):
            strings_temp = strings_temp + recording_list[1][i] + ","
        self.lineEdit_recording_right.setText(strings_temp)

        strings_temp = ""
        for i in range(len(stimulating_list[0])):
            strings_temp = strings_temp + stimulating_list[0][i] + ","
        self.lineEdit_stimulating_left.setText(strings_temp)

        strings_temp = ""
        for i in range(len(stimulating_list[1])):
            strings_temp = strings_temp + stimulating_list[1][i] + ","
        self.lineEdit_stimulating_right.setText(strings_temp)


    def remove_data(self):
        item = self.listWidget.currentItem()

        if item:
            text = item.text()
            row = self.listWidget.currentRow()
            self.listWidget.takeItem(row)

            if self.sys_device == SYSTEM_DEVICE.MEA2100:
                full_path = os.path.join("./out/MEA2100", text + ".npz")
            elif self.sys_device == SYSTEM_DEVICE.INTAN:
                full_path = os.path.join("./out/INTAN", text + ".npz")
            elif self.sys_device == SYSTEM_DEVICE.MAXWELL:
                full_path = os.path.join("./out/MAXWELL", text + ".npz")
            if os.path.exists(full_path):
                os.remove(full_path)

    def get_selected_stimulating_para(self):
        return self.stimulating_data