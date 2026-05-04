# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

from operator import index
from tkinter.messagebox import NO
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *


from src.Ui_SignalWidget import Ui_SignalWidget


class SignalWidget(QWidget, Ui_SignalWidget):
    save_sti_para = pyqtSignal(int)
    close_widget = pyqtSignal(bool)

    def __init__(self, parent=None, para=None):
        super(SignalWidget, self).__init__(parent)
        self.setupUi(self)
        self.setWindowModality(Qt.WindowModality.WindowModal)

        self.para = para

        if self.para is None:
            self.frequency = 10
            self.ISI = 1000 / self.frequency
            self.spb_frequency.setValue(self.frequency)
            self.spb_ISI.setValue(self.ISI)

            self.uint_time = self.spb_cycles.value() * (self.spb_dura_1.value() + self.spb_dura_2.value() + self.spb_dura_3.value())  # ms
            self.spb_unit_time.setValue(self.uint_time)
            self.spb_unit_time.setEnabled(False)
        else:
            self.frequency = para["frequency"]; self.spb_frequency.setValue(para["frequency"])
            self.ISI = para["ISI"]; self.spb_ISI.setValue(para["ISI"])
            self.uint_time = para["one_uint_time"]
            self.spb_unit_time.setValue(self.uint_time)
            self.spb_unit_time.setEnabled(False)

            self.spb_cycles.setValue(para["cycles"])
            self.spb_amp_1.setValue(para["amplitude_1"])
            self.spb_amp_2.setValue(para["amplitude_2"])
            self.spb_dura_1.setValue(para["duration_1"])
            self.spb_dura_2.setValue(para["duration_2"])
            self.spb_dura_3.setValue(para["duration_3"])

        self.init_widget()


    def init_widget(self):
        self.charView = QChartView(self.widget_signal)
        self.charView.setContentsMargins(0, 0, 0, 0)
        self.charView.chart().setMargins(QMargins(0, 0, 0, 0))

        self.gridLayout_view = QGridLayout(self.widget_signal)
        self.gridLayout_view.setObjectName("gridLayout_view")
        self.gridLayout_view.addWidget(self.charView, 0, 0, 1, 1)

        self.series = QLineSeries(self)
        x_global = 0; point_list = []
        amp_1 = self.spb_amp_1.value()
        amp_2 = self.spb_amp_2.value()
        duration_1 = self.spb_dura_1.value()
        duration_2 = self.spb_dura_2.value()
        duration_3 = self.spb_dura_3.value()
        ISI = self.spb_ISI.value() * 1000
        for i in range(self.spb_cycles.value()):
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

            x_global = x_global + ISI    # 单个方波间隔
            point8 = QPointF(x_global, 0)
            points = [point0, point1, point2, point3, point4, point5, point6, point7, point8]

            point_list.extend(points)

            # self.point_0 = QPointF(0.0, 0.0)    # 起点
            # self.point_1 = QPointF(0.0, self.spb_amp_1.value())
            # self.point_2 = QPointF(self.spb_dura_1.value(), self.spb_amp_1.value())
            # self.point_3 = QPointF(self.spb_dura_1.value() + self.spb_dura_2.value(), self.spb_amp_2.value())
            # self.point_4 = QPointF(self.spb_dura_1.value() + self.spb_dura_2.value() + self.spb_dura_3.value(), self.spb_amp_2.value())
            # self.point_5 = QPointF(self.spb_dura_1.value() + self.spb_dura_2.value() + self.spb_dura_3.value(), 0)
            # point_list = [self.point_0, self.point_1, self.point_2, self.point_3, self.point_4, self.point_5]
        self.series.append(point_list)

        self.x_aix = QValueAxis(self)
        self.x_aix.setRange(0, x_global)
        self.x_aix.setLabelFormat("%d")
        self.x_aix.setTickCount(11)
        self.x_aix.setMinorTickCount(0)
        self.x_aix.setTitleText("Duration Time(μs)")

        self.y_aix = QValueAxis(self)
        self.y_aix.setRange(-1000, 1000)
        self.y_aix.setLabelFormat("%d")
        self.y_aix.setTickCount(11)
        self.y_aix.setMinorTickCount(0)
        # self.y_aix.setTitleText("Amplitude (mV)")

        self.charView.chart().setAxisX(self.x_aix)
        self.charView.chart().setAxisY(self.y_aix)

        # self.charView.chart().createDefaultAxes()
        self.charView.chart().setTitle("Stimulating Signal")

        self.spb_amp_1.valueChanged.connect(self.update_view)
        self.spb_amp_2.valueChanged.connect(self.update_view)
        self.spb_dura_1.valueChanged.connect(self.update_view)
        self.spb_dura_2.valueChanged.connect(self.update_view)
        self.spb_dura_3.valueChanged.connect(self.update_view)

        self.spb_cycles.valueChanged.connect(self.update_view)
        self.spb_ISI.valueChanged.connect(self.update_view)
        self.spb_unit_time.valueChanged.connect(self.update_view)

        self.charView.chart().addSeries(self.series)

        self.series.attachAxis(self.x_aix)
        self.series.attachAxis(self.y_aix)
        self.charView.chart().legend().hide()
        self.charView.show()

        # for connect
        self.spb_frequency.valueChanged.connect(self.update_frequency)
        self.spb_ISI.valueChanged.connect(self.update_ISI)
        # self.spb_dura_1.valueChanged.connect(self.update_uint_time)
        # self.spb_dura_2.valueChanged.connect(self.update_uint_time)
        # self.spb_dura_3.valueChanged.connect(self.update_uint_time)
        # self.spb_cycles.valueChanged.connect(self.update_uint_time)

        self.pb_ok.clicked.connect(self.pb_ok_clicked)
        self.pb_cancel.clicked.connect(self.pb_cancel_clicked)

    def update_view(self):
        x_global = 0; point_list = []
        amp_1 = self.spb_amp_1.value()
        amp_2 = self.spb_amp_2.value()
        duration_1 = self.spb_dura_1.value()
        duration_2 = self.spb_dura_2.value()
        duration_3 = self.spb_dura_3.value()
        ISI = self.spb_ISI.value() * 1000
        for i in range(self.spb_cycles.value()):
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

            x_global = x_global + ISI    # 单个方波间隔
            point8 = QPointF(x_global, 0)
            points = [point0, point1, point2, point3, point4, point5, point6, point7, point8]

            point_list.extend(points)
        self.update_uint_time()


        # self.point_0 = QPointF(0.0, 0.0)    # 起点
        # self.point_1 = QPointF(0.0, self.spb_amp_1.value())
        # self.point_2 = QPointF(self.spb_dura_1.value(), self.spb_amp_1.value())
        # self.point_3 = QPointF(self.spb_dura_1.value() + self.spb_dura_2.value(), self.spb_amp_2.value())
        # self.point_4 = QPointF(self.spb_dura_1.value() + self.spb_dura_2.value() + self.spb_dura_3.value(), self.spb_amp_2.value())
        # self.point_5 = QPointF(self.spb_dura_1.value() + self.spb_dura_2.value() + self.spb_dura_3.value(), 0)

        # point_list = [self.point_0, self.point_1, self.point_2, self.point_3, self.point_4, self.point_5]
        self.series.replace(point_list)
        self.x_aix.setRange(0, x_global)
        self.charView.update()

    def update_frequency(self, value):
        self.frequency = value
        self.ISI = 1000 / self.frequency
        self.spb_ISI.setValue(self.ISI)
    
    def update_ISI(self, value):
        self.ISI = value
        self.frequency = 1000 / self.ISI
        self.spb_frequency.setValue(self.frequency)

    def update_uint_time(self):
        self.uint_time = self.spb_cycles.value() * (self.spb_dura_1.value() + self.spb_dura_2.value() + self.spb_dura_3.value() + self.ISI * 1000)  # μs
        self.spb_unit_time.setValue(self.uint_time / 1000)

    def pb_ok_clicked(self):
        if self.para is None:  # 表示新建参数
            self.save_sti_para.emit(1)
        else:
            self.save_sti_para.emit(2)    # 不添加新的按钮
        self.close()

    def pb_cancel_clicked(self):
        self.save_sti_para.emit(0)
        self.close()
    
    def get_para(self):
        self.para = {
            "amplitude_1":self.spb_amp_1.value(),
            "amplitude_2":self.spb_amp_2.value(),
            "duration_1":self.spb_dura_1.value(),
            "duration_2":self.spb_dura_2.value(),
            "duration_3":self.spb_dura_3.value(),
            "cycles":self.spb_cycles.value(),
            "ISI":self.ISI,
            "frequency":self.frequency,
            "one_uint_time":self.uint_time}
        return self.para
    
    # def close(self):
    #     self.close_widget.emit(True)

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.close_widget.emit(True)
        return super().closeEvent(a0)