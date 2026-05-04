# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2024
# Written by Guiping Cao
# Time: 2024.06.18
# --------------------------------------------------------

from operator import index
from tkinter.messagebox import NO
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *


from src.Ui_INTAN_Sti_Setting import Ui_StimulateSettingINTAN


class INTANSignalWidget(QWidget, Ui_StimulateSettingINTAN):
    save_sti_para = pyqtSignal(int)
    close_widget = pyqtSignal(bool)

    def __init__(self, parent=None, para=None):
        super(INTANSignalWidget, self).__init__(parent)
        self.setupUi(self)
        self.setWindowModality(Qt.WindowModality.WindowModal)

        self.para = para

        self.groupBox.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.groupBox_2.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.groupBox_3.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.groupBox_4.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.groupBox_5.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.groupBox_6.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.groupBox_7.setStyleSheet("QGroupBox { font-weight: bold; }")
        self.groupBox_8.setStyleSheet("QGroupBox { font-weight: bold; }")

        self.pb_ok.clicked.connect(self.pb_ok_clicked)
        self.pb_cancel.clicked.connect(self.pb_cancel_clicked)

    def pb_ok_clicked(self):
        if self.para is None:  # 表示新建参数
            self.save_sti_para.emit(1)
        else:
            self.save_sti_para.emit(2)    # 不添加新的按钮
        self.close()

    def pb_cancel_clicked(self):
        self.save_sti_para.emit(0)
        self.close()

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.close_widget.emit(True)
        return super().closeEvent(a0)
    
    def get_para(self):
        self.setting1 = {"D1": self.spb_s1_d1.value(),
                         "D2": self.spb_s1_d1.value(),
                         "A1": self.spb_s1_a1.value(),
                         "A2": self.spb_s1_a2.value(),
                         "Pre1": self.spb_s1_pre1.value(),
                         "Post1": self.spb_s1_post1.value()}
        self.setting2 = {"D1": self.spb_s2_d1.value(),
                         "D2": self.spb_s2_d1.value(),
                         "A1": self.spb_s2_a1.value(),
                         "A2": self.spb_s2_a2.value(),
                         "Pre1": self.spb_s2_pre1.value(),
                         "Post1": self.spb_s2_post1.value()}
        self.setting3 = {"D1": self.spb_s3_d1.value(),
                         "D2": self.spb_s3_d1.value(),
                         "A1": self.spb_s3_a1.value(),
                         "A2": self.spb_s3_a2.value(),
                         "Pre1": self.spb_s3_pre1.value(),
                         "Post1": self.spb_s3_post1.value()}
        self.setting4 = {"D1": self.spb_s4_d1.value(),
                         "D2": self.spb_s4_d1.value(),
                         "A1": self.spb_s4_a1.value(),
                         "A2": self.spb_s4_a2.value(),
                         "Pre1": self.spb_s4_pre1.value(),
                         "Post1": self.spb_s4_post1.value()}
        self.setting5 = {"D1": self.spb_s5_d1.value(),
                         "D2": self.spb_s5_d1.value(),
                         "A1": self.spb_s5_a1.value(),
                         "A2": self.spb_s5_a2.value(),
                         "Pre1": self.spb_s5_pre1.value(),
                         "Post1": self.spb_s5_post1.value()}
        self.setting6 = {"D1": self.spb_s6_d1.value(),
                         "D2": self.spb_s6_d1.value(),
                         "A1": self.spb_s6_a1.value(),
                         "A2": self.spb_s6_a2.value(),
                         "Pre1": self.spb_s6_pre1.value(),
                         "Post1": self.spb_s6_post1.value()}
        self.setting7 = {"D1": self.spb_s7_d1.value(),
                         "D2": self.spb_s7_d1.value(),
                         "A1": self.spb_s7_a1.value(),
                         "A2": self.spb_s7_a2.value(),
                         "Pre1": self.spb_s7_pre1.value(),
                         "Post1": self.spb_s7_post1.value()}
        self.setting8 = {"D1": self.spb_s8_d1.value(),
                         "D2": self.spb_s8_d1.value(),
                         "A1": self.spb_s8_a1.value(),
                         "A2": self.spb_s8_a2.value(),
                         "Pre1": self.spb_s8_pre1.value(),
                         "Post1": self.spb_s8_post1.value()}

        self.para = {
            "setting1":self.setting1,
            "setting2":self.setting2,
            "setting3":self.setting3,
            "setting4":self.setting4,
            "setting5":self.setting5,
            "setting6":self.setting6,
            "setting7":self.setting7,
            "setting8":self.setting8}
        return self.para