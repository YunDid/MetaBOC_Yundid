# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

import os
import numpy as np

from operator import index
from pickle import FALSE
from tkinter.messagebox import NO
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from src.Ui_StimulateSetting import Ui_StimulateDialog
from src.SignalWidget import SignalWidget
from src.INTAN_Sti_Setting import INTANSignalWidget
from src.SignalButton import SignalButtonPara

from src.robot.task import SYSTEM_DEVICE
# from src.electrode_intan import Electrode_Intan_Widget


class StimulateSettingDialog(QDialog, Ui_StimulateDialog):
    close_dialog = pyqtSignal(bool)

    def __init__(self, parent=None, sys_device=None, maxwell_preset=None):
        super(StimulateSettingDialog, self).__init__(parent)
        self.setupUi(self)

        self.sys_device = sys_device
        self.maxwell_preset = maxwell_preset  # Maxwell：预填左右记录/刺激电极（来自设备切换选择）

        self.intan_ele_widget = None
        self.initial_device_ele(self.sys_device)


        self.frame_signal.hide()
        self.frame_para.setStyleSheet(
                "QFrame{background-color: rgb(240, 240, 240);}\n")
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
                "background-color: rgb(240, 140, 140);\n"
                "}\n"
                "")
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
                "background-color: rgb(240, 140, 140);\n"
                "}\n"
                "")
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
                "background-color: rgb(240, 140, 140);\n"
                "}\n"
                "")

        self.pb_button = {"21":self.pb_21,"31":self.pb_31,"41":self.pb_41,
                            "51":self.pb_51,"61":self.pb_61,"71":self.pb_71,

                "12":self.pb_12,"22":self.pb_22,"32":self.pb_32,"42":self.pb_42,
                "52":self.pb_52,"62":self.pb_62,"72":self.pb_72,"82":self.pb_82,

                "13":self.pb_13,"23":self.pb_23,"33":self.pb_33,"43":self.pb_43,
                "53":self.pb_53,"63":self.pb_63,"73":self.pb_73,"83":self.pb_83,
                
                "14":self.pb_14,"24":self.pb_24,"34":self.pb_34,"44":self.pb_44,
                "54":self.pb_54,"64":self.pb_64,"74":self.pb_74,"84":self.pb_84,
                
                "25":self.pb_25,"35":self.pb_35,"45":self.pb_45,"55":self.pb_55,
                "65":self.pb_65,"75":self.pb_75,"85":self.pb_85,
                
                "16":self.pb_16,"26":self.pb_26,"36":self.pb_36,"46":self.pb_46,
                "56":self.pb_56,"66":self.pb_66,"76":self.pb_76,"86":self.pb_86,
                
                "17":self.pb_17,"27":self.pb_27,"37":self.pb_37,"47":self.pb_47,
                "57":self.pb_57,"67":self.pb_67,"77":self.pb_77,"87":self.pb_87,
                
                "28":self.pb_28,"38":self.pb_38,"48":self.pb_48,
                "58":self.pb_58,"68":self.pb_68,"78":self.pb_78,
                
                # for intan
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
                "B-030":self.pb_030_b, "B-031":self.pb_031_b,

                }

        # 用于记录电极
        self.recording_left_list = []
        self.recording_right_list = []
        self.stimulating_left_list = []
        self.stimulating_right_list = []

        # Maxwell：用设备切换时选好的电极预填四个缓冲（图1 不在网格选电极，靠这个带进 npz）
        if self.maxwell_preset is not None:
            rec = self.maxwell_preset.get("recording_list", [[], []])
            stim = self.maxwell_preset.get("stimulating_list", [[], []])
            self.recording_left_list = list(rec[0]) if len(rec) > 0 else []
            self.recording_right_list = list(rec[1]) if len(rec) > 1 else []
            self.stimulating_left_list = list(stim[0]) if len(stim) > 0 else []
            self.stimulating_right_list = list(stim[1]) if len(stim) > 1 else []

        self.bg_record_sti = QButtonGroup(self)
        self.bg_record_sti.addButton(self.radioButton_recording_left)
        self.bg_record_sti.addButton(self.radioButton_recording_right)
        self.bg_record_sti.addButton(self.radioButton_stimulating_left)
        self.bg_record_sti.addButton(self.radioButton_stimulating_right)
        self.bg_record_sti.buttonClicked.connect(self.record_sti_change)

        self.button_group = QButtonGroup(self)
        for key in self.pb_button.keys():
            self.pb_button[key].setCheckable(True)
            self.button_group.addButton(self.pb_button[key])
        
        self.button_group.buttonToggled.connect(self.push_button_clicked)
        self.button_group.setExclusive(False)

        self.pb_ok.clicked.connect(self.ok)
        self.pb_cancel.clicked.connect(self.cancel)

        self.pb_add_sti.clicked.connect(self.add_stimulating_signal)

        self.save_stimulating_state = False

        self.signal_widget = None
        self.intan_signal_widget = None
        self.signal_hbox = QHBoxLayout(self.frame_signal)
        self.signal_hbox.setObjectName("signal_hbox")
        self.signal_hbox.setContentsMargins(5, 0, 5, 0)
        spacerItem = QSpacerItem(20, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.signal_hbox.addSpacerItem(spacerItem)

        self.button_group = QButtonGroup(self)
        self.button_group.buttonClicked.connect(self.modify_para)
        self.button_index = None    # 点击的button id

        self.setWindowModality(Qt.WindowModality.ApplicationModal)

        self.spb_repeat.valueChanged.connect(self.repeat_changed)
        self.spb_repeat.setEnabled(False)
        self.lineEdit_sti_name.textEdited.connect(self.sti_name_changed)


    def initial_device_ele(self, dev):
        
        if dev == SYSTEM_DEVICE.MEA2100:
            self.frame_intan.hide()
            self.frame_intan_b.hide()
            self.frame_array.show()
            self.resize(488, 700)
        elif dev == SYSTEM_DEVICE.MAXWELL:
            # Maxwell HD-MEA：无 8x8 物理网格、无 Intan 双列。记录电极由 cfg routing 决定、
            # 刺激电极在设备切换时已注入，图1 在 Maxwell 下只编辑「双相脉冲参数」（复用 MCS
            # SignalWidget），不在此选电极 → 隐藏全部电极网格。
            self.frame_intan.hide()
            self.frame_intan_b.hide()
            self.frame_array.hide()
            self.resize(488, 600)
        else:
            self.frame_intan.show()
            self.frame_intan_b.show()
            self.frame_array.hide()
            self.resize(488, 600)



    def push_button_clicked(self, button):
        text = button.text()
        state = button.isChecked()

        if text in self.recording_left_list and not self.radioButton_recording_left.isChecked():
            self.recording_left_list.remove(text)
        if text in self.recording_right_list and not self.radioButton_recording_right.isChecked():
            self.recording_right_list.remove(text)
        if text in self.stimulating_left_list and not self.radioButton_stimulating_left.isChecked():
            self.stimulating_left_list.remove(text)
        if text in self.stimulating_right_list and not self.radioButton_stimulating_right.isChecked():
            self.stimulating_right_list.remove(text)

        # 根据选择的模式，设置相应的电极，如已被设置，需及时判断更改
        if self.radioButton_recording_left.isChecked():
            self.recording_left_list.append(text)
            button.setStyleSheet(                # 浅绿 (173, 255, 172)     # 深绿 (73, 227, 155)
                "QPushButton{\n"                 # 浅红 (255, 184, 184)     # 深红 (237, 86, 86) 
                "border-radius:25px;\n"          # pressed: 绿 (117, 245, 185);  红 (240, 140, 140)
                "background-color: rgb(255, 184, 184);\n"
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
                "background-color: rgb(240, 140, 140);\n"
                "}\n")
        elif self.radioButton_recording_right.isChecked():
            self.recording_right_list.append(text)

            button.setStyleSheet(
                "QPushButton{\n"
                "border-radius:25px;\n"
                "background-color: rgb(237, 86, 86);\n"
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
                "background-color: rgb(240, 140, 140);\n"
                "}\n")
        elif self.radioButton_stimulating_left.isChecked():
            self.stimulating_left_list.append(text)
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
                "}\n"
                "\n"
                "QPushButton:pressed{\n"
                "color:black;\n"
                "border-radius:25px;\n"
                "background-color: rgb(117, 245, 185);\n"
                "}\n")
        elif self.radioButton_stimulating_right.isChecked():
            self.stimulating_right_list.append(text)
            button.setStyleSheet(
                "QPushButton{\n"
                "border-radius:25px;\n"
                "background-color: rgb(73, 227, 155);\n"
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
                "background-color: rgb(117, 245, 185);\n"
                "}\n")
    
    # recording 和 stimulating设置切换
    def record_sti_change(self, button):
        if button.text() == "Recording Left":
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
                "background-color: rgb(240, 140, 140);\n"
                "}\n"
                "")
        elif button.text() == "Recording Right":
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
                    "background-color: rgb(240, 140, 140);\n"
                    "}\n"
                    "")
        elif button.text() == "Stimulating Left":
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
                    "background-color: rgb(117, 245, 185);\n"
                    "}\n"
                    "")
        elif button.text() == "Stimulating Right":
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
                    "background-color: rgb(117, 245, 185);\n"
                    "}\n"
                    "")

    def ok(self):
        name = self.lineEdit_sti_name.text()

        if name == "":
            self.lineEdit_sti_name.setStyleSheet("border: 1px solid red;")
            QMessageBox.information(self, "Warning", "Stimulate Name Should Not be None!")
            return

        # # 保存结果至 txt
        # save_path_record = "./out/" + name + "_record.txt"
        # r_file = open(save_path_record, "w")
        # for i in range(len(self.recording_list)):
        #     text = self.recording_list[i] + ","
        #     r_file.write(text)
        # r_file.close()

        # save_path_sti = "./out/" + name + "_sti.txt"
        # s_file = open(save_path_sti, "w")
        # for i in range(len(self.stimulating_list)):
        #     text = self.stimulating_list[i] + ","
        #     s_file.write(text)
        # s_file.close()

        state = self.save_para_to_npz(name)
        if state:
            self.save_stimulating_state = True
            self.close()

    def cancel(self):
        self.close()

    def closeEvent(self, a0: QCloseEvent) -> None:
        self.close_dialog.emit(True)
        return super().closeEvent(a0)

    def save_para_to_npz(self, name):
        if self.sys_device == SYSTEM_DEVICE.MEA2100:
            out_name = os.path.join("./out/MEA2100", name + ".npz")
        elif self.sys_device == SYSTEM_DEVICE.INTAN:
            out_name = os.path.join("./out/INTAN", name + ".npz")
        elif self.sys_device == SYSTEM_DEVICE.MAXWELL:
            os.makedirs("./out/MAXWELL", exist_ok=True)  # 首次保存时目录可能不存在
            out_name = os.path.join("./out/MAXWELL", name + ".npz")

        if os.path.exists(out_name):
            self.lineEdit_sti_name.setStyleSheet("border: 1px solid red;")
            QMessageBox.information(self,'Warning','There already have the same file, please set another stimulate name again!')
            return False

        para, sti_para, recording_list, stimulating_list = self.get_all_para()

        np.savez(out_name, para=para, sti_para=[sti_para], recording_left_list=recording_list[0], 
            recording_right_list=recording_list[1], stimulating_left_list=stimulating_list[0],
            stimulating_right_list=stimulating_list[1])
        
        return True

    def get_all_para(self):
        out = []
        button = self.button_group.buttons()
        for i in range(len(button)):
            para = button[i].get_para()
            out.append(para)
        
        sti_para = {"stimulate_name":self.lineEdit_sti_name.text(),
            "reapeat_times":self.spb_repeat.value(),
            "total_time":self.spb_total_time.value()}

        recording_list = [self.recording_left_list, self.recording_right_list]
        stimulating_list = [self.stimulating_left_list, self.stimulating_right_list]
        return out, sti_para, recording_list, stimulating_list


    def add_stimulating_signal(self):
        # Maxwell 复用 MCS 双相脉冲参数 UI（stimulation_maxwell 按 MCS sti_para schema 移植）
        if self.sys_device in (SYSTEM_DEVICE.MEA2100, SYSTEM_DEVICE.MAXWELL):
            self.signal_widget = SignalWidget()
            self.signal_widget.save_sti_para.connect(self.save_stimulating_para)
            self.signal_widget.close_widget.connect(self.close_signal_widget)
            self.signal_widget.show()
        else:
            self.intan_signal_widget = INTANSignalWidget()
            self.intan_signal_widget.save_sti_para.connect(self.save_stimulating_para_intan)
            self.intan_signal_widget.close_widget.connect(self.close_signal_widget_intan)
            self.intan_signal_widget.show()

    def save_stimulating_para_intan(self, state):
        if state == 1:  # 新建参数，添加相应按钮
            signal_button = SignalButtonPara(self.frame_signal)
            signal_button.set_para(self.intan_signal_widget.get_para())
            signal_button.show()

            self.button_group.addButton(signal_button)
            self.signal_hbox.addWidget(signal_button)
            self.frame_signal.show()
            # self.spb_repeat.setEnabled(True)
            self.repeat_changed(0)    # 更新一下总的时间
        elif state == 2:    # 不新建参数
            self.button_group.button(self.button_index).set_para(self.intan_signal_widget.get_para())
        elif state == 0:
            pass

    def close_signal_widget_intan(self):
        if self.intan_signal_widget is not None:
            self.intan_signal_widget.save_sti_para.disconnect(self.save_stimulating_para_intan)
            self.intan_signal_widget.close_widget.disconnect(self.close_signal_widget_intan)
            del self.intan_signal_widget
            self.intan_signal_widget = None

    def close_signal_widget(self):
        if self.signal_widget is not None:
            self.signal_widget.save_sti_para.disconnect(self.save_stimulating_para)
            self.signal_widget.close_widget.disconnect(self.close_signal_widget)
            del self.signal_widget
            self.signal_widget = None


    # 保存设置的刺激信号参数
    def save_stimulating_para(self, state):

        if state == 1:  # 新建参数，添加相应按钮
            signal_button = SignalButtonPara(self.frame_signal)
            signal_button.set_para(self.signal_widget.get_para())
            signal_button.show()

            self.button_group.addButton(signal_button)
            self.signal_hbox.addWidget(signal_button)
            self.frame_signal.show()
            self.spb_repeat.setEnabled(True)
            self.repeat_changed(0)    # 更新一下总的时间
        elif state == 2:    # 不新建参数
            self.button_group.button(self.button_index).set_para(self.signal_widget.get_para())
        elif state == 0:
            pass
    

    # 点击pushbutton, 再次编辑
    def modify_para(self, value):
        para = value.get_para()
        self.button_index = self.button_group.id(value)
        if self.signal_widget is None:
            self.signal_widget = SignalWidget(para=para)
            self.signal_widget.save_sti_para.connect(self.save_stimulating_para)
            self.signal_widget.close_widget.connect(self.close_signal_widget)
            self.signal_widget.show()
        
    
    # 单元组信号重复次数变化，需重新计算总时间
    def repeat_changed(self, value):
        para, sti_para, recording_list, stimulating_list = self.get_all_para()
        x_global = 0; 
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
                    x_global = x_global + duration_1

                    x_global = x_global + duration_2

                    x_global = x_global + duration_3
                   
                    x_global = x_global + ISI*1000   # 单个方波间隔

        self.spb_total_time.setValue(int(x_global/1000))
    
    def sti_name_changed(self):
        self.lineEdit_sti_name.setStyleSheet("border: 1px solid grey;")