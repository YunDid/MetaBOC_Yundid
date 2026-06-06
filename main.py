# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

# pyrcc5 -o res.py res.qrc
# pip install PyQt5 -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# pip install PyQt5-tools -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# pip install pythonnet -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# pip install pandas -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# pip install matplotlib -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# pip install scipy -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# pip install pyQtChart -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# pip install pyqtgraph -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# pip install h5py -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# pip install pytorch_lightning -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# pip install scikit-learn -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# pip install seaborn -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# pip install pyOpenGL  -i https://pypi.douban.com/simple
# pip install casadi  -i https://pypi.douban.com/simple
# pip install do_mpc -i https://pypi.douban.com/simple
# pip install watchdog

# pip install -r requirements.txt

# for sys
from ast import Not
import sys
import os
import math
from tkinter.messagebox import NO
import time

# for src
from src.Ui_MainWindow import Ui_MainWindow
from src.ImageWidget import ImageWidget
from src.StimulateSetting import StimulateSettingDialog
from src.SelectStimulatingPara import SelectStimulatingPara
from src.ChannelSignalDialog import ChannelSignalDialog
from src.IntanChannelSignalDialog import INTANChannelSignalDialog
from src.SpikeDetectionSetting import SpikeDetectionSetting
from src.StimulateDialog import StimulateDialog
from src.DynamicDialog import DynamicDialog
from src.StiSpikesSignalDataDialog import StiSpikesSignalDataDialog
from src.grasp_distance_dialog import GraspDistanceDialog
from src.Arm3DPostionWidget import Arm3DPosWidget

# 导入字体配置
from font_config import get_global_font_style, create_global_font

from src.robot.task import TASK, MAP, GRASP_STATE, GRASP_CONTROL, SYSTEM_DEVICE

# for pyqt
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets
from PyQt5 import QtCore



__appname__ = 'Meta-BOC'      # Brain Organoid for Controlling with Hybrid Intelligence Platform  人工脑智能复合体信息交互平台
# __appname__ = 'Brain-Chip'      # Brain Organoid for Controlling with Hybrid Intelligence Platform  人工脑智能复合体信息交互平台
# __appname__ = 'Brain-IIIP'    # Artificial Brain Intelligence Information Interaction Platform  人工脑智能复合体信息交互平台

# git add .
# git commit -m "xx"
# git push -u origin main

class MainWindowClass(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindowClass, self).__init__(parent)
        self.setupUi(self)
        self.setWindowTitle(__appname__)
        # Linux 下 showFullScreen 会隐藏标题栏，影响 GUI 操作；只在 Windows 启用全屏
        from src.platform_config import IS_WINDOWS
        if IS_WINDOWS:
            self.showFullScreen()
        # test
        # self.setWindowIcon(QIcon(":/icons/resources/fineLabel.png"))
        self.setWindowState(Qt.WindowMaximized)
        self.setStyleSheet("background-color: rgb(248, 248, 248);")
        
        # 设置全局字体样式
        self.setGlobalFont()
        
        print('ss')
        self.Initial()
        # 创建一个定时器用于更新机械臂位置

    def Initial(self):
        self.imageWidget = ImageWidget(self.frame)
        self.gridLayoutFrame = QtWidgets.QGridLayout(self.frame)
        self.gridLayoutFrame.setObjectName("gridLayoutFrame")
        self.gridLayoutFrame.addWidget(self.imageWidget, 0, 0, 1, 1)
        self.gridLayoutFrame.setContentsMargins(0, 0, 0, 0)

        self.timer_run = None
        self.update_time = 100  # ms
        self.pb_stop.setEnabled(False)

        self.pb_start.setStyleSheet("background-color: None;border-color: rgb(170, 255, 255);")
        self.pb_stop.setStyleSheet("background-color: None;border-color: rgb(170, 255, 255);")
        self.pb_reset.setStyleSheet("background-color: None;border-color: rgb(170, 255, 255);")
        self.pb_reset_all.setStyleSheet("background-color: None;border-color: rgb(170, 255, 255);")
        self.comboBox_run_direction.setStyleSheet("background-color: None;border-color: rgb(170, 255, 255);")

        # for connect
        self.dsb_x.valueChanged.connect(self.x_position_change)
        self.dsb_y.valueChanged.connect(self.y_position_change)

        self.dsb_left_wheel_fre.valueChanged.connect(self.left_wheel_frequency_change)
        self.dsb_right_wheel_fre.valueChanged.connect(self.right_wheel_frequency_change)
        self.imageWidget.mea_control.connect(self.update_wheel_fre)    # MEA控制robot轮速，更新显示的轮速

        self.pb_start.clicked.connect(self.start_event)
        self.pb_stop.clicked.connect(self.stop_event)
        self.pb_reset.clicked.connect(self.reset_event)

        self.comboBox_run_direction.currentIndexChanged.connect(self.change_robot_direction)

        self.button_group = QButtonGroup(self)
        self.button_group.addButton(self.radioButton_human)
        self.button_group.addButton(self.radioButton_mea)
        self.button_group.addButton(self.radioButton_human_real)
        self.button_group.addButton(self.radioButton_real_control)
        self.button_group.buttonClicked.connect(self.update_control_mode)

        # 在设置电极之后，才可以切换mea control
        self.radioButton_mea.setEnabled(False)
        self.radioButton_real_control.setEnabled(False)    # 连接外部设备成功之后，才能够切换

        self.imageWidget.velocity.connect(self.velocity_change)
        self.imageWidget.robot_angles.connect(self.angles_change)
        self.imageWidget.left_distance.connect(self.distance_left)
        self.imageWidget.right_distance.connect(self.distance_right)
        self.imageWidget.current_pos.connect(self.robot_pos)
        self.imageWidget.pos_left.connect(self.left_pos)
        self.imageWidget.pos_right.connect(self.right_pos)
        self.imageWidget.outof_border.connect(self.out_of_border)
        self.imageWidget.left_wheel_hit.connect(self.left_wheel_hits)
        self.imageWidget.right_wheel_hit.connect(self.right_wheel_hits)
        self.imageWidget.real_robot_connect_server.connect(self.update_robot_connect_state)    # 真实机器人连接状态
        self.imageWidget.cell_state.connect(self.update_cell_state)

        self.imageWidget.zoomChanged.connect(self.ZoomChanged)
        self.zoomValue = 1.0                #缩放比例

        # for robot training and testing
        self.spb_train_time.setEnabled(False)
        self.spb_test_time.setEnabled(True)
        self.start_time = 0    # 用于记录训练或测试的起始时间
        self.all_time = 0      # 总累计时间（中间可能暂停）
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.ResetStatusBar)  # 计时结束调用operate()方法

        self.button_group_tt = QButtonGroup(self)
        self.button_group_tt.addButton(self.rdb_train_mode)
        self.button_group_tt.addButton(self.rdb_test_mode)
        self.button_group_tt.buttonClicked.connect(self.update_train_test_mode)
        self.pb_reset_all.clicked.connect(self.reset_all)
        self.rdb_train_mode.setChecked(True)

        self.button_group_dynamic_model = QButtonGroup(self)
        self.button_group_dynamic_model.addButton(self.rdb_dynamic_model)
        self.button_group_dynamic_model.addButton(self.rdb_non_dynamic_model)
        self.button_group_dynamic_model.buttonClicked.connect(self.update_dynamic_model_mode)
        self.rdb_non_dynamic_model.setChecked(True)

        self.stimulate_setting_dialog = None
        self.stimulate_select_dialog = SelectStimulatingPara(self)
        self.stimulate_select_dialog.select_sti_para.connect(self.update_show_sti_para)
        self.stimulate_select_dialog.hide()

        self.channel_signal_dialog = None    # 用于MEA2100的信号显示
        self.INTAN_channel_signal_dialog = None    # 用于INTAN系统的信号显示
        self.stimulate_dialog = None
        self.dynamic_dialog = None
        self.spike_detection_setting_widget = None
        self.spike_para = {"Sample_Rate":25000,
            "Filter_Order":4,
            "Min_Frequency":300,
            "Max_Frequency":3000,
            "Multiplier":5,
            "Refractor_Time":2} # 用来保存spike detection的超参数
        self.imageWidget.mea_ic.set_spike_detection_para(self.spike_para)

        # for action
        self.actionExit.triggered.connect(self.exit)
        


        self.actionAdd_Stimulating.triggered.connect(self.add_stimulating)
        self.actionSelect_Stimulating.triggered.connect(self.select_stimulating)
        self.actionStimulate.triggered.connect(self.do_stimulate_dialog)

        self.actionShow_Signal.triggered.connect(self.show_channel_signal)

        self.actionSpike_Detection_Setting.triggered.connect(self.spike_detection_setting)

        self.actionEncode_Decode_Setting.triggered.connect(self.encoding_decoding)

        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        # for export data
        self.actionSave_Raw_Data.triggered.connect(self.export_data_setting)
        self.actionSave_State.triggered.connect(self.save_state)

        # for dynamic model
        self.actionDynamic_Model.triggered.connect(self.dynamic_create_dialog)

        # for real robot connect
        self.lineEdit_ip_adress.setText("192.168.4.1")
        self.lineEdit_port.setText("9000")
        self.lineEdit_ip_adress.editingFinished.connect(self.update_ip_port)
        self.lineEdit_port.editingFinished.connect(self.update_ip_port)
        self.pb_connect.setStyleSheet("background-color: None;border-color: rgb(170, 255, 255);")
        self.label_state.setStyleSheet("background-color:rgb(225, 225, 225);")

        # for cell state
        self.label_cell_state.setStyleSheet("background-color: rgb(225, 225, 225);")
  
        # for robot obstacle
        self.robot_sti_spike_dialog = StiSpikesSignalDataDialog(self, self.imageWidget.mea_ic)
        self.robot_sti_spike_dialog.hide()
        self.actionShow_Stimulus_Spike_Signal.triggered.connect(self.show_sti_spike_robot)    # 可视化显示机器人避障的刺激信号和spike控制信号

        self.actionSave_Spikes.triggered.connect(self.save_spikes_data)

        # MaxOne 原生记录（mx.Saving）：Export 菜单第 4 项，设备隔离、仅 Maxwell 生效。
        # 勾选时选目录 + 通道范围；实际开录绑定到运行生命周期（start_event 开、Stop/超时停）。
        # 与 C++ 探头独立；MCS/INTAN 下整条逻辑为安全空操作（hasattr 守卫）。
        self.actionMaxOne_Recording = QAction(self)
        self.actionMaxOne_Recording.setCheckable(True)
        self.actionMaxOne_Recording.setObjectName("actionMaxOne_Recording")
        self.actionMaxOne_Recording.setText("MaxOne Recording")
        self.menuExport.addAction(self.actionMaxOne_Recording)
        self.actionMaxOne_Recording.toggled.connect(self.toggle_maxone_recording)
        self.maxone_rec_enabled = False
        self.maxone_rec_dir = None
        self.maxone_rec_all_channels = True   # True=全 1024 通道；False=仅在用电极通道
        # 手动 Stop / Reset / Reset All 都收尾原生记录（超时停在 update_run 内处理）；
        # 出界自动 reset 走 out_of_border 信号、不经这些按钮，故不停录 → 单次 run 一个文件。
        self.pb_stop.clicked.connect(self._maxone_record_stop)
        self.pb_reset.clicked.connect(self._maxone_record_stop)
        self.pb_reset_all.clicked.connect(self._maxone_record_stop)

        # for grasp view
        self.grasp_distance_signal_dialog = GraspDistanceDialog(self)
        self.grasp_distance_signal_dialog.hide()
        self.arm_pos_3d_widget = Arm3DPosWidget(self)
        self.arm_pos_3d_widget.hide()
        self.imageWidget.set_arm_pos_3d_widget(self.arm_pos_3d_widget)
        


        self.actionShow_Grasp_Distance.triggered.connect(self.show_grasp_distance)
        self.actionShow_Arm_Pos_in_3D_Space.triggered.connect(self.show_arm_pos_3D)

        # 机械臂的可视化，在主窗口进行
        self.gridLayoutFrame.addWidget(self.arm_pos_3d_widget, 0, 0, 1, 1)

        # for task mode changing
        self.task_mode_group = QActionGroup(self)
        self.task_mode_group.addAction(self.actionObstacle_Avoidance)
        self.task_mode_group.addAction(self.actionObject_Tracking)
        self.task_mode_group.addAction(self.actionObject_Grasping)
        self.task_mode_group.triggered.connect(self.task_mode_changed)

        self.button_group_task = QButtonGroup(self)
        self.button_group_task.addButton(self.rdb_avoidance)
        self.button_group_task.addButton(self.rdb_tracking)
        self.button_group_task.addButton(self.rdb_grasping)
        self.button_group_task.buttonClicked.connect(self.task_mode_changed_bt)
        self.rdb_avoidance.setChecked(True)

        # for map changing
        self.map_mode_group = QActionGroup(self)
        self.map_mode_group.addAction(self.actionEmpty_Map)
        self.map_mode_group.addAction(self.actionRandom_Map)
        self.map_mode_group.addAction(self.actionHuman_Map)
        self.map_mode_group.addAction(self.actionRegular_Map)
        self.map_mode_group.triggered.connect(self.map_mode_changed)

        self.update_visual_item()  # 初始不显示机械臂角度

        self.dsb_robot_arm_angle.valueChanged.connect(self.robot_arm_angle_human)

        # 切换连接的系统
        # Phase B：手补 Maxwell 菜单项（不动 .ui，不重生成 Ui_MainWindow.py）。
        # Phase C 后续若把 .ui 规范化，这段可以拿掉。
        self.actionMaxwell = QAction(self)
        self.actionMaxwell.setCheckable(True)
        self.actionMaxwell.setObjectName("actionMaxwell")
        self.actionMaxwell.setText("Maxwell MaxOne")
        self.menuSystem.addAction(self.actionMaxwell)

        self.action_group_dev = QActionGroup(self)
        self.action_group_dev.addAction(self.actionMEA_2100)
        self.action_group_dev.addAction(self.actionINTAN_System)
        self.action_group_dev.addAction(self.actionMaxwell)
        self.action_group_dev.triggered.connect(self.device_changed)

        self.actionSet_INTAN_Data_Path.triggered.connect(self.set_intan_data_path)
        self.INTAN_data_path = None    # 需先选择数据保存路径

        # Maxwell 会话参数缓存（Phase B：用 QFileDialog 让用户选 cfg）
        self.maxwell_cfg_path = None
        self.maxwell_recording_list = None  # Stage 2：设备切换时选的左右记录电极 [left_ids, right_ids]
        self.maxwell_stim_lr = None         # 设备切换时选的左右轮刺激电极 [[left],[right]]（字符串）


    def exit(self):
        self.close()

    def closeEvent(self, a0: QCloseEvent) -> None:
        self._maxone_record_stop()  # 关闭前收尾 MaxOne 记录文件（若在录）
        self.imageWidget.close()
        del self.imageWidget
        self.imageWidget = None

        return super().closeEvent(a0)


    def x_position_change(self, value):
        self.imageWidget.update_position_x(value)
        self.update()

    def y_position_change(self, value):
        self.imageWidget.update_position_y(value)
        self.update()

    def left_wheel_frequency_change(self, value):
        self.imageWidget.update_wheel_fre_left(value, self.update_time)

    def right_wheel_frequency_change(self, value):
        self.imageWidget.update_wheel_fre_right(value, self.update_time)
    
    def distance_left(self, value):
        self.dsb_distance_left.setValue(value)

    def distance_right(self, value):
        self.dsb_distance_right.setValue(value)

    def velocity_change(self, value):
        self.dsb_velocity.setValue(value)
    
    def angles_change(self, value):
        va = value % (2.0 * math.pi)
        self.dsb_angles.setValue(va * 180.0 / math.pi)
    
    def update_control_mode(self):
        if self.radioButton_human.isChecked() or self.radioButton_human_real.isChecked():
            self.dsb_left_wheel_fre.setEnabled(True)
            self.dsb_right_wheel_fre.setEnabled(True)
        else:
            self.dsb_left_wheel_fre.setEnabled(False)
            self.dsb_right_wheel_fre.setEnabled(False)
        
        # 控制模式切换：人为和MEA虚拟环境、真实电子设备
        if self.radioButton_human.isChecked() or self.radioButton_mea.isChecked():
            self.imageWidget.initial_human()
        if self.radioButton_real_control.isChecked() or self.radioButton_human_real.isChecked():
            self.imageWidget.initial_real_robot(self.lineEdit_ip_adress.text(), self.lineEdit_port.text())

    def robot_pos(self, x, y):
        self.dsb_x.setValue(x)
        self.dsb_y.setValue(y)
    
    def left_pos(self, x, y):
        self.dsb_left_x.setValue(x)
        self.dsb_left_y.setValue(y)
    
    def right_pos(self, x, y):
        self.dsb_right_x.setValue(x)
        self.dsb_right_y.setValue(y)

    def start_event(self):
        # MaxOne 原生记录：随实验开录（须在闭环发第一个刺激前 open，刺激 mx.Event 标识才会
        # 写进文件）。幂等——出界自动 reset→start_event 不会重开文件。
        self._maxone_record_start()
        if self.timer_run is not None:
            del self.timer_run
            self.timer_run = None
        self.timer_run = QTimer(self)
        self.timer_run.timeout.connect(self.update_run)
        self.timer_run.start(self.update_time)
        self.pb_start.setEnabled(False)
        self.pb_stop.setEnabled(True)
        self.rdb_test_mode.setEnabled(False)
        self.rdb_train_mode.setEnabled(False)

        self.current_time = time.time()


    def stop_event(self):
        if self.timer_run is not None:
            self.timer_run.timeout.disconnect(self.update_run)
            self.timer_run = None
            self.pb_stop.setEnabled(False)
            self.pb_start.setEnabled(True)
            self.rdb_test_mode.setEnabled(True)
            self.rdb_train_mode.setEnabled(True)
            self.all_time = self.all_time + time.time() - self.current_time

            # 停止外部机器人的运动
            self.imageWidget.stop_robot_move()
            # 最后保存机器人的路径点，每100ms记录一次位置
            self.imageWidget.robot.save_points()

            if self.actionObject_Tracking.isChecked():
                times = time.localtime(time.time())
                hour = times.tm_hour
                half_hour = times.tm_min // 30  # 0 for 00-29, 1 for 30-59
                name = f"{times.tm_year}_{times.tm_mon}_{times.tm_mday}_{hour}_{half_hour}"
                self.imageWidget.tracking.save_tracking_distance_txt(os.path.join("./out_tracking", name + ".txt"))

            if self.actionObject_Grasping.isChecked():
                times = time.localtime(time.time())
                hour = times.tm_hour
                half_hour = times.tm_min // 30  # 0 for 00-29, 1 for 30-59
                name = f"{times.tm_year}_{times.tm_mon}_{times.tm_mday}_{hour}_{half_hour}"
                self.imageWidget.grasping.save_grabing_txt(os.path.join("./out_grasping", name + ".txt"))
                self.imageWidget.grasping.update_cur_state(GRASP_STATE.Grasping)

    def out_of_border(self):
        self.reset_event()
        self.start_event()

    def reset_event(self):
        self.stop_event()

        # reset parameter
        self.imageWidget.reset_state()

        self.dsb_x.setValue(self.imageWidget.robot.x)
        self.dsb_y.setValue(self.imageWidget.robot.y)
        self.dsb_angles.setValue(self.imageWidget.robot.angles)
        self.dsb_velocity.setValue(self.imageWidget.robot.center_velocity)
        self.dsb_distance_left.setValue(0)
        self.dsb_distance_right.setValue(0)
        self.dsb_left_x.setValue(self.imageWidget.robot.wheel_lux)
        self.dsb_left_y.setValue(self.imageWidget.robot.wheel_luy)
        self.dsb_right_x.setValue(self.imageWidget.robot.wheel_rux)
        self.dsb_right_y.setValue(self.imageWidget.robot.wheel_ruy)
        self.dsb_left_wheel_fre.setValue(self.imageWidget.robot.w_left)
        self.dsb_right_wheel_fre.setValue(self.imageWidget.robot.w_right)

    def reset_all(self):
        self.reset_event()

        self.all_time = 0
        self.dsb_current_time.setValue(self.all_time)

        self.imageWidget.robot.hits = 0
        self.spb_number_hints.setValue(self.imageWidget.robot.hits)
        self.spb_distance_tracking.setValue(int(self.imageWidget.tracking.get_mean_distance() or 0))
        self.imageWidget.robot.points_his.clear()


    def update_run(self):
        v = self.dsb_velocity.value()
        left_fre = self.dsb_left_wheel_fre.value()
        right_fre = self.dsb_right_wheel_fre.value()
        if self.radioButton_human.isChecked():
            self.imageWidget.update_robot(left_fre, right_fre, self.update_time, 0)   # 0为人为控制
        elif self.radioButton_mea.isChecked():
            self.imageWidget.update_robot(left_fre, right_fre, self.update_time, 1)   # 1为mea控制虚拟环境
        elif self.radioButton_human_real.isChecked():
            self.imageWidget.update_robot(left_fre, right_fre, self.update_time, 2)   # 2为人为控制真实环境
        elif self.radioButton_real_control.isChecked():
            self.imageWidget.update_robot(left_fre, right_fre, self.update_time, 3)   # 3为mea控制真实环境
        if self.timer_run is not None:
            self.timer_run.start(self.update_time)

        tim = (self.all_time + time.time() - self.current_time) / 60
        self.dsb_current_time.setValue(tim)
        self.spb_number_hints.setValue(self.imageWidget.robot.hits)
        self.spb_distance_tracking.setValue(int(self.imageWidget.tracking.get_mean_distance() or 0))

        self.update_arm_visual()

        if self.rdb_test_mode.isChecked():
            if tim >= self.spb_test_time.value(): # 超时则停止
                self._maxone_record_stop()
                self.stop_event()
        else:
            if tim >= self.spb_train_time.value():
                self._maxone_record_stop()
                self.stop_event()

    def update_arm_visual(self):
        if self.actionObject_Grasping.isChecked():
            if self.arm_pos_3d_widget is not None:
                arm_data, control_count, obr,direction = self.imageWidget.get_arm_data_from_real_robot()

                self.arm_pos_3d_widget.update_arm_pos(arm_data, control_count, obr, direction)
    
    def update_wheel_fre(self, left, right):
        # 直接控制为实际的轮速
        self.dsb_left_wheel_fre.setValue(left)
        self.dsb_right_wheel_fre.setValue(right)

        # # MEA 增量控制轮速
        # l = self.dsb_left_wheel_fre.value()
        # r = self.dsb_right_wheel_fre.value()
        # self.dsb_left_wheel_fre.setValue(left + l)
        # self.dsb_right_wheel_fre.setValue(right + r)


    # ************************ for action **************************
    def add_stimulating(self):
        if self.stimulate_setting_dialog is None:
            if self.actionMEA_2100.isChecked():
                sys_ = SYSTEM_DEVICE.MEA2100
            elif self.actionINTAN_System.isChecked():
                sys_ = SYSTEM_DEVICE.INTAN
            elif self.actionMaxwell.isChecked():
                # Maxwell：图1 复用 MCS 脉冲参数 UI（隐藏物理网格），编辑双相脉冲刺激参数
                sys_ = SYSTEM_DEVICE.MAXWELL

            # Maxwell：把设备切换时选的左右记录/刺激电极预填进图1，使其随 npz 流到图2 显示
            maxwell_preset = None
            if sys_ == SYSTEM_DEVICE.MAXWELL:
                maxwell_preset = {
                    "recording_list": self.maxwell_recording_list or [[], []],
                    "stimulating_list": self.maxwell_stim_lr or [[], []],
                }
            self.stimulate_setting_dialog = StimulateSettingDialog(self, sys_, maxwell_preset=maxwell_preset)
            self.stimulate_setting_dialog.close_dialog.connect(self.close_sti_setting_dialog)
            self.stimulate_setting_dialog.show()

    def close_sti_setting_dialog(self):
        if self.stimulate_setting_dialog is not None:
            if self.stimulate_setting_dialog.save_stimulating_state:
                para, sti_para, recording_list, stimulating_list = self.stimulate_setting_dialog.get_all_para()
                self.stimulate_select_dialog.add_list(para, sti_para, recording_list, stimulating_list)

            del self.stimulate_setting_dialog
            self.stimulate_setting_dialog = None


    def select_stimulating(self):
        self.stimulate_select_dialog.show()

    def update_show_sti_para(self, state):
        """
        设置了惩罚刺激信号之后，更新任务态模式
        """
        sti_para = self.stimulate_select_dialog.get_selected_stimulating_para()
        # Stage 2：Maxwell 下用设备切换时选的左右记录电极覆盖 sti_para 的 recording_list，
        # 使其随同一 Parameter 流到 set_record_para → get_recording 启用左右分组。
        if self.actionMaxwell.isChecked() and self.maxwell_recording_list is not None:
            sti_para.recording_list = self.maxwell_recording_list
        self.lineEdit_select_sti.setText(sti_para.sti_para["stimulate_name"])
        self.imageWidget.mea_ic.set_stimulating_signal(sti_para)
        self.ShowMessageToStatusBar("Set stimulating parameter successed!...", False)

        self.radioButton_mea.setEnabled(True)
        self.radioButton_real_control.setEnabled(True)

    # 用于显示各通道信号
    def show_channel_signal(self):
        if self.actionMEA_2100.isChecked():
            if self.channel_signal_dialog is None:
                self.channel_signal_dialog = ChannelSignalDialog(self, self.imageWidget.mea_ic.recording)
                self.channel_signal_dialog.close_dialog.connect(self.close_channel_signal_dialog)
                self.channel_signal_dialog.show()
        elif self.actionINTAN_System.isChecked():
            if self.INTAN_channel_signal_dialog is None:
                self.INTAN_channel_signal_dialog = INTANChannelSignalDialog(self, self.imageWidget.mea_ic.recording)
                self.INTAN_channel_signal_dialog.close_dialog.connect(self.close_intan_channel_signal_dialog)
                self.INTAN_channel_signal_dialog.show()
        elif self.actionMaxwell.isChecked():
            QMessageBox.information(
                self,
                "Maxwell stub stage",
                "Maxwell 通道信号可视化需要 C++ binary 数据流（Phase D 计划）。\n"
                "Phase B 阶段不开放本功能。",
            )

    def close_channel_signal_dialog(self):
        if self.channel_signal_dialog is not None:
            del self.channel_signal_dialog
            self.channel_signal_dialog = None


    def close_intan_channel_signal_dialog(self):
        if self.INTAN_channel_signal_dialog is not None:
            del self.INTAN_channel_signal_dialog
            self.INTAN_channel_signal_dialog = None

    # 设置检测spike超参
    def spike_detection_setting(self):
        if self.spike_detection_setting_widget is None:
            if self.spike_para is None:
                self.spike_detection_setting_widget = SpikeDetectionSetting(recording=self.imageWidget.mea_ic.recording)
                self.spike_detection_setting_widget.close_widget.connect(self.close_spike_detection_widget)
                self.spike_detection_setting_widget.show()
            else:
                self.spike_detection_setting_widget = SpikeDetectionSetting(para=self.spike_para, recording=self.imageWidget.mea_ic.recording)
                self.spike_detection_setting_widget.close_widget.connect(self.close_spike_detection_widget)
                self.spike_detection_setting_widget.show()
        else:
            self.spike_detection_setting_widget.show()

    def close_spike_detection_widget(self, state):
        if self.spike_detection_setting_widget is not None:
            if state:
                self.spike_para = self.spike_detection_setting_widget.get_spike_det_para()
                self.imageWidget.mea_ic.set_spike_detection_para(self.spike_para)
                del self.spike_detection_setting_widget
                self.spike_detection_setting_widget = None
            else:
                del self.spike_detection_setting_widget
                self.spike_detection_setting_widget = None


    def do_stimulate_dialog(self):
        if self.stimulate_dialog is None:
            para = self.stimulate_select_dialog.get_selected_stimulating_para()
            self.stimulate_dialog = StimulateDialog(self, para=para, spike_para=self.spike_para, recording=self.imageWidget.mea_ic.recording, stimulating=self.imageWidget.mea_ic.stimulation)
            self.stimulate_dialog.close_widget.connect(self.close_do_stimulate_dialog)
            self.stimulate_dialog.show()

    def close_do_stimulate_dialog(self):
        if self.stimulate_dialog is not None:
            del self.stimulate_dialog
            self.stimulate_dialog = None

    def encoding_decoding(self):
        self.imageWidget.mea_ic.show_encode_decode_setting()

    def ShowMessageToStatusBar(self, message, ifWarn=False):
        self.statusbar.showMessage(message)
        if ifWarn:
            self.statusbar.setStyleSheet("Background:rgb(252, 155, 155)")
            self.timer.start(2000)  # 设置计时间隔并启动
        else:
            self.statusbar.setStyleSheet("Background:rgb(89, 163, 154)")
            self.timer.start(2000)  # 设置计时间隔并启动

    def ResetStatusBar(self):
        self.timer.stop()
        self.statusbar.showMessage("")
        self.statusbar.setStyleSheet("Background:rgb(236, 236, 236)")

    def update_train_test_mode(self):
        if self.rdb_train_mode.isChecked():
            self.spb_train_time.setEnabled(True)
            self.spb_test_time.setEnabled(False)
            self.imageWidget.set_train_test_mode(True)
        else:
            self.spb_train_time.setEnabled(False)
            self.spb_test_time.setEnabled(True)
            self.imageWidget.set_train_test_mode(False)

        self.all_time = 0    # 切换模式，重置时间
        self.dsb_current_time.setValue(self.all_time)
        self.spb_number_hints.setValue(0)
        self.spb_distance_tracking.setValue(-1)

    def update_dynamic_model_mode(self):
        if self.rdb_dynamic_model.isChecked():
            self.imageWidget.set_dynamic_model(True)
        elif self.rdb_non_dynamic_model.isChecked():
            self.imageWidget.set_dynamic_model(False)


    def left_wheel_hits(self, state):
        self.ShowMessageToStatusBar("Left wheel hits!...", ifWarn=True)

    def right_wheel_hits(self, state):
        self.ShowMessageToStatusBar("Right wheel hits!...", ifWarn=True)

    def change_robot_direction(self, value):
        text = self.comboBox_run_direction.currentText()
        if text == "Drive":
            self.imageWidget.robot.set_robot_direction(1)
        else:
            self.imageWidget.robot.set_robot_direction(-1)


    def export_data_setting(self):
        file_path, ok_ = QFileDialog.getSaveFileName(self, "File Save", "C:/", "*.h5", options=QFileDialog.DontUseNativeDialog)

        if file_path:
            path = file_path + ok_[-3:]
            self.imageWidget.mea_ic.recording.set_save_path(path)
        else:
            self.ShowMessageToStatusBar("Please set output file path!...", ifWarn=True)
            return

    def save_state(self):
        state = self.actionSave_State.isChecked()
        if state:
            self.imageWidget.mea_ic.recording.set_save_state(True)
        else:
            self.imageWidget.mea_ic.recording.set_save_state(False)


    # for dynamic model
    def dynamic_create_dialog(self):
        if self.dynamic_dialog is None:
            para = self.stimulate_select_dialog.get_selected_stimulating_para()
            # para = None
            self.dynamic_dialog = DynamicDialog(self, para=para, spike_para=self.spike_para, recording=self.imageWidget.mea_ic.recording, stimulating=self.imageWidget.mea_ic.stimulation)
            self.dynamic_dialog.close_widget.connect(self.close_do_dynamic_dialog)
            self.dynamic_dialog.show()

    def close_do_dynamic_dialog(self):
        if self.dynamic_dialog is not None:
            del self.dynamic_dialog
            self.dynamic_dialog = None
    
    def update_robot_connect_state(self, state):
        if state:
            self.label_state.setStyleSheet("background-color:green")
        else:
            self.label_state.setStyleSheet("background-color:orange")

    def update_cell_state(self, state):
        if state:
            self.label_cell_state.setStyleSheet("background-color:green")
        else:
            self.label_cell_state.setStyleSheet("background-color:orange")

    def update_ip_port(self):
        self.imageWidget.update_ip_port(self.lineEdit_ip_adress.text(), self.lineEdit_port.text())

    def show_sti_spike_robot(self):
        self.robot_sti_spike_dialog.show()

    def save_spikes_data(self):
        file_path, ok_ = QFileDialog.getSaveFileName(self, "File Save", "C:/", "*.h5", options=QFileDialog.DontUseNativeDialog)

        if file_path:
            path = file_path + ok_[-3:]
            self.imageWidget.mea_ic.set_spike_save_file_name(path)
        else:
            self.ShowMessageToStatusBar("Please set output file path!...", ifWarn=True)
            return

    # ---- MaxOne 原生记录（mx.Saving）：Export 菜单第 4 项 ----
    def toggle_maxone_recording(self, checked):
        """勾选：选目录 + 通道范围，置 enabled；取消：关闭并（若在录）立即停。"""
        if not checked:
            self.maxone_rec_enabled = False
            self._maxone_record_stop()
            self.ShowMessageToStatusBar("MaxOne 记录已关闭。", False)
            return
        if not self.actionMaxwell.isChecked():
            QMessageBox.information(
                self, "MaxOne Recording",
                "该记录功能仅用于 Maxwell MaxOne 设备，请先在 System 菜单切到「Maxwell MaxOne」并连接。")
            self.actionMaxOne_Recording.setChecked(False)
            return
        dir_path = QFileDialog.getExistingDirectory(self, "选择 MaxOne 记录保存目录", "")
        if not dir_path:
            self.ShowMessageToStatusBar("未选择目录，MaxOne 记录未开启。", ifWarn=True)
            self.actionMaxOne_Recording.setChecked(False)
            return
        # 通道范围：全部 1024 vs 仅本次在用电极对应通道
        box = QMessageBox(self)
        box.setWindowTitle("记录通道范围")
        box.setText("这次记录哪些通道？")
        box.setInformativeText(
            "全部通道：完整 1024 通道原始数据（文件大、最全）。\n"
            "仅在用电极：只录本次选的记录+刺激电极对应通道（文件小、对齐你的实验）。")
        btn_all = box.addButton("全部通道 (1024)", QMessageBox.AcceptRole)
        btn_used = box.addButton("仅在用电极通道", QMessageBox.AcceptRole)
        box.addButton("取消", QMessageBox.RejectRole)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is btn_all:
            self.maxone_rec_all_channels = True
        elif clicked is btn_used:
            self.maxone_rec_all_channels = False
        else:
            self.actionMaxOne_Recording.setChecked(False)
            return
        self.maxone_rec_dir = dir_path
        self.maxone_rec_enabled = True
        scope = "全部 1024 通道" if self.maxone_rec_all_channels else "仅在用电极通道"
        self.ShowMessageToStatusBar(
            "MaxOne 记录已就绪（{}）：每次 Start 随实验开录、Stop/超时结束。目录：{}".format(scope, dir_path), False)

    def _maxone_record_filename(self):
        """文件名 {时间}_{test/train}_{任务}_{刺激标签}；刺激标签做路径字符清洗。"""
        import re
        times = time.localtime(time.time())
        timetag = "{}_{}_{}_{}_{}".format(
            times.tm_year, times.tm_mon, times.tm_mday, times.tm_hour, times.tm_min // 30)
        mode = "train" if self.rdb_train_mode.isChecked() else "test"
        if self.actionObstacle_Avoidance.isChecked():
            task = "avoid"
        elif self.actionObject_Tracking.isChecked():
            task = "track"
        elif self.actionObject_Grasping.isChecked():
            task = "grasp"
        else:
            task = "task"
        stim_label = (self.lineEdit_select_sti.text() or "").strip() or "nostim"
        stim_label = re.sub(r"[^0-9A-Za-z_.-]", "_", stim_label)
        return "{}_{}_{}_{}".format(timetag, mode, task, stim_label)

    def _maxone_record_start(self):
        """Start 时随实验开录（幂等）。仅 Maxwell 设备 + 已勾选 enabled 时生效；其余安全空操作。"""
        if not self.maxone_rec_enabled or not self.actionMaxwell.isChecked():
            return
        rec = getattr(self.imageWidget.mea_ic, "recording", None)
        if rec is None or not hasattr(rec, "start_native_recording"):
            return
        channels = None if self.maxone_rec_all_channels else rec.used_channels()
        rec.start_native_recording(self.maxone_rec_dir, self._maxone_record_filename(), channels)

    def _maxone_record_stop(self):
        """测试/训练结束（手动 Stop 或超时）时停录、收尾文件。出界 reset 不调用本方法。幂等。"""
        rec = getattr(self.imageWidget.mea_ic, "recording", None)
        if rec is not None and hasattr(rec, "stop_native_recording"):
            rec.stop_native_recording()

    def ZoomChanged(self, delta):
        units = delta / (8 * 15)
        scale = 10
        self.adjustScale(scale * units)

    def adjustScale(self, value=1):
        value = self.zoomValue + value * 0.005

        if value <= 0.05:
            self.ShowMessageToStatusBar("The image can no longer be scaled down!", True)
            return

        self.zoomValue = value
        self.imageWidget.scale = value
        self.imageWidget.adjustSize()

        # if self.zoomSate == ZoomState.zoomInitial or self.zoomSate == ZoomState.zoomOrigin:
        #     # self.scrollArea.horizontalScrollBar().setValue(380)
        #     # self.scrollArea.verticalScrollBar().setValue(750)

        #     self.scrollArea.horizontalScrollBar().setValue(650)
        #     self.scrollArea.verticalScrollBar().setValue(900)
        self.imageWidget.update()

    def task_mode_changed(self):
        if self.actionObstacle_Avoidance.isChecked():
            self.imageWidget.set_task_mode(TASK.Obstacle_Avoidance)
            self.actionHuman_Map.setChecked(True)
            self.rdb_avoidance.setChecked(True)
            self.arm_pos_3d_widget.hide()
            self.ShowMessageToStatusBar("Change task mode into Obstacle Avoidance!...")
        elif self.actionObject_Tracking.isChecked():
            self.imageWidget.set_task_mode(TASK.Object_Tracking)
            self.actionEmpty_Map.setChecked(True)
            self.rdb_tracking.setChecked(True)
            self.arm_pos_3d_widget.hide()
            self.ShowMessageToStatusBar("Change task mode into Object Tracking!...")
        elif self.actionObject_Grasping.isChecked():
            self.imageWidget.set_task_mode(TASK.Object_Grasping)
            self.actionEmpty_Map.setChecked(True)
            self.rdb_grasping.setChecked(True)
            self.arm_pos_3d_widget.show()
            if self.rdb_train_mode.isChecked():
                self.arm_pos_3d_widget.init_glview_random()
            else:
                self.arm_pos_3d_widget.init_glview()
            self.ShowMessageToStatusBar("Change task mode into Object Grasping!...")
        
        self.update_visual_item()
        self.map_mode_changed()

    def update_visual_item(self):
        if self.actionObject_Grasping.isChecked():
            self.dsb_robot_arm_angle.show()
            self.label_25.show()
        else:
            self.dsb_robot_arm_angle.hide()
            self.label_25.hide()

    def task_mode_changed_bt(self):
        if self.rdb_avoidance.isChecked():
            self.imageWidget.set_task_mode(TASK.Obstacle_Avoidance)
            self.actionObstacle_Avoidance.setChecked(True)
            self.actionHuman_Map.setChecked(True)
            self.arm_pos_3d_widget.hide()
        elif self.rdb_tracking.isChecked():
            self.imageWidget.set_task_mode(TASK.Object_Tracking)
            self.actionObject_Tracking.setChecked(True)
            self.actionEmpty_Map.setChecked(True)
            self.arm_pos_3d_widget.hide()
        elif self.rdb_grasping.isChecked():
            self.imageWidget.set_task_mode(TASK.Object_Grasping)
            self.actionObject_Grasping.setChecked(True)
            self.actionEmpty_Map.setChecked(True)
            self.arm_pos_3d_widget.show()
            if self.rdb_train_mode.isChecked():
                self.arm_pos_3d_widget.init_glview_random()
            else:
                self.arm_pos_3d_widget.init_glview()

        self.update_visual_item()
        self.map_mode_changed()

    def map_mode_changed(self):
        if self.actionEmpty_Map.isChecked():
            self.imageWidget.set_map_mode(MAP.Empty_Map)
            self.ShowMessageToStatusBar("Change map mode into Empty Map!...")
        elif self.actionRandom_Map.isChecked():
            self.imageWidget.set_map_mode(MAP.Random_Map)
            self.ShowMessageToStatusBar("Change map mode into Random Map!...")
        elif self.actionHuman_Map.isChecked():
            self.imageWidget.set_map_mode(MAP.Human_Map)
            self.ShowMessageToStatusBar("Change map mode into Human Map!...")
        elif self.actionRegular_Map.isChecked():
            self.imageWidget.set_map_mode(MAP.Regular_Map)
            self.ShowMessageToStatusBar("Change map mode into Regular Map!...")

    def show_grasp_distance(self):
        self.grasp_distance_signal_dialog.show()

    def show_arm_pos_3D(self):
        self.arm_pos_3d_widget.show()

    def robot_arm_angle_human(self):
        value = self.dsb_robot_arm_angle.value()
        self.imageWidget.update_arm_angle(value)

    # 系统切换
    def device_changed(self):
        state = self.actionMEA_2100.isChecked()
        if state:
            self.imageWidget.update_system(SYSTEM_DEVICE.MEA2100)
            if self.stimulate_select_dialog is not None:
                self.stimulate_select_dialog.update_system(SYSTEM_DEVICE.MEA2100)

        state = self.actionINTAN_System.isChecked()
        if state:
            self.imageWidget.update_system(SYSTEM_DEVICE.INTAN)
            self.stimulate_select_dialog.update_system(SYSTEM_DEVICE.INTAN)

        state = self.actionMaxwell.isChecked()
        if state:
            cfg_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select MaxLab Live cfg file",
                "",
                "Maxwell cfg (*.cfg);;All files (*)",
            )
            if not cfg_path:
                # 用户取消选择 → 回退到 MEA2100，避免半连接状态
                self.actionMEA_2100.setChecked(True)
                return

            # 解析 cfg 电极（刺激选择器与记录选择器共用同一份）
            try:
                from src.system_device.maxwell.cfg_loader import extract_electrodes
                cfg_electrodes = extract_electrodes(cfg_path)
            except Exception as exc:
                QMessageBox.warning(self, "cfg 解析失败",
                                    "无法从 cfg 解析电极：{}".format(exc))
                self.actionMEA_2100.setChecked(True)
                return
            if not cfg_electrodes:
                QMessageBox.warning(self, "cfg 无电极", "cfg 未解析出任何电极。")
                self.actionMEA_2100.setChecked(True)
                return

            from src.MaxwellRecordElectrodeDialog import MaxwellRecordElectrodeDialog

            # 刺激电极：当前范式固定 2 个（左轮/右轮各 1），从 cfg 列表单选。左右切换通过
            # stim_unit 输出开关实现（共享 DAC0），同一时刻只允许一侧 connect=True 避免串扰。
            # 对应卡片：[[Maxwell - 路由层与 unit 输出层的双层 connect 语义]] /
            #           [[MetaBOC - cfg 文件解析与显式路由策略]]
            stim_picker = MaxwellRecordElectrodeDialog(
                cfg_electrodes, self, single=True,
                title="Maxwell 刺激电极 — 左轮 / 右轮",
                intro="给「左轮」「右轮」各选一个刺激电极（cfg 已 routing 范围内）。\n"
                      "左轮触发左侧环境/惩罚/奖励刺激，右轮触发右侧；双侧奖励同时驱动两个。\n"
                      "两电极不能映射到同一 stim_unit（冲突由 MaxLab Live mapping_preflight 预筛）。",
            )
            stim_picker.exec_()
            stim_lr = stim_picker.get_recording_list()
            if stim_lr is None:
                # 刺激电极是 connect 路由的必需项，取消则回退
                self.actionMEA_2100.setChecked(True)
                return
            stim_left_str, stim_right_str = stim_lr[0][0], stim_lr[1][0]
            left_id, right_id = int(stim_left_str), int(stim_right_str)
            stim_electrodes = [left_id, right_id]
            role_mapping = {"left_wheel": left_id, "right_wheel": right_id}
            self.maxwell_stim_lr = [[stim_left_str], [stim_right_str]]

            # 记录电极：多选指派左右轮（决定 get_recording 的左右分组）。顶部显示已选刺激电极。
            # 跳过 → maxwell_recording_list=None → get_recording 走全通道合并 fallback。
            self.maxwell_recording_list = None
            rec_picker = MaxwellRecordElectrodeDialog(
                cfg_electrodes, self,
                info_text="已选刺激电极 —— 左轮 {} / 右轮 {}".format(stim_left_str, stim_right_str),
            )
            rec_picker.exec_()
            rl = rec_picker.get_recording_list()
            if rl is not None:
                self.maxwell_recording_list = rl

            self.maxwell_cfg_path = cfg_path
            self.imageWidget.mea_ic.set_maxwell_session_params(
                cfg_path=cfg_path,
                record_electrodes=None,
                stim_electrodes=stim_electrodes,
                role_mapping=role_mapping,
            )
            self.imageWidget.update_system(SYSTEM_DEVICE.MAXWELL)
            # 图2 切到 Maxwell 模式：从 ./out/MAXWELL 加载已保存的刺激参数（与 MEA2100/INTAN 对齐）
            self.stimulate_select_dialog.update_system(SYSTEM_DEVICE.MAXWELL)

            # 刺激组有效性反馈：connect 成功 → recording 非 None；两刺激电极映射到同一
            # stim_unit（冲突）或 cfg/探头失败 → recording 为 None（终端 [STIM-CHECK]/报错有详情）。
            if self.imageWidget.mea_ic.recording is None:
                QMessageBox.warning(
                    self, "Maxwell 会话启动失败",
                    "Maxwell 连接/路由失败 —— 可能两个刺激电极映射到同一 stim_unit（冲突），\n"
                    "或 cfg / 探头问题。请看终端日志，重选刺激电极或检查 cfg。",
                )
                self.actionMEA_2100.setChecked(True)
            else:
                self.ShowMessageToStatusBar(
                    "Maxwell 已连接，刺激组有效（左右轮 unit 不冲突，详见终端 [STIM-CHECK]）", False)
    
    def setGlobalFont(self):
        """设置全局字体样式：中文使用微软雅黑，英文使用Times New Roman"""
        # 创建字体对象
        font = create_global_font()
        
        # 应用字体到整个应用程序
        QApplication.setFont(font)
        
        # 应用字体到当前窗口及其所有子控件
        self.setFont(font)
        
        # 获取字体样式表
        style_sheet = get_global_font_style()
        
        # 合并样式表
        self.setStyleSheet(self.styleSheet() + style_sheet)
        
        print("已应用全局字体：微软雅黑 + Times New Roman")

    def set_intan_data_path(self):
        # dir_path = QFileDialog.getExistingDirectory(self, "Select Dir", "C:/", options=QFileDialog.DontUseNativeDialog) # QFileDialog.ShowDirsOnly
        dir_path = "F:\\Intan"
        # dir_path = "E:\\TCP\\Data\\1"
        # dir_path = "D:/INTAN"
        if dir_path:
            self.INTAN_data_path = dir_path
            self.imageWidget.set_intan_data_path(self.INTAN_data_path)
            self.ShowMessageToStatusBar("The INTAN data will be saved in: " + dir_path)
        else:
            self.ShowMessageToStatusBar("You have not select INTAN data path, please select again!...", ifWarn=True)
    


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setApplicationName(__appname__)
#    app.setWindowIcon(newIcon("app"))
    win = MainWindowClass()
    win.show()

    sys.exit(app.exec_())