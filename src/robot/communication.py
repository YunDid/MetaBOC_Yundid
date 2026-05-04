# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

import numpy as np
import os
import h5py
import time
from datetime import datetime

from threading import Thread

from src.infor_com_mea.recording import Recording
from src.infor_com_mea.stimulation import Stimulation
from src.robot.encode_decode import EncodingDecoding

from src.robot.task import TASK, MAP, SYSTEM_DEVICE
from src.robot.dynamic_model_control import DynamicM_Control

from src.system_device.intan_sys import INTAN_System


class Communication(object):

    def __init__(self, in_l=0, in_r=0, out_l=0, out_r=0):
        super(Communication, self).__init__()
        self.in_dis_left = in_l
        self.in_dis_right = in_r

        self.mea_left = out_l
        self.mea_right = out_r

        self.device_type = SYSTEM_DEVICE.MEA2100
        self.recording = Recording()
        self.stimulation = Stimulation()
        self.stimulation.set_recording(self.recording)

        self.intan_data_path = None

        # 选择和设定当前连接的系统
        device_cont = self.recording.get_device_count()
        if device_cont < 1:
            print("MEA device is not found!...")
        else:
            self.stimulation.initial_device()

        self.mode_train = True    # 是否是训练模式

        self.map = None
        self.times = 0    #  1s内的调用次数，达到10时重置为0

        self.dynamic_model_used = False    # 是否使用动力学模型
        self.dynamic_model = None

        self.angle_left = None    # 机器人左右两侧的角度信息
        self.angle_right = None

        self.left_sti = []
        self.right_sti = []
        self.left_spikes = []    # 记录spike，用于可视化显示
        self.right_spikes = []

        self.l_spikes = []    # 记录spike，用于保存发放率和期望值
        self.r_spikes = []
        self.spike_length = 23  # 记录长度，与动力学模型中的设置保持一致

        self.task = TASK.Obstacle_Avoidance    # 初始任务

        self.spike_save_name = "./out_spikes/spikes_left_right.h5"

        timestamp = time.time()
        dt_object = datetime.fromtimestamp(timestamp)
        formatted_date = dt_object.strftime("%Y-%m-%d %H-%M-%S")
        self.save_spike_ref_path_left = os.path.join("./out_spike_ref", "left-" + formatted_date[:-9] + "-" + formatted_date[-8:] + ".txt")
        self.save_spike_ref_path_right = os.path.join("./out_spike_ref", "right-" + formatted_date[:-9] + "-" + formatted_date[-8:] + ".txt")


        path = "./encode_decode/en_de_coding.npz"
        parameter = None
        if os.path.exists(path):
            parameter = np.load(path, allow_pickle=True)
        self.en_de_code = EncodingDecoding(para=parameter)

    def close(self):
        if self.device_type == SYSTEM_DEVICE.INTAN:
            self.intan_sys.stop_connect()

    def update_systems(self, systems):
        if systems == SYSTEM_DEVICE.MEA2100:
            self.connect_to_mea2100()
        elif systems == SYSTEM_DEVICE.INTAN:
            self.connect_to_intan()

    def set_intan_data_path(self, dir_data):
        """
        设置ITNAN系统的数据保存路径，用于监控实时生成的数据
        """
        self.intan_data_path = dir_data
        if self.device_type == SYSTEM_DEVICE.INTAN:
            self.recording.recording.set_monitoring_directory(dir_data)
    
    def connect_to_mea2100(self):
        if self.device_type == SYSTEM_DEVICE.INTAN:
            
            # 首先安全断开连接
            self.intan_sys.stop_connect()

            self.device_type = SYSTEM_DEVICE.MEA2100

            self.recording = Recording()
            self.stimulation = Stimulation()
            self.stimulation.set_recording(self.recording)

            # 选择和设定当前连接的系统
            device_cont = self.recording.get_device_count()
            if device_cont < 1:
                print("MEA device is not found!...")
            else:
                self.stimulation.initial_device()

    def connect_to_intan(self):
        if self.intan_data_path is None:
            print("Please select data path for INTAN system!...")

        if self.device_type == SYSTEM_DEVICE.MEA2100:
            self.device_type = SYSTEM_DEVICE.INTAN
            del self.recording; del self.stimulation
            self.recording = None
            self.stimulation = None

            self.intan_sys = INTAN_System()
            self.recording = self.intan_sys.recording
            self.stimulation = self.intan_sys.stimulating
            self.recording.recording.set_monitoring_directory(self.intan_data_path)

            print(30*"==")
            print("The intan system is connected!...")
            print(30*"==")

    def set_obstacle_map(self, map):
        self.map = map

    def set_task(self, task):
        self.task = task
        self.en_de_code.set_task(task)

    def show_encode_decode_setting(self):
        self.en_de_code.show()

    # 用于设置奖惩刺激的信号
    def set_stimulating_signal(self, sig):
        self.stimulation.set_sti_signal(sig)
        self.recording.set_record_para(sig)

    def set_spike_detection_para(self, para):
        self.recording.set_spike_detection_para(para)

    def set_training_mode(self, state):
        self.mode_train = state
        print("Train state:" , state)

    # 从虚拟环境输出  距离值，并转化为刺激信号（距离信息的刺激），输出至刺激器进行刺激
    def update_distance(self, left, left_hit, right, right_hit, border_out, ang_dis_left, ang_dis_right, left_id_change, right_id_change,ang_dis =0):
        """
        左右最近距离、Robot是否出界
        ang_dis_left, ang_dis_right: 障碍物距离车子正前方的角度，左侧为负值，右侧为正值
        """

        self.angle_left = ang_dis_left
        self.angle_right = ang_dis_right

        if self.mode_train:  # 仅在训练模式下，进行奖惩刺激
            # if left_hit:
            #     self.stimulation.update_stimulation_left()    # 撞击到障碍物时的操作，相关信号需传输至刺激器，奖惩刺激
            #     self.save_spikes_left_right(0, 0, 1, 0)    # 保存spikes数据，便于后续分析
            # if right_hit:
            #     self.stimulation.update_stimulation_right()
            #     self.save_spikes_left_right(0, 0, 0, 1)    # 保存spikes数据，便于后续分析
            
            # 故意撞击
            # if right_hit:
            #     self.stimulation.update_stimulation_left()    # 撞击到障碍物时的操作，相关信号需传输至刺激器，奖惩刺激
            #     self.save_spikes_left_right(0, 0, 1, 0)    # 保存spikes数据，便于后续分析
            # if left_hit:
            #     self.stimulation.update_stimulation_right()
            #     self.save_spikes_left_right(0, 0, 0, 1)    # 保存spikes数据，便于后续分析

            if (left_hit or right_hit) and self.task == TASK.Obstacle_Avoidance:
                if left_hit:
                    self.stimulation.update_stimulation_left()    # 撞击到障碍物时的操作，相关信号需传输至刺激器，奖惩刺激
                elif right_hit:
                    self.stimulation.update_stimulation_right()
                self.save_spikes_left_right(0, 0, 1, 1)    # 保存spikes数据，便于后续分析
            elif (left_id_change or right_id_change) and self.task == TASK.Obstacle_Avoidance:
                if left_id_change and right_id_change:
                    self.stimulation.update_stimulation_left_right_reward()  # 屏蔽奖励刺激
                    pass


            if self.task == TASK.Object_Tracking:
                """
                目标跟踪的奖惩刺激，需有事件触发条件
                """
                tracking_threshold = 40
                if left < tracking_threshold:
                    self.stimulation.update_stimulation_left()
                if right < tracking_threshold:
                    self.stimulation.update_stimulation_right()
                self.save_spikes_left_right(0, 0, 1, 1)

            if self.task == TASK.Object_Grasping:
                """
                目标抓取的奖惩刺激 与目标物体差距在20度时施加惩罚刺激,这个参数需要调试来决定
                """
                if abs(ang_dis) > 20:
                    if left != 0:
                        self.stimulation.update_stimulation_left()
                    elif right != 0:
                        self.stimulation.update_stimulation_right()
                pass

        if (left_hit or right_hit) and self.task == TASK.Obstacle_Avoidance and self.mode_train:
            _wait_t0 = time.perf_counter()
            time.sleep(4)
            from src.system_device.timing_logger import TimingLogger
            TimingLogger.get().add_protocol_wait((time.perf_counter() - _wait_t0) * 1000.0)
            pass
        elif left_id_change and right_id_change and self.mode_train:
            """
            当小车未发生撞击，且障碍物id发生变化，则小车暂停；
            如果是测试阶段，则小车只是暂停，不施加奖励刺激；
            如果是训练阶段，会自动施加奖励刺激
            """
            _wait_t0 = time.perf_counter()
            time.sleep(0.11)
            from src.system_device.timing_logger import TimingLogger
            TimingLogger.get().add_protocol_wait((time.perf_counter() - _wait_t0) * 1000.0)

        if self.times >= 3:    # 控制信号更新频次
            self.times = 0

            self.in_dis_left = left
            self.in_dis_right = right

            if border_out:
                pass    # 机器人出界，做出相应的操作(当前直接重置游戏)


            DYNAMIC_USED = False
            if self.dynamic_model_used and self.dynamic_model is not None:
                if left <= right:
                    sti_l_amp, sti_l_dur, sti_fre_l = self.dynamic_model.get_sti_signal_left()
                    print('self.dynamic_model.get_sti_signal_left() stim_parm:',sti_l_amp)
                    print('self.dynamic_model.get_sti_signal_left() dur_parm:',sti_l_dur)
                    print('self.dynamic_model.get_sti_signal_left() fre:',sti_fre_l)
                    if sti_l_amp is not None:
                        if len(self.left_sti) > 100:
                            self.left_sti.pop(0)
                            self.right_sti.pop(0)
                        self.left_sti.append(sti_fre_l[0][0][0])    # 随机频率较难直接计算，先不进行可视化
                        self.right_sti.append(0)
                        DYNAMIC_USED = True
                        self.stimulation.update_record_stimulation_dynamic_model_left(sti_l_amp, sti_l_dur)
                        self.times = self.times + 1
                        return
                else:
                    sti_r_amp, sti_r_dur, sti_fre_r = self.dynamic_model.get_sti_signal_right()
                    if sti_r_amp is not None:
                        if len(self.left_sti) > 100:
                            self.left_sti.pop(0)
                            self.right_sti.pop(0)
                        self.left_sti.append(0)    # 随机频率较难直接计算，先不进行可视化
                        self.right_sti.append(sti_fre_r[1][0][0])
                        DYNAMIC_USED = True
                        
                        self.stimulation.update_record_stimulation_dynamic_model_right(sti_r_amp, sti_r_dur)
                        self.times = self.times + 1
                        return

            from src.system_device.timing_logger import TimingLogger as _TL
            from time import perf_counter as _pc
            _stim_logger = _TL.get()

            if not DYNAMIC_USED:
                _t_enc = _pc()
                sti_l, sti_r = self.en_de_code.encode(left, right, ang_dis_left, ang_dis_right)  # 环境信息编码为刺激频率
                _stim_logger.mark("stage4_encode_ms", (_pc() - _t_enc) * 1000.0)


            if len(self.left_sti) > 100:
                self.left_sti.pop(0)
                self.right_sti.pop(0)

            self.left_sti.append(sti_l)    # 随机频率较难直接计算，先不进行可视化
            self.right_sti.append(sti_r)
            _t_stim = _pc()
            self.stimulation.update_record_stimulation(sti_l, sti_r)
            _stim_logger.mark("stage4_stim_tcp_ms", (_pc() - _t_stim) * 1000.0)
        
        self.times = self.times + 1

        # 测试用
        # self.stimulation.update_stimulation_left()

    def set_spike_save_file_name(self, name):
        self.spike_save_name = name

    def save_spikes_left_right(self, left, right, l_re_pun, r_re_pun):
        try:
            if not os.path.exists(self.spike_save_name):
                h5_file = h5py.File(self.spike_save_name, "a")
                h5_file.create_dataset("data", (4,1), maxshape=(4, None), chunks=True, dtype='int32')    # 0 left 1 right
                h5_file['data'][0, 0] = left
                h5_file['data'][1, 0] = right
                h5_file['data'][2, 0] = l_re_pun   # 是否有奖惩刺激
                h5_file['data'][3, 0] = r_re_pun
                h5_file.close()
                print("==="*30)
            else:
                with h5py.File(self.spike_save_name, "a") as hf:
                    t1 = time.time()
                    hf['data'].resize((hf['data'].shape[1] + 1), axis = 1)  # for left & right
                    hf['data'][0, -1:] = left
                    hf['data'][1, -1:] = right
                    hf['data'][2, -1:] = l_re_pun   # 是否有奖惩刺激
                    hf['data'][3, -1:] = r_re_pun
        except:
            print("Saving data error happend!...")


    def get_mea_control_infor(self, left_hit, right_hit, min_dis_left, min_dis_right):
        if self.task == TASK.Obstacle_Avoidance:
            return self.get_mea_control_infor_obstacle_avoidance(left_hit, right_hit, min_dis_left, min_dis_right)
        elif self.task == TASK.Object_Tracking:
            print(f'min_dis_left={min_dis_left},min_dis_right={min_dis_right}')
            return self.get_mea_control_infor_object_tracking(left_hit, right_hit, min_dis_left, min_dis_right)
        elif self.task == TASK.Object_Grasping:
            """
            抓取任务，仅需传入角度误差，min_dis_left表示角度差
            """
            return self.get_mea_control_infor_object_grasping(min_dis_left)



    # 用以保存非AI条件下的spike发放和期望值
    def save_spike_ref_wo_AI(self, left_spike, right_spike, min_dis_left, min_dis_right):
        if len(self.l_spikes) > self.spike_length:
            self.l_spikes.pop(0)
            self.r_spikes.pop(0)

            self.l_spikes.append(left_spike)
            self.r_spikes.append(right_spike)
        else:
            self.l_spikes.append(left_spike)
            self.r_spikes.append(right_spike) 

        if len(self.l_spikes) > self.spike_length and self.times >= 3:
            # 距离和响应呈正相关关系，l  r  已映射到特定的频率范围，参考编码参数
            l, r = self.en_de_code.encode(min_dis_left, min_dis_right, self.angle_left, self.angle_right)

            left_history = np.array(self.l_spikes)
            right_history = np.array(self.r_spikes)
            mean_l = left_history.mean(axis=0) # 间隔100ms记录1s滑窗的spike数据（共20个）
            mean_r = right_history.mean(axis=0)
            scale_l = l / (mean_l.sum() + 0.0001)
            scale_r = r / (mean_r.sum() + 0.0001)
            ref_l = scale_l * mean_l
            ref_r = scale_r * mean_r

            self.save_spike_reference("left", self.l_spikes, ref_l)    # 保存了当前时刻的spike，和下一时刻的期望值
            self.save_spike_reference("right", self.r_spikes, ref_r)

    def save_spike_reference(self, ele, spike, reference):
        
        # 获取当前时间戳
        current_time = time.time()
        timestamp = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        out_spike = timestamp + ": "
        for i in range(len(spike)):
            cur = spike[i]
            out_spike = out_spike + str(cur[0]) + ","

        out_spike = out_spike + ":" + str(np.round(reference[0],2)) + "\n"
        if ele == "left":
            with open(self.save_spike_ref_path_left, "a") as f:
                f.write(out_spike)
        elif ele == "right":
            with open(self.save_spike_ref_path_right, "a") as f:
                f.write(out_spike)


    # 从MEA获得输出值, 并改变轮速
    def get_mea_control_infor_obstacle_avoidance(self, left_hit, right_hit, min_dis_left, min_dis_right):
        """
        每100ms更新一次
        """
        from src.system_device.timing_logger import TimingLogger
        from time import perf_counter
        _logger = TimingLogger.get()

        left_spike, right_spike = self.recording.get_recording()    # 从MEA获取数据：特定时间内的

        # 动力学模型控制，获取输入的spike信号
        if self.dynamic_model_used:
            if self.dynamic_model is not None:
                print(100*"=")
                self.dynamic_model.update_distance(min_dis_left, min_dis_right, self.angle_left, self.angle_right)
                self.dynamic_model.update_spike_data(left_spike, right_spike)                

        # 保存电极的期望发放率和实际spike发放率： 不影响刺激信号的生成，只是重新计算和记录，用于实验室分析
        self.save_spike_ref_wo_AI(left_spike, right_spike, min_dis_left, min_dis_right)

        left, right = np.sum(left_spike), np.sum(right_spike)

        self.mea_left = left
        self.mea_right = right

        self.save_spikes_left_right(left, right, 0, 0)    # 保存spikes数据，便于后续分析

        if len(self.left_spikes) > 100:
            self.left_spikes.pop(0)
            self.right_spikes.pop(0)

        self.left_spikes.append(left)
        self.right_spikes.append(right)

        _t_dec = perf_counter()
        ctrol_left, ctrol_right = self.en_de_code.decode(left, right, min_dis_left, min_dis_right)
        _logger.mark("stage3_decode_ms", (perf_counter() - _t_dec) * 1000.0)


        # if left_hit or right_hit:
        #     gap = 0.5
        #     if ctrol_left > ctrol_right:
        #         ctrol_left = ctrol_left + gap
        #         ctrol_right = ctrol_right - gap
        #     else:
        #         ctrol_right = ctrol_right + gap
        #         ctrol_left = ctrol_left - gap
        return ctrol_left, ctrol_right, left, right      # 返回更新的控制信号，仅用于改变轮速



    def get_mea_control_infor_object_tracking(self, left_hit, right_hit, min_dis_left, min_dis_right):
        left, right = self.recording.get_recording()    # 从MEA获取数据：特定时间内的

        left, right = np.sum(left), np.sum(right)

        self.mea_left = left
        self.mea_right = right

        self.save_spikes_left_right(left, right, 0, 0)    # 保存spikes数据，便于后续分析

        if len(self.left_spikes) > 100:
            self.left_spikes.pop(0)
            self.right_spikes.pop(0)

        self.left_spikes.append(left)
        self.right_spikes.append(right)

        ctrol_left, ctrol_right = self.en_de_code.decode(left, right, min_dis_left, min_dis_right)

        return ctrol_left, ctrol_right, left, right      # 返回更新的控制信号，仅用于改变轮速


    def get_mea_control_infor_object_grasping(self, dis_angle):
        left, right = self.recording.get_recording()    # 从MEA获取数据：特定时间内的

        left, right = np.sum(left), np.sum(right)

        self.mea_left = left
        self.mea_right = right

        self.save_spikes_left_right(left, right, 0, 0)    # 保存spikes数据，便于后续分析

        if len(self.left_spikes) > 100:
            self.left_spikes.pop(0)
            self.right_spikes.pop(0)

        self.left_spikes.append(left)
        self.right_spikes.append(right)

        ctrol_left, ctrol_right = self.en_de_code.decode(left, right, dis_angle, None)

        return ctrol_left, ctrol_right, left, right      # 返回更新的控制信号，仅用于改变机械臂角度

    def get_stifre_ctrol_signal(self):
        """
        获取刺激频率和控制信号，进行可视化显示
        """
        return self.left_sti, self.right_sti, self.left_spikes, self.right_spikes

    def set_dynamic_model_state(self, state):
        self.dynamic_model_used = state
        if state and self.dynamic_model is None:
            self.dynamic_model = DynamicM_Control(self.recording, self.stimulation, self.en_de_code)
        if not state:
            if self.dynamic_model is not None:
                del self.dynamic_model
                self.dynamic_model = None

