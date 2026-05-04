# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2024
# Written by Guiping Cao
# Time: 2024.05.23
# --------------------------------------------------------


import time
import os
import numpy as np

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *

from src.system_device.stimulation_intan_api import StimulationIntan
from src.enum.constants import (
    StimulationType, StimulationPosition, StimulationSpec,
    TriggerType, KeypressSource, DigitalOutput, DigitalIn
)

# 单独构建的平台与INTAN Stimulation API对接的文件，用以实现平台内的特定函数


class Stimulating_Intan_Platform(QObject):
    stimulate_finshed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(Stimulating_Intan_Platform, self).__init__(parent)

        self.stimulating_stim = StimulationIntan()
        self.stimulating_record = StimulationIntan()
        
        self.stimulating_stim.set_socket("127.0.0.1", 5000)
        self.stimulating_record.set_socket("127.0.0.1", 5001)
        
        self.connect_socket_stim = self.stimulating_stim.connect_to_server()
        self.connect_socket_record = self.stimulating_record.connect_to_server()
        self.debug = False
        

        self.sti_sig = None    # 刺激参数
        self.amplitude = None; self.duration = None   # 惩罚刺激信号

        self.amp_reward = None; self.dur_reward = None   # 奖励信号
        
        self.left_electrode_key = None; self.right_electrode_key =None  # 左右刺激电极

        base_channel = 32
        self.channel_map = {
                "A-000": 0, "A-001": 1, "A-002":2,
                "A-003": 3, "A-004": 4, "A-005": 5,
                "A-006": 6, "A-007": 7, "A-008": 8,
                "A-009": 9, "A-010": 10, "A-011": 11,
                "A-012": 12, "A-013": 13, "A-014": 14,
                "A-015": 15, "A-016": 16, "A-017": 17,
                "A-018": 18, "A-019": 19, "A-020": 20,
                "A-021": 21, "A-022": 22, "A-023": 23,
                "A-024": 24, "A-025": 25, "A-026": 26,
                "A-027": 27, "A-028": 28, "A-029": 29,
                "A-030": 30, "A-031": 31, 
                
                "B-000": 0 + base_channel, "B-001": 1 + base_channel, "B-002":2 + base_channel,
                "B-003": 3 + base_channel, "B-004": 4 + base_channel, "B-005": 5 + base_channel,
                "B-006": 6 + base_channel, "B-007": 7 + base_channel, "B-008": 8 + base_channel,
                "B-009": 9 + base_channel, "B-010": 10 + base_channel, "B-011": 11 + base_channel,
                "B-012": 12 + base_channel, "B-013": 13 + base_channel, "B-014": 14 + base_channel,
                "B-015": 15 + base_channel, "B-016": 16 + base_channel, "B-017": 17 + base_channel,
                "B-018": 18 + base_channel, "B-019": 19 + base_channel, "B-020": 20 + base_channel,
                "B-021": 21 + base_channel, "B-022": 22 + base_channel, "B-023": 23 + base_channel,
                "B-024": 24 + base_channel, "B-025": 25 + base_channel, "B-026": 26 + base_channel,
                "B-027": 27 + base_channel, "B-028": 28 + base_channel, "B-029": 29 + base_channel,
                "B-030": 30 + base_channel, "B-031": 31 + base_channel,
                }


        self.ele_key_sti = None    # 刺激电极，用于刺激时刻的数据保存
        self.sti_data_amp = None
        self.sti_data_dur = None
        self.sti_intan_amp_8 = {}      # 8种刺激信号
        self.sti_intan_dur_8 = {}      # 8种刺激信号

        self.recording = None

    def _debug(self, *args):
        if self.debug:
            print(*args)



    def __del__(self):
        self.stimulating_stim.stopRecord()

        del self.stimulating_stim
        self.stimulating_stim = None
    
    def stopRun(self):
        self.stimulating_stim.stopRun()
    
    def stopRecord(self):
        self.stimulating_record.stopRecord()
        self.stimulating_stim.stopRecord()
        
    def close_connection(self):
        self.stimulating_stim.close_connection()
        self.stimulating_record.close_connection()
        
    def set_recording(self, recording):
        self.recording = recording

    # 通过软件设置的刺激信号，进行解析
    # TODO 需要将该信号转化为INTAN可以接收的刺激信号
    def set_sti_signal(self, signal):
        self.sti_sig = signal
        parameter = signal.para
        sti_para= signal.sti_para    # 包含刺激名称和组重复次数

        # 解析8种参数设置
        for key in parameter[0]:
            para = parameter[0][key]
            d1 = para['D1']
            d2 = para['D2']
            a1 = para['A1']
            a2 = para['A2']
            pre1 = para['Pre1']
            post1 = para['Post1']

            amplitude = [0, a1, a2, 0]          # μA 
            duration = [pre1, d1, d2, post1]    # μs
            # amplitude = [a1, a2, 0]          # μA 
            # duration = [d1, d2, post1]    # μs
            self.sti_intan_amp_8[key] = amplitude
            self.sti_intan_dur_8[key] = duration

        # 电极设置
        l_electrode = signal.stimulating_list[0]
        r_electrode = signal.stimulating_list[1]

        self.left_electrode_key = []; self.right_electrode_key = []  # 保存了刺激电极index key
        for i in range(len(l_electrode)):
            self.left_electrode_key.append(l_electrode[i])
        
        for i in range(len(r_electrode)):
            self.right_electrode_key.append(r_electrode[i])

        self.set_punishment_sti()   # 设置2种刺激
        self.set_reward_sti()       # 设置1种刺激
        time.sleep(0.1)  # Give some time for the command to be processed
        
        self.stimulating_record.enableDigitalInChanel(DigitalIn.DIGITAL_CHANNEL_01.value)
        self.stimulating_record.startRecord()

    def set_punishment_sti(self):
        
        """
        惩罚刺激：5Hz, 70uA, 4s
        """
        left = 5; 
        l_time = 1000000 / left

        amplidude = 180.0  # μA

        # 刺激信号单元,定义100ms的信号去刺激，下一次更新即可
        left_amp = [];left_dur = []
        for i in range(left * 4):    # 100ms内的刺激次数
            # l_amplitude = Array[Int32]([1000, 1000, 0])    # 幅值 1000μV
            # l_duration = Array[UInt64]([20000, 20000, l_time])  # 200μs
            l_amplitude =[amplidude, amplidude, 0]    # 幅值 1000μV
            l_duration = [200, 200, l_time]  # 200μs
            left_amp.extend(l_amplitude)
            left_dur.extend(l_duration)

        self.amplitude = left_amp
        self.duration = left_dur

        print("Punishment stimulus signal has been set!...")

    def set_reward_sti(self):

        """
        奖励刺激：100Hz, 70uA, 100ms
        """
        left = 100;  right = 100
        l_time = 1000000 / left

        amplidude = 90.0  # μA

        # 刺激信号单元,定义100ms的信号去刺激，下一次更新即可
        left_amp = [];left_dur = []
        for i in range(int(left*0.1 + 0.5)):    # 100ms内的刺激次数
            # l_amplitude = Array[Int32]([1000, 1000, 0])    # 幅值 1000μV
            # l_duration = Array[UInt64]([20000, 20000, l_time])  # 200μs
            l_amplitude =[amplidude, amplidude, 0]    # 幅值 1000μV
            l_duration = [200, 200, l_time]  # 200μs
            left_amp.extend(l_amplitude)
            left_dur.extend(l_duration)

        self.amp_reward = left_amp
        self.dur_reward = left_dur

        print("Reward stimulus signal has been set!...")

    def update_amplitude_duration(self, m_amplitude, m_duration):
        """
        m_amplitude: μV
        m_duration: μs
        """
        # 解析设置，变为奖惩刺激参数
        amp = np.array(m_amplitude[0])     # 转变为μV
        dur = np.array(m_duration[0])      # 转变为μs

        # self.amplitude = Array[Int32](amp)
        # self.duration = Array[UInt64](dur)
        # TODO
        self.amplitude = amp
        self.duration = dur

        print("Setting dynamic model stimulation signal!...")

    def update_record_stimulation(self, left, right):
        """左右两侧传感器的信号经编码后， 传入并通过device输出刺激
        left: 刺激频率，整数
        right: 刺激频率，整数
        """

        l_time = int(1000000 / left)
        r_time = int(1000000 / right)    # μs，间隔时间

        amplidude = 70.0  # μA

        # 刺激信号单元,定义100ms的信号去刺激，下一次更新即可
        left_amp = [];left_dur = []
        for i in range(int(left*0.1 + 0.5)):    # 500ms内的刺激次数

            l_amplitude =[amplidude, amplidude, 0]    # 幅值 1000μV
            l_duration = [200, 200, l_time]  # 200μs
            left_amp.extend(l_amplitude)
            left_dur.extend(l_duration)

        right_amp = [];right_dur = []
        for i in range(int(right*0.1 + 0.5)):
            # r_amplitude = Array[Int32]([1000, 1000, 0])
            # r_duration = Array[UInt64]([200, 200, r_time])
            r_amplitude = [amplidude, amplidude, 0]
            r_duration = [200, 200, r_time]
            right_amp.extend(r_amplitude)
            right_dur.extend(r_duration)
        
        # TODO
        left_amp = left_amp
        left_dur = left_dur

        right_amp = right_amp
        right_dur = right_dur


        # 环境信息的刺激控制
        if left >= right:
            self.update_stimulation_stg1(self.left_electrode_key, left_amp, left_dur)
        else:
            self.update_stimulation_stg2(self.right_electrode_key, right_amp, right_dur)


    def update_record_stimulation_dynamic_model_left(self, ampli, duri):
        """左右两侧传感器的信号经编码后， 传入并通过device输出刺激
        left: 刺激频率，整数
        right: 刺激频率，整数
        动力学模型产生的优化环境刺激
        """
        
        # 解析设置，变为奖惩刺激参数
        amp = np.array(ampli[0]) * 0.001  # 转变为μA
        dur = np.array(duri[0])      # 转变为μs
        if amp is None or len(amp) == 0 or dur is None or len(dur) == 0:
            return  # 或者跳过本次发送
        # amp = Array[Int32](amp)
        # dur = Array[UInt64](dur)
        # TODO
        amp = np.abs(amp)
        dur = dur
        self._debug('update_record_stimulation_dynamic_model_left amp:', amp)
        self._debug('update_record_stimulation_dynamic_model_left dur:', dur)
        # 环境信息的刺激控制
        self.update_stimulation_stg1(self.left_electrode_key, amp, dur)

    def update_record_stimulation_dynamic_model_right(self, ampli, duri):
        """左右两侧传感器的信号经编码后， 传入并通过device输出刺激
        left: 刺激频率，整数
        right: 刺激频率，整数
        动力学模型产生的优化环境刺激,右侧
        """
        
        # 解析设置，变为奖惩刺激参数
        amp = np.array(ampli[0]) * 0.001     # 转变为μA
        dur = np.array(duri[0])      # 转变为μs
        if amp is None or len(amp) == 0 or dur is None or len(dur) == 0:
            return  # 或者跳过本次发送
        # amp = Array[Int32](amp)
        # dur = Array[UInt64](dur)
        # TODO
        amp = np.abs(amp)
        dur = dur

        # 环境信息的刺激控制
        self.update_stimulation_stg2(self.right_electrode_key, amp, dur)


    def update_stimulation_left(self):
        """
        在出现撞击时，进行惩罚刺激
        """

        self.update_stimulation_stg1_punish(self.left_electrode_key, self.amplitude, self.duration)


    def update_stimulation_right(self):
        """
        在出现撞击时，进行惩罚刺激
        """

        self.update_stimulation_stg2_punish(self.right_electrode_key, self.amplitude, self.duration)


    def update_stimulation_left_reward(self):
        """
        在障碍物ID变化时，进行奖励刺激
        """

        self.update_stimulation_stg1_reward(self.left_electrode_key, self.amp_reward, self.dur_reward)


    def update_stimulation_right_reward(self):
        """
        在障碍物ID变化时，进行奖励刺激
        """

        self.update_stimulation_stg2_reward(self.right_electrode_key, self.amp_reward, self.dur_reward)

    def update_stimulation_left_right_reward(self):
        """
        在障碍物ID变化时，进行奖励刺激，同时刺激所有非刺激电极
        """
        all_ele = False
        if not all_ele:
            """同时刺激所连接的两个电极"""
            self.update_stimulation_stg1_stg2_reward(self.left_electrode_key, self.right_electrode_key, self.amp_reward, self.dur_reward)
        else:
            """同时刺激所有电极"""
            self.update_stimulation_stg1_reward_all_eles(self.amp_reward, self.dur_reward)

    # 用于单次刺激
    def update_stimulation_one_time(self, cur_electrode_key):
        self.update_stimulation_stg2(cur_electrode_key, self.amplitude, self.duration)


    # 基础功能
   
    # 惩罚刺激，设置不同的标识符
    def update_stimulation_stg1_punish(self, eles_key, amplitude, duration):
        """
        左轮惩罚刺激信号
        # 1. 先配置后施加
        # 2. 检测控制器运行状态
        # 3. 设置配置参数，指定电极，启动允许刺激，指定电极上的刺激参数（幅值，持续时间）
        # 4. 启动控制器，记录刺激时刻，并触发
        # 5. 设置控制器为停止态
        
        duration 与 amplitude 共同构成一个脉冲串，一一对应刺激序列，可以把amplitude第三个位置设置为0，表示休息期
        """
        self._debug("=====================update_stimulation_stg1_punish")
        self.stimulating_stim.request_controller_stop()
        # 刺激类型设置
        left_punish = StimulationSpec(StimulationType.PUNISHMENT, StimulationPosition.LEFT)
        # 进行电极参数配置
        left_pulse_count = len(amplitude)/3
        self._debug("left punish pulse count:", left_pulse_count)
        if left_pulse_count >= 2:
            # 多脉冲串
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, amplitude, duration, KeypressSource.KEYPRESS_F1.value, int(left_pulse_count), left_punish)
        else:
            # 单脉冲
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, amplitude, duration, KeypressSource.KEYPRESS_F1.value, int(left_pulse_count), left_punish)
            
        self.stimulating_stim.startRun()
        self.stimulating_stim.trigger_stimulation(TriggerType.F1.value)
        
        # 刺激时刻标识 通过事件的形式标识正 负 0 脉冲 最终需要和记录到的通道数据进行对齐
 

    def update_stimulation_stg2_punish(self, eles_key, amplitude, duration):
        """
        右轮惩罚刺激信号
        # 1. 先配置后施加
        # 2. 检测控制器运行状态
        # 3. 设置配置参数，指定电极，启动允许刺激，指定电极上的刺激参数（幅值，持续时间）
        # 4. 启动控制器，记录刺激时刻，并触发
        # 5. 设置控制器为停止态
        
        duration 与 amplitude 共同构成一个脉冲串，一一对应刺激序列，可以把amplitude第三个位置设置为0，表示休息期
        """
        self._debug("=====================update_stimulation_stg2_punish")
        self.stimulating_stim.request_controller_stop()
        # 刺激类型设置
        right_punish = StimulationSpec(StimulationType.PUNISHMENT, StimulationPosition.RIGHT)
        # 进行电极参数配置
        left_pulse_count = len(amplitude)/3
        self._debug("right punish pulse count:", left_pulse_count)
        if left_pulse_count >= 2:
            # 多脉冲串
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, amplitude, duration, KeypressSource.KEYPRESS_F2.value, int(left_pulse_count), right_punish)
        else:
            # 单脉冲
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, amplitude, duration, KeypressSource.KEYPRESS_F2.value, int(left_pulse_count), right_punish)
            
        self.stimulating_stim.startRun()
        self.stimulating_stim.trigger_stimulation(TriggerType.F2.value)
        
        # 刺激时刻标识 通过事件的形式标识正 负 0 脉冲 最终需要和记录到的通道数据进行对齐

    def update_stimulation_stg1_reward(self, eles_key, amplitude, duration):
        """
        左轮奖励刺激信号
        # 1. 先配置后施加
        # 2. 检测控制器运行状态
        # 3. 设置配置参数，指定电极，启动允许刺激，指定电极上的刺激参数（幅值，持续时间）
        # 4. 启动控制器，记录刺激时刻，并触发
        # 5. 设置控制器为停止态
        
        duration 与 amplitude 共同构成一个脉冲串，一一对应刺激序列，可以把amplitude第三个位置设置为0，表示休息期
        """
        self._debug("=====================update_stimulation_stg1_reward")
        self.stimulating_stim.request_controller_stop()
        # 刺激类型设置
        left_reward = StimulationSpec(StimulationType.REWARD, StimulationPosition.LEFT)
        # 进行电极参数配置
        left_pulse_count = len(amplitude)/3
        self._debug("left reward pulse count:", left_pulse_count)
        if left_pulse_count >= 2:
            # 多脉冲串
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, amplitude, duration, KeypressSource.KEYPRESS_F1.value, int(left_pulse_count), left_reward)
        else:
            # 单脉冲
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, amplitude, duration, KeypressSource.KEYPRESS_F1.value, int(left_pulse_count), left_reward)
            
        self.stimulating_stim.startRun()
        self.stimulating_stim.trigger_stimulation(TriggerType.F1.value)


    def update_stimulation_stg2_reward(self, eles_key, amplitude, duration):
        """
        右轮奖励刺激信号
        # 1. 先配置后施加
        # 2. 检测控制器运行状态
        # 3. 设置配置参数，指定电极，启动允许刺激，指定电极上的刺激参数（幅值，持续时间）
        # 4. 启动控制器，记录刺激时刻，并触发
        # 5. 设置控制器为停止态
        
        duration 与 amplitude 共同构成一个脉冲串，一一对应刺激序列，可以把amplitude第三个位置设置为0，表示休息期
        """
        self._debug("=====================update_stimulation_stg2_reward")
        self.stimulating_stim.request_controller_stop()
        # 刺激类型设置
        right_reward = StimulationSpec(StimulationType.REWARD, StimulationPosition.RIGHT)
        # 进行电极参数配置
        left_pulse_count = len(amplitude)/3
        self._debug("right reward pulse count:", left_pulse_count)
        if left_pulse_count >= 2:
            # 多脉冲串
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, amplitude, duration, KeypressSource.KEYPRESS_F2.value, int(left_pulse_count), right_reward)
        else:
            # 单脉冲
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, amplitude, duration, KeypressSource.KEYPRESS_F2.value, int(left_pulse_count), right_reward)
            
        self.stimulating_stim.startRun()
        self.stimulating_stim.trigger_stimulation(TriggerType.F2.value)
        
        # 刺激时刻标识 通过事件的形式标识正 负 0 脉冲 最终需要和记录到的通道数据进行对齐

    def update_stimulation_stg1_stg2_reward(self, left_ele, right_ele, amplitude, duration):
        """
        奖励刺激信号，同时刺激左右电极
        # 1. 先配置后施加
        # 2. 检测控制器运行状态
        # 3. 设置配置参数，指定电极，启动允许刺激，指定电极上的刺激参数（幅值，持续时间）
        # 4. 启动控制器，记录刺激时刻，并触发
        # 5. 设置控制器为停止态
        
        duration 与 amplitude 共同构成一个脉冲串，一一对应刺激序列，可以把amplitude第三个位置设置为0，表示休息期
        """
        self._debug("=====================update_stimulation_stg1_stg2_reward")
        self.stimulating_stim.request_controller_stop()
        # 设置刺激类型枚举
        ALL_reward = StimulationSpec(StimulationType.REWARD, StimulationPosition.ALL)
        # 取消所有刺激参数配置
        eles_key = left_ele + right_ele
        # self.stimulating_stim.cancel_config(eles_key)
        # 进行电极参数配置
        left_pulse_count = len(amplitude)/3
        self._debug("all reward pulse count:", left_pulse_count)
        if left_pulse_count >= 2:
            # 多脉冲串
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, amplitude, duration, KeypressSource.KEYPRESS_F3.value, int(left_pulse_count), ALL_reward)
        else:
            # 单脉冲
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, amplitude, duration, KeypressSource.KEYPRESS_F3.value, int(left_pulse_count), ALL_reward)
            
        self.stimulating_stim.startRun()
        self.stimulating_stim.trigger_stimulation(TriggerType.F3.value)

    def update_stimulation_stg1_reward_all_eles(self):
        """
        奖励刺激信号，同时刺激所有电极
        # 1. 先配置后施加
        # 2. 检测控制器运行状态
        # 3. 设置配置参数，指定电极，启动允许刺激，指定电极上的刺激参数（幅值，持续时间）
        # 4. 启动控制器，记录刺激时刻，并触发
        # 5. 设置控制器为停止态
        
        duration 与 amplitude 共同构成一个脉冲串，一一对应刺激序列，可以把amplitude第三个位置设置为0，表示休息期
        """
        return
    
        self._debug("=====================update_stimulation_stg1_reward_all_eles")
        
        eles_key = ["A-000", "A-001", "A-002", "A-003", "A-004", "A-005", "A-006",
        "A-007", "A-008", "A-009", "A-010", "A-011", "A-012", "A-013",
        "A-014", "A-015", "A-016", "A-017", "A-018", "A-019", "A-020", "A-021", 
        "A-022", "A-023", "A-024", "A-025", "A-026", "A-027", "A-028", "A-029",
        "A-030", "A-031", "B-000", "B-001", "B-002", "B-003", "B-004", "B-005", "B-006",
        "B-007", "B-008", "B-009", "B-010", "B-011", "B-012", "B-013", "B-014", 
        "B-015", "B-016", "B-017", "B-018", "B-019", "B-020", "B-021", "B-022",
        "B-023", "B-024", "B-025", "B-026", "B-027", "B-028", "B-029", "B-030", "B-031"]
        
        self.stimulating_stim.request_controller_stop()
        # 取消所有刺激参数配置
        # self.stimulating_stim.cancel_config(eles_key)
        # 进行电极参数配置
        left_pulse_count = len(self.amplitude)/3
        self._debug("all-ele reward pulse count:", left_pulse_count)
        
        if left_pulse_count >= 2:
            # 多脉冲串
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, self.amplitude, self.duration, self.source8[2], int(left_pulse_count))
        else:
            # 单脉冲
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, self.amplitude, self.duration, self.source8[2], int(left_pulse_count))
            
        self.stimulating_stim.startRun()
        self.stimulating_stim.trigger_stimulation(self.trigger8[2])
    
    def update_stimulation_stg1(self, eles_key, amplitude, duration):
        """
        左轮环境刺激信号
        # 1. 先配置后施加
        # 2. 检测控制器运行状态
        # 3. 设置配置参数，指定电极，启动允许刺激，指定电极上的刺激参数（幅值，持续时间）
        # 4. 启动控制器，记录刺激时刻，并触发
        # 5. 设置控制器为停止态
        
        duration 与 amplitude 共同构成一个脉冲串，一一对应刺激序列，可以把amplitude第三个位置设置为0，表示休息期
        """
        self._debug("=====================update_stimulation_stg1")
        self.stimulating_stim.request_controller_stop()
        # 刺激类型设置
        left_env = StimulationSpec(StimulationType.ENVIRONMENT, StimulationPosition.LEFT)
        # 进行电极参数配置
        left_pulse_count = len(amplitude)/3
        self._debug("left env pulse count:", left_pulse_count)
        self._debug('update_stimulation_stg1 amplitude:', amplitude)
        self._debug('update_stimulation_stg1 duration:', duration)
        if left_pulse_count >= 2:
            # 多脉冲串
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, amplitude, duration, KeypressSource.KEYPRESS_F1.value, int(left_pulse_count), left_env)
        else:
            # 单脉冲
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, amplitude, duration, KeypressSource.KEYPRESS_F1.value, int(left_pulse_count), left_env)
            
        self.stimulating_stim.startRun()
        self.stimulating_stim.trigger_stimulation(TriggerType.F1.value)
        
        # 记录刺激时刻

    def update_stimulation_stg2(self, eles_key, amplitude, duration):
        """
        右轮环境刺激信号
        # 1. 先配置后施加
        # 2. 检测控制器运行状态
        # 3. 设置配置参数，指定电极，启动允许刺激，指定电极上的刺激参数（幅值，持续时间）
        # 4. 启动控制器，记录刺激时刻，并触发
        # 5. 设置控制器为停止态
        """
        self._debug("=====================update_stimulation_stg2")
        self.stimulating_stim.request_controller_stop()
        # 刺激类型设置
        right_env = StimulationSpec(StimulationType.ENVIRONMENT, StimulationPosition.RIGHT)
        # 进行电极参数配置
        right_pulse_count = len(amplitude)/3
        self._debug("right env pulse count:", right_pulse_count)
        self._debug('update_stimulation_stg2 amplitude:', amplitude)
        self._debug('update_stimulation_stg2 duration:', duration)
        if right_pulse_count >= 2:
            # 多脉冲串
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, amplitude, duration, KeypressSource.KEYPRESS_F2.value, int(right_pulse_count), right_env)
        else:
            # 单脉冲
            self.stimulating_stim.configure_stimulation(eles_key, DigitalOutput.DIGITAL_CHANNEL_01.value, amplitude, duration, KeypressSource.KEYPRESS_F2.value, int(right_pulse_count), right_env)

        self.stimulating_stim.startRun()
        # 这里触发的是脉冲串刺激   
        self.stimulating_stim.trigger_stimulation(TriggerType.F2.value)
        
        
        
