# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------


import time
import os
import clr
import numpy as np
import random

import ctypes

from System import Action
from System import *
from System.Collections import Generic
# from System import ComponentModel, Data, Drawing, Linq, Text
from System.Threading import Tasks

from src.infor_com_mea.clr_array_to_numpy import asNumpyArray

clr.AddReference(os.getcwd() + '/bin/x64/McsUsbNet.dll')
from Mcs.Usb import ElectrodeDacMuxEnumNet, ElectrodeModeEnumNet
from Mcs.Usb import CMcsUsbListNet
from Mcs.Usb import DeviceEnumNet

from Mcs.Usb import CStg200xBasicNet, CStg200xDownloadNet
from Mcs.Usb import McsBusTypeEnumNet
from Mcs.Usb import STG_DestinationEnumNet

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *




class Stimulation(QObject):
    stimulate_finshed = pyqtSignal(bool)

    def __init__(self, parent=None):
        super(Stimulation, self).__init__(parent)
        # QThread.__init__(self, parent=None)

        self.deviceList = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
        print("found %d stimulating devices" % (self.deviceList.Count))

        for i in range(self.deviceList.Count):
            listEntry = self.deviceList.GetUsbListEntry(i)
            print("Device: %s   Serial: %s" % (listEntry.DeviceName, listEntry.SerialNumber))

        self.sti_sig = None    # 刺激参数
        self.amplitude = None; self.duration = None   # 解析出的幅值和脉宽
        self.amp_reward = None; self.dur_reward = None   # 奖励信号
        self.left_electrode_key = None; self.right_electrode_key =None  # 左右刺激电极

        self.channel_map = {"47":1, "48":2, "46":3, "45":4, "38":5, "37":6, "28":7,
                    "36":8, "27":9, "17":10, "26":11, "16":12, "35":13, "25":14, "15":15,
                    "14":16, "24":17, "34":18, "13":19, "23":20, "12":21, "22":22, "33":23, 
                    "21":24, "32":25, "31":26, "44":27, "43":28, "41":29, "42":30, "52":31,
                    "51":32, "53":33, "54":34, "61":35, "62":36, "71":37, "63":38, "72":39,
                    "82":40, "73":41, "83":42, "64":43, "74":44, "84":45, "85":46, "75":47, 
                    "65":48, "86":49, "76":50, "87":51, "77":52, "66":53, "78":54, "67":55,
                    "68":56, "55":57, "56":58, "58":59, "57":60}

        self.ele_key_sti = None    # 刺激电极，用于刺激时刻的数据保存
        self.sti_data_amp = None
        self.sti_data_dur = None

        self.recording = None

        self.has_set_sti_ele_stg1 = False    # 判断是否已经设置电极
        self.has_set_sti_ele_stg2 = False    # 判断是否已经设置电极

        # self.initial_device()   # 由communication控制初始化
        self.connect_state = False
    
    def __del__(self):
        if self.connect_state:
            print("Reset all electrodes!...")
            self.reset_electrode_stimulating_state()
        else:
            print("The stimulator is not connected!...")



    def initial_device(self):
        self.connect_state = True

        self.device = CStg200xDownloadNet()
        # self.device = CStg200xStreamingNet()  # not recommend for stg

        self.device.Stg200xPollStatusEvent += self.PollHandler

        self.device.Connect(self.deviceList.GetUsbListEntry(0))

        memory = self.device.GetTotalMemory()
        # segment_num = 2
        # segmentmemory = np.uint16(memory / segment_num)
        # self.device.SegmentDefine(segmentmemory)

        nchannels = self.device.GetNumberOfAnalogChannels()
        nsync = self.device.GetNumberOfSyncoutChannels()
        channel_cap = []
        syncout_cap = []
        # for i in range(5):
        #     # self.device.SegmentSelect(i)
        #     # segment_mem = self.device.GetMemory()
        #
        #     for j in range(nchannels):
        #         channel_cap.append(segment_mem / (nchannels + nsync))
        #
        #     for m in range(nsync):
        #         syncout_cap.append(segment_mem / (nchannels + nsync))
        #
        #     self.device.SetCapacity(channel_cap, syncout_cap)

        self.voltageRange = self.device.GetVoltageRangeInMicroVolt(0)
        self.voltageResulution = self.device.GetVoltageResolutionInMicroVolt(0)
        self.currentRange = self.device.GetCurrentRangeInNanoAmp(0)
        self.currentResolution = self.device.GetCurrentResolutionInNanoAmp(0)

        print('Voltage Mode:  Range: %d mV  Resolution: %1.2f mV' % (
        self.voltageRange / 1000, self.voltageResulution / 1000.0))
        print('Current Mode:  Range: %d uA  Resolution: %1.2f uA' % (
        self.currentRange / 1000, self.currentResolution / 1000.0))

        # set up the trigger
        trigger_inputs = self.device.GetNumberOfTriggerInputs()
        print("There is %f trigger inputs."%(trigger_inputs))
        number_stg_source = self.device.GetNumberOfStimulationSourcesPerElectrode()

        channelmap = Array[UInt32]([1, 2])
        syncoutmap = Array[UInt32]([1, 2])
        repeat = Array[UInt32]([1, 1])

        # self.device.SetupTrigger(0,  Array[UInt32]([255]), Array[UInt32]([255]), Array[UInt32]([10]))
        self.device.SetupTrigger(0,  channelmap, syncoutmap, repeat)
        self.device.SetVoltageMode()
        self.reset_electrode_stimulating_state()


    def PollHandler(self, status, stgStatusNet, index_list):
        # print('%x %s' % (status, str(stgStatusNet.TiggerStatus[0])))
        print('%s %s' % (status, str(stgStatusNet.TiggerStatus[0])))
        # print(type(status))

        # for stg 1
        if status == 16777217 or status == "":
            print("Stg 1 Start!..", time.time())
            # 清空缓存，并传入对应的信号
            # self.recording.clear_recording_buffer_get_sti_time(self.ele_key_sti, self.sti_data_amp, self.sti_data_dur)
        elif status == 16777216:
            self.stimulate_finshed.emit(True)
            print("Stg 1 End!..", time.time())

        # for stg 2
        if status == 33554436 or status == "":
            print("Stg 2 Start!..", time.time())
            # 清空缓存，并传入对应的信号
            # self.recording.clear_recording_buffer_get_sti_time(self.ele_key_sti, self.sti_data_amp, self.sti_data_dur)
        elif status == 33554432:
            self.stimulate_finshed.emit(True)
            print("Stg 2 End!..", time.time())



        # if status == 2000004 or status == "":
        #     print("Sti Start!..", time.time())
        #     # 清空缓存，并传入对应的信号
        #     self.recording.clear_recording_buffer_get_sti_time(self.ele_key_sti, self.sti_data_amp, self.sti_data_dur)
        # elif status == 2000000:
        #     print("Sti End!..", time.time())


    def set_recording(self, recording):
        self.recording = recording

    def disconnect(self):
        self.device.Disconnect()

    # 通过软件设置的刺激信号，进行解析
    # def set_sti_signal(self, signal):
    #     self.sti_sig = signal
    #     parameter = signal.para
    #     sti_para= signal.sti_para    # 包含刺激名称和组重复次数

    #     # 解析设置，变为奖惩刺激参数
    #     amp = []; dur = []
    #     for k in range(sti_para["reapeat_times"]):  # 总的group的重复次数
    #         for i in range(len(parameter)):    # 循环多个不同的单元信号，构成一个group
    #             tp = parameter[i]
    #             for j in range(tp["cycles"]):  # 每个单元信号，有固定信号循环次数
    #                 amplitude = [tp['amplitude_1']*1000, 0, tp['amplitude_2']*1000, 0]   # μV 
    #                 duration = [tp['duration_1'], tp['duration_2'], tp['duration_3'], tp["ISI"]*1000]     # μs

    #                 amp.extend(amplitude)
    #                 dur.extend(duration)

    #     self.amplitude = Array[Int32](amp)
    #     self.duration = Array[UInt64](dur)

    #     # 电极设置
    #     l_electrode = signal.stimulating_list[0]
    #     r_electrode = signal.stimulating_list[1]

    #     self.left_electrode_key = []; self.right_electrode_key = []  # 保存了刺激电极index key
    #     for i in range(len(l_electrode)):
    #         self.left_electrode_key.append(l_electrode[i])
        
    #     for i in range(len(r_electrode)):
    #         self.right_electrode_key.append(r_electrode[i])

    #     self.set_reward_sti()

    import random

    # 通过软件设置的刺激信号，进行解析
    def set_sti_signal(self, signal):
        self.sti_sig = signal
        parameter = signal.para
        sti_para = signal.sti_para    # 包含刺激名称和组重复次数

        # ----------------------------
        # 范式控制参数
        # ----------------------------
        duty_cycle = 0.6            # 占空比：slot里有多少比例打刺激
        min_burst_len = 2           # burst 最小连续刺激数（>=2）
        freq_hz = 5                 # 5Hz 时间基准
        # ✅ 简单开关：True=随机时序；False=固定时序（每个slot都刺激）        ================================惩罚刺激随机与否修改位置=================================
        random_mode = True
        # ----------------------------

        def make_burst_mask(n_slots, duty, min_len, rng):
            """
            生成长度为 n_slots 的 0/1 mask：
            - 1 表示该 5Hz slot 打刺激
            - 0 表示该 5Hz slot 空档
            mask 的 1 会以 burst(连续1) 的形式出现，burst 起点随机
            """
            if n_slots <= 0:
                return []

            n_on = int(round(n_slots * duty))
            n_on = max(0, min(n_on, n_slots))
            n_off = n_slots - n_on

            # 极端情况：不刺激 or 全刺激
            if n_on == 0:
                return [0] * n_slots
            if n_on == n_slots:
                return [1] * n_slots

            # 如果要求 burst 最小长度，但 n_on < min_len，无法满足
            # 这里选择：降级成一个 burst（长度=n_on）
            if n_on < min_len:
                return [1] * n_on + [0] * (n_slots - n_on)

            # burst 数量 b 的上限：
            # 1) 每个 burst 至少 min_len 个 1 => b <= floor(n_on / min_len)
            # 2) burst 之间至少留 1 个 0（否则会合并为更长 burst）
            #    n_off 个 0 最多能分隔成 b-1 个内部间隔 => b <= n_off + 1
            max_b = min(n_on // min_len, n_off + 1)
            b = rng.randint(1, max_b)

            # 先给每个 burst 分配 min_len 个 1
            burst_lens = [min_len] * b
            remaining = n_on - min_len * b

            # 把剩余的 1 随机分配到各个 burst 上（让 burst 长度有随机性）
            for _ in range(remaining):
                burst_lens[rng.randrange(b)] += 1

            # 现在分配 0：总共有 n_off 个 0
            # 内部 gap（burst 之间）至少 1 个 0，共 (b-1) 个内部 gap
            internal_min = b - 1
            # 理论上 b <= n_off + 1 已保证 n_off >= internal_min
            zeros_left = n_off - internal_min

            # gap 一共有 b+1 段：前导、(b-1)个内部、尾随
            gaps = [0] * (b + 1)
            # 先给每个内部 gap 1 个 0
            for gi in range(1, b):
                gaps[gi] = 1

            # 剩余 0 随机撒到所有 gap（包括前导/尾随/内部）
            for _ in range(zeros_left):
                gaps[rng.randrange(b + 1)] += 1

            # 拼接 mask： 0... + 111.. + 0.. + 111.. + ... + 0...
            mask = []
            mask.extend([0] * gaps[0])
            for idx, L in enumerate(burst_lens):
                mask.extend([1] * L)
                mask.extend([0] * gaps[idx + 1])

            # 保险：裁剪到 n_slots（理论上应正好等长）
            return mask[:n_slots]

        # 用于可复现随机（如果你想每次都不一样就删掉 seed 或改成 None）
        rng = random.Random(None)

        # 解析设置，变为奖惩刺激参数
        amp = []
        dur = []

        # 固定 5Hz slot 时长（单位 μs）
        slot_us = int(round(1_000_000 / freq_hz))  # 5Hz -> 200000 μs

        for k in range(sti_para["reapeat_times"]):  # 总的group的重复次数
            for i in range(len(parameter)):         # 循环多个不同的单元信号，构成一个group
                tp = parameter[i]

                # cycles = 20 -> 4s（在你设定 freq=5Hz 前提下）
                n_slots = int(tp["cycles"])

                # ✅ 开关逻辑：random_mode 决定 mask 如何生成
                if random_mode:
                    mask = make_burst_mask(
                        n_slots=n_slots,
                        duty=duty_cycle,
                        min_len=min_burst_len,
                        rng=rng
                    )
                else:
                    # 固定模式：每个 slot 都刺激（严格 5Hz 节拍）
                    mask = [1] * n_slots

                # 取单个 pulse 的三个阶段 duration（单位 μs）
                d1 = int(tp["duration_1"])
                d2 = int(tp["duration_2"])
                d3 = int(tp["duration_3"])

                pulse_us = d1 + d2 + d3
                if pulse_us >= slot_us:
                    raise ValueError(
                        f"Pulse duration (d1+d2+d3={pulse_us} μs) must be < slot_us ({slot_us} μs). "
                        f"Otherwise 5Hz slot cannot be formed."
                    )

                # 关键：不要用界面 ISI，slot 内最后一段 gap 由 5Hz 决定
                gap_us = slot_us - pulse_us  # 保证每个 slot 总长度 = 200ms

                # 幅值（单位：μV）
                a1_uv = int(tp["amplitude_1"] * 1000)
                a2_uv = int(tp["amplitude_2"] * 1000)

                # 逐 slot 拼接刺激序列
                for j in range(n_slots):
                    if mask[j] == 1:
                        amplitude = [a1_uv, 0, a2_uv, 0]
                    else:
                        # 空档 slot：幅值全 0，但 duration 仍占据一个完整 5Hz slot
                        amplitude = [0, 0, 0, 0]

                    duration = [d1, d2, d3, gap_us]  # μs

                    amp.extend(amplitude)
                    dur.extend(duration)

        self.amplitude = Array[Int32](amp)
        self.duration = Array[UInt64](dur)

        # 电极设置
        l_electrode = signal.stimulating_list[0]
        r_electrode = signal.stimulating_list[1]

        self.left_electrode_key = []   # 保存了刺激电极 index key
        self.right_electrode_key = []
        for idx in l_electrode:
            self.left_electrode_key.append(idx)

        for idx in r_electrode:
            self.right_electrode_key.append(idx)

        self.set_reward_sti()


    def set_reward_sti(self):
        """
        奖励刺激：100Hz, 75mV, 100ms
        """
        left = 100 # 刺激频率              ================================= 频率修改位置 =================================
        l_time = 1000000 / left # 对应刺激频率下的脉冲间隔

        amplidude = 75000.0  # μV

        # 刺激信号单元,定义100ms的信号去刺激，下一次更新即可
        left_amp = [];left_dur = []
        for i in range(int(left*0.1)):    # 100ms内的刺激次数
            # l_amplitude = Array[Int32]([1000, 1000, 0])    # 幅值 1000μV
            # l_duration = Array[UInt64]([20000, 20000, l_time])  # 200μs
            l_amplitude =[-amplidude, amplidude, 0]    # 幅值 1000μV
            l_duration = [200, 200, l_time]  # 200μs
            left_amp.extend(l_amplitude)
            left_dur.extend(l_duration)

        self.amp_reward = Array[Int32](left_amp)
        self.dur_reward = Array[UInt64](left_dur)

        print("Reward stimulus signal has been set!...")


    def update_amplitude_duration(self, m_amplitude, m_duration):
        """
        m_amplitude: μV
        m_duration: μs
        """
        # 解析设置，变为奖惩刺激参数
        amp = np.array(m_amplitude[0])     # 转变为μV
        dur = np.array(m_duration[0])      # 转变为μs

        self.amplitude = Array[Int32](amp)
        self.duration = Array[UInt64](dur)

        print("Setting dynamic model stimulation signal!...")



    def update_record_stimulation(self, left, right):
        """左右两侧传感器的信号经编码后， 传入并通过device输出刺激
        left: 刺激频率，整数
        right: 刺激频率，整数
        """
        # # 生成随机频率 (4-40Hz)
        # left = random.randint(4, 40)    # 左侧随机频率
        # right = random.randint(4, 40)   # 右侧随机频率
        # print(f"随机刺激频率 - 左侧: {left}Hz, 右侧: {right}Hz")  # 调试输出

        l_time = 1000000 / left
        r_time = 1000000 / right    # μs，间隔时间

        # amplidude = 500000.0  # μV
        amplidude = 75000.0  # μV
        # amplidude = 500000.0  # μV
        # amplidude = 1200000.0  # μV

        # 刺激信号单元,定义100ms的信号去刺激，下一次更新即可
        left_amp = [];left_dur = []
        for i in range(int(left*0.1 + 0.5)):    # 500ms内的刺激次数
            # l_amplitude = Array[Int32]([1000, 1000, 0])    # 幅值 1000μV
            # l_duration = Array[UInt64]([20000, 20000, l_time])  # 200μs
            l_amplitude =[amplidude, -amplidude, 0]    # 幅值 1000μV
            l_duration = [200, 200, l_time]  # 200μs
            left_amp.extend(l_amplitude)
            left_dur.extend(l_duration)

        right_amp = [];right_dur = []
        for i in range(int(right*0.1 + 0.5)):
            # r_amplitude = Array[Int32]([1000, 1000, 0])
            # r_duration = Array[UInt64]([200, 200, r_time])
            r_amplitude = [amplidude, -amplidude, 0]
            r_duration = [200, 200, r_time]
            right_amp.extend(r_amplitude)
            right_dur.extend(r_duration)

        left_amp = Array[Int32](left_amp)
        left_dur = Array[UInt64](left_dur)

        right_amp = Array[Int32](right_amp)
        right_dur = Array[UInt64](right_dur)

        # 环境信息的刺激控制
        if left >= right:
            self.update_stimulation_stg1(self.left_electrode_key, left_amp, left_dur)
        else:
            self.update_stimulation_stg2(self.right_electrode_key, right_amp, right_dur)


        # self.update_stimulation_stg1(self.left_electrode_key, left_amp, left_dur)
        # self.update_stimulation_stg2(self.right_electrode_key, right_amp, right_dur)

    def update_record_stimulation_dynamic_model_left(self, ampli, duri):
        """左右两侧传感器的信号经编码后， 传入并通过device输出刺激
        left: 刺激频率，整数
        right: 刺激频率，整数
        动力学模型产生的优化环境刺激
        """

        # 解析设置，变为奖惩刺激参数
        amp = np.array(ampli[0])     # 转变为μV
        dur = np.array(duri[0])      # 转变为μs

        amp = Array[Int32](amp)
        dur = Array[UInt64](dur)

        # 环境信息的刺激控制
        self.update_stimulation_stg1(self.left_electrode_key, amp, dur)

    def update_record_stimulation_dynamic_model_right(self, ampli, duri):
        """左右两侧传感器的信号经编码后， 传入并通过device输出刺激
        left: 刺激频率，整数
        right: 刺激频率，整数
        动力学模型产生的优化环境刺激,右侧
        """

        # 解析设置，变为奖惩刺激参数
        amp = np.array(ampli[0])     # 转变为μV
        dur = np.array(duri[0])      # 转变为μs

        amp = Array[Int32](amp)
        dur = Array[UInt64](dur)

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
        在障碍物ID变化时，进行奖励刺激，两侧同时刺激
        """
        all_ele = True
        if not all_ele:
            """同时刺激所连接的两个电极"""
            self.update_stimulation_stg1_stg2_reward(self.left_electrode_key, self.right_electrode_key, self.amp_reward, self.dur_reward)
        else:
            """同时刺激所有电极"""
            self.update_stimulation_stg1_reward_all_eles(self.amp_reward, self.dur_reward)


    # 用于单次刺激
    def update_stimulation_one_time(self, cur_electrode_key):
        self.update_stimulation_stg2(cur_electrode_key, self.amplitude, self.duration)


    def update_stimulation_stg1(self, eles_key, amplitude, duration):
        """
        环境刺激信号
        """
        # self.device.SendStop(UInt32(1))
        if not self.has_set_sti_ele_stg1:
            if not self.has_set_sti_ele_stg1 and not self.has_set_sti_ele_stg2:
                self.reset_electrode_stimulating_state()    # 都未连接的情况下，断开所有刺激器的连接

            self.has_set_sti_ele_stg1 = True


            for i in range(len(eles_key)):
                electrode = self.channel_map[eles_key[i]] - 1
                self.device.SetElectrodeMode(UInt32(electrode), ElectrodeModeEnumNet.emManual)

                # ElectrodeDacMux: DAC to use for stimulation
                self.device.SetElectrodeDacMux(UInt32(electrode), UInt32(0), ElectrodeDacMuxEnumNet.Stg1)

                # ElectrodeEnable: enable electrode for stimulation
                self.device.SetElectrodeEnable(UInt32(electrode), UInt32(0), True)

                # BlankingEnable: false: do not blank the ADC signal while stimulation is running
                self.device.SetBlankingEnable(UInt32(electrode), False)

                # AmplifierProtectionSwitch: false: Keep ADC connected to electrode even while stimulation is running
                self.device.SetEnableAmplifierProtectionSwitch(UInt32(electrode), False)

        channelmap = Array[UInt32]([1, 2])
        syncoutmap = Array[UInt32]([1, 2])
        repeat = Array[UInt32]([1, 1])
        self.device.SetVoltageMode()
        self.device.SetupTrigger(0,  channelmap, syncoutmap, repeat)
        # self.device.SetupTrigger(0, Array[UInt32]([255]), Array[UInt32]([255]), Array[UInt32]([10]))
        self.device.PrepareAndSendData(0, amplitude, duration, STG_DestinationEnumNet.channeldata_voltage)
        
        # 记录刺激时刻
        amp = asNumpyArray(amplitude, ctypes.c_int32); sync = []
        for i in range(len(amp)):
            if amp[i] > 0:
                sync.append(1 << 2)    #  1 << 8  2 << 8
            elif amp[i] < 0:
                sync.append(2 << 2)
            else:
                sync.append(0)
        sync = Array[UInt32](sync)
        self.device.PrepareAndSendData(0, sync, duration, STG_DestinationEnumNet.syncoutdata)

        self.device.SendStart(1)

        # 清空缓存，并传入对应的信号
        self.ele_key_sti = eles_key
        self.sti_data_amp = amplitude
        self.sti_data_dur = duration
        self.recording.clear_recording_buffer_get_sti_time(eles_key, amplitude, duration)


    def update_stimulation_stg2(self, eles_key, amplitude, duration):
        """
        环境刺激信号
        """
        # self.device.SendStop(UInt32(1))
        if not self.has_set_sti_ele_stg2:
            if not self.has_set_sti_ele_stg1 and not self.has_set_sti_ele_stg2:
                self.reset_electrode_stimulating_state()    # 都未连接的情况下，断开所有刺激器的连接

            self.has_set_sti_ele_stg2 = True


            for i in range(len(eles_key)):
                electrode = self.channel_map[eles_key[i]] - 1
                self.device.SetElectrodeMode(UInt32(electrode), ElectrodeModeEnumNet.emManual)

                # ElectrodeDacMux: DAC to use for stimulation
                self.device.SetElectrodeDacMux(UInt32(electrode), UInt32(0), ElectrodeDacMuxEnumNet.Stg2)

                # ElectrodeEnable: enable electrode for stimulation
                self.device.SetElectrodeEnable(UInt32(electrode), UInt32(0), True)

                # BlankingEnable: false: do not blank the ADC signal while stimulation is running
                self.device.SetBlankingEnable(UInt32(electrode), False)

                # AmplifierProtectionSwitch: false: Keep ADC connected to electrode even while stimulation is running
                self.device.SetEnableAmplifierProtectionSwitch(UInt32(electrode), False)

        channelmap = Array[UInt32]([1, 2])
        syncoutmap = Array[UInt32]([1, 2])
        repeat = Array[UInt32]([1, 1])
        self.device.SetVoltageMode()
        self.device.SetupTrigger(0,  channelmap, syncoutmap, repeat)
        # self.device.SetupTrigger(0, Array[UInt32]([255]), Array[UInt32]([255]), Array[UInt32]([10]))
        self.device.PrepareAndSendData(1, amplitude, duration, STG_DestinationEnumNet.channeldata_voltage)


        amp = asNumpyArray(amplitude, ctypes.c_int32); sync = []
        for i in range(len(amp)):
            if amp[i] > 0:
                sync.append(3 << 2)
            elif amp[i] < 0:
                sync.append(4 << 2)
            else:
                sync.append(0)
        sync = Array[UInt32](sync)
        self.device.PrepareAndSendData(1, sync, duration, STG_DestinationEnumNet.syncoutdata)

        self.device.SendStart(2)
        print("Start time SendStart", time.time())

        # 清空缓存，并传入对应的信号
        self.ele_key_sti = eles_key
        self.sti_data_amp = amplitude
        self.sti_data_dur = duration
        self.recording.clear_recording_buffer_get_sti_time(eles_key, amplitude, duration)


    # 惩罚刺激，设置不同的标识符
    def update_stimulation_stg1_punish(self, eles_key, amplitude, duration):
        """
        在出现撞击时，进行奖惩刺激
        """
        # self.device.SendStop(UInt32(1))
        if not self.has_set_sti_ele_stg1:
            if not self.has_set_sti_ele_stg1 and not self.has_set_sti_ele_stg2:
                self.reset_electrode_stimulating_state()    # 都未连接的情况下，断开所有刺激器的连接

            self.has_set_sti_ele_stg1 = True


            for i in range(len(eles_key)):
                electrode = self.channel_map[eles_key[i]] - 1
                self.device.SetElectrodeMode(UInt32(electrode), ElectrodeModeEnumNet.emManual)

                # ElectrodeDacMux: DAC to use for stimulation
                self.device.SetElectrodeDacMux(UInt32(electrode), UInt32(0), ElectrodeDacMuxEnumNet.Stg1)

                # ElectrodeEnable: enable electrode for stimulation
                self.device.SetElectrodeEnable(UInt32(electrode), UInt32(0), True)

                # BlankingEnable: false: do not blank the ADC signal while stimulation is running
                self.device.SetBlankingEnable(UInt32(electrode), False)

                # AmplifierProtectionSwitch: false: Keep ADC connected to electrode even while stimulation is running
                self.device.SetEnableAmplifierProtectionSwitch(UInt32(electrode), False)

        channelmap = Array[UInt32]([1, 2])
        syncoutmap = Array[UInt32]([1, 2])
        repeat = Array[UInt32]([1, 1])
        self.device.SetVoltageMode()
        self.device.SetupTrigger(0,  channelmap, syncoutmap, repeat)
        # self.device.SetupTrigger(0, Array[UInt32]([255]), Array[UInt32]([255]), Array[UInt32]([10]))
        self.device.PrepareAndSendData(0, amplitude, duration, STG_DestinationEnumNet.channeldata_voltage)
        
        # 记录刺激时刻
        amp = asNumpyArray(amplitude, ctypes.c_int32); sync = []
        for i in range(len(amp)):
            if amp[i] > 0:
                sync.append(5 << 2)    #  1 << 8  2 << 8
            elif amp[i] < 0:
                sync.append(6 << 2)
            else:
                sync.append(0)
        sync = Array[UInt32](sync)
        self.device.PrepareAndSendData(0, sync, duration, STG_DestinationEnumNet.syncoutdata)

        self.device.SendStart(1)

        # 清空缓存，并传入对应的信号
        self.ele_key_sti = eles_key
        self.sti_data_amp = amplitude
        self.sti_data_dur = duration
        self.recording.clear_recording_buffer_get_sti_time(eles_key, amplitude, duration)


    def update_stimulation_stg2_punish(self, eles_key, amplitude, duration):
        """
        在出现撞击时，进行奖惩刺激
        """
        # self.device.SendStop(UInt32(1))
        if not self.has_set_sti_ele_stg2:
            if not self.has_set_sti_ele_stg1 and not self.has_set_sti_ele_stg2:
                self.reset_electrode_stimulating_state()    # 都未连接的情况下，断开所有刺激器的连接

            self.has_set_sti_ele_stg2 = True


            for i in range(len(eles_key)):
                electrode = self.channel_map[eles_key[i]] - 1
                self.device.SetElectrodeMode(UInt32(electrode), ElectrodeModeEnumNet.emManual)

                # ElectrodeDacMux: DAC to use for stimulation
                self.device.SetElectrodeDacMux(UInt32(electrode), UInt32(0), ElectrodeDacMuxEnumNet.Stg2)

                # ElectrodeEnable: enable electrode for stimulation
                self.device.SetElectrodeEnable(UInt32(electrode), UInt32(0), True)

                # BlankingEnable: false: do not blank the ADC signal while stimulation is running
                self.device.SetBlankingEnable(UInt32(electrode), False)

                # AmplifierProtectionSwitch: false: Keep ADC connected to electrode even while stimulation is running
                self.device.SetEnableAmplifierProtectionSwitch(UInt32(electrode), False)

        channelmap = Array[UInt32]([1, 2])
        syncoutmap = Array[UInt32]([1, 2])
        repeat = Array[UInt32]([1, 1])
        self.device.SetVoltageMode()
        self.device.SetupTrigger(0,  channelmap, syncoutmap, repeat)
        # self.device.SetupTrigger(0, Array[UInt32]([255]), Array[UInt32]([255]), Array[UInt32]([10]))
        self.device.PrepareAndSendData(1, amplitude, duration, STG_DestinationEnumNet.channeldata_voltage)
        

        amp = asNumpyArray(amplitude, ctypes.c_int32); sync = []
        for i in range(len(amp)):
            if amp[i] > 0:
                sync.append(7 << 2)
            elif amp[i] < 0:
                sync.append(8 << 2)
            else:
                sync.append(0)
        sync = Array[UInt32](sync)
        self.device.PrepareAndSendData(1, sync, duration, STG_DestinationEnumNet.syncoutdata)

        self.device.SendStart(2)
        print("Start time SendStart", time.time())

        # 清空缓存，并传入对应的信号
        self.ele_key_sti = eles_key
        self.sti_data_amp = amplitude
        self.sti_data_dur = duration
        self.recording.clear_recording_buffer_get_sti_time(eles_key, amplitude, duration)


    # 惩罚刺激，设置不同的标识符
    def update_stimulation_stg1_reward(self, eles_key, amplitude, duration):
        """
        在出现撞击时，进行奖励刺激
        """
        # self.device.SendStop(UInt32(1))
        if not self.has_set_sti_ele_stg1:
            if not self.has_set_sti_ele_stg1 and not self.has_set_sti_ele_stg2:
                self.reset_electrode_stimulating_state()    # 都未连接的情况下，断开所有刺激器的连接

            self.has_set_sti_ele_stg1 = True

            for i in range(len(eles_key)):
                electrode = self.channel_map[eles_key[i]] - 1
                self.device.SetElectrodeMode(UInt32(electrode), ElectrodeModeEnumNet.emManual)

                # ElectrodeDacMux: DAC to use for stimulation
                self.device.SetElectrodeDacMux(UInt32(electrode), UInt32(0), ElectrodeDacMuxEnumNet.Stg1)

                # ElectrodeEnable: enable electrode for stimulation
                self.device.SetElectrodeEnable(UInt32(electrode), UInt32(0), True)

                # BlankingEnable: false: do not blank the ADC signal while stimulation is running
                self.device.SetBlankingEnable(UInt32(electrode), False)

                # AmplifierProtectionSwitch: false: Keep ADC connected to electrode even while stimulation is running
                self.device.SetEnableAmplifierProtectionSwitch(UInt32(electrode), False)

        channelmap = Array[UInt32]([1, 2])
        syncoutmap = Array[UInt32]([1, 2])
        repeat = Array[UInt32]([1, 1])
        self.device.SetVoltageMode()
        self.device.SetupTrigger(0,  channelmap, syncoutmap, repeat)
        # self.device.SetupTrigger(0, Array[UInt32]([255]), Array[UInt32]([255]), Array[UInt32]([10]))
        self.device.PrepareAndSendData(0, amplitude, duration, STG_DestinationEnumNet.channeldata_voltage)
        
        # 记录刺激时刻
        amp = asNumpyArray(amplitude, ctypes.c_int32); sync = []
        for i in range(len(amp)):
            if amp[i] > 0:
                sync.append((1 << 2) - 1)    #  1 << 8  2 << 8
            elif amp[i] < 0:
                sync.append((2 << 2) - 1)
            else:
                sync.append(0)
        sync = Array[UInt32](sync)
        self.device.PrepareAndSendData(0, sync, duration, STG_DestinationEnumNet.syncoutdata)

        self.device.SendStart(1)

        # 清空缓存，并传入对应的信号
        self.ele_key_sti = eles_key
        self.sti_data_amp = amplitude
        self.sti_data_dur = duration
        self.recording.clear_recording_buffer_get_sti_time(eles_key, amplitude, duration)


    def update_stimulation_stg2_reward(self, eles_key, amplitude, duration):
        """
        在出现撞击时，进行奖励刺激
        """
        # self.device.SendStop(UInt32(1))
        if not self.has_set_sti_ele_stg2:
            if not self.has_set_sti_ele_stg1 and not self.has_set_sti_ele_stg2:
                self.reset_electrode_stimulating_state()    # 都未连接的情况下，断开所有刺激器的连接

            self.has_set_sti_ele_stg2 = True

            for i in range(len(eles_key)):
                electrode = self.channel_map[eles_key[i]] - 1
                self.device.SetElectrodeMode(UInt32(electrode), ElectrodeModeEnumNet.emManual)

                # ElectrodeDacMux: DAC to use for stimulation
                self.device.SetElectrodeDacMux(UInt32(electrode), UInt32(0), ElectrodeDacMuxEnumNet.Stg2)

                # ElectrodeEnable: enable electrode for stimulation
                self.device.SetElectrodeEnable(UInt32(electrode), UInt32(0), True)

                # BlankingEnable: false: do not blank the ADC signal while stimulation is running
                self.device.SetBlankingEnable(UInt32(electrode), False)

                # AmplifierProtectionSwitch: false: Keep ADC connected to electrode even while stimulation is running
                self.device.SetEnableAmplifierProtectionSwitch(UInt32(electrode), False)

        channelmap = Array[UInt32]([1, 2])
        syncoutmap = Array[UInt32]([1, 2])
        repeat = Array[UInt32]([1, 1])
        self.device.SetVoltageMode()
        self.device.SetupTrigger(0,  channelmap, syncoutmap, repeat)
        # self.device.SetupTrigger(0, Array[UInt32]([255]), Array[UInt32]([255]), Array[UInt32]([10]))
        self.device.PrepareAndSendData(1, amplitude, duration, STG_DestinationEnumNet.channeldata_voltage)
        

        amp = asNumpyArray(amplitude, ctypes.c_int32); sync = []
        for i in range(len(amp)):
            if amp[i] > 0:
                sync.append((3 << 2) - 1)
            elif amp[i] < 0:
                sync.append((4 << 2) - 1)
            else:
                sync.append(0)
        sync = Array[UInt32](sync)
        self.device.PrepareAndSendData(1, sync, duration, STG_DestinationEnumNet.syncoutdata)

        self.device.SendStart(2)
        print("Start time SendStart", time.time())

        # 清空缓存，并传入对应的信号
        self.ele_key_sti = eles_key
        self.sti_data_amp = amplitude
        self.sti_data_dur = duration
        self.recording.clear_recording_buffer_get_sti_time(eles_key, amplitude, duration)


    def update_stimulation_stg1_stg2_reward(self, left_ele, right_ele, amplitude, duration):
        """
        在出现撞击时，进行奖励刺激
        """
        # self.device.SendStop(UInt32(1))
        if not self.has_set_sti_ele_stg1:
            if not self.has_set_sti_ele_stg1 and not self.has_set_sti_ele_stg2:
                self.reset_electrode_stimulating_state()    # 都未连接的情况下，断开所有刺激器的连接

            self.has_set_sti_ele_stg1 = True


            for i in range(len(left_ele)):
                electrode = self.channel_map[left_ele[i]] - 1
                self.device.SetElectrodeMode(UInt32(electrode), ElectrodeModeEnumNet.emManual)

                # ElectrodeDacMux: DAC to use for stimulation
                self.device.SetElectrodeDacMux(UInt32(electrode), UInt32(0), ElectrodeDacMuxEnumNet.Stg1)

                # ElectrodeEnable: enable electrode for stimulation
                self.device.SetElectrodeEnable(UInt32(electrode), UInt32(0), True)

                # BlankingEnable: false: do not blank the ADC signal while stimulation is running
                self.device.SetBlankingEnable(UInt32(electrode), False)

                # AmplifierProtectionSwitch: false: Keep ADC connected to electrode even while stimulation is running
                self.device.SetEnableAmplifierProtectionSwitch(UInt32(electrode), False)

        channelmap = Array[UInt32]([1, 2])
        syncoutmap = Array[UInt32]([1, 2])
        repeat = Array[UInt32]([1, 1])
        self.device.SetVoltageMode()
        self.device.SetupTrigger(0,  channelmap, syncoutmap, repeat)
        # self.device.SetupTrigger(0, Array[UInt32]([255]), Array[UInt32]([255]), Array[UInt32]([10]))
        self.device.PrepareAndSendData(0, amplitude, duration, STG_DestinationEnumNet.channeldata_voltage)
        
        # 记录刺激时刻
        amp = asNumpyArray(amplitude, ctypes.c_int32); sync = []
        for i in range(len(amp)):
            if amp[i] > 0:
                sync.append((1 << 2) - 1)    #  1 << 8  2 << 8
            elif amp[i] < 0:
                sync.append((2 << 2) - 1)
            else:
                sync.append(0)
        sync = Array[UInt32](sync)
        self.device.PrepareAndSendData(0, sync, duration, STG_DestinationEnumNet.syncoutdata)


        # for stg2 setting
        if not self.has_set_sti_ele_stg2:
            if not self.has_set_sti_ele_stg1 and not self.has_set_sti_ele_stg2:
                self.reset_electrode_stimulating_state()    # 都未连接的情况下，断开所有刺激器的连接

            self.has_set_sti_ele_stg2 = True
            

            for i in range(len(right_ele)):
                electrode = self.channel_map[right_ele[i]] - 1
                self.device.SetElectrodeMode(UInt32(electrode), ElectrodeModeEnumNet.emManual)

                # ElectrodeDacMux: DAC to use for stimulation
                self.device.SetElectrodeDacMux(UInt32(electrode), UInt32(0), ElectrodeDacMuxEnumNet.Stg2)

                # ElectrodeEnable: enable electrode for stimulation
                self.device.SetElectrodeEnable(UInt32(electrode), UInt32(0), True)

                # BlankingEnable: false: do not blank the ADC signal while stimulation is running
                self.device.SetBlankingEnable(UInt32(electrode), False)

                # AmplifierProtectionSwitch: false: Keep ADC connected to electrode even while stimulation is running
                self.device.SetEnableAmplifierProtectionSwitch(UInt32(electrode), False)

        channelmap = Array[UInt32]([1, 2])
        syncoutmap = Array[UInt32]([1, 2])
        repeat = Array[UInt32]([1, 1])
        self.device.SetVoltageMode()
        self.device.SetupTrigger(0,  channelmap, syncoutmap, repeat)
        # self.device.SetupTrigger(0, Array[UInt32]([255]), Array[UInt32]([255]), Array[UInt32]([10]))
        self.device.PrepareAndSendData(1, amplitude, duration, STG_DestinationEnumNet.channeldata_voltage)
        

        amp = asNumpyArray(amplitude, ctypes.c_int32); sync = []
        for i in range(len(amp)):
            if amp[i] > 0:
                sync.append((3 << 2) - 1)
            elif amp[i] < 0:
                sync.append((4 << 2) - 1)
            else:
                sync.append(0)
        sync = Array[UInt32](sync)
        self.device.PrepareAndSendData(1, sync, duration, STG_DestinationEnumNet.syncoutdata)

        self.device.SendStart(3)    # 同时开启两个刺激器

        # # 清空缓存，并传入对应的信号
        self.ele_key_sti = left_ele    # 原保存时间戳的功能已不适用，所以此处左右电极不使用，可理解为占位符
        self.sti_data_amp = amplitude
        self.sti_data_dur = duration
        self.recording.clear_recording_buffer_get_sti_time(left_ele, amplitude, duration)


    def update_stimulation_stg1_reward_all_eles(self, amplitude, duration):
        """
        在出现撞击时，进行奖励刺激, 刺激器1 连接所有电极进行刺激
        """
        # self.device.SendStop(UInt32(1))

        self.reset_electrode_stimulating_state()
        self.has_set_sti_ele_stg1 = False
        self.has_set_sti_ele_stg2 = False

        eles_key = ["47", "48", "46", "45", "38", "37", "28",
                    "36", "27", "17", "26", "16", "35", "25",
                    "14", "24", "34", "13", "23", "12", "22", "33", 
                    "21", "32", "31", "44", "43", "41", "42", "52",
                    "51", "53", "54", "61", "62", "71", "63", "72",
                    "82", "73", "83", "64", "74", "84", "85", "75", 
                    "65", "86", "76", "87", "77", "66", "78", "67",
                    "68", "55", "56", "58", "57"]

        for i in range(len(eles_key)):
            electrode = self.channel_map[eles_key[i]] - 1
            self.device.SetElectrodeMode(UInt32(electrode), ElectrodeModeEnumNet.emManual)

            # ElectrodeDacMux: DAC to use for stimulation
            self.device.SetElectrodeDacMux(UInt32(electrode), UInt32(0), ElectrodeDacMuxEnumNet.Stg1)

            # ElectrodeEnable: enable electrode for stimulation
            self.device.SetElectrodeEnable(UInt32(electrode), UInt32(0), True)

            # BlankingEnable: false: do not blank the ADC signal while stimulation is running
            self.device.SetBlankingEnable(UInt32(electrode), False)

            # AmplifierProtectionSwitch: false: Keep ADC connected to electrode even while stimulation is running
            self.device.SetEnableAmplifierProtectionSwitch(UInt32(electrode), False)

        channelmap = Array[UInt32]([1, 2])
        syncoutmap = Array[UInt32]([1, 2])
        repeat = Array[UInt32]([1, 1])
        self.device.SetVoltageMode()
        self.device.SetupTrigger(0,  channelmap, syncoutmap, repeat)
        # self.device.SetupTrigger(0, Array[UInt32]([255]), Array[UInt32]([255]), Array[UInt32]([10]))
        self.device.PrepareAndSendData(0, amplitude, duration, STG_DestinationEnumNet.channeldata_voltage)
        
        # 记录刺激时刻
        amp = asNumpyArray(amplitude, ctypes.c_int32); sync = []
        for i in range(len(amp)):
            if amp[i] > 0:
                sync.append((1 << 2) - 1)    #  1 << 8  2 << 8
            elif amp[i] < 0:
                sync.append((2 << 2) - 1)
            else:
                sync.append(0)
        sync = Array[UInt32](sync)
        self.device.PrepareAndSendData(0, sync, duration, STG_DestinationEnumNet.syncoutdata)

        self.device.SendStart(1)

        # 清空缓存，并传入对应的信号
        self.ele_key_sti = eles_key
        self.sti_data_amp = amplitude
        self.sti_data_dur = duration
        self.recording.clear_recording_buffer_get_sti_time(eles_key, amplitude, duration)


    def reset_electrode_stimulating_state(self):
        for i in self.channel_map:
            ele = self.channel_map[i]
            self.device.SetElectrodeMode(UInt32(ele), ElectrodeModeEnumNet.emAutomatic)

            self.device.SetElectrodeDacMux(UInt32(ele), UInt32(0), ElectrodeDacMuxEnumNet.Ground)

            # ElectrodeEnable: enable electrode for stimulation
            self.device.SetElectrodeEnable(UInt32(ele), UInt32(0), False)

            # BlankingEnable: false: do not blank the ADC signal while stimulation is running
            self.device.SetBlankingEnable(UInt32(ele), True)

            # AmplifierProtectionSwitch: false: Keep ADC connected to electrode even while stimulation is running
            self.device.SetEnableAmplifierProtectionSwitch(UInt32(ele), True)