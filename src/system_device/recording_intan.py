# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2024
# Written by Guiping Cao
# Time: 2024.04.14
# --------------------------------------------------------

import os
import numpy as np
import time
import copy

from src.system_device.RealRHXDataRead import RealTimeDataReader
from src.system_device.recording_intan_thread import ReadIntanDataThread

from src.infor_com_mea.spike_detection import SpikeDetection


from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *


class RecordingIntan(QObject):
    def __init__(self):
        super(RecordingIntan, self).__init__()

        self.recording = RealTimeDataReader()
        self.recording.start()

        self.recording_para = None        # 记录电极
        self.spike_detection_para = None  # spike检测参数

        # for data saving
        self.save_path = os.getcwd() + "\\" + str(time.time()) + ".h5"            # 保存数据的路径
        self.save_state = False    # 控制是否保存数据，默认不保存

        self.spike_det_left = SpikeDetection()
        self.spike_det_right = SpikeDetection()

        self.Samplingrate = 30000    # 25000
        self.callbackThreshold = self.Samplingrate // 10
        self.frame_ret = None    # 实际读取的帧数

        self.one_second_npdata = None    # 1s内数据的缓存池

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
        
        base_path = os.path.dirname(self.save_path)
        np.savez(base_path + "\channel_map_intan.npz", channel_map=[self.channel_map])

        self.recording_thread = ReadIntanDataThread(self.recording)
        self.recording_thread.start()

    def __del__(self):
        self.recording.file_monitor.stop()

        self.stop_recording()

    def set_save_path(self, path):
        """
        由于INTAN系统已自动记录了数据，此处暂时无需再保存
        """
        self.save_path = path
        # self.channel_data_saving_thread.set_save_path(path)

    def set_save_state(self, state):
        self.save_state = state

    def stop_recording(self):
        del self.recording_thread
        self.recording_thread = None

    def set_record_para(self, sig):
        self.recording_para = sig


    # 获取MEA的输出信号，并检测Spike
    def get_recording(self):
        """
        返回记录信号, out_left, out_right
        """
        data = self.get_recording_signal_all_channel()
        sti_time = self.get_recording_sti_time_data()

        # 需对数据进行筛选，剔除刺激后10ms内数据 TODO 剔除刺激伪迹
        data = self.get_spike_data_from_channel_data(data, sti_time)

        if sti_time.max() > 0:
            print(30*"===")

        recording_left_node = self.recording_para.recording_list
        left = recording_left_node[0]
        right = recording_left_node[1]

        # 每个通道单独计算，用于动力学模型
        left_spike = []
        try:
            for i in range(len(left)):
                l_data = data[self.channel_map[left[i]]]
                # spikes_num = self.spike_det.run(l_data)
                self.spike_det_left.set_data(l_data)
                self.spike_det_left.start()
                spikes_num = self.spike_det_left.get_spike_num()
                left_spike.append(spikes_num)
        except:
            print("Error on left spikes detection...")

        right_spike = []
        try:
            for i in range(len(right)):
                r_data = data[self.channel_map[right[i]]]
                # spikes_num = self.spike_det.run(r_data)
                self.spike_det_right.set_data(r_data)
                self.spike_det_right.start()
                spikes_num = self.spike_det_right.get_spike_num()
                right_spike.append(spikes_num)
        except:
            print("Error on right spikes detection...")


        # # 每个通道数据合并，只计算一次spike
        # left_spike = []
        # try:
        #     left_data = None
        #     for i in range(len(left)):
        #         l_data = data[self.channel_map[left[i]] - 1]
        #         if left_data is None:
        #             left_data = l_data
        #         else:
        #             left_data = np.append(left_data, l_data)
        #         # spikes_num = self.spike_det.run(l_data)
        #     self.spike_det_left.set_data(left_data)
        #     self.spike_det_left.start()
        #     spikes_num = self.spike_det_left.get_spike_num()
        #     left_spike.append(spikes_num)
        # except:
        #     print("Error on left spikes detection...")

        # right_spike = []
        # try:
        #     right_data = None
        #     for i in range(len(right)):
        #         r_data = data[self.channel_map[right[i]] - 1]
        #         if right_data is None:
        #             right_data = r_data
        #         else:
        #             right_data = np.append(right_data, r_data)
        #         # spikes_num = self.spike_det.run(r_data)
        #     self.spike_det_right.set_data(right_data)
        #     self.spike_det_right.start()
        #     spikes_num = self.spike_det_right.get_spike_num()
        #     right_spike.append(spikes_num)
        # except:
        #     print("Error on right spikes detection...")

        print("spike num left:", np.sum(left_spike))
        print("spike num right:", np.sum(right_spike))

        return left_spike, right_spike
    
    # 剔除刺激后的10ms内数据 （用于获取刺激标识符，剔除刺激伪迹，INTAN的实现未知）
    def get_spike_data_from_channel_data(self, channel_data, sti_time):    
        """
        刺激尾迹剔除 - 最简洁版本
        
        Args:
            channel_data: 神经信号 shape=(通道数, 样本数)
            digital: 数字信号 shape=(1, 样本数)，值为0或1
            artifact_ms: 剔除时长，默认10ms
        """
        # 1. 检测上升沿（0→1）
        sti_signal = sti_time[0]  # 取第一个通道
        edges = np.diff(sti_signal)
        rising_edges = np.where(edges == 1)[0] + 1  # 上升沿位置
        
        # 2. 剔除每个上升沿后的10ms数据
        samples_to_remove = int(30 * 10)  # 30000Hz下，10ms = 300个样本
        
        for start in rising_edges:
            end = min(start + samples_to_remove, channel_data.shape[1])
            channel_data[:, start:end] = 0
        
        return channel_data
    
    
        # 取 digital_in 第一行的数据，没问题
        # sti_signal = sti_time[0]
        
        # non_zero = np.where(sti_signal > 0)[0]    # 此帧为首次刺激时刻，每个频率时刻，需要重置
        # out_frame = []
        # for i in range(len(non_zero)):
        #     if len(out_frame) == 0:
        #         out_frame.append(non_zero[i])
        #     elif non_zero[i] > out_frame[-1] + 20:    # 判断若20帧内，即800μs内2非刺激信号(400μs有延迟)
        #         out_frame.append(non_zero[i])

        # # 根据找出的检测位置，获得特定时间内的，目标区域的检测数据，用于spike检测
        # out_raw_data = []
        # resolution = self.Samplingrate / 1000    # 每ms多少帧
        # if len(out_frame) > 0:    # 有刺激时刻
        #     for i in range(len(out_frame)):
        #         cur = out_frame[i]    # 刺激起始位置
        #         next_frame = cur + int(10 * resolution)
        #         # next_frame = cur + int(100 * resolution)

        #         if next_frame < len(channel_data[0]) - 1:
        #             channel_data[:, cur:next_frame] = 0
        #         else:
        #             channel_data[:, cur:] = 0

        # return channel_data
            

    def set_spike_detection_para(self, para):
        self.spike_detection_para = para
        self.spike_det_left.update_para(sample_rate=para['Sample_Rate'], 
            order=para['Filter_Order'], min_fre=para['Min_Frequency'],
            max_fre=para['Max_Frequency'], multiplier=para['Multiplier'],
            period=para['Refractor_Time'])

        self.spike_det_right.update_para(sample_rate=para['Sample_Rate'], 
            order=para['Filter_Order'], min_fre=para['Min_Frequency'],
            max_fre=para['Max_Frequency'], multiplier=para['Multiplier'],
            period=para['Refractor_Time'])


    # 获取一次读取时的数据(指定通道)，约100ms
    def get_recording_signal(self, channel):
        temp = self.recording_thread.get_100_ms_data()
        self.frame_ret = len(temp[0])
        x = np.arange(0, self.frame_ret, 1)
        
        return x, temp[channel]
    

    # 用于机器人避障时，读取MEA数据
    def get_recording_signal_all_channel(self):
        self.one_second_npdata = self.recording_thread.get_one_second_data()
        return self.one_second_npdata    # 1s内的数据
    
    def get_recording_sti_time_data(self):
        sti_time_data = self.recording_thread.get_one_second_sti_timdata()
        return sti_time_data

    def get_one_second_signal(self, channel):
        x = np.arange(0, self.Samplingrate, 1)
        if self.recording_thread is not None:
            self.one_second_npdata = self.recording_thread.get_one_second_data()

            # 测试在线去除刺激伪迹的可视化
            # sti_time_data = self.recording_thread.get_one_second_sti_timdata()
            # self.one_second_npdata = self.get_spike_data_from_channel_data(self.one_second_npdata, sti_time_data)


            return x, self.one_second_npdata[channel]



# 待完成：剔除刺激伪迹问题！