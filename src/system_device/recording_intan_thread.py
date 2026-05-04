# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2024
# Written by Guiping Cao
# Time: 2024.04.30
# --------------------------------------------------------

import imp
import os
import copy

import scipy.io as scio

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import butter, lfilter

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *

import time


class ReadIntanDataThread(QThread):
    def __init__(self, recording):
        QThread.__init__(self, parent=None)

        self.recording = recording

        self.Samplingrate = 30000 # 这里设置默认值，后续根据读取的数据更改
        self.mChannels = 32
        self.one_second_npdata = np.zeros((self.mChannels, self.Samplingrate), np.int32)    # 保存一秒钟的数据
        self.numpy_data = None    # 实时最新100ms的数据
        self.sti_time_data = None    # 刺激时刻数据，刺激通道不确定，根据设定会返回不同的维度
        self.one_second_sti_time = np.zeros((1, self.Samplingrate), np.int32)    # 默认一个通道，后续按需调整

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_channel_data)    # 每隔一定时间，自动读取数据
        self.timer.start(90)    # 每200ms读取一次
    
    def on_channel_data(self):
        once_data, sti_time, timestamp, digital = self.recording.read_data(100)    # 输入参数表示100ms
        # print("Read data time:", time.time(), 30*"==")

        # # 根据读取的数据，更新采样率
        # if self.recording.sample_rate is not None:
        #     self.Samplingrate = self.recording.sample_rate
        #     self.one_second_npdata = np.zeros(self.Samplingrate * self.mChannels, np.int32)    # 保存一秒钟的数据

        # self.numpy_data = copy.deepcopy(once_data)
        # self.sti_time_data = copy.deepcopy(sti_time)
        
        self.numpy_data = once_data
        self.sti_time_data = digital

        if once_data is not None:
            length = int(self.Samplingrate * 0.1)
            self.update_one_second_data(self.numpy_data, self.sti_time_data, length)

            print("Shape of once_data:", np.shape(once_data))
        else:
            print("The current data is limited!...")


    def update_one_second_data(self, data, sti_time_data, length):
        tp = self.one_second_npdata[:, length:]    # 需要拿一部分出去，这是剩下需要保存的
        self.one_second_npdata[:, :-length] = tp
        self.one_second_npdata[:, -length:] = data[:self.mChannels, :]

        # for sti time data
        C, N = self.one_second_sti_time.shape    # 保存的1s的刺激时刻数据
        c_sti, n_sti = sti_time_data.shape       # 当前实时记录的刺激时刻数据
        if C == c_sti:  # 说明预设的通道与实际的刺激通道数量相同
            pass
        else:
            for i in range(c_sti):
                self.one_second_sti_time = np.concatenate([self.one_second_sti_time, self.one_second_sti_time[0:1, :]], axis=0)
                c_tp, n_tp = self.one_second_sti_time.shape
                if c_tp == c_sti:
                    break

        tp_sti = self.one_second_sti_time[:, length:]    # 需要拿一部分出去，这是剩下需要保存的
        self.one_second_sti_time[:, :-length] = tp_sti
        self.one_second_sti_time[:, -length:] = sti_time_data


    def get_one_second_data(self):
        return self.one_second_npdata
    
    def get_one_second_sti_timdata(self):
        return self.one_second_sti_time

    def get_100_ms_data(self):
        return self.numpy_data