# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

import imp
import os

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


class SpikeDetection(QThread):
    def __init__(self, sample_rate=25000, order=4, min_fre=300, max_fre=3000, multiplier=5, period=2):
        QThread.__init__(self, parent=None)
        
        self.sample_rate = sample_rate    # 设备的采样率
        self.min_fre = min_fre                  # 最小截断频率
        self.max_fre = max_fre                 # 最大截断频率
        self.multiplier = multiplier            # 乘数因子
        self.period = period                    # 不应期时间，ms
        self.order = order                       # 滤波函数的阶数

        self.ref_period = self.period * 0.001 * self.sample_rate    # 计算间隔帧数
        up = self.max_fre * 2 / self.sample_rate
        down = self.min_fre * 2 / self.sample_rate
        self.b, self.a = butter(self.order, [down, up], "bandpass")

        self.data = None
        self.spike_num = 0

    def update_para(self, sample_rate=10000, order=4, min_fre=300, max_fre=3000, multiplier=5, period=2):
        self.sample_rate = sample_rate    # 设备的采样率
        self.min_fre = min_fre                  # 最小截断频率
        self.max_fre = max_fre                 # 最大截断频率
        self.multiplier = multiplier            # 乘数因子
        self.period = period                    # 不应期时间，ms
        self.order = order                       # 滤波函数的阶数

    def set_data(self, data):
        self.data = data


    def run(self):
        from time import perf_counter as _pc
        _t_start = _pc()
        try:
            # t1 = time.time()
            filtered = signal.filtfilt(self.b, self.a, self.data)
            # print("Filter time", time.time() - t1)
            sigma = np.std(filtered)

            threshold = -self.multiplier * sigma
            spikes = np.zeros_like(filtered, dtype=np.uint8)
            spikes[filtered < threshold] = 1

            length = len(spikes)
            index = np.where(spikes > 0)[0]
            # t2 = time.time()
            for i in range(len(index)):
                pos = index[i]
                if spikes[pos] == 1:
                    ref_start = pos + 1
                    ref_end = int(pos + self.ref_period)
                    if ref_end > length:
                        spikes[ref_start:] = 0
                    else:
                        spikes[ref_start:ref_end] = 0

            if len(spikes) == 0:
                self.spike_num = 0
            else:
                self.spike_num = np.sum(spikes)
        except:
            print("Error on spike detection!...")
        finally:
            try:
                from src.system_device.timing_logger import TimingLogger
                TimingLogger.get().log_spike_async((_pc() - _t_start) * 1000.0)
            except Exception:
                pass
            # return self.spike_num

        # print("for time", time.time() - t2)
        # print("Spike num is 2", spike_num)

        # return filtered, spike_num
        # return self.spike_num

    def get_spike_num(self):
        return self.spike_num





#  for spike detections
class DetectionSpike_old(object):
    def __init__(self, sample_rate=25000, order=4, min_fre=300, max_fre=6000, multiplier=5, period=2):
        self.sample_rate = sample_rate    # 设备的采样率
        self.min_fre = min_fre                  # 最小截断频率
        self.max_fre = max_fre                 # 最大截断频率
        self.multiplier = multiplier            # 乘数因子
        self.period = period                    # 不应期时间，ms
        self.order = order                       # 滤波函数的阶数

        self.ref_period = self.period * 0.001 * self.sample_rate    # 计算间隔帧数
        up = self.max_fre * 2 / self.sample_rate
        down = self.min_fre * 2 / self.sample_rate
        self.b, self.a = butter(self.order, [down, up], "bandpass")

    def spike_detection(self, data):
        # t1 = time.time()
        filtered = signal.filtfilt(self.b, self.a, data)
        # print("Filter time", time.time() - t1)
        sigma = np.std(filtered)

        threshold = -self.multiplier * sigma
        spikes = np.zeros_like(filtered, dtype=np.uint8)
        spikes[filtered < threshold] = 1

        length = len(filtered)
        # t2 = time.time()
        for i in range(length):
            if spikes[i] == 1:
                ref_start = i + 1
                ref_end = int(i + self.ref_period)
                if ref_end > length:
                    spikes[ref_start:] = 0
                else:
                    spikes[ref_start:ref_end] = 0

        spike_num = np.sum(spikes)

        # print("for time", time.time() - t2)
        # print("Spike num is", spike_num)

        return filtered, spike_num


