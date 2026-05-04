# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

import time
import os
from xmlrpc.client import Boolean
import clr
import ctypes
import copy

from pandas import UInt32Dtype
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import h5py

matplotlib.use('agg')

from System import *
clr.AddReference('System.Collections')
from System.Collections.Generic import List

from src.infor_com_mea.clr_array_to_numpy import asNumpyArray
from src.infor_com_mea.spike_detection import SpikeDetection
from src.infor_com_mea.save_channel_data import ChannelDataSaving

clr.AddReference(os.getcwd() + '/bin/x64/McsUsbNet.dll')
from Mcs.Usb import CMcsUsbListNet
from Mcs.Usb import DeviceEnumNet

from Mcs.Usb import CMeaDeviceNet
from Mcs.Usb import McsBusTypeEnumNet
from Mcs.Usb import DataModeEnumNet
from Mcs.Usb import SampleSizeNet, DigitalDatastreamEnableEnumNet

X = np.arange(0, 10, 0.01)  # X shape： (N,)
Ys = [np.sin(X + k / 10.0) for k in range(100)]

class Recording(object):
    def __init__(self):
        self.deviceList = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
        self.DataModeToSampleSizeDict = {
            DataModeEnumNet.Unsigned_16bit : SampleSizeNet.SampleSize16Unsigned,
            DataModeEnumNet.Signed_32bit :  SampleSizeNet.SampleSize32Signed}

        print("found %d devices" % (self.deviceList.Count))

        for i in range(self.deviceList.Count):
            listEntry = self.deviceList.GetUsbListEntry(i)
            print("Device: %s   Serial: %s" % (listEntry.DeviceName, listEntry.SerialNumber))

        # self.dataMode = DataModeEnumNet.Unsigned_16bit
        self.dataMode = DataModeEnumNet.Signed_32bit

        self.device = CMeaDeviceNet(McsBusTypeEnumNet.MCS_USB_BUS)

        self.numpy_data = None    # 用于记录每次读取的数据
        self.frame_ret = None

        self.record_300ms_mark = False        # 用于记录特定时间内的数据
        self.record_300ms_np = None

        self.recording_para = None        # 记录电极
        self.spike_detection_para = None  # spike检测参数
        
        # for data saving
        self.save_path = os.getcwd() + "\\" + str(time.time()) + ".h5"            # 保存数据的路径
        self.save_state = False    # 控制是否保存数据，默认不保存
                
        self.spike_det_left = SpikeDetection()
        self.spike_det_right = SpikeDetection()

        if self.deviceList.Count >= 1:
            self.device_setting()

    def get_device_count(self):
        return self.deviceList.Count

    def set_save_path(self, path):
        self.save_path = path
        self.channel_data_saving_thread.set_save_path(path)
    
    def set_save_state(self, state):
        self.save_state = state

    def device_setting(self):
        self.device.ChannelDataEvent += self.OnChannelData
        self.device.ErrorEvent += self.OnError

        self.device.Connect(self.deviceList.GetUsbListEntry(0))    # 通过index,连接指定设备

        self.Samplingrate = 25000

        self.device.SetSamplerate(self.Samplingrate, 1, 0)  # 设置采样率

        adcRange = self.device.GetVoltageRangeInMicroVolt(0)
        self.miliGain = self.device.GetGain()
        self.voltage_range = 1000 * adcRange / self.miliGain # in microvolt as well
        self.constant = self.voltage_range / pow(2, 23)

        voltageRanges = self.device.HWInfo().GetAvailableVoltageRangesInMicroVoltAndStringsInMilliVolt(self.miliGain)
        for i in range(0, len(voltageRanges)):
            print("(" + str(i) + ") " + voltageRanges[i].VoltageRangeDisplayStringMilliVolt)

        self.device.SetVoltageRangeByIndex(8, 0)
        self.voltage_min = -voltageRanges[0].VoltageRangeInMicroVolt / 1000    # mV
        self.voltage_max = voltageRanges[0].VoltageRangeInMicroVolt / 1000     # mV

        ch = 0
        stat, channel = self.device.HWInfo().GetNumberOfHWADCChannels(ch)
                    
        self.device.SetDataMode(self.dataMode, 0)
        # self.device.SetNumberOfChannels(channel)
        self.device.SetNumberOfAnalogChannels(60, 0, 0, 8, 0)

        self.device.EnableDigitalIn(DigitalDatastreamEnableEnumNet.DigitalIn | DigitalDatastreamEnableEnumNet.DigitalOut |
                                    DigitalDatastreamEnableEnumNet.Hs1SidebandLow | DigitalDatastreamEnableEnumNet.Hs1SidebandHigh, UInt32(0))
        # self.device.EnableDigitalIn(Boolean(True), np.uint32(0))
        self.device.EnableChecksum(True, 0)


        block = self.device.GetChannelsInBlock(0)
        print("Channels in Block: ", block)

        ana = 0; digi = 0; che = 0; tim = 0
        _, ana, digi, che,  tim, block = self.device.GetChannelLayout(ana, digi, che,  tim, block, 0)
                

        self.callbackThreshold = self.Samplingrate // 10

        if self.dataMode == DataModeEnumNet.Unsigned_16bit:
            mChannels = self.device.GetChannelsInBlock(0)
        else: # dataMode == DataModeEnumNet.Signed_32bit
            mChannels = self.device.GetChannelsInBlock(0) // 2
        self.total_channel = mChannels
        print("Number of Channels: ", mChannels)
        self.mChannels = mChannels

        self.channel_data_saving_thread = ChannelDataSaving(save_path=self.save_path, 
            sampling_rate=self.Samplingrate, constant=self.constant, channels=self.mChannels)


        self.one_second_npdata = np.zeros(self.Samplingrate*mChannels, np.int32)    # 保存一秒钟的数据

        self.device.SetSelectedData(mChannels, self.callbackThreshold * 10, self.callbackThreshold, self.DataModeToSampleSizeDict[self.dataMode], block)
        # self.device.SetSelectedChannels(mChannels, self.callbackThreshold * 10, self.callbackThreshold, self.DataModeToSampleSizeDict[self.dataMode], block)

        self.device.ChannelBlock_SetCommonThreshold(self.callbackThreshold)
        self.device.ChannelBlock_SetCheckChecksum(che, tim)

        # selChannels = np.zeros((mChannels), np.uint8)
        # self.device.SetSelectedChannels(0, 10 * self.callbackThreshold, self.callbackThreshold)

        self.channel_map = {"47":1, "48":2, "46":3, "45":4, "38":5, "37":6, "28":7,
                    "36":8, "27":9, "17":10, "26":11, "16":12, "35":13, "25":14, "15":15,
                    "14":16, "24":17, "34":18, "13":19, "23":20, "12":21, "22":22, "33":23, 
                    "21":24, "32":25, "31":26, "44":27, "43":28, "41":29, "42":30, "52":31,
                    "51":32, "53":33, "54":34, "61":35, "62":36, "71":37, "63":38, "72":39,
                    "82":40, "73":41, "83":42, "64":43, "74":44, "84":45, "85":46, "75":47, 
                    "65":48, "86":49, "76":50, "87":51, "77":52, "66":53, "78":54, "67":55,
                    "68":56, "55":57, "56":58, "58":59, "57":60}

        base_path = os.path.dirname(self.save_path)
        np.savez(base_path + "\channel_map.npz", channel_map=[self.channel_map])
        self.start_dacq()

    def start_dacq(self):
        self.device.StartDacq()
        print("Data acquisition thread and sampling start!...")


    def stop_dacq(self):
        self.device.StopDacq()
        print("Data acquisition thread and sampling is stopped!...")
    
    def disconnect(self):
        self.device.Disconnect()
        print("Device is disconnected!...")

    def plot_channel_data(self, channel, ret_frame, np_data):
        height = 85; width = 280

        channel_data = np.zeros((ret_frame), np.uint16)
        print("frames", ret_frame)
        x = np.arange(0, ret_frame, 1)
        for i in range(ret_frame):
            channel_data[i] = np_data[i * self.total_channel + channel]
            # x[i] = np.uint16(i * width / ret_frame)

        min_data = np.min(channel_data)
        max_data = np.max(channel_data)
        process_data = (channel_data - min_data + 1) * height / (max_data - min_data + 2)

        plt.plot(x, process_data)
        plt.show()

    def plot_channel_data_voltage(self, channel, ret_frame, np_data):
        width = 280; voltage_range = self.voltage_max - self.voltage_min

        # channel_data = np.zeros((ret_frame), np.int32)
        # x = np.arange(0, ret_frame, 1)
        # for i in range(ret_frame):
        #     channel_data[i] = np_data[i * self.total_channel + channel]

        channel_data = np.reshape(np_data,(int(np_data.shape[0]/ret_frame), ret_frame), "F")

        channel_data = channel_data / self.miliGain * 1000    # μV

        # plt.plot(x, channel_data)
        # plt.show()
        # print(np.max(channel_data))
        # print(np.min(channel_data))


    # 获取MEA的输出信号，并检测Spike
    def get_recording(self):
        """
        返回记录信号, out_left, out_right
        """
        data = self.get_recording_signal_all_channel()

        # 需对数据进行筛选，剔除刺激后10ms内数据
        data = self.get_spike_data_from_channel_data(data)

        recording_left_node = self.recording_para.recording_list
        left = recording_left_node[0]
        right = recording_left_node[1]

        DYNAMICS_MODEL = False    # 控制spike是单独计算，还是多个通道一起计算；一般来说一起算，更准确，延时更低；但动力学模型需要单独算

        # 每个通道单独计算，用于动力学模型
        if DYNAMICS_MODEL:
            left_spike = []
            try:
                for i in range(len(left)):
                    l_data = data[self.channel_map[left[i]] - 1]
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
                    r_data = data[self.channel_map[right[i]] - 1]
                    # spikes_num = self.spike_det.run(r_data)
                    self.spike_det_right.set_data(r_data)
                    self.spike_det_right.start()
                    spikes_num = self.spike_det_right.get_spike_num()
                    right_spike.append(spikes_num)
            except:
                print("Error on right spikes detection...")
        else:
            # 每个通道数据合并，只计算一次spike
            left_spike = []
            try:
                left_data = None
                for i in range(len(left)):
                    l_data = data[self.channel_map[left[i]] - 1]
                    if left_data is None:
                        left_data = l_data
                    else:
                        left_data = np.append(left_data, l_data)
                    # spikes_num = self.spike_det.run(l_data)
                self.spike_det_left.set_data(left_data)
                self.spike_det_left.start()
                spikes_num = self.spike_det_left.get_spike_num()
                left_spike.append(spikes_num)
            except:
                print("Error on left spikes detection...")

            right_spike = []
            try:
                right_data = None
                for i in range(len(right)):
                    r_data = data[self.channel_map[right[i]] - 1]
                    if right_data is None:
                        right_data = r_data
                    else:
                        right_data = np.append(right_data, r_data)
                    # spikes_num = self.spike_det.run(r_data)
                self.spike_det_right.set_data(right_data)
                self.spike_det_right.start()
                spikes_num = self.spike_det_right.get_spike_num()
                right_spike.append(spikes_num)
            except:
                print("Error on right spikes detection...")

        print("spike num left:", np.sum(left_spike))
        print("spike num right:", np.sum(right_spike))

        return left_spike, right_spike


    # 剔除刺激后的10ms内数据
    def get_spike_data_from_channel_data(self, channel_data):
        # 根据记录的刺激电极时刻，获取第一个刺激时刻信号,20帧内，不重复检测
        
        # 原69通道，可直接获取
        sti_signal = channel_data[69]

        # # 更改标识符之后，需检测出两个记录通道的刺激时刻，取并集
        # sti_sg1 = channel_data[70]
        # sti_sg2 = channel_data[71]
        # sti_signal = channel_data[69]    # 理论上全部为 0， 需测试一下
        # sti_signal[sti_sg1 > 0] = 1
        # sti_signal[sti_sg2 > 0] = 1      # 取两者并集


        non_zero = np.where(sti_signal > 0)[0]    # 此帧为首次刺激时刻，每个频率时刻，需要重置
        out_frame = []
        for i in range(len(non_zero)):
            if len(out_frame) == 0:
                out_frame.append(non_zero[i])
            elif non_zero[i] > out_frame[-1] + 20:    # 判断若20帧内，即800μs内2非刺激信号(400μs有延迟)
                out_frame.append(non_zero[i])

        # 根据找出的检测位置，获得特定时间内的，目标区域的检测数据，用于spike检测
        out_raw_data = []
        resolution = self.Samplingrate / 1000    # 每ms多少帧
        if len(out_frame) > 0:    # 有刺激时刻
            for i in range(len(out_frame)):
                cur = out_frame[i]    # 刺激起始位置
                next_frame = cur + int(50 * resolution)

                if next_frame < len(channel_data[0]) - 1:
                    channel_data[:, cur:next_frame] = 0
                else:
                    channel_data[:, cur:] = 0

        return channel_data



    def OnChannelData(self, x, cbHandle, numSamples):
        if self.dataMode == DataModeEnumNet.Unsigned_16bit:
            num = self.device.ChannelBlock_AvailFrames(0)
            # print("num", num)
            # data, frames_ret = self.device.ChannelBlock_ReadFramesUI16(cbHandle, self.callbackThreshold, np.int32(0))
            # np_data = asNumpyArray(data, ctypes.c_uint16)
            # self.plot_channel_data(68, frames_ret, np_data)
            pass
            # time.sleep(0.2)
            # print(".Net numSamples 16: %d frames_ret: %d size: %d Data: %04x %04x Checksum: %04x %04x %04x %04x" % (numSamples, frames_ret, len(np_data), np_data[0], np_data[1], np_data[2], np_data[3], np_data[4], np_data[5]))
        else: # dataMode == DataModeEnumNet.Signed_32bit
            data, self.frame_ret = self.device.ChannelBlock_ReadFramesI32(0, self.callbackThreshold, np.int32(0))
            temp = asNumpyArray(data, ctypes.c_int32)
            self.numpy_data = copy.deepcopy(temp)
            # print(self.numpy_data.shape)

            #  0 - 59: Elektrode Channels
            #  60 - 67 IFB Analog IFB channels
            #  68 Digital In/Out
            #  69 - 74 Sideband channels
            #  75 - 76 Checksum channels

            self.read_data_func()

            # self.plot_channel_data_voltage(16, frames_ret, np_data)
            # print(".Net numSamples 32: %d frames_ret: %d size: %d Data: %08x %08x Checksum: %08x %08x" % (numSamples, frames_ret, len(np_data), np_data[0], np_data[1], np_data[2], np_data[3]))

    # 读取出数据后，需要进行的各种数据操作
    def read_data_func(self):
        # save data 数据保存
        try:
            if self.save_state:
                self.channel_data_saving_thread.set_new_data(self.numpy_data)
                self.channel_data_saving_thread.start()

            # 用于保存，清空缓存后，特定时间内的数据
            if self.record_300ms_mark:
                if self.record_300ms_np is None:
                    self.record_300ms_np = copy.copy(self.numpy_data)
                else:
                    self.record_300ms_np = np.append(self.record_300ms_np, copy.copy(self.numpy_data))
                # print("record_300ms_np shape:", self.record_300ms_np.shape)

            # 缓存池，保存了1s的数据
            length = self.frame_ret * self.mChannels
            tp = self.one_second_npdata[length:]
            self.one_second_npdata[:-length] = tp
            self.one_second_npdata[-length:] = self.numpy_data
        except:
            print("Error happend on reading data from MEA!...")


    def OnError(self, msg, info):
        print(msg, info)

    def set_record_para(self, sig):
        self.recording_para = sig

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
        x = np.arange(0, self.frame_ret, 1)
        # for i in range(self.frame_ret):
        #     channel_data[i] = self.numpy_data[i * self.total_channel + channel]
        temp = self.numpy_data
        channel_data = np.reshape(temp,(int(temp.shape[0]/self.frame_ret), self.frame_ret), "F")
        channel_data = channel_data * self.constant
        
        # channel_data = channel_data * 2 / self.miliGain * 1000    # μV

        # 测试stimulate 检测位置是否准确
        # self.detect_stimulating_pos(channel_data[], 100)

        return x, channel_data[channel - 1]


    # 用于机器人避障时，读取MEA数据
    def get_recording_signal_all_channel(self):
        temp = self.one_second_npdata    # 1s内的数据
        frame_ret = self.Samplingrate
        channel_data = np.reshape(temp,(int(temp.shape[0]/frame_ret), frame_ret), "F")
        channel_data = channel_data * self.constant
        
        # channel_data = channel_data * 2 / self.miliGain * 1000    # μV

        return channel_data

    # 获取1s钟的数据
    def get_one_second_signal(self, channel):
        x = np.arange(0, self.Samplingrate, 1)

        temp = self.one_second_npdata
        channel_data = np.reshape(temp,(int(temp.shape[0]/self.Samplingrate), self.Samplingrate), "F")

        channel_data = channel_data * self.constant


        # 3种方法分析
        # import Mcs
        # status, res, unit = self.device.GetResolutionPerDigit(0, Mcs.Usb.DacqGroupChannelEnumNet.HeadstageElectrodeGroup,0,0)
        # method2 = res*pow(10, -unit) * channel_data
        # method1 = channel_data * self.constant
        # ours = channel_data * 2 / self.miliGain * 1000

        
        # channel_data = channel_data * 2 / self.miliGain * 1000    # μV

        # 显示剔除刺激尾迹后的数据
        # channel_data = self.get_spike_data_from_channel_data(channel_data)

        return x, channel_data[channel - 1]
    

    # 调用函数，清空缓存，并记录所有数据，进行保存
    def clear_recording_buffer(self, clear_state):
        # 此处通过SendStart()调用，仅用来设置数据的保存和标记情况
        # if clear_state:
        #     num = self.device.ChannelBlock_AvailFrames(0)
        #     if num > 0:
        #         data, self.frame_ret = self.device.ChannelBlock_ReadFramesI32(0, self.callbackThreshold, np.int32(0))
        #         temp = asNumpyArray(data, ctypes.c_int32)
        #         self.numpy_data = copy.copy(temp)
                
        #         # 保存此段即将清空的数据
        #         if self.save_state: 
        #             self.channel_data_saving_thread.set_new_data(self.numpy_data)
        #             self.channel_data_saving_thread.start()
                
        #         # 缓存池，保存了1s的数据
        #         length = self.frame_ret * self.mChannels
        #         tp = self.one_second_npdata[length:]
        #         self.one_second_npdata[-length:] = self.numpy_data
        #         self.one_second_npdata[:-length] = tp

        #     self.device.ClearBuffers()
        
        self.record_300ms_mark = clear_state

    # 调用函数，清空缓存，并记录所有数据，进行保存
    def clear_recording_buffer_get_sti_time(self, eles_key, amplitude, duration):
        """
        2024.08.17 注释掉，因为清空时数据偶尔会报错，导致spike检测等出错
        """
        # num = self.device.ChannelBlock_AvailFrames(0)
        # if num > 0:
        #     data, self.frame_ret = self.device.ChannelBlock_ReadFramesI32(0, self.callbackThreshold, np.int32(0))
        #     temp = asNumpyArray(data, ctypes.c_int32)
        #     self.numpy_data = copy.copy(temp)
                        
        #     # 缓存池，保存了1s的数据
        #     length = self.frame_ret * self.mChannels
        #     tp = self.one_second_npdata[length:]
        #     self.one_second_npdata[-length:] = self.numpy_data
        #     self.one_second_npdata[:-length] = tp

        # # # 根据信号计算刺激位置
        # frame_time = 1.0 / self.Samplingrate   # 每一帧数据对应的时间 μs
        # dur = copy.copy(asNumpyArray(duration, ctypes.c_uint64))
        # amp = copy.copy(asNumpyArray(amplitude, ctypes.c_int32))

        # t = 0; sti_times = []
        # for i in range(len(dur)):
        #     if amp[i] != 0:  # 刺激幅值不为0的部分
        #         if i == 0:
        #             sti_times.append(0.0)
        #         else:
        #             sti_times.append(t * 0.000001)   # to s
        #     t = t + dur[i]

        # # 保存此段即将清空的数据
        # if self.save_state: 
        #     self.channel_data_saving_thread.set_sti_timestamp(sti_times, eles_key)
        #     # self.channel_data_saving_thread.set_sti_timestamp(0, eles_key)
        #     self.channel_data_saving_thread.set_new_data(self.numpy_data)
        #     self.channel_data_saving_thread.start()
        # self.device.ClearBuffers()

        pass

    # 重置临时数据读取缓存池
    def reset_target_time_process(self):
        self.record_300ms_mark = False

        if self.record_300ms_np is not None:
            del self.record_300ms_np
            self.record_300ms_np = None
            print("record_300ms_np is reset!...")
  

    # 获取特定时间内的信号
    def get_target_time_signal(self):
        resolution = self.Samplingrate / 1000    # 每ms多少帧

        frame = int(self.record_300ms_np.shape[0] / self.mChannels)

        channel_data = np.reshape(self.record_300ms_np,(int(self.record_300ms_np.shape[0]/frame), frame), "F")
        channel_data = channel_data * self.constant                 # 新计算方法
    
        # channel_data = channel_data * 2 / self.miliGain * 1000    # μV  旧计算方法

        return resolution, channel_data



    def detect_stimulating_pos(self, data, groups):
        """
        通过算法，检测信号的刺激时刻或刺激位置；
        输入：某通道的数据,分组个数
        输出：检测出的刺激位置 index
        """

        std_value = np.std(data)

        length = len(data)  // groups
        new_data = data[:length*groups].reshape(length, groups)
        mean_data = np.mean(new_data, axis=1)

        out_pos = []
        for i in range(1, len(mean_data)):
            dis = mean_data[i] - mean_data[i - 1]
            if dis > std_value and mean_data[i] > std_value and mean_data[i - 1] < std_value:
                out_pos.append(i*groups)

        return out_pos