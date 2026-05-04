import time
import os
import clr
import numpy as np

from System import Action
from System import *
from System.Collections import Generic
from System import ComponentModel, Data, Drawing, Linq, Text
from System.Threading import Tasks


clr.AddReference(os.getcwd() + '/bin/x64/McsUsbNet.dll')
from Mcs.Usb import ElectrodeDacMuxEnumNet, ElectrodeModeEnumNet
from Mcs.Usb import CMcsUsbListNet
from Mcs.Usb import DeviceEnumNet

from Mcs.Usb import CStg200xBasicNet, CStg200xDownloadNet
from Mcs.Usb import McsBusTypeEnumNet
from Mcs.Usb import STG_DestinationEnumNet


def PollHandler(status, stgStatusNet, index_list):
    print('%x %s' % (status, str(stgStatusNet.TiggerStatus[0])))



class Stimulation(object):
    def __init__(self):
        self.deviceList = CMcsUsbListNet(DeviceEnumNet.MCS_DEVICE_USB)
        print("found %d stimulating devices" % (self.deviceList.Count))

        for i in range(self.deviceList.Count):
            listEntry = self.deviceList.GetUsbListEntry(i)
            print("Device: %s   Serial: %s" % (listEntry.DeviceName, listEntry.SerialNumber))

        self.sti_sig = None    # 刺激参数
        self.amplitude = None; self.duration = None   # 解析出的幅值和脉宽
        self.left_electrode = None; self.right_electrode =None  # 左右刺激电极

        self.channel_map = {"47":1, "48":2, "46":3, "45":4, "38":5, "37":6, "28":7,
                    "36":8, "27":9, "17":10, "26":11, "16":12, "35":13, "25":14, "15":15,
                    "14":16, "24":17, "34":18, "13":19, "23":20, "12":21, "22":22, "33":23, 
                    "21":24, "32":25, "31":26, "44":27, "43":28, "41":29, "42":30, "52":31,
                    "51":32, "53":33, "54":34, "61":35, "62":36, "71":37, "63":38, "72":39,
                    "82":40, "73":41, "83":42, "64":43, "74":44, "84":45, "85":46, "75":47, 
                    "65":48, "86":49, "76":50, "87":51, "77":52, "66":53, "78":54, "67":55,
                    "68":56, "55":57, "56":58, "58":59, "57":60}

        # self.initial_device()

    def initial_device(self):
        self.device = CStg200xDownloadNet()
        # self.device = CStg200xStreamingNet()  # not recommend for stg

        self.device.Stg200xPollStatusEvent += PollHandler

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

        # channelmap = Array[UInt32]([1, 0, 0, 0])
        # syncoutmap = Array[UInt32]([1, 0, 0, 0])
        # repeat = Array[UInt32]([10, 0, 0, 0])

        electrodes = Array[UInt32]([7, 6, 5, 4])
        for i in range(len(electrodes)):
            electrode = electrodes[i]

            # ElectrodeMode: emManual: electrode is permanently selected for stimulation
            self.device.SetElectrodeMode(UInt32(electrode), ElectrodeModeEnumNet.emManual)

            # ElectrodeDacMux: DAC to use for stimulation
            self.device.SetElectrodeDacMux(UInt32(electrode), UInt32(0), ElectrodeDacMuxEnumNet.Stg1)

            # ElectrodeEnable: enable electrode for stimulation
            self.device.SetElectrodeEnable(UInt32(electrode), UInt32(0), True)

            # BlankingEnable: false: do not blank the ADC signal while stimulation is running
            self.device.SetBlankingEnable(UInt32(electrode), False)

            # AmplifierProtectionSwitch: false: Keep ADC connected to electrode even while stimulation is running
            self.device.SetEnableAmplifierProtectionSwitch(UInt32(electrode), False)

        channelmap = Array[UInt32]([0, 2, 4, 0, 0, 0])
        syncoutmap = Array[UInt32]([0, 0, 4, 0, 0, 0])
        repeat = Array[UInt32]([10, 10, 10, 0, 0, 0])

        amplitude0 = Array[Int32]([-50000, 1000,0])          # μV   50mV
        sync0 = Array[Int32]([255, 255, 0])                  # Digital  
        amplitude1 = Array[Int32]([1000, -1000,0])          # μV   
        duration = Array[UInt64]([100000, 100000,1000000])     # μs

        self.device.SetupTrigger(0,  Array[UInt32]([255]), Array[UInt32]([255]), Array[UInt32]([10]))
        # self.device.SetupTrigger(0,  channelmap, syncoutmap, repeat)
        self.device.SetVoltageMode()
        self.device.PrepareAndSendData(0, amplitude0, duration, STG_DestinationEnumNet.channeldata_voltage)
        self.device.PrepareAndSendData(0, amplitude0, duration, STG_DestinationEnumNet.syncoutdata)
        self.device.PrepareAndSendData(1, amplitude1, duration, STG_DestinationEnumNet.channeldata_voltage)

    def send_start(self):
        self.device.SendStart(1)

    def disconnect(self):
        self.device.Disconnect()

    def set_sti_signal(self, signal):
        self.sti_sig = signal
        parameter = signal.para
        sti_para= signal.sti_para    # 包含刺激名称和组重复次数

        # 解析设置，变为奖惩刺激参数
        amp = []; dur = []
        for k in range(sti_para["repeat_times"]):  # 总的group的重复次数
            for i in range(len(parameter)):    # 循环多个不同的单元信号，构成一个group
                tp = parameter[i]
                for j in range(tp["cycles"]):  # 每个单元信号，有固定信号循环次数
                    amplitude = [tp['amplitude_1']*1000, 0, tp['amplitude_2']*1000, 0]   # μV 
                    duration = [tp['duration_1'], tp['duration_2'], tp['duration_3'], tp["ISI"]*1000]     # μs

                    amp.extend(amplitude)
                    dur.extend(duration)

        self.amplitude = Array[Int32](amp)
        self.duration = Array[UInt64](dur)

        # 电极设置
        l_electrode = signal.stimulating_list[0]
        r_electrode = signal.stimulating_list[1]

        self.left_electrode = []; self.right_electrode = []
        for i in range(len(l_electrode)):
            self.left_electrode.append(self.channel_map[l_electrode[i]])
        
        for i in range(len(r_electrode)):
            self.right_electrode.append(self.channel_map[r_electrode[i]])


    def update_record_stimulation(self, left, right):
        """输入左右两侧传感器的 刺激信号, 并通过device输出刺激
        """
        DACResolution = self.device.GetDACResolution()

        self.device.SendStart(1)

    def reset_electrode_stimulating_state(self, electrodes):
        for i in range(len(electrodes)):
            ele = Array[UInt32]([electrodes[i]])
            self.device.SetElectrodeMode(UInt32(ele), ElectrodeModeEnumNet.emAutomatic)

            # ElectrodeEnable: enable electrode for stimulation
            self.device.SetElectrodeEnable(UInt32(ele), UInt32(0), False)

            # BlankingEnable: false: do not blank the ADC signal while stimulation is running
            self.device.SetBlankingEnable(UInt32(ele), True)

            # AmplifierProtectionSwitch: false: Keep ADC connected to electrode even while stimulation is running
            self.device.SetEnableAmplifierProtectionSwitch(UInt32(electrode), True)


    def update_stimulation_left(self):
        """
        在出现撞击时，进行奖惩刺激
        """

        for i in range(len(self.left_electrode)):
            electrode = Array[UInt32]([self.left_electrode[i]])
            self.device.SetElectrodeMode(UInt32(electrode), ElectrodeModeEnumNet.emManual)

            # ElectrodeDacMux: DAC to use for stimulation
            self.device.SetElectrodeDacMux(UInt32(electrode), UInt32(0), ElectrodeDacMuxEnumNet.Stg1)

            # ElectrodeEnable: enable electrode for stimulation
            self.device.SetElectrodeEnable(UInt32(electrode), UInt32(0), True)

            # BlankingEnable: false: do not blank the ADC signal while stimulation is running
            self.device.SetBlankingEnable(UInt32(electrode), False)

            # AmplifierProtectionSwitch: false: Keep ADC connected to electrode even while stimulation is running
            self.device.SetEnableAmplifierProtectionSwitch(UInt32(electrode), False)

        self.device.SetupTrigger(0,  Array[UInt32]([255]), Array[UInt32]([255]), Array[UInt32]([1]))
        self.device.PrepareAndSendData(0, self.amplitude, self.duration, STG_DestinationEnumNet.channeldata_voltage)
        self.send_start(1)
    
    def update_stimulation_right(self):
        """
        在出现撞击时，进行奖惩刺激
        """

        for i in range(len(self.right_electrode)):
            electrode = Array[UInt32]([self.right_electrode[i]])
            self.device.SetElectrodeMode(UInt32(electrode), ElectrodeModeEnumNet.emManual)

            # ElectrodeDacMux: DAC to use for stimulation
            self.device.SetElectrodeDacMux(UInt32(electrode), UInt32(0), ElectrodeDacMuxEnumNet.Stg1)

            # ElectrodeEnable: enable electrode for stimulation
            self.device.SetElectrodeEnable(UInt32(electrode), UInt32(0), True)

            # BlankingEnable: false: do not blank the ADC signal while stimulation is running
            self.device.SetBlankingEnable(UInt32(electrode), False)

            # AmplifierProtectionSwitch: false: Keep ADC connected to electrode even while stimulation is running
            self.device.SetEnableAmplifierProtectionSwitch(UInt32(electrode), False)

        self.device.SetupTrigger(0,  Array[UInt32]([255]), Array[UInt32]([255]), Array[UInt32]([1]))
        self.device.PrepareAndSendData(0, self.amplitude, self.duration, STG_DestinationEnumNet.channeldata_voltage)
        self.send_start(1)