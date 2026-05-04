# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2024
# Written by Guiping Cao
# Time: 2024.04.14
# --------------------------------------------------------

import time
import socket
import os

import tkinter as tk
from tkinter import filedialog

from src.system_device.RHXRunAndStimulate import RunAndStimulateDemo
from src.enum.constants import (
    StimulationType, StimulationPosition, StimulationSpec,
    TriggerType, KeypressSource, DigitalOutput
)


class StimulationIntan(object):
    def __init__(self):
        self.ip = ""
        self.port = 0
        self.command_buffer_size = 1024    # 指令相关
        self.command_settle_enabled = False

    def set_socket(self, ip, port):
        self.ip = ip
        self.port = port
        
    def connect_to_server(self):
        """
        连接到TCP命令服务器。

        :param ip_address: TCP服务器的IP地址。
        :param port: TCP服务器的端口号。
        :return: 连接到服务器的套接字对象。
        """

        print('Connecting to TCP command server...')
        scommand = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            scommand.connect((self.ip, self.port))
            self.connect_socket = scommand
            print('Connected successfully.')
        except socket.error as err:
            print(f"Connection failed with error: {err}")
            scommand = None
        return scommand

    def get_SampleRateHertz(self, scommand, command_buffer_size = 1024):
        """
        获取采样率（赫兹）。

        参数:
        - scommand: socket 对象，用于与外部设备进行通信。
        - command_buffer_size: 整数，接收缓冲区大小，默认为 1024。

        返回值:
        - 整数，采样率（赫兹）。如果转换失败，返回 None。
        """

        scommand.sendall(b'get SampleRateHertz')
        command_return = str(scommand.recv(command_buffer_size), "utf-8")

        # 提取采样率
        try:
            sample_rate = int(command_return.split()[-1])
        except (IndexError, ValueError) as e:
            print(f"Error extracting sample rate: {e}")
            sample_rate = None

        return sample_rate


    def verify_controller_type(self):
        """
        验证连接的RHX软件是否使用的是刺激/录制控制器。

        :param scommand: 已连接到TCP服务器的套接字对象。
        :param command_buffer_size: 从TCP命令套接字读取的缓冲区大小。
        :return: none
        """

        print('Verifying controller type...')
        self.connect_socket.sendall(b'get type')
        command_return = str(self.connect_socket.recv(self.command_buffer_size), "utf-8")
        is_stim = command_return == "Return: Type ControllerStimRecord"
        if not is_stim:
            raise InvalidControllerType(
                'This example script should only be used with a '
                'Stimulation/Recording Controller.'
            )
        print('Controller type verified as Stimulation/Recording Controller.')



    def ensure_controller_stopped(self):
        """
        通过查询其运行模式来确保控制器没有运行。
        如果控制器正在运行，它会发送一个命令来停止它。

        :param scommand: 已连接到TCP服务器的套接字对象。
        :param command_buffer_size: 从TCP命令套接字读取的缓冲区大小。
        :return: none
        """

        print('Checking controller run mode...')
        self.connect_socket.sendall(b'get runmode;')
        command_return = str(self.connect_socket.recv(self.command_buffer_size), "utf-8")
        is_stopped = "Return: RunMode Stop" in command_return

        if not is_stopped:
            time.sleep(0.01)
            print('Controller is running. Sending stop command...')
            self.connect_socket.sendall(b'set runmode stop;')
            time.sleep(0.01)  # Give some time for the command to be processed
            print('Controller stopped.')
        else:
            print('Controller is already stopped.')

    def request_controller_stop(self):
        """
        Fire-and-forget stop request for closed-loop stimulation updates.

        The main loop only needs to enqueue the RHX command sequence. Querying
        run mode and waiting for a reply belongs to setup/diagnostic paths, not
        to the timing-critical closed-loop path.
        """
        self.connect_socket.sendall(b'set runmode stop;')

    def configureTrainStimulation(self, channel, source, amplitude, duration, pulseTrain, numberOfstimpulses, stimenabled = False):

        """
        生成配置单个通道高频刺激设置的命令字符串，例如设置为 256 个刺激脉冲的序列。

        :param scommand: 已连接到TCP服务器的套接字对象。此函数生成命令字符串，但不发送它；
                        调用者负责使用此套接字发送命令。
        :param channel: 要配置的通道名称，例如 'A-010'。
        :param source: 刺激的来源，例如 'keypressf1'。
        :param amplitude: 刺激第一阶段的幅度，单位为微安。
        :param duration: 刺激第一阶段的持续时间，单位为微秒。
        :param pulseTrain: “SinglePulse” - 默认 or “PulseTrain”
        :param train_num: 刺激脉冲数 0-256。
        :param stimenabled: 一个布尔值，指示是否启用通道的刺激（True）或禁用（False）。

        :return: 包含配置刺激通道所需的所有命令的字符串。命令用分号连接，并准备通过TCP发送。

        注意：此函数仅生成配置字符串。发送此字符串通过`scommand`套接字是调用者的责任。
            确保通过执行返回的命令上传刺激参数以使其生效。返回字符串中命令的顺序对于正确配置非常重要。
            参数可以增加，因为还有其他刺激参数可以调整，对应增加接口中指令设置即可.
            最后指令返回，主调函数负责发送指令，注意设置顺序.
        """


        # 必须有的，设置通道刺激为可触发态.
        com_stimenabled = f'set {channel}.stimenabled {stimenabled};'


        # 其他指令该处插入即可.
        com_source = f'set {channel}.source {source};'

        com_firstphaseamplitudemicroamps = f'set {channel}.firstphaseamplitudemicroamps {amplitude};'

        com_firstphasedurationmicroseconds = f'set {channel}.firstphasedurationmicroseconds {duration};'

        com_PulseOrTrain = f'set {channel}.PulseOrTrain {pulseTrain};'

        com_numberOfstimpulses = f'set {channel}.NumberOfStimPulses {numberOfstimpulses};'

        # 必须上传后才能生效.
        com_uploadstimparameters = f'execute uploadstimparameters {channel};'

        com_config = com_stimenabled + com_source + com_firstphaseamplitudemicroamps + com_firstphasedurationmicroseconds + com_PulseOrTrain + com_numberOfstimpulses + com_uploadstimparameters;

        return com_config


    def configureSingleStimulation(self, scommand, channel, source, amplitude, duration, stimenabled = False):

        """
        生成配置单个通道刺激设置的命令字符串。

        :param scommand: 已连接到TCP服务器的套接字对象。此函数生成命令字符串，但不发送它；
                        调用者负责使用此套接字发送命令。
        :param channel: 要配置的通道名称，例如 'A-010'。
        :param source: 刺激的来源，例如 'keypressf1'。
        :param amplitude: 刺激第一阶段的幅度，单位为微安。
        :param duration: 刺激第一阶段的持续时间，单位为微秒。
        :param stimenabled: 一个布尔值，指示是否启用通道的刺激（True）或禁用（False）。

        :return: 包含配置刺激通道所需的所有命令的字符串。命令用分号连接，并准备通过TCP发送。

        注意：此函数仅生成配置字符串。发送此字符串通过`scommand`套接字是调用者的责任。
            确保通过执行返回的命令上传刺激参数以使其生效。返回字符串中命令的顺序对于正确配置非常重要。
            参数可以增加，因为还有其他刺激参数可以调整，对应增加接口中指令设置即可.
            最后指令返回，主调函数负责发送指令，注意设置顺序.
        """


        # 必须有的，设置通道刺激为可触发态.
        com_stimenabled = f'set {channel}.stimenabled {stimenabled};'

        # 其他指令该处插入即可.
        com_source = f'set {channel}.source {source};'

        com_firstphaseamplitudemicroamps = f'set {channel}.firstphaseamplitudemicroamps {amplitude};'

        com_firstphasedurationmicroseconds = f'set {channel}.firstphasedurationmicroseconds {duration};'

        # 必须上传后才能生效.
        com_uploadstimparameters = f'execute uploadstimparameters {channel};'

        com_config = com_stimenabled + com_source + com_firstphaseamplitudemicroamps + com_firstphasedurationmicroseconds + com_uploadstimparameters;

        return com_config


    def once_stimulate(self, channels):
        """
        eg: channels = ['A-000']
        TODO: 根据输入的通道数不同，自适应生成相应的控制指令；按键绑定，输入两个；
        参数修改，对应关系，刺激信号转化；
        """

        command = self.configureTrainStimulation(channels[0], 'f1', 10, 500, 'PulseTrain', 256, True)
        self.connect_socket.sendall(command.encode())    # 发送信号

        self.connect_socket.sendall(b'set runmode run;')

        self.trigger_stimulation('f1')

    def _configureStimulation(self, channel, source, digital_out, duration_ttl, amplitude, duration, numberOfstimpulses, pulseTrain, stimenabled = True):

        """
        生成配置通道刺激设置的命令字符串。
        此接口不对外部开放。

        :param scommand: 已连接到TCP服务器的套接字对象。此函数生成命令字符串，但不发送它；
                         调用者负责使用此套接字发送命令。
        :param channel: 要配置的通道名称，例如 'A-010'。
        :param source: 刺激的来源，例如 'keypressf1'。
        :param digital_out: 数字信号输出 0 or 1。
        :param amplitude: 刺激电流幅度列表，分别代表第一个脉冲幅度与第二个脉冲幅度，单位为微安。 注意非负。
        :param duration: 时间列表，分别代表第一个刺激脉宽时间，第二个刺激脉宽时间，刺激前放大器稳定时间和刺激后放大器稳定时间，单位为微秒。
        :param pulseTrain: “SinglePulse” - 默认 or “PulseTrain”。
        :param numberOfstimpulses: 刺激脉冲数 0-256，仅在 pulseTrain 为 “PulseTrain” 时有效。
        :param stimenabled: 一个布尔值，指示是否启用通道的刺激（True）或禁用（False）。

        :return: 包含配置刺激通道所需的所有命令的字符串。命令用分号连接，并准备通过TCP发送。

        注意：此函数仅生成配置字符串。发送此字符串通过`scommand`套接字是调用者的责任。
             确保通过执行返回的命令上传刺激参数以使其生效。返回字符串中命令的顺序对于正确配置非常重要。
             参数可以增加，因为还有其他刺激参数可以调整，对应增加接口中指令设置即可。
             最后指令返回，主调函数负责发送指令，注意设置顺序。
        """

        # 必须有的，设置通道刺激为可触发态
        com_stimenabled = f'set {channel}.stimenabled {stimenabled};'
        com_digitalStimEnabled = f'set {digital_out}.stimenabled {stimenabled};'
        com_digitalEnabled = f'set {digital_out}.Enabled {stimenabled};'
        
        # 其他指令该处插入即可
        com_source = f'set {channel}.source {source};'
        com_digitalSource = f'set {digital_out}.source {source};'

        # 刺激电流幅值
        com_firstphaseamplitudemicroamps = f'set {channel}.firstphaseamplitudemicroamps {amplitude[0]};'
        com_secondphaseamplitudemicroamps = f'set {channel}.secondphaseamplitudemicroamps {amplitude[1]};'

        # 刺激脉宽
        com_firstphasedurationmicroseconds = f'set {channel}.firstphasedurationmicroseconds {duration[0]};'
        com_secondphasedurationmicroseconds = f'set {channel}.secondphasedurationmicroseconds {duration[1]};'
        com_digitalDurationmicroseconds = f'set {digital_out}.FirstPhaseDurationMicroseconds {duration_ttl};'

        # 刺激后放大器稳定时间
        com_prestimampsettlemicroseconds = f'set {channel}.PreStimAmpSettleMicroseconds 0;'
        com_poststimampsettlemicroseconds = f'set {channel}.poststimampsettlemicroseconds 1000;'

        # 单点与高频刺激类别
        com_pulseortrain = f'set {channel}.PulseOrTrain {pulseTrain};'
        com_digitalPulseortrain = f'set {digital_out}.PulseOrTrain {pulseTrain};'
        
        # 检查是否为高频刺激
        if pulseTrain == "PulseTrain":

            com_numberOfstimpulses = f'set {channel}.NumberOfStimPulses {numberOfstimpulses};'
            com_digitalNumberOfstimpulses = f'set {digital_out}.NumberOfStimPulses {numberOfstimpulses};'
            com_MaintainAmpSettle  = f'set {channel}.MaintainAmpSettle True;'
            com_TrainPeriod = f'set {channel}.PulseTrainPeriodMicroseconds {duration[2]};'
            com_digitalTrainPeriod = f'set {digital_out}.PulseTrainPeriodMicroseconds {duration[2]};'
            com_uploadstimparameters = f'execute uploadstimparameters {channel};'
            com_digitalUploadstimparameters = f'execute uploadstimparameters {digital_out};'
            
            com_config = (
                        com_stimenabled + com_source + com_firstphaseamplitudemicroamps + com_secondphaseamplitudemicroamps + com_firstphasedurationmicroseconds + com_secondphasedurationmicroseconds
                        + com_digitalEnabled + com_digitalStimEnabled + com_digitalSource + com_digitalDurationmicroseconds + com_digitalPulseortrain + com_digitalNumberOfstimpulses + com_digitalTrainPeriod
                        + com_prestimampsettlemicroseconds + com_poststimampsettlemicroseconds + com_MaintainAmpSettle + com_TrainPeriod
                        + com_pulseortrain + com_numberOfstimpulses +com_uploadstimparameters + com_digitalUploadstimparameters)
        else:

            # 单点刺激直接上传
            com_uploadstimparameters = f'execute uploadstimparameters {channel};'
            com_digitalUploadstimparameters = f'execute uploadstimparameters {digital_out};'
            com_config = (
                    com_stimenabled + com_source + com_firstphaseamplitudemicroamps + com_secondphaseamplitudemicroamps + com_firstphasedurationmicroseconds + com_secondphasedurationmicroseconds
                    + com_digitalEnabled + com_digitalStimEnabled + com_digitalSource + com_digitalDurationmicroseconds + com_digitalPulseortrain
                    + com_prestimampsettlemicroseconds + com_poststimampsettlemicroseconds
                    + com_pulseortrain + com_uploadstimparameters + com_digitalUploadstimparameters)

        return com_config

    def cancel_config(self, channels):
        """
        根据输入的通道、取消参数设置，以便下一次设置启动。
        """

        com_configs = []

        for channel in channels:
            com_poststimampsettlemicroseconds = f'set {channel}.poststimampsettlemicroseconds 1000;'
            com_RefractoryPeriodMicroseconds = f'set {channel}.RefractoryPeriodMicroseconds 1000;'
            com_uploadstimparameters = f'execute uploadstimparameters {channel};'
            com_config = f'set {channel}.stimenabled False;' + com_poststimampsettlemicroseconds + com_RefractoryPeriodMicroseconds + com_uploadstimparameters
            com_configs.append(com_config)

        com_config = ';'.join(com_configs)
        self.connect_socket.sendall(com_config.encode())
        time.sleep(0.1)
    
    
    def configure_stimulation(self, channels, digital_out, amplitude, duration, trigger, numberOfstimpulses, stim_spec: StimulationSpec):
        """
        根据输入的通道、幅值和duration，生成对应的刺激信号，并发送给ITNAN,待触发；

        为多个通道生成配置刺激设置的命令字符串。

        :param scommand: 已连接到TCP服务器的套接字对象。此函数生成命令字符串，但不发送它；
                        调用者负责使用此套接字发送命令。
        :param channels: 一个包含要配置的通道名称的列表，例如 ['A-010', 'A-011']。
        :param digital_out: 标明数字输出通道用于标识刺激时刻, ''。
        :param amplitude: 刺激电流幅度列表，分别代表第一个脉冲幅度与第二个脉冲幅度，单位为微安。 注意非负。
        :param duration: 时间列表，分别代表第一个刺激脉宽时间，第二个刺激脉宽时间，刺激前放大器稳定时间和刺激后放大器稳定时间，单位为微秒。
        :param trigger: 字符串，表示触发器 'keypressf1 - keypressf8'。
        :param numberOfstimpulses:  脉冲串个数

        :return: 包含配置多个通道刺激设置所需的所有命令的字符串。命令用分号连接，并准备通过TCP发送。

        注意：此函数仅生成配置字符串。发送此字符串通过`scommand`套接字是调用者的责任。
            确保通过执行返回的命令上传刺激参数以使其生效。返回字符串中命令的顺序对于正确配置非常重要。
        """
        
        # 根据刺激类型设置不同的duration_ttl参数
        duration_ttl = self._get_duration_ttl_by_spec(stim_spec)
        
        com_configs = []

        if numberOfstimpulses >= 2:
            for channel in channels:
                com_config = self._configureStimulation(channel, trigger, digital_out, duration_ttl, amplitude, duration, numberOfstimpulses, "PulseTrain", stimenabled=True)
                com_configs.append(com_config) 
        else:
            for channel in channels:
                com_config = self._configureStimulation(channel, trigger, digital_out, duration_ttl, amplitude, duration, numberOfstimpulses, "SinglePulse", stimenabled=True)
                com_configs.append(com_config)

        com_config = ';'.join(com_configs)
        self.connect_socket.sendall(com_config.encode())

        # return ';'.join(com_configs)

    def _get_duration_ttl_by_spec(self, stim_spec: StimulationSpec) -> int:
        """
        根据刺激规格返回对应的duration_ttl参数
        考虑类型和位置的完整组合
        
        :param stim_spec: 刺激规格对象
        :return: duration_ttl值（微秒）
        """
        # 定义完整的组合映射表
        duration_mapping = {
            # 左侧刺激
            (StimulationType.REWARD, StimulationPosition.LEFT): 100,
            (StimulationType.PUNISHMENT, StimulationPosition.LEFT): 200,  
            (StimulationType.ENVIRONMENT, StimulationPosition.LEFT): 300,
            
            # 右侧刺激
            (StimulationType.REWARD, StimulationPosition.RIGHT): 400,
            (StimulationType.PUNISHMENT, StimulationPosition.RIGHT): 500,
            (StimulationType.ENVIRONMENT, StimulationPosition.RIGHT): 600,
            
            # 全局刺激（可能需要更长时间）
            (StimulationType.REWARD, StimulationPosition.ALL): 700,
            (StimulationType.PUNISHMENT, StimulationPosition.ALL): 800,
            (StimulationType.ENVIRONMENT, StimulationPosition.ALL): 900,
        }
        
        # 获取对应的duration_ttl，如果组合不存在则使用默认值
        combination_key = (stim_spec.type, stim_spec.position)
        return duration_mapping.get(combination_key, 100)  # 默认100μs

    def trigger_stimulation(self, key):

        """
        触发刺激命令。此函数发送TCP命令来触发已经配置好的刺激。

        注意：在调用此函数之前，应确保系统的运行模式已经设置为'record'或'run'。
        刺激施加后，调用方负责在合适的时刻将运行模式切换回'stop'或其他状态。

        :param scommand: 已连接到TCP服务器的套接字对象。用于发送TCP命令。
        :param key: 触发刺激的键值，例如 'F1'，用于指定触发特定刺激的键。
        """

        com_trigger = f'execute manualstimtriggerpulse {key};'
        self.connect_socket.sendall(com_trigger.encode())    # 发送触发信号
        # Do not wait here: after the trigger command is accepted by RHX, pulse
        # timing is governed by the stimulator hardware rather than Python.


    def close_connection(self):
        if self.connect_socket:
            print('Disconnecting from TCP command server...')
            self.connect_socket.close()
            print('Disconnected successfully.')
        else:
            print("None socket instance to disconnect!...")

    def startRun(self):
        self.connect_socket.sendall(b'set runmode run;')
        if self.command_settle_enabled:
            time.sleep(0.05)  # Optional diagnostic/setup settling delay.


    def stopRun(self):
        self.connect_socket.sendall(b'set runmode stop')
        if self.command_settle_enabled:
            time.sleep(0.05)  # Optional diagnostic/setup settling delay.
    
    
    def startRecord(self):
        self.connect_socket.sendall(b'set runmode record;')
        time.sleep(0.1)  # Give some time for the command to be processed


    def stopRecord(self):
        self.connect_socket.sendall(b'set runmode stop')
        time.sleep(0.1)  # Give some time for the command to be processed
        
    def enableDigitalInChanel(self, digital_in):
        com_digitalEnabled = f'set {digital_in}.Enabled True;'
        self.connect_socket.sendall(com_digitalEnabled.encode())
        time.sleep(0.01)  # Give some time for the command to be processed
        


class InvalidControllerType(Exception):
    """Exception returned when received controller type is not
    ControllerStimRecord (this script only works with Stim systems).
    """
