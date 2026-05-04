import time
import socket
import os

import tkinter as tk
from tkinter import filedialog
















def setFilePath(scommand, baseFileName, path):
    """
    设置文件的基本名称和路径，用于在RHX软件中保存数据文件。

    :param scommand: 已连接到TCP服务器的套接字对象。用于向RHX软件发送设置文件路径和名称的TCP命令。
    :param baseFileName: 要设置的数据文件的基本名称。这个名称将用于生成最终的数据文件名，但不包括文件扩展名。
        - 拓展名跟随文件格式的设置
        - 文件名 Intan 回默认添加时间戳，例如 最终名称为 FileName_240331_162550，年-月-日-时-分-秒
    :param path: 数据文件将被保存的目录路径。应确保这个路径已存在，并且应用程序有权写入该路径。

    在发送设置基本文件名和路径的命令之前，会先确保控制器处于停止状态。命令被合并后一起发送，以确保设置生效。
    """
    # 确保控制器已停止
    ensure_controller_stopped(scommand, COMMAND_BUFFER_SIZE)

    # 生成并发送设置基本文件名和路径的命令
    com_baseFileName = f'set filename.basefilename {baseFileName};'
    com_path = f'set filename.path {path};'
    com_setFile = com_baseFileName + com_path
    scommand.sendall(com_setFile.encode())

def setSaveFileFormat(scommand, savleTypeIndex, latencyIndex, NewDirectory, SaveWidebandAmplifierWaveforms, NewSaveFilePeriodMinutes):
    """
    设置RHX软件中保存数据文件的格式和相关配置。

    :param scommand: 已连接到TCP服务器的套接字对象。用于向RHX软件发送设置保存文件格式的TCP命令。
    :param savleTypeIndex: 文件格式索引。可选值包括 0 ("Traditional"), 1 ("OneFilePerSignalType"), 2 ("OneFilePerChannel")。
    :param latencyIndex: 写入磁盘延迟级别的索引。可选值包括 0 ("Highest"), 1 ("High"), 2 ("Medium"), 3 ("Low"), 4 ("Lowest")。
    :param NewDirectory: 是否为每次记录创建新的目录。True 或 False。
    :param SaveWidebandAmplifierWaveforms: 是否保存宽带放大器波形数据。True 或 False。
    :param NewSaveFilePeriodMinutes: 新文件创建周期，以分钟为单位。数值类型。

    在发送设置保存文件格式和相关配置的命令之前，会先确保控制器处于停止状态。命令被合并后一起发送，以确保设置生效。
    """
    # 确保控制器已停止
    ensure_controller_stopped(scommand, COMMAND_BUFFER_SIZE)

    # 设置索引列表
    FileFormat_list = ["Traditional", "OneFilePerSignalType", "OneFilePerChannel"]
    Latency_list = ["Highest", "High", "Medium", "Low", "Lowest"]

    # 生成并发送设置基本文件名和路径的命令
    com_FileFormat = f'set FileFormat {FileFormat_list[savleTypeIndex]};'
    com_Latency = f'set WriteToDiskLatency {Latency_list[latencyIndex]};'
    com_NewDirectory = f'set CreateNewDirectory {NewDirectory};'
    com_NewSaveFilePeriodMinutes = f'set NewSaveFilePeriodMinutes {NewSaveFilePeriodMinutes};'

    # 设置存储波形数据类型，其他数据类型，直接添加即可，此处以放大器数据为例.
    com_SaveWidebandAmplifierWaveforms =  f'set SaveWidebandAmplifierWaveforms {SaveWidebandAmplifierWaveforms};'

    com_setFile = com_FileFormat + com_Latency + com_NewDirectory + com_NewSaveFilePeriodMinutes + com_SaveWidebandAmplifierWaveforms
    scommand.sendall(com_setFile.encode())



def RunAndStimulateDemo():

    # Connect to TCP command server - default home IP address at port 5000
    scommand = connect_to_server()
    # Query controller type from RHX software.
    # Throw an error and exit if controller type is not Stim.
    verify_controller_type(scommand, COMMAND_BUFFER_SIZE)
    # Query runmode from RHX software
    ensure_controller_stopped(scommand, COMMAND_BUFFER_SIZE)
    # -----------------------------------------------------------------------------------------
    # Test1
    # TestDemo1(scommand)
    # Test2
    TestDemo2(scommand)
    # -----------------------------------------------------------------------------------------
    # Close TCP socket
    disconnect_from_server(scommand)

def TestDemo2(scommand):
    baseFileName = f"FileName"
    path = f"E:/TCP/Data"

    setFilePath(scommand, baseFileName, path)
    setSaveFileFormat(scommand,2,4,True,True,5)

    startRecord(scommand)
    time.sleep(2)
    stopRecord(scommand)

def TestDemo1(scommand):

    channels = ['A-000']

    command = configureTrainStimulation(scommand, channels[0], 'keypressf1', 10, 500, 'PulseTrain', 256, True)
    scommand.sendall(command.encode())

    scommand.sendall(b'set runmode run;')

    TriggerStimulation(scommand,'keypressf1')

if __name__ == '__main__':
    # Declare buffer size for reading from TCP command socket
    # This is the maximum number of bytes expected for 1 read. 1024 is plenty
    # for a single text command.
    # Increase if many return commands are expected.
    COMMAND_BUFFER_SIZE = 1024

    RunAndStimulateDemo()

