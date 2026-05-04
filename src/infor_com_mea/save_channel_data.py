# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *

import time
import os
import h5py
import numpy as np



class ChannelDataSaving(QThread):
    def __init__(self, save_path=None, sampling_rate=None, constant=None, channels=71):
        QThread.__init__(self, parent=None)

        self.save_path = save_path
        self.save_sti_path = save_path[:-3] + "_sti_timestamp.txt"
        self.new_data = None    # 每次传入的新数据
        # self.h5_file = None

        self.sti_time_stamp = None    # 保存刺激时刻的数据
        self.sti_ele = None           # 保存刺激时的电极

        self.Samplingrate = sampling_rate
        self.constant = constant

        self.frame_time = 1.0 / self.Samplingrate
        self.current_frame_time = 0   # 时间戳
        self.channels = channels
    
    def set_save_path(self, path):
        self.save_path = path
        if not os.path.exists(self.save_path):
            h5_file = h5py.File(self.save_path, "a")
            para = h5_file.create_dataset("parameter", (2))
            para[0] = self.Samplingrate; para[1] = self.constant
            h5_file.create_dataset("data", (self.channels,10000), maxshape=(self.channels,None), chunks=True, dtype='int32')
            h5_file.close()

        # sti data path
        self.save_sti_path = path[:-3] + "_sti_timestamp.txt"

    
    def set_new_data(self, data):
        self.new_data = data
        self.current_frame_time = self.current_frame_time + self.frame_time * data.shape[0] / self.channels
        # print("data shape", data.shape)
        # print("current time", self.current_frame_time)

    def set_sti_timestamp(self, time_stamp, ele_key):
        self.sti_time_stamp = np.array(time_stamp)
        cur_time_stamp = self.current_frame_time + self.sti_time_stamp

        with open(self.save_sti_path, "a") as f:
            for j in range(len(ele_key)):
                f.write(ele_key[j] + ",")
            f.write("\n")

            for i in range(len(cur_time_stamp)):
                f.write(str(cur_time_stamp[i]) + "\n")

    def run(self):
        try:
            if not os.path.exists(self.save_path):
                h5_file = h5py.File(self.save_path, "a")
                para = h5_file.create_dataset("parameter", (2))
                para[0] = self.Samplingrate; para[1] = self.constant
                h5_file.create_dataset("data", (self.channels,10000), maxshape=(self.channels,None), chunks=True, dtype='int32')
                h5_file.close()
                print("==="*30)
            else:
                with h5py.File(self.save_path, "a") as hf:
                    t1 = time.time()
                    frame = int(self.new_data.shape[0] / self.channels)
                    channel_data = np.reshape(self.new_data,(int(self.new_data.shape[0]/frame), frame), "F")
                        
                    hf['data'].resize((hf['data'].shape[1] + channel_data.shape[1]), axis = 1)
                    hf['data'][:, -channel_data.shape[1]:] = channel_data

                    # print("time", time.time() - t1)
        except:
            print("Saving data error happend!...")

            # t1 = time.time()
            # self.h5_file['data'].resize((self.h5_file['data'].shape[0] + self.new_data.shape[0]), axis = 0)
            # self.h5_file['data'][-self.new_data.shape[0]: ] = self.new_data
        

    
    def close_file(self):
        self.h5_file.close()