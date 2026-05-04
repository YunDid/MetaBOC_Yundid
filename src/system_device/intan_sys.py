# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2024
# Written by Guiping Cao
# Time: 2024.04.14
# --------------------------------------------------------

import numpy as np
import os
import h5py
import time


from src.robot.encode_decode import EncodingDecoding

from src.robot.task import TASK, MAP, SYSTEM_DEVICE

from src.system_device.recording_intan import RecordingIntan
from src.system_device.stimulating_intan import Stimulating_Intan_Platform


class INTAN_System(object):

    def __init__(self,):
        super(INTAN_System, self).__init__()

        self.recording = RecordingIntan()
        self.stimulating = Stimulating_Intan_Platform()

    def stop_connect(self):
        self.recording.recording.file_monitor.stop()
        self.recording.recording.stop_data_loading_thread()

        self.stimulating.stopRecord()
        self.stimulating.close_connection()

        del self.recording; self.recording =None
        del self.stimulating; self.stimulating = None

        