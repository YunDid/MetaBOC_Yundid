# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2024.01.02
# --------------------------------------------------------


from cmath import sqrt
from faulthandler import disable
import random
import math
import time
import numpy as np
from scipy.interpolate import interp1d

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

from src.robot.task import TASK, MAP, GRASP_STATE, GRASP_CONTROL


class Grasping(object):

    def __init__(self):
        # self.cur_state = GRASP_STATE.Reset
        self.cur_state = GRASP_STATE.Grasping
        self.cur_control = GRASP_CONTROL.Mode

        self.arm_step = [1, 2, 3, 4, 5, 6, 7, 8]
        # 1~7：7个抓取状态
        # 8：竖直状态

        self.cur_step_id = 0    # 当前的arm_step id

    def update_cur_control_mode(self, state):
        """
        全自由度控制 & 离散状态控制
        """
        self.cur_control = state

    def update_cur_state(self, state):
        """
        抓取存在三个状态：初始、方向调整、抓取动作
        """
        self.cur_state = state

    def update_grasp(self):
        if self.cur_state == GRASP_STATE.Reset:
            pass
        elif self.cur_state == GRASP_STATE.Direction_Adjust:
            pass
        elif self.cur_state == GRASP_STATE.Grasping:
            pass

    def get_control_signal(self):
        if self.cur_control == GRASP_CONTROL.Mode:
            if self.cur_state == GRASP_STATE.Grasping:
                speed = 1  
                cur_step = self.arm_step[self.cur_step_id]
                free_6 = [0, 0, 0, 0, 0, 0]

                self.cur_step_id = self.cur_step_id + 1
                if self.cur_step_id == 8:    # 抓取完成后，回复至原竖直状态
                    self.cur_step_id = 0
                    self.cur_state = GRASP_STATE.Reset
            elif self.cur_state == GRASP_STATE.Reset:
                speed = 1  
                cur_step = 8
                free_6 = [0, 0, 0, 0, 0, 0]
            
            return free_6, cur_step, speed
        
        elif self.cur_control == GRASP_CONTROL.Freedom:
            cur_step = 0 # 自由控制状态
            speed = 0
            free_6 = [0, 0, 0, 0, 0, 0]  # 存放6个自由度的控制数值，整型
            return free_6, cur_step, speed 

    