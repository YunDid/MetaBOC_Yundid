# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------


from operator import index
from tkinter.messagebox import NO
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *

from src import res

class SignalButtonPara(QPushButton):
    save_sti_para = pyqtSignal(bool)


    def __init__(self, parent=None):
        super(SignalButtonPara, self).__init__(parent)

        self.setIcon(QIcon(":/icons/resources/wave.png"))

        self.setMinimumWidth(36)
        self.setMinimumHeight(36)
        self.setMaximumHeight(36)
        self.setMaximumWidth(36)

        self.index = 0

        self.para = {}

    def set_para(self, para):
        self.para = para

    def get_para(self):
        return self.para