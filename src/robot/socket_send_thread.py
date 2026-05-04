# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

import imp
import os

import socket

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *

import time

class SocketSendThread(QThread):
    def __init__(self, client=None):
        QThread.__init__(self)

        self.client = client

        self.mess = ""
        self.r_mess = None
        self.is_receive = False


    def set_ip_address(self, ip_adress, port):
        self.ip_adress = ip_adress
        self.port = port

    def connect(self):
        retcode = self.client.connect_ex((self.ip_adress, int(self.port))) 
        
        if retcode == 0:    # 连接成功
            print("Connected!...")
            return True
        else:
            print("Connect failed!...")
            return False

    def disconnect(self):
        self.client.close()

    def set_data(self, mess):
        self.mess = mess

    def run(self):
        """
        发送控制信号：{a,b,c,d} 四个轮子的轮速
        """
        print("Begin send!...")


        t2 = time.time()
        self.client.send(self.mess.encode("utf-8"))
        print("send time", time.time() - t2)


    def get_recieve_data(self):
        return self.r_mess

    def recive_message(self):
        """
        获取机器人返回的数据
        """
        # msg = self.client.recv(1024)    # 原方法
        # de_msg = total_data.decode("utf-8")

        total_data = bytes()
        while True:
        # 将收到的数据拼接起来
            # print(30*"==")
            data = self.client.recv(1024, 0)
            total_data += data
            print(data)
            if len(data) < 1024:
                break

        # print("total data", total_data)
        return total_data