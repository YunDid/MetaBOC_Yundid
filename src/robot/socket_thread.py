# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

import imp
import os

import socket
import select

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *

import time

class SocketThread(QThread):
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
        print("Begin send and recieve!...")


        t1 = time.time()
        self.r_mess = self.recive_message()

        self.is_receive = True
        print("recieve time", time.time() - t1)
        print("recive sucessed!...")

        t2 = time.time()
        self.client.send(self.mess.encode("utf-8"))
        self.is_receive = False
        print("send time", time.time() - t2)

        # return r_mess

    def get_recieve_data(self):
        return self.r_mess

    def recive_message(self):
        """
        获取机器人返回的数据
        """
        # msg = self.client.recv(1024)    # 原方法
        # de_msg = total_data.decode("utf-8")

        total_data = bytes()

        r_inputs = set()
        r_inputs.add(self.client)
        w_inputs = set()
        w_inputs.add(self.client)
        e_inputs = set()
        e_inputs.add(self.client)

        while True:
            try:
                r_list, w_list, e_list = select.select(r_inputs, w_inputs, e_inputs, 1)
                for event in r_list:
                    try:
                        data = event.recv(1024)

                        total_data += data
                        # print(data)
                        if len(data) < 1024:
                            break
                    except Exception as e:
                        print(e)
                    if data:
                        print(data)
                        print("收到信息")
                    else:
                        print("远程断开连接")
                        r_inputs.clear()

                print("w")
                if len(w_list) > 0:     # 产生了可写的事件，即连接完成
                    print(w_list)
                    w_inputs.clear()    # 当连接完成之后，清除掉完成连接的socket
                print("e")
                if len(e_list) > 0:     # 产生了错误的事件，即连接错误
                    print(e_list)
                    e_inputs.clear()    # 当连接有错误发生时，清除掉发生错误的socket
                
                print(total_data)
                return total_data
            except OSError as e:
                print(e)

        # while True:
        # # 将收到的数据拼接起来

        #     data = self.client.recv(1024)

        #     total_data += data
        #     print(data)
        #     if len(data) < 1024:
        #         break



        # # print("total data", total_data)
        # return total_data