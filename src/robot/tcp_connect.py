# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------

import socket
import threading
import sys
import sched
import time
import datetime

class TcpLogic():
    def __init__(self):
        
        self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_th = None
        self.link = False  # 用于标记是否开启了连接
        
        self.receive_data = None
    
    def disconnect(self):
        # self.client_th.join()
        self.tcp_socket.close()
        # self.tcp_socket.shutdown(2)

    def tcp_client_start(self, ip, port):
        """
        功能函数，TCP客户端连接其他服务端的方法
        :return:
        """
        # self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        state = -1    # 0 表示连接成功，-1 不成功
        try:
            address = (ip, int(port))
        except Exception as ret:
            msg = '请检查目标IP，目标端口\n'
            print(msg)
        else:
            try:
                msg = '正在连接目标服务器\n'
                print(msg)
                # self.tcp_socket.connect(address)
                state = self.tcp_socket.connect_ex((ip, int(port))) 
            except Exception as ret:
                msg = '无法连接目标服务器\n'
                print(msg)
            else:
                state = 0
                self.link = True
                self.client_th = threading.Thread(target=self.tcp_client_receive)
                self.client_th.start()
                msg = 'TCP客户端已连接IP:%s 端口:%s\n' % address
                print(msg)
        return state

    def get_receive_data(self):
        return self.receive_data

    def tcp_client_receive(self):
        """
        功能函数，用于TCP客户端创建子线程的方法，阻塞式接收
        :return:
        """
        msg_time = None
        try:
            while True:
                recv_msg = self.tcp_socket.recv(1024)
                if recv_msg:
                    self.receive_data = recv_msg.decode('utf-8')

                    # now = datetime.datetime.now()
                    # msg_time = now.strftime("%Y-%m-%d %H:%M:%S")
                    # # print(msg_time)
                else:
                    self.tcp_socket.close()
                    msg = 'Disconnect with server!... \n'
                    print(msg)
                    break
        except:
            print("End the tcp receive thread!...")

    def tcp_send(self, mess):
        """
        功能函数，用于TCP服务端和TCP客户端发送消息
        :return: None
        """
        if self.link is False:
            msg = '请选择服务，并点击连接网络\n'
            print(msg)
        else:
            try:
                # send_msg = ("00000000000000000000000000 ".encode('utf-8'))
                self.tcp_socket.send(mess)
                # msg = 'TCP客户端已发送\n'
                print("send sucess!...")
            except Exception as ret:
                msg = '发送失败\n'
                print(msg)