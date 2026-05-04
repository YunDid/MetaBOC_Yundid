# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2022
# Written by Guiping Cao
# Time: 2022.04.17
# --------------------------------------------------------


import copy
import time
import math
import socket
import random
from src.robot.robot import Robot

from src.robot.socket_thread import SocketThread
from src.robot.socket_send_thread import SocketSendThread
from src.robot.tcp_connect import TcpLogic

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


# 真实机器人

class RealRobot(Robot):

    def __init__(self, ip_adress="115.29.109.104", port="6513"):
        self.scale = 0.3  # cm
        axis_length = 182 * self.scale  # robot轴长,mm
        wheel_radius = 75 / 2.0 * self.scale  # 车轮半径 mm

        Robot.__init__(self, axis_length, wheel_radius)

        self.first_pos_update = False

        self.ip_adress = ip_adress
        self.port = port

        self.axis_length = axis_length

        self.ip_adress = "115.29.109.104"
        self.port = "6513"

        self.type = "REAL_ROBOT"

        self.r_mess = None

        self.real_robot_mess = None  # 真实机器人的位置和角度信息

        self.angles = 0

        self.tcp = TcpLogic()  # 用于连接server，发送和接收数据

    def update_start_pos(self):
        self.wheel_lux = self.start_x + self.wheel_radius
        self.wheel_luy = self.start_y - self.axis_length / 2
        self.wheel_ldx = self.start_x - self.wheel_radius
        self.wheel_ldy = self.start_y - self.axis_length / 2

        self.wheel_rux = self.start_x + self.wheel_radius
        self.wheel_ruy = self.start_y + self.axis_length / 2
        self.wheel_rdx = self.start_x - self.wheel_radius
        self.wheel_rdy = self.start_y + self.axis_length / 2

    def connect_robot(self):
        """
        与真实的机器人建立连接
        """
        state = self.tcp.tcp_client_start(self.ip_adress, self.port)

        if state == 0:  # 连接成功
            print("Connected!...")
            return True
        else:
            print("Connect failed!...")
            return False

    def stop_robot(self):
        print("Stop the real robot!...")
        a = 0
        b = 0
        c = 0
        d = 0
        messg = "(" + self.control_mess_encoding(a) + "," + self.control_mess_encoding(
            b) + "," + self.control_mess_encoding(c) + "," + self.control_mess_encoding(d) + ")"
        self.tcp.tcp_send(messg.encode('utf-8'))

    def set_ip_port(self, ip, port):
        self.ip_adress = ip
        self.port = port

    #######################################################
    #   Function: 修改传输速度格式
    #   Input   : 要发送的速度
    #   Output  : 一个字符串
    ########################################################################
    def control_mess_encoding(self, v):

        # 保留小数点后三位
        v = round(v, 3)
        # 如果为负数设置标志位为1
        if (v < 0):
            v1 = 1
            v = -v
            # 否则设置标志位为0
        else:
            v1 = 0
        # 扩大1000被，方便传输
        v = int(v * 1000)
        # 最高位
        v2 = int(v / 10000 % 10)
        # 次高位
        v3 = int(v / 1000 % 10)
        v4 = int(v / 100 % 10)
        v5 = int(v / 10 % 10)
        v6 = int(v / 1 % 10)
        v = str(v1) + str(v2) + str(v3) + str(v4) + str(v5) + str(v6)
        return v

    ########################################################################
    #   Function: 获取雷达数据
    #   Input   : 接收到的数据
    #   Output  : list，每个点用tuple表示，第一个数据为距离，第二个数据为角度0
    ########################################################################
    def get_obstacle_distance_data(self, total_data):
        # rec = total_data.decode("utf-8")                        # 将接收到的数据转化

        data_split = total_data.split("\n")
        # print(f'data_split===[{data_split}]')
        for i in range(len(data_split)):
            tp = data_split[i]
            print
            if len(tp) > 0:
                # print(f'tp===[{tp}]')
                data = self.get_raw_data(tp)
                if data is not None:
                    # print(f'data===[{data}]')
                    return data

        return None

    def get_raw_data(self, rec):
        """
        获取雷达障碍物信息
        """
        Point = []  # 初始化接收列表
        if rec[:2] == '{B' and rec[-1] == '}':  # 判断是否是一组完整的数据
            msg_points = rec.split(";")  # 按照';' 进行分割
            print(f'msg_point==[{msg_points}]')
            for msg_point in range(len(msg_points)):  # 对每一个分割的点进行处理

                point = msg_points[msg_point]
                index_douhao = point.find(',')  # 每个点的格式为（距离，角度），逗号前的数据为距离，后面的为角度信息

                tuple1 = point[1:index_douhao]  # 提取距离信息

                tuple2 = point[index_douhao + 1:-1]  # 提取角度信息，并舍弃小数部分

                # 判断并转换为整数
                if tuple2.startswith('-'):
                    if tuple1.isdigit():
                        tuple1 = int(tuple1)
                        try:
                            tuple2 = int(tuple2)
                            print(f'tuple2 ==[{tuple2}], True')
                            # print(f'Converted integer: {int_value}')
                            tupl = (tuple1, tuple2)  # 将提取的信息存在一个元组中

                            Point.append(tupl)
                            continue
                        except ValueError:
                            print(f'tuple2 ==[{tuple2}], False')

                # 判断提取的是否是数字，并转化为浮点数
                if tuple1.isdigit() and tuple2.isdigit():
                    tuple1 = int(tuple1)
                    tuple2 = int(tuple2)
                # 若不为数字，则跳过这个点继续循环
                else:
                    continue
                tupl = (tuple1, tuple2)  # 将提取的信息存在一个元组中

                Point.append(tupl)  # 将元组添加到接收列表
        else:
            return None
        return Point

        ########################################################################

    #   Function: 获取轮速信息
    #   Input   : 接收到的数据
    #   Output  : 长度为4的list，依次为A、B、C、D的轮速
    ########################################################################
    def  get_velocity_data(self, total_data):
        # rec = total_data.decode("utf-8")                        # 将接收到的数据转化

        rec = total_data

        Point = []  # 初始化接收列表
        try:
            rec_split = rec.split("\n")
        except Exception as e:
            # 输出简单的错误信息
            print(f"请确认发送的数据是否符合当前任务！！！！！！！！！ An error occurred: {e}")
        for i in range(len(rec_split)):
            data = rec_split[i]

            if data[:3] == '{C:' and data[-1] == '}':  # 判断是否是一组完整的数据，以{}包裹的是完整信息
                valid_data = data[3:]
                print(f'data={data}')
                msg_wheel = valid_data.split(';')  # 从第一个数据的第一个字符到最后一个字符，前24个字符为速度信息

                # 分割后的第2个元素为A轮的信息，示例：(0.0)
                temp = msg_wheel[3]
                Motor_A = float(temp[1:-1])  # 去掉括号为数字信息，并转化为浮点数

                # 分割后的第3个元素为B轮的信息，示例：(0.0)
                temp = msg_wheel[4]
                Motor_B = float(temp[1:-1])  # 去掉括号为数字信息，并转化为浮点数

                # 分割后的第4个元素为C轮的信息，示例：(0.0)
                temp = msg_wheel[5]
                Motor_C = float(temp[1:-1])  # 去掉括号为数字信息，并转化为浮点数

                # 分割后的第5个元素为D轮的信息，示例：(0.0)
                temp = msg_wheel[6]
                Motor_D = float(temp[1:-2])  # 去掉括号为数字信息，并转化为浮点数

                # 添加到列表
                Point.append(Motor_A)
                Point.append(Motor_B)
                Point.append(Motor_C)
                Point.append(Motor_D)

        return Point

        ########################################################################

    #   Function: 获取XY，机器人的位置信息
    #   Input   : 接收到的数据
    #   Output  : XY
    ########################################################################
    def get_xy_data(self, total_data):
        rec = total_data

        pt_x = []
        pt_y = []
        pt_z = []

        rec_split = rec.split("\n")
        # 初始化接收列表

        for i in range(len(rec_split)):
            data = rec_split[i]
            if data[:3] == '{C:' and data[-1] == '}':  # 判断是否是一组完整的数据，以{}包裹的是完整信息
                print(f'C = {data}')
                valide_data = data[3:]
                mess = valide_data.split(";")
                x = int(mess[0])
                y = int(mess[1])
                z = int(mess[2])

                pt_x.append(x)
                pt_y.append(y)
                pt_z.append(z)

        return pt_x, pt_y, pt_z

    def disconnect(self):
        self.tcp.disconnect()

    def tcp_receive(self):
        """
        接收雷达信息，用于避障和跟踪
        """

        # 接收雷达信号
        r_mess = self.tcp.get_receive_data()
        print(f'r_mess={r_mess}')

        if r_mess is None:
            r_mess = self.r_mess
        else:
            self.r_mess = r_mess
        vl, vr, obs_data_tuple = self.get_velocity_data_same_time(r_mess)

        # get arm data
        arm_data = self.get_arm_data(r_mess)

        # get velocity (从雷达获取机器人信息)

        velocity = {"vl": vl, "vr": vr, "obs": obs_data_tuple}

        return arm_data, velocity

        # MODE = "ARM"  # ARM or VELOCITY
        # if MODE == "VELOCITY":
        #     # 轮速、障碍物信息的获取（雷达信息进行数据解）

        # elif MODE == "ARM":  # 获取机械臂的信息

        #     return arm_data

    def get_velocity_data_same_time(self, r_mess):
        # 更新虚拟地图中机器人位置的方法
        VELOCITY_scale_factor = 1.0
        velocity = self.get_velocity_data(r_mess)  # A B C D, 前轮为B C
        vl = velocity[1]
        vr = velocity[2]

        obs_data_tuple = self.get_obstacle_distance_data(r_mess)
        x, y, z = self.get_xy_data(r_mess)  # 获取机器人的位置信息
        # y[0] = -1.0 * y[0]

        print("x", x, "    y:" , y, "   z:", z)
        # z[0] = - z[0] * 1.7
        # z[0] = - z[0]

        # angle_in_radians = math.radians(z[0])
        # print("angle_in_radians:", angle_in_radians)
        # self.angles = z[0] * 0.001 + math.pi / 2  # 弧度信息，机器人的全局角度信息（浮点数不好传输，所以乘了1000传过来的，现在再除回去）
        self.angles = z[-1] * 0.001
        temp_x = -1.0 * x[-1] * self.scale * VELOCITY_scale_factor
        temp_y = -1.0 * y[-1] * self.scale * VELOCITY_scale_factor # 机器人运动方向的角度

        self.x, self.y = self.transpose_x_y(temp_x, temp_y)
        self.x +=200
        self.y += 200
        # 首次连接，更新机器人初始位置
        if not self.first_pos_update:
            self.first_pos_update = True
            self.start_x = self.x
            self.start_y = self.y
            self.update_start_pos()
        # self.angles = (z[0] * 0.001) * math.pi / 180    # 转化为角度

        return vl, vr, obs_data_tuple

    def transpose_x_y(self, x, y):
        """
        解析机器人运动传入的x和y, 解析为实际的x y 坐标
        """
        # x分解
        x_x = x * math.cos(self.angles)
        x_y = x * math.sin(self.angles)

        # y分解
        y_x = y * math.cos(math.pi * 0.5 - self.angles)
        y_y = y * math.sin(math.pi * 0.5 - self.angles)

        new_x = x_x + y_x + 100
        new_y = x_y - y_y + 100

        return new_x, new_y

    def send_control_mess(self, left, right):
        if left is None:
            self.w_right = right
        elif right is None:
            self.w_left = left
        else:
            self.w_left = left
            self.w_right = right

        # 计算左右轮速度
        # vl = self.w_left * self.wheel_radius / 1000
        # vr = self.w_right * self.wheel_radius / 1000   # m/s

        # vl = (self.w_left - 0.5) * self.run_directions
        # vr = (self.w_right - 0.5) * self.run_directions

        # 设定机器人速度范围：0~0.3

        vl = self.w_left * self.wheel_radius * self.run_directions / 50
        vr = self.w_right * self.wheel_radius * self.run_directions / 50

        # vl = self.w_left * self.wheel_radius * self.run_directions / 50
        # vr = self.w_right * self.wheel_radius * self.run_directions / 50
        # vl*=0.4
        # vr*=0.4

        # vl = round(vl, 4)    # v*1000 m/s    vl mm/s
        # vr = round(vr, 4)    # v*1000 m/s

        print("vl:", vl)
        print("vr:", vr)
        a = 0;
        b = vl;
        c = vr
        d = 0
        messg = '(' + self.control_mess_encoding(a) + "," + self.control_mess_encoding(
            b) + "," + self.control_mess_encoding(c) + "," + self.control_mess_encoding(d) + ')'


        self.tcp.tcp_send(messg.encode('utf-8'))


        if self.run_directions == -1:
            # time.sleep(1.5)    # 机器人回退时间，通过imageWidget中的计时器控制，从而信号和轨迹不中断显示
            pass

    # 通过频率计算小车位置,输入为左右轮的轮速角速度
    def update_wheels(self):
        """
        通过从真实的机器人接收障碍物信息和机器人位置信息（x,y），更新平台中同步显示的位置和距离信息
        """

        # if left is None:
        #     self.w_right = right
        # elif right is None:
        #     self.w_left = left
        # else:
        #     self.w_left = left
        #     self.w_right = right

        # # 计算左右轮速度
        # # vl = self.w_left * self.wheel_radius / 1000
        # # vr = self.w_right * self.wheel_radius / 1000   # m/s

        # # vl = (self.w_left - 0.5) * self.run_directions
        # # vr = (self.w_right - 0.5) * self.run_directions

        # vl = (self.w_left - 0.5) * self.run_directions
        # vr = (self.w_right - 0.5) * self.run_directions

        # # vl = round(vl, 4)    # v*1000 m/s    vl mm/s
        # # vr = round(vr, 4)    # v*1000 m/s

        # print("vl:", vl)
        # print("vr:", vr)
        # a = 0; b = vl; c = vr; d = 0
        # messg = '(' + self.control_mess_encoding(a)+","+self.control_mess_encoding(b)+","+self.control_mess_encoding(c)+","+self.control_mess_encoding(d) + ')'

        arm_data, robot_mess = self.tcp_receive()
        vl, vr, obs_data_tuple = robot_mess["vl"], robot_mess["vr"], robot_mess["obs"]
        print(self.angles, 30*"==")

        self.wheel_lux = self.x + math.cos(math.pi / 2.0 - self.angles - self.alpha) * self.big_length
        self.wheel_luy = self.y - math.sin(math.pi / 2.0 - self.angles - self.alpha) * self.big_length
        self.wheel_ldx = self.x + math.cos(math.pi / 2.0 - self.angles + self.alpha) * self.big_length
        self.wheel_ldy = self.y - math.sin(math.pi / 2.0 - self.angles + self.alpha) * self.big_length

        self.wheel_rux = self.x - math.cos(math.pi / 2.0 - self.angles + self.alpha) * self.big_length
        self.wheel_ruy = self.y + math.sin(math.pi / 2.0 - self.angles + self.alpha) * self.big_length
        self.wheel_rdx = self.x - math.cos(math.pi / 2.0 - self.angles - self.alpha) * self.big_length
        self.wheel_rdy = self.y + math.sin(math.pi / 2.0 - self.angles - self.alpha) * self.big_length

        points = QPoint(self.x, self.y)
        self.points.append(points)
        self.angles_list.append(self.angles)

        self.real_robot_mess = self.x, self.y, self.angles

        return obs_data_tuple, (self.x, self.y, self.angles), self.scale  # 返回真实的障碍物信息

    def update_arm_from_real_robot(self):
        arm_data, robot_mess = self.tcp_receive()
        return arm_data, robot_mess

    ########################################################################
    #   Function: 获取机械臂信息
    #   Input   : 接收到的数据
    #   Output  : 字符串的长度是34，依次为A、B、C、D的轮速
    #             arm="{A:0000:0000:0000:0000:0000:0000}"
    ########################################################################
    def get_arm_data(self, total_data):
        """
        解析获取到的arm信息
        """
        # rec = total_data.decode("utf-8")      # 将接收到的数据转化, 默认已转化
        rec = total_data  # 将接收到的数据转化

        Point = []  # 初始化接收列表
        # if rec.startswith('{A') and rec.endswith('}'):
        if rec.startswith('{A') and rec.endswith('\n'):
            wheel = rec[3:32]
            msg_wheel = wheel.split(':')

            for motor_angle in msg_wheel:
                Point.append(motor_angle)

            get_arm_datas = "{" + ":".join(map(str, Point)) + "}"  # 六个自由度度的角度信息，单位：弧度
            return get_arm_datas

    def get_real_robot_mess(self):
        return self.real_robot_mess

    def generate_random_obstacle_mess(self, length=100):
        out = []

        for i in range(length):
            angle = random.randint(0, 360)
            dis = random.randint(50, 200)
            mess = (angle / 180.0 * math.pi, dis)
            out.append(mess)

        return out

    #######################################################
    #   Function: 修改传输机械臂6自由度控制格式
    #   Input   : 要发送的机械臂的单个自由度的控制角度
    #   Output  : 一个字符串
    ########################################################################
    def change_arm(self, a):

        # 保留小数点后三位
        a = round(a, 2)
        # 如果为负数设置标志位为1
        if (a < 0):
            a1 = 1
            a = -a
            # 否则设置标志位为0
        elif a == 2000:
            a1 = 2
        else:
            a1 = 0
        # 扩大1000被，方便传输-1.57*100=-157
        a = int(a * 100)
        # 最高位
        a2 = int(a / 100 % 10)
        # 次高位
        a3 = int(a / 10 % 10)
        a4 = int(a / 1 % 10)
        a = str(a1) + str(a2) + str(a3) + str(a4)
        return a

    def tcp_send_arm_control(self, free_signal):
        """
        状态控制
        mess: 共8个数值, 整型;
        1~6:  控制各自由度;
        7: 控制模式 取值0~8, 0自由控制, 1~7控制抓取过程, 8竖直状态;
        8位: 控制速度 取值0 慢 1 快；
        """

        free_6 = free_signal["free"]
        mode = free_signal["cur_step"]
        speed = free_signal["speed"]
        sendmsg = "[" + self.change_arm(free_6[0]) + "," + self.change_arm(free_6[1]) + "," + self.change_arm(
            free_6[2]) + "," \
                  + self.change_arm(free_6[3]) + "," + self.change_arm(free_6[4]) + "," + self.change_arm(
            free_6[5]) + "," \
                  + str(mode) + "," + str(speed) + "]"
        # print("send mess:", sendmsg)
        self.tcp.tcp_send(sendmsg.encode("utf-8"))

# 待办
# 1. 机械臂和目标物体的雷达信息的同时获取；（wy)
# 2. 自由控制时，其他自由度的自动设置；（传过去0或-1，其余自由度保持当前位姿）（wy）
# 3. 自动调节控制模式；（gp）
# 4. MEA分类控制（抓取任务下的编解码），包括角度方案的控制问题：电极、分类任务；（gp）
# 5. 控制信号的输出;
# 6. MEA的联调(避障、跟踪、抓取)；