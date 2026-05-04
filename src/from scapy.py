from scapy.all import sniff
from datetime import datetime
import time

prev_prev_timestamp = None  # 用于记录上上一个数据包的时间戳
prev_timestamp = None  # 用于记录上一个数据包的时间戳

def packet_callback(packet):
    global prev_prev_timestamp, prev_timestamp
    if packet.haslayer('TCP') and packet['IP'].dst == "115.29.109.104" and packet['TCP'].dport == 6513:
        current_timestamp = datetime.now()
        timestamp_str = current_timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(f"[{timestamp_str}] {packet.summary()}")
        if packet.haslayer('Raw'):  # 判断是否有应用层数据（payload）
            payload = packet['Raw'].load.decode('utf-8', 'replace')  # 尝试以UTF-8解码，无法解码的字符用替换字符表示
            print("Data:")
            lines = payload.splitlines()
            for line in lines:
                print(f"    {line.strip()}")  # 去除每行两端可能的空白字符后再打印，使输出更规范
        
        if prev_prev_timestamp and prev_timestamp:
            time_diff = (current_timestamp - prev_prev_timestamp).total_seconds()
            print(f"相隔两个数据的时间戳差值（秒）: {time_diff}")
        
        prev_prev_timestamp = prev_timestamp
        prev_timestamp = current_timestamp

# 开始监听，你可以根据实际需求调整或者设置为0表示持续监听
sniff(filter="tcp", prn=packet_callback, count=0)