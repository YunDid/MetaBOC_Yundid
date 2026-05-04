"""
HDF5 File  Structure
===================
├── assay
│   └── inputs
│
├── bits
│
├── data_store
│   └── data0000                 ← 本次 recording
│       ├── events               ← 刺激时刻1：事件列表，每个事件包含 frameno（物理帧编号）、eventtype、eventid、eventmessage 等字段   关注1
│       ├── groups
│       │   └── all_channels     ← 记录组（1024 通道）
│       │       ├── raw          ← 原始信号 (1024 × 100000)  刺激时刻和这里对应对其   关注2
│       │       ├── channels     ← 通道列表 (1024)
│       │       ├── frame_nos    ← 刺激时刻2：很关键！！！！！！这里记录了每个采样点的物理帧编号，可以用于对其刺激时刻（event下的frameno对应到这里frame_nos的索引，再对应到rawdata的采样帧号）  关注4
│       │       └── triggered    ← 是否触发标志 
│       │
│       ├── settings             ← 本次记录参数
│       │   ├── sampling         ← 采样率
│       │   ├── gain             ← 增益
│       │   ├── lsb              ← ADC 转换比例
│       │   ├── hpf              ← 高通滤波参数
│       │   ├── spike_threshold  ← spike 阈值
│       │   └── mapping          ← 通道 ↔ 电极 ↔ 坐标 映射
│       │
│       ├── spikes               ← spike 检测结果（当前为空）  ????????????   关注3
│       ├── events               ← 事件记录（当前为空）     ?????????????
│       ├── start_time           ← 记录开始时间
│       ├── stop_time            ← 记录结束时间
│       ├── well_id              ← 所属 well
│       └── recording_id         ← 记录编号
│
├── environment
│   ├── diagnosis
│   └── temperature
│
├── wellplate
│   ├── id
│   ├── variant
│   ├── version
│   └── well000
│       ├── id
│       ├── name
│       ├── group_name
│       ├── group_color
│       └── control
│
├── recordings
│   └── rec0000
│
├── wells
│   └── well000
│
├── notes
├── hdf_version
├── mxw_version
└── version

"""


import h5py
import os
import matplotlib.pyplot as plt
import numpy as np

filename = "/home/maxwell/Data/test_soft_recording/3/data.raw.h5"
filename = "/home/maxwell/Data/test_recording/test/random_test.raw.h5"
# filename = "/home/maxwell/Data/qgj/test_stim/260211/25907/Record/000014/data.raw.h5"

if not os.path.exists(filename):
    raise FileNotFoundError("文件不存在")

def find_frame_index(filename, frameno):
    with h5py.File(filename, "r") as f:
        frame_nos = f["data_store/data0000/groups/all_channels/frame_nos"][:]

    idx = np.where(frame_nos == frameno)[0]

    if len(idx) == 0:
        return False, None
    else:
        return True, int(idx[0])
    
# print(find_frame_index(filename, 304604080))

def show(name, obj):
    indent = "  " * name.count("/")

    if isinstance(obj, h5py.Group):
        print(f"{indent}[Group] {name}")
    else:
        print(f"{indent}[Dataset] {name}")
        print(f"{indent}  → 形状: {obj.shape}")
        print(f"{indent}  → 数据类型: {obj.dtype}")

    if obj.attrs:
        print(f"{indent}  → 属性:")
        for k, v in obj.attrs.items():
            print(f"{indent}     - {k}: {v}")

with h5py.File(filename, "r") as f:
    print("====== 文件整体属性 ======")
    for k, v in f.attrs.items():
        print(f"- {k}: {v}")

    print("\n====== 文件结构 ======")
    f.visititems(show)

    sampling = f["data_store/data0000/settings/sampling"][0]
    print(f"\n采样率: {sampling} Hz")

    # triggered = f["data_store/data0000/groups/all_channels/triggered"][0]
    # print(f"是否触发: {triggered}")

    events = f["data_store/data0000/events"][:]

    print("共有事件条数:", len(events))

    # for i, ev in enumerate(events):
    #     print(f"\n事件 {i}")
    #     print("  frameno      :", ev["frameno"])
    #     print("  eventtype    :", ev["eventtype"])
    #     print("  eventid      :", ev["eventid"])
    #     print("  eventmessage :", ev["eventmessage"])
    
    # raw = f["data_store/data0000/groups/all_channels/raw"]
    # sampling = f["data_store/data0000/settings/sampling"][0]
    # lsb = f["data_store/data0000/settings/lsb"][0]
    # gain = f["data_store/data0000/settings/gain"][0]
    
    start_time = f["data_store/data0000/start_time"][0]
    stop_time  = f["data_store/data0000/stop_time"][0]
    print(f"\n记录开始时间: {start_time}")
    print(f"记录结束时间: {stop_time}")


