# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MetaBOC（Meta Brain Organoid Control）是一个面向体外神经网络功能评估的闭环神经接口平台。平台通过微电极阵列（MEA）及柔性电极将人脑类器官与机器人接口耦合，构建感觉运动反馈回路，实现类器官在闭环环境中的具身学习与行为控制。

### 研究背景

脑类器官（brain organoid）是由人多能干细胞（hESC/hiPSC）分化培养的三维神经组织，可重现人脑的部分细胞多样性、突触连接和电生理活动。现有研究已证明类器官具备自发振荡、突触可塑性（LTP/LTD）等基础能力，但缺乏**将类器官嵌入感觉运动环路并量化其行为控制能力**的方法学工具。MetaBOC 正是为填补这一空白而设计的闭环平台。

### 平台定位

- **核心价值**：为神经科学研究提供标准化的体外神经网络行为学评估工具，而非单纯的软件系统
- **闭环原理**：类器官的自发 spike 活动经解码后转化为机器人运动指令，机器人的环境传感信号经编码后转化为电刺激反馈至类器官，形成 100ms 周期的实时闭环
- **多任务支持**：避障（Obstacle Avoidance）、目标跟踪（Object Tracking）、机械臂抓取（Object Grasping），通过编解码参数切换实现
- **开源设计**：模块化架构，支持不同硬件接入和新任务扩展

### 硬件主线

平台基于不同电生理采集设备形成多条硬件主线：
- **二维片上脑**：MCS MEA2100 系统（`src/infor_com_mea/`）— 60 通道平面微电极阵列，25kHz 采样，适用于 2D 贴壁培养的神经网络
- **三维片上脑**：Intan RHX 系统（`src/system_device/`）— 64 通道立体电极（双端口 A/B），30kHz 采样，配合柔性网状电极和探针电极用于 3D 类器官
- **扩展方向**：MaxOne 高密度 MEA 设备接入（同时支持二维与三维，开发中）
  - Maxwell API 文档与手册位于 PKM 外部路径注册表中「Maxwell 参考资料」条目所指位置（Sphinx HTML 站点 + PDF 手册）
  - PKM 端摘要索引：`Atlas/Sources/Maxwell/SUMMARIES.md`

### 关键创新

1. **多模态神经接口统一抽象**：同一套 Communication 类兼容 MCS 和 Intan 两套硬件，通过枚举动态切换，新设备接入只需实现 Recording/Stimulation 接口
2. **双向编解码管线**：环境信号→刺激信号的编码（距离-角度联合调制）与神经活动→运动指令的解码（动量差分法）分离设计，可独立替换
3. **AI 代理模型增强闭环**：基于生物启发的兴奋-抑制循环神经网络（EI_RNN）学习类器官输入-输出动态映射，通过模型预测控制（MPC）在线生成最优刺激序列，提升控制精度
4. **奖惩反馈机制**：reward/punishment/environment 三类刺激协同驱动类器官学习

### 论文关联

本代码库服务于两个知识管理项目（由 CC-PKM 端管理）：
- **metaboc**：平台代码拆解与二次开发
- **thesis**：硕士论文撰写（论文主体基于本平台的架构、方法与实验验证）

## 协作架构（CC-Code 端）

本仓库为纯开发工作台。PKM（Second Cortex）作为知识容器，集中管理所有参考资料、原子卡片、项目调度与里程碑。项目全景上下文见 PKM 中的 `Atlas/Dossiers/Dossier - MetabOC 平台资产总览.md`。

**本端职责**：代码阅读、架构分析、模块拆解、二次开发
**不做**：原子卡片创建、项目状态管理、里程碑更新（均由 CC-PKM 端负责）

**输出约束**：
- 使用工程化技术语言，禁止 emoji，保持客观严谨
- 去时间化：不使用"今天""昨天"等时间词汇
- 结构化优先：标题+要点列表，避免大段落叙事，但是每个标题下的段落叙述可以详细
- 模块拆解产出粒度："一个模块一张卡片"，可直接作为 PKM 原子卡片正文
- **认知负载转移**：用户是木桶短板侧，CC 必须主动承担框架设计、方案构思、骨架列举等认知负载工作。需要信息时向用户提出具体问题清单，而非要求用户自行构思框架。CC 基于业界最佳实践输出完整方案，用户只做审核决策。

**相关命令**：使用 `/debrief` 生成结构化产出，供搬运到 CC-PKM 端沉淀

## MaxOne / Maxwell API 概要

MaxOne 是 MaxWell Biosystems 的单孔高密度微电极阵列平台，作为 MetaBOC 的第三条硬件主线接入。

**通信架构**：`Python/C++ 脚本 → mxwserver（后台进程） → MaxOne/MaxTwo 设备`。所有 API 调用前 mxwserver 必须运行。

**Python API**（`import maxlab as mx`，开环与配置）：
1. `mx.initialize()` — 芯片重置
2. `mx.Array("stimulation")` — 配置记录电极（≤1020）和刺激电极（≤32），`array.route()` + `array.download()`
3. `mx.Sequence(name)` — 构建刺激序列（`mx.chip.DAC(...)` + `mx.DelaySamples(...)`）
4. `mx.save_recording(path)` / `mx.stop_recording()` — 录制控制
5. `mx.send(obj)` — 发送 ApiObject 到服务器

**C/C++ API**（闭环低延迟）：
- `DataStreamerRaw` — 每帧 1024 通道振幅值（50 us 采样周期）
- `DataStreamerFiltered` — 仅 spike 事件，支持 FIR/IIR 滤波器
- 典型模式：Python 设置阵列配置 + C++ 二进制监听数据流并触发闭环响应

**硬件约束**：≤1020 记录通道，≤32 刺激单元（每电极映射唯一单元），DAC 10 位（0-1023），有效增益 1/7/112/512/1024/1025/2048，仅 Python 3。

## Running the Application

```bash
python main.py
```

## Dependencies

没有 requirements.txt 或 setup.py。依赖安装命令记录在 `main.py` 文件头部注释中。核心依赖：
- PyQt5, PyQt5-tools（GUI）
- pythonnet（C#/.NET 互操作，硬件控制）
- numpy, pandas, scipy, h5py（数据处理）
- matplotlib, pyqtgraph, pyQtChart, pyOpenGL（可视化）
- pytorch_lightning, scikit-learn（ML/DL）
- casadi, do_mpc（MPC 优化）
- watchdog, loguru（文件监控、日志）

## Architecture

### 模块化架构（论文对应：BioBOC / StimBOC / InterBOC / RoBOC）

1. **BioBOC — 多模态神经接口模块**
   - **MCS MEA2100 系统（2D）**（`src/infor_com_mea/`）
     - `recording.py` — 25kHz 实时采集，60 通道，pythonnet 调用 C# DLL
     - `spike_detection.py` — 带通滤波（300-3000Hz）+ 5σ 阈值检测
     - `stimulation.py` — STG200x 刺激器控制
   - **Intan RHX 系统（3D）**（`src/system_device/`）
     - `recording_intan.py` — 30kHz 流式采集，64 通道（A/B 双端口）
     - `RealTimeDataReader.py` — 文件监控 + 数据加载
     - `stimulating_intan.py` — RHX API 刺激接口
     - 伪迹消隐：`get_spike_data_from_channel_data()` 检测数字触发信号，消隐刺激后 10ms 窗口
   - 两套系统通过 `SYSTEM_DEVICE` 枚举切换（`src/robot/task.py`）

2. **StimBOC — 电刺激编码模块**（`src/infor_com_mea/stimulation.py`）
   - 5Hz 时隙范式（200ms/slot），duty cycle 60%
   - 突发掩码（`make_burst_mask()`）：随机突发模式防止适应
   - 双相脉冲：+75mV → -75mV，200μs/相
   - 三类刺激：reward（100Hz 固定）/ punishment（碰撞触发）/ environment（距离编码，4-40Hz）
   - MCS 与 Intan 的刺激实现差异

3. **InterBOC — 编解码与神经动力学模块**
   - **编解码管线**（`src/robot/encode_decode.py`）
     - 编码：距离 + 角度联合调制 → 刺激频率（`encode_obstacle_avoidance_with_angleAndDis()`）
     - 解码：动量差分法，当前 spike 率 vs 滑动 3 窗口均值 → 轮速（`decode_obstacle_avoidance_same_side_history_gap()`）
     - 编解码参数存储：`encode_decode/en_de_coding.npz`
   - **神经动力学代理模型**（`src/dynamic_model/`）
     - `Models.py` — EI_RNN（兴奋-抑制约束，4:1 比例，稀疏连接）
     - `Trainer.py` — 训练循环，MSE + L1/L2 正则化
     - `dynamic_model_run.py` — 运行时封装，QThread 异步推理
     - `mpc_setup_scipy.py` — MPC 优化器（do_mpc + COBYLA），10 步预测地平线，控制通道 [freq, amp] × left/right
     - `checkpoints/` — 预训练权重

4. **RoBOC — 机器人控制模块**（`src/robot/`）
   - `robot.py` — 虚拟差速驱动机器人（运动学、碰撞检测）
   - `real_robot.py` — 真实机器人 TCP 通信接口
   - `task.py` — 任务枚举（避障 / 跟踪 / 抓取）+ 地图枚举 + 设备枚举

5. **闭环协调中枢**（`src/robot/communication.py`）
   - `Communication` 类：100ms 周期，串联 Recording → SpikeDetection → Decode → Robot → Encode → Stimulation
   - 支持传统编码路径和代理模型增强路径的动态切换

6. **GUI 层**（PyQt5）
   - `main.py` → `MainWindowClass` 入口
   - `src/ImageWidget.py` — OpenGL 3D 可视化
   - 多个 Dialog 类（`StimulateDialog`, `DynamicDialog`, `ChannelSignalDialog` 等）

### 数据流

```
BioBOC 记录 → spike 检测 → InterBOC 解码 → RoBOC 执行
    ↑                                          ↓
StimBOC 刺激 ← InterBOC 编码（或 MPC 优化） ← 环境反馈
```

### 关键设计模式

- Qt signals/slots 事件驱动
- 线程化异步 I/O（SocketThread, 文件监控）
- 枚举管理任务/模式切换
- Communication 类抽象硬件接口，实现设备无关的闭环流程

## Testing

测试脚本在 `scripts/` 目录下，为独立运行的 ad-hoc 脚本（无 pytest/unittest 框架）：
```bash
python scripts/mtest_spikes.py      # spike 检测测试
python scripts/mtest_datamanager_h5.py  # HDF5 数据管理测试
```

## Data Directories

- `out/` — MEA2100 和 INTAN 记录输出
- `out_grasping/`, `out_tracking/`, `out_spikes/` — 实验输出
- `encode_decode/` — 编解码数据文件
- `channel_map.npz` / `channel_map_intan.npz` — 通道映射数据

## Code Conventions

- 中英文混合注释
- UI 文件使用 Qt Designer 编辑 `.ui` → pyuic5 生成 `Ui_*.py`（不要手动编辑生成文件）
