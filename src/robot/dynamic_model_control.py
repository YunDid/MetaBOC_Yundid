# --------------------------------------------------------
# Brain for Controlling with Hybrid Intelligence Platform 
# Copyright (c) 2024
# Written by Guiping Cao
# Time: 2024.01.19
# --------------------------------------------------------

import numpy as np
import os
import h5py
import time
from datetime import datetime

from src.platform_config import MCS_AVAILABLE

if MCS_AVAILABLE:
    from src.infor_com_mea.recording import Recording
    from src.infor_com_mea.stimulation import Stimulation
else:
    Recording = None
    Stimulation = None

from src.robot.encode_decode import EncodingDecoding

from src.robot.task import TASK, MAP

# for dynamic models
# 防御性 import：dynamic_model 链依赖 pytorch_lightning，版本漂移或 API
# 不兼容时不应阻塞 GUI 启动。dynamic_model_used 默认 False，仅当用户显式
# 启用 MPC 时才需要 NeuroDynamicModel；失败时打印警告并将其置为 None，
# 让 DynamicM_Control.initial_model 在使用时给出明确报错。
try:
    from src.dynamic_model.dynamic_model_run import NeuroDynamicModel
    AI_MODEL_AVAILABLE = True
except Exception as _ai_import_exc:  # noqa: BLE001
    print(
        "AI dynamic model unavailable (NeuroDynamicModel import failed): {!r}. "
        "MPC features will be disabled until the AI module is fixed.".format(_ai_import_exc)
    )
    NeuroDynamicModel = None
    AI_MODEL_AVAILABLE = False


class DynamicM_Control(object):

    def __init__(self, recording, stimulation, en_decode):
        super(DynamicM_Control, self).__init__()

        # self.para = para
        # self.left_ele = self.para.recording_list
        # self.right_ele = self.para.stimulating_list

        self.recording = recording
        self.stimulating = stimulation
        self.encoder_decoder = en_decode
        
        self.l_spikes = []     # {“42”: [0, 0, 2, 5, ...]}
        self.r_spikes = []    # {“42”: [0, 0, 2, 5, ...]}
        self.spike_length = 23    # spike长度单位

        self.angle_left = None    # 机器人的距离信息（角度、和距离）
        self.angle_right = None
        self.left_distance = None
        self.right_distance = None

        self.update_times = 0  # 每3*100ms 更新一次模型计算

        self.amp_left = None
        self.dur_left = None   # 生成的刺激信号
        self.fre_left = None
        self.amp_right = None
        self.dur_right = None   
        self.fre_right = None# 生成的刺激信号, 可能需要两个模型分别计算左侧和右侧的刺激生成


        timestamp = time.time()
        dt_object = datetime.fromtimestamp(timestamp)
        formatted_date = dt_object.strftime("%Y-%m-%d %H-%M-%S")
        self.save_spike_ref_path_left = os.path.join("./out_spike_ref", "AI-left-" + formatted_date[:-9] + "-" + formatted_date[-8:] + ".txt")
        self.save_spike_ref_path_right = os.path.join("./out_spike_ref", "AI-right-" + formatted_date[:-9] + "-" + formatted_date[-8:] + ".txt")


        self.initial_model()


    def initial_model(self):

        # 0316-M1
        # l_ckpt_json = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250428/left.json'
        # l_ckpt_path = 'C:\\Users\Administrator\Desktop\nationalBrainCode\src\dynamic_model\checkpoints\250428\left.pth'

        # r_ckpt_json = 'C:\\Users\Administrator\Desktop\nationalBrainCode\src\dynamic_model\checkpoints\250428\right.json'
        # r_ckpt_path = 'C:\\Users\Administrator\Desktop\nationalBrainCode\src\dynamic_model\checkpoints\250428\right.pth'

        # 0424-M1
        # l_ckpt_json = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250514_M1/left.json'
        # l_ckpt_path = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250514_M1/left.pth'

        # r_ckpt_json = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250514_M1/right.json'
        # r_ckpt_path = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250514_M1/right.pth'

        # # 0424-M4
        # l_ckpt_json = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250514_M4/left.json'
        # l_ckpt_path = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250514_M4/left.pth'

        # r_ckpt_json = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250514_M4/right.json'
        # r_ckpt_path = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250514_M4/right.pth'

        # M
        # l_ckpt_json = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250526/m2_left.json'
        # l_ckpt_path = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250526/m2_left.pth'

        # r_ckpt_json = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250526/m2_right.json'
        # r_ckpt_path = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250526/m2_right.pth'
        
        # S
        # l_ckpt_json = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250526/single4_left.json'
        # l_ckpt_path = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250526/single4_left.pth'

        # r_ckpt_json = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250526/single4_right.json'
        # r_ckpt_path = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/250526/single4_right.pth'



        # 0626m1

        l_ckpt_json = os.path.join('src', 'dynamic_model', 'checkpoints', '20250626', 'm3_left.json')
        l_ckpt_path = os.path.join('src', 'dynamic_model', 'checkpoints', '20250626', 'm3_left.pth')

        r_ckpt_json = os.path.join('src', 'dynamic_model', 'checkpoints', '20250626', 'm3_right.json')
        r_ckpt_path = os.path.join('src', 'dynamic_model', 'checkpoints', '20250626', 'm3_right.pth')

        # 0626m2

        # l_ckpt_json = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/20250626/m2_left.json'
        # l_ckpt_path = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/20250626/m2_left.pth'

        # r_ckpt_json = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/20250626/m2_right.json'
        # r_ckpt_path = 'C:/Users/Administrator/Desktop/nationalBrainCode/src/dynamic_model/checkpoints/20250626/m2_right.pth'


        # 0626m3
        # l_ckpt_json = 'H:/MetaBOC/resources/20250626/m3_left.json'
        # l_ckpt_path = 'H:/MetaBOC/resources/20250626/m3_left.pth'

        # r_ckpt_json = 'H:/MetaBOC/resources/20250626/m3_right.json'
        # r_ckpt_path = 'H:/MetaBOC/resources/20250626/m3_right.pth'

        # 平台当前选择的电极
        self.sti_left_ele_ori = self.stimulating.left_electrode_key
        self.sti_right_ele_ori = self.stimulating.right_electrode_key
        # print('stim_left:',int(self.sti_left_ele_ori[0]))
        # 临时测试
        sti_list = ['26', '27', '52', '53']
        recording_list = ['14', '32', '37', '62', '67']

        # left_channel
        self.record_left_ori = self.recording.recording_para.recording_list[0]
        self.record_right_ori = self.recording.recording_para.recording_list[1]

        # left_channel
        self.sti_left_ele = self.stimulating.left_electrode_key
        self.sti_right_ele = self.stimulating.right_electrode_key

        # 以左侧电极测试
        import time 
        start_time = time.time()
        if os.path.exists(l_ckpt_json) and os.path.exists(l_ckpt_path) and os.path.exists(r_ckpt_json) and os.path.exists(r_ckpt_path):
            print('model_file_exists.')
        else:
            print('model_file_not_exists.')
        # self.dynamic_model_run_left = NeuroDynamicModel(l_ckpt_json, l_ckpt_path, recording=self.recording,stim_electrodes=[[[int(self.sti_left_ele_ori[0])]],[[int(self.sti_right_ele_ori[0])]]],channels=self.record_left_ori)
        # self.dynamic_model_run_right = NeuroDynamicModel(r_ckpt_json, r_ckpt_path, recording=self.recording,stim_electrodes=[[[int(self.sti_left_ele_ori[0])]],[[int(self.sti_right_ele_ori[0])]]],channels=self.record_right_ori)
        self.dynamic_model_run_left = NeuroDynamicModel(l_ckpt_json, l_ckpt_path, recording=self.recording,stim_electrodes=self.sti_left_ele,channels=self.record_left_ori)
        self.dynamic_model_run_right = NeuroDynamicModel(r_ckpt_json, r_ckpt_path, recording=self.recording,stim_electrodes=self.sti_right_ele,channels=self.record_right_ori)
        end_time = time.time()

        # # 以左侧电极测试
        # import time 
        # start_time = time.time()
        # if os.path.exists(l_ckpt_json) and os.path.exists(l_ckpt_path) and os.path.exists(r_ckpt_json) and os.path.exists(r_ckpt_path):
        #     print('model_file_exists.')
        # else:
        #     print('model_file_not_exists.')
        # self.dynamic_model_run_left = NeuroDynamicModel(l_ckpt_json, l_ckpt_path, recording=self.recording,stim_electrodes=[[[int(self.sti_left_ele_ori[0])]],[[int(self.sti_right_ele_ori[0])]]],channels=self.record_left_ori)
        # self.dynamic_model_run_right = NeuroDynamicModel(r_ckpt_json, r_ckpt_path, recording=self.recording,stim_electrodes=[[[int(self.sti_left_ele_ori[0])]],[[int(self.sti_right_ele_ori[0])]]],channels=self.record_right_ori)
        # end_time = time.time()

        print('time of loading dynamical model:',(end_time-start_time)*1000,'ms')
        print('load model from %s',l_ckpt_path)
        # self.sti_left_ele = self.read_stim_ele(self.dynamic_model_run_left.stim_electrodes[0])
        # self.sti_right_ele = self.read_stim_ele(self.dynamic_model_run_right.stim_electrodes[0])
        self.rec_left_ele = self.dynamic_model_run_left.selected_electrodes
        self.rec_right_ele = self.dynamic_model_run_right.selected_electrodes
        # print('left ele:', sti_left_ele)
        # print('right ele:', sti_left_ele)
        
        # 判断系统中选定的电极与模型加载的电极是否一致
        # sti_left_state = self.check_ele_loaded_from_model(self.sti_left_ele_ori, self.sti_left_ele)
        # sti_right_state = self.check_ele_loaded_from_model(self.sti_right_ele_ori, self.sti_right_ele)
        # rec_left_state = self.check_ele_loaded_from_model(self.record_left_ori, self.rec_left_ele)
        # rec_right_state = self.check_ele_loaded_from_model(self.record_right_ori, self.rec_right_ele)

        # if sti_left_state and sti_right_state and rec_left_state and rec_right_state:
        #     print("All electrode loaded sucessfully!!!")
        # else:
        #     assert sti_left_state and sti_right_state and rec_left_state and rec_right_state, "Wrong with electrode loaded!..."


        print("Initial dynamic model succeed!...")

    def check_ele_loaded_from_model(self, ori_ele, model_ele):
        for i in range(len(ori_ele)):
            if ori_ele[i] != model_ele[i]:
                return False

        return True
                

    def read_stim_ele(self, stim_electrodes):
        stim_ele = []
        for i in range(len(stim_electrodes)):
            stim_ = ''
            for j in range(len(stim_electrodes[i])):
                stim_ += str(stim_electrodes[i][j])
            stim_ele.append(stim_)
        return stim_ele


    def update_ele(self, sti_left, sti_right, rec_left, rec_right):
        """
        eg: ['14', '33', '34', '42', '51', '52']  个数不限，取决于选定的电极数量
        """
        self.sti_left_ele = sti_left
        self.sti_right_ele = sti_right

        self.record_left = rec_left
        self.record_right = rec_left

    def update_spike_data(self, left, right):
        """
        每100ms更新一次，输入为list
        """
        if len(self.l_spikes) > self.spike_length:
            self.l_spikes.pop(0)
            self.r_spikes.pop(0)

            self.l_spikes.append(left)
            self.r_spikes.append(right)
        else:
            self.l_spikes.append(left)
            self.r_spikes.append(right) 

        # 每三次更新一次
        if self.update_times >= 3:
            self.update_times = 0
            self.update_sti()
        else:
            self.update_times += 1

    def update_distance(self, min_dis_left, min_dis_right, ang_dis_left, ang_dis_right):
        self.left_distance = min_dis_left
        self.right_distance = min_dis_right
        self.angle_left = ang_dis_left
        self.angle_right = ang_dis_right


    def save_spike_reference(self, ele, spike, reference):
        
         # 获取当前时间戳
        current_time = time.time()
        timestamp = datetime.fromtimestamp(current_time).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        out_spike = timestamp + ": "
        for i in range(len(spike)):
            cur = spike[i]
            out_spike = out_spike + str(cur[0]) + ","

        out_spike = out_spike + ":" + str(np.round(reference[0],2)) + "\n"
        if ele == "left":
            with open(self.save_spike_ref_path_left, "a") as f:
                f.write(out_spike)
        elif ele == "right":
            with open(self.save_spike_ref_path_right, "a") as f:
                f.write(out_spike)

    def update_sti(self):
        """
        动力学模型根据输入的spike，产生刺激信号
        """
        if len(self.l_spikes) > self.spike_length:
            # 距离和响应呈正相关关系，l  r  已映射到特定的频率范围，参考编码参数
            l, r = self.encoder_decoder.encode(self.left_distance, self.right_distance, self.angle_left, self.angle_right)
            if l<0 or r <0:
                print(60*"=",l," ",r)
            # print('left: ',self.l_spikes)
            left_history = np.array(self.l_spikes)
            # print('left: ',left_history)
            # print('left: ',left_history.shape)
            # print('left: ',np.reshape(self.l_spikes,(4,6))[-1:,:].T)
            right_history = np.array(self.r_spikes)
            mean_l = left_history.mean(axis=0) # 间隔100ms记录1s滑窗的spike数据（共20个）
            mean_r = right_history.mean(axis=0)
            scale_l = l / (mean_l.sum() + 0.0001)
            scale_r = r / (mean_r.sum() + 0.0001)
            ref_l = 5*np.maximum(scale_l * mean_l,0)
            ref_r = 5*np.maximum(scale_r * mean_r,0)
            print('reference left: %.2f, right: %.2f ',ref_l,ref_r)
            # self.dynamic_model_run_left.set_reference(l / 6.0)
            self.dynamic_model_run_left.set_reference(ref_l)
            # self.dynamic_model_run_left.set_input_spike_data(np.reshape(self.l_spikes,(4,6))[-1:,:].T)
            self.dynamic_model_run_left.set_input_spike_data(np.reshape(self.l_spikes, (-1, 6))[-4:, :].T)
            
            amp_left, dur_left, _, fre_left = self.dynamic_model_run_left.run(control_strategy='Tracking')    # 单位： μV   μs

            print(30*"==", self.l_spikes, ref_l)
            self.save_spike_reference("left", self.l_spikes, ref_l)

            self.amp_left = [amp_left[0]]
            self.dur_left = [dur_left[0]]
            self.fre_left=fre_left
            # print('amp:',amp_left)
            # print('dur_left:',dur_left)
            # self.dynamic_model_run_right.set_reference(r / 6.0)
            self.dynamic_model_run_right.set_reference(ref_r)
            # self.dynamic_model_run_right.set_input_spike_data(np.reshape(self.r_spikes,(4,6))[-1:,:].T)
            self.dynamic_model_run_right.set_input_spike_data(np.reshape(self.r_spikes, (-1, 6))[-4:, :].T)
            amp_right, dur_right, _, fre_right = self.dynamic_model_run_right.run(control_strategy='Tracking')    # 单位： μV   μs
            self.amp_right = [amp_right[1]]
            self.dur_right = [dur_right[1]]
            self.fre_right=fre_right
            self.save_spike_reference("right", self.r_spikes, ref_r)

    def get_sti_signal_left(self):
        return self.amp_left, self.dur_left, self.fre_left

    def get_sti_signal_right(self):
        return self.amp_right, self.dur_right, self.fre_right

# 待办：
# 1. 模型的使用, 两侧；  (rp)
# 2. 模型对应电极的输入，不同数量； 模型自动加载通道；（rp）  
# 3. 输入信号 spike：1s滑窗数据，需更新； (?)
# 4. 刺激信号生成时，如何考虑距离信息，而不是单纯的spike信息； (rp)
# 5. 当刺激频率和幅值都变化时，需在stimulation中设定该信号；  (cgp)
# 6. 原来通过生成刺激频率，自动生成刺激信号；现在直接生成刺激信号； （cgp）



# # --------------------------------------------------------
# # Brain for Controlling with Hybrid Intelligence Platform 
# # Copyright (c) 2024
# # Written by Guiping Cao
# # Time: 2024.01.19
# # --------------------------------------------------------

# import numpy as np
# import os
# import h5py
# import time

# from src.infor_com_mea.recording import Recording
# from src.infor_com_mea.stimulation import Stimulation
# from src.robot.encode_decode import EncodingDecoding

# from src.robot.task import TASK, MAP

# # for dynamic models
# from src.dynamic_model.dynamic_model_run import NeuroDynamicModel


# class DynamicM_Control(object):

#     def __init__(self, recording, stimulation, en_decode):
#         super(DynamicM_Control, self).__init__()

#         # self.para = para
#         # self.left_ele = self.para.recording_list
#         # self.right_ele = self.para.stimulating_list

#         self.recording = recording
#         self.stimulating = stimulation
#         self.encoder_decoder = en_decode
        
#         self.l_spikes = []     # {“42”: [0, 0, 2, 5, ...]}
#         self.r_spikes = []    # {“42”: [0, 0, 2, 5, ...]}
#         self.spike_length = 23    # spike长度单位

#         self.angle_left = None    # 机器人的距离信息（角度、和距离）
#         self.angle_right = None
#         self.left_distance = None
#         self.right_distance = None

#         self.update_times = 0  # 每3*100ms 更新一次模型计算

#         self.amp_left = None
#         self.dur_left = None   # 生成的刺激信号
#         self.fre_left = None
#         self.amp_right = None
#         self.dur_right = None   
#         self.fre_right = None# 生成的刺激信号, 可能需要两个模型分别计算左侧和右侧的刺激生成

#         self.initial_model()


#     def initial_model(self):
#         l_ckpt_json = 'src/dynamic_model/checkpoints/test-0123.json'
#         l_ckpt_path = 'src\dynamic_model\checkpoints\0123-data.json'

#         r_ckpt_json = 'src/dynamic_model/checkpoints/test-0123.json'
#         r_ckpt_path = 'src\dynamic_model\checkpoints\0123-data.json'

#         # 平台当前选择的电极
#         self.sti_left_ele_ori = self.stimulating.left_electrode_key
#         self.sti_right_ele_ori = self.stimulating.right_electrode_key

#         self.record_left_ori = self.recording.recording_para.recording_list[0]
#         self.record_right_ori = self.recording.recording_para.recording_list[1]

#         # 临时测试
#         sti_list = ['26', '27', '52', '53']
#         recording_list = ['14', '32', '37', '62', '67']

#         # 以左侧电极测试
#         import time
#         start_time = time.time()
#         self.dynamic_model_run_left = NeuroDynamicModel(l_ckpt_json, l_ckpt_path, recording=self.recording)
#         self.dynamic_model_run_right = NeuroDynamicModel(r_ckpt_json, r_ckpt_path, recording=self.recording)
#         end_time = time.time()
#         print('time of loading dynamical model:',end_time-start_time)
#         self.sti_left_ele = self.read_stim_ele(self.dynamic_model_run_left.stim_electrodes[0])
#         self.sti_right_ele = self.read_stim_ele(self.dynamic_model_run_right.stim_electrodes[0])
#         self.rec_left_ele = self.dynamic_model_run_left.selected_electrodes
#         self.rec_right_ele = self.dynamic_model_run_right.selected_electrodes
#         # print('left ele:', sti_left_ele)
#         # print('right ele:', sti_left_ele)
        
#         # 判断系统中选定的电极与模型加载的电极是否一致
#         sti_left_state = self.check_ele_loaded_from_model(self.sti_left_ele_ori, self.sti_left_ele)
#         sti_right_state = self.check_ele_loaded_from_model(self.sti_right_ele_ori, self.sti_right_ele)
#         rec_left_state = self.check_ele_loaded_from_model(self.record_left_ori, self.rec_left_ele)
#         rec_right_state = self.check_ele_loaded_from_model(self.record_right_ori, self.rec_right_ele)

#         # if sti_left_state and sti_right_state and rec_left_state and rec_right_state:
#         #     print("All electrode loaded sucessfully!!!")
#         # else:
#         #     assert sti_left_state and sti_right_state and rec_left_state and rec_right_state, "Wrong with electrode loaded!..."


#         print("Initial dynamic model succeed!...")

#     def check_ele_loaded_from_model(self, ori_ele, model_ele):
#         for i in range(len(ori_ele)):
#             if ori_ele[i] != model_ele[i]:
#                 return False

#         return True
                

#     def read_stim_ele(self, stim_electrodes):
#         stim_ele = []
#         for i in range(len(stim_electrodes)):
#             stim_ = ''
#             for j in range(len(stim_electrodes[i])):
#                 stim_ += str(stim_electrodes[i][j])
#             stim_ele.append(stim_)
#         return stim_ele


#     def update_ele(self, sti_left, sti_right, rec_left, rec_right):
#         """
#         eg: ['14', '33', '34', '42', '51', '52']  个数不限，取决于选定的电极数量
#         """
#         self.sti_left_ele = sti_left
#         self.sti_right_ele = sti_right

#         self.record_left = rec_left
#         self.record_right = rec_left

#     def update_spike_data(self, left, right):
#         """
#         每100ms更新一次，输入为list
#         """
#         if len(self.l_spikes) > self.spike_length:
#             self.l_spikes.pop(0)
#             self.r_spikes.pop(0)

#             self.l_spikes.append(left)
#             self.r_spikes.append(right)
#         else:
#             self.l_spikes.append(left)
#             self.r_spikes.append(right) 

#         # 每三次更新一次
#         if self.update_times >= 3:
#             self.update_times = 0
#             self.update_sti()
#         else:
#             self.update_times += 1

#     def update_distance(self, min_dis_left, min_dis_right, ang_dis_left, ang_dis_right):
#         self.left_distance = min_dis_left
#         self.right_distance = min_dis_right
#         self.angle_left = ang_dis_left
#         self.angle_right = ang_dis_right


#     def update_sti(self):
#         """
#         动力学模型根据输入的spike，产生刺激信号
#         """
#         if len(self.l_spikes) > self.spike_length:
#             # 距离和响应呈正相关关系，l  r  已映射到特定的频率范围，参考编码参数
#             l, r = self.encoder_decoder.encode(self.left_distance, self.right_distance, self.angle_left, self.angle_right)

#             left_history = np.array(self.l_spikes)
#             right_history = np.array(self.r_spikes)
#             mean_l = left_history.mean(axis=0) # 间隔100ms记录1s滑窗的spike数据（共20个）
#             mean_r = right_history.mean(axis=0)
#             scale_l = l / (mean_l.sum() + 0.0001)
#             scale_r = r / (mean_r.sum() + 0.0001)
#             ref_l = scale_l * mean_l
#             ref_r = scale_r * mean_r

#             # self.dynamic_model_run_left.set_reference(l / 6.0)
#             self.dynamic_model_run_left.set_reference(ref_l)
#             self.dynamic_model_run_left.set_input_spike_data(self.l_spikes)
#             self.amp_left, self.dur_left, _, self.fre_left = self.dynamic_model_run_left.run()    # 单位： μV   μs

#             # self.dynamic_model_run_right.set_reference(r / 6.0)
#             self.dynamic_model_run_right.set_reference(ref_r)
#             self.dynamic_model_run_right.set_input_spike_data(self.r_spikes)
#             self.amp_right, self.dur_right, _, self.fre_right = self.dynamic_model_run_right.run()    # 单位： μV   μs

#     def get_sti_signal_left(self):
#         return self.amp_left, self.dur_left, self.fre_left

#     def get_sti_signal_right(self):
#         return self.amp_right, self.dur_right, self.fre_right

# # 待办：
# # 1. 模型的使用, 两侧；  (rp)
# # 2. 模型对应电极的输入，不同数量； 模型自动加载通道；（rp）  
# # 3. 输入信号 spike：1s滑窗数据，需更新； (?)
# # 4. 刺激信号生成时，如何考虑距离信息，而不是单纯的spike信息； (rp)
# # 5. 当刺激频率和幅值都变化时，需在stimulation中设定该信号；  (cgp)
# # 6. 原来通过生成刺激频率，自动生成刺激信号；现在直接生成刺激信号； （cgp）
