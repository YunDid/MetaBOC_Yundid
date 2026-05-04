import sys
import time, copy
import PyQt5
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *

import numpy as np
# from src.recording import Recording
# from src.stimulation import Stimulation
# from src.spike_detection import SpikeDetection
# from src.dynamic_model.DynamicModel import DynamicModel
from src.dynamic_model.Trainer import ModelTrainer
import torch
from src.dynamic_model.mpc_setup_scipy import my_mpc
from src.dynamic_model.optim_setup import optim_setup




class NeuroDynamicModel(QThread):
    def __init__(self, ckpt_json, ckpt_path, recording,stim_electrodes,channels):
        '''

        :param ckpt_json: 模型对应json文件
        :param ckpt_path: 模型参数存储位置
        :param stim_electrodes: 刺激电极组，目前是两组四个电极 eg：['26', '27', '52', '53']
         :param rec_electrodes: 记录电极 eg：['14', '32', '37', '62', '67']
        '''
        super().__init__()
        # super(QThread,self).__init__()
        self.time_step = 250
        self.expected_time_horizon = 250

        self.store_data = np.empty((60, 0))  # 拼接每10ms的数据
        # self.iter_count = 0
        self.data_length = 0

        self.stim_reset = True
        self.random_stim = False

        self.stim_electrodes =None # None #stim_electrodes#['26', '27', '52', '53']  # [32, 33, 19, 26]
        self.selected_electrodes = None #rec_electrodes#['14', '32', '37', '62', '67']

        # self.input_amplitudes=np.
        self.dynamic_model = None
        self.init_dynamic_model(ckpt_json, ckpt_path,stim_ele_set=stim_electrodes,selected_channels=channels)
        print('channel:',channels)
        self.electrodes = channels# self.selected_electrodes#[i for i in range(5)]  # 设置感兴趣的记录电极
        self.sample_data_ = np.zeros((len(self.electrodes), 150))  # 每一个时间窗的数据做完spike detection，把FR存进来
        self.sample_data_smooth = np.empty((len(self.electrodes), 0))

        self.sample_data_ori = np.empty((len(self.electrodes), 0))
        # network_params=None
        # network_params = {'Wrec': self.dynamic_model.model.rnn_cell.get_w_rec().detach().numpy(),
        #                   # Network Connectivity
        #                   'Wrec_bias': self.dynamic_model.model.rnn_cell.bias_rec.detach().numpy(),
        #                   # State Transition Bias
        #                   'B': self.dynamic_model.model.rnn_cell.B.detach().numpy(),  # 输入矩阵的Weight
        #                   'B_bias': self.dynamic_model.model.rnn_cell.B_bias.detach().numpy(),  # 输入矩阵的bias
        #                   'C': self.dynamic_model.model.rnn_cell.C.detach().numpy(),  # 输出矩阵的Weight
        #                   'C_bias': self.dynamic_model.model.rnn_cell.C_bias.detach().numpy(),  # 输出矩阵的bias
        #                   }
        network_params={'Wrec':self.dynamic_model.model.rnn_cell.get_w_rec().detach().numpy(),# Network Connectivity
                            'Wrec_bias':self.dynamic_model.model.rnn_cell.bias_rec.detach().numpy(), # State Transition Bias
                            'B':self.dynamic_model.model.rnn_cell.B.detach().numpy(),# 输入矩阵的Weight
                            'B_bias':self.dynamic_model.model.rnn_cell.B_bias.detach().numpy(),# 输入矩阵的bias
                            'C':torch.relu(self.dynamic_model.model.rnn_cell.C).detach().numpy(),# 输出矩阵的Weight
                            'C_bias':torch.relu(self.dynamic_model.model.rnn_cell.C_bias).detach().numpy(),# 输出矩阵的bias
                        }
        
        self.data_preprocess_scale = 1
        self.ref=np.array([6,4,4,6,5,5]) #2 3 2 3 # 6 5 5 6 # 6 4 3 5
        self.predict_length=10
        self.mpc_ = my_mpc(network_params,self.predict_length,control_length=self.predict_length,ref=self.ref,control_strategy='Tracking',control_channel=None,control_type='closed_loop')# 需要把网络的参数传进去
        self.mpc_.emit_optimal_input_.connect(self.set_optimal_result_from_mpc)
        
        #self.mpc_ = my_mpc(network_params)  # 需要把网络的参数传进去
        # self.mpc_.set_reference(np.array([10,10,10,10,10]).T)
        # self.mpc_._signal.connect(self.set_optimal_result_from_mpc)
        self.mpc_.start()
        self.return_an_optimal_stim_signal = False

        # self.optim_ = optim_setup(network_params)
        # self.optim_.emit_optimal_input_.connect(self.set_optimal_result_from_optimization_framework)
        # self.optim_.start()
        self.mpc_.init_stim(num_stim_channels=None,num_stim=1)
        u, self.input_amplitudes, self.input_durations, self.amplitudes, self.frequencys,loss = self.mpc_.get_optimal_control()
        
        self.testRecording = 0
        self.dynamic_model_input_dimention = self.dynamic_model.model.ext_input_dim
        self.external_input_signal = np.zeros((2, self.dynamic_model.model.data_len))
        self.optimal_input_signal = np.empty((self.dynamic_model_input_dimention, 0))

        self.control_strategy='Tracking'
        # 不需要读取数据
        # self.recording = Recording(my_timer=self.time_step)
        # self.stimulation = Stimulation(select_electrodes=self.stim_electrodes)
        # self.stimulation.set_recording(self.recording)
        self.recording = recording

        # self.stimulation2=Stimulation()
        # self.stimulation2.set_recording(self.recording)

        # self.spike_detection = SpikeDetection()
        # self.spike_detection.start()

        self.input_data = copy.deepcopy(self.recording.channel_map)
        for i in self.input_data:
            self.input_data[i] = []

        # start qtimer
        self.timer15ms = QTimer()
        # self.timer15ms.timeout.connect(self.record_and_stim)
        # self.timer15ms.start(10) # 10ms 定时器
        # end all initialization


    # def init_dynamic_model(self, ckpt_json, ckpt_path,stim_ele_set = [[[84]],[[41]]],selected_channels=None):
    def init_dynamic_model(self, ckpt_json, ckpt_path,stim_ele_set,selected_channels=None):
        
        if self.dynamic_model is None:
            # self.dynamic_model = DynamicModel(optimize=True)
            self.dynamic_model = ModelTrainer(simple_params=ckpt_json, json_file_type=True)
            self.dynamic_model.load_model(checkpoint_path=ckpt_path)
            # 加载默认模型
            # self.dynamic_model.set_init_model()
            print("Load dynamic model successed!...")

            # self.dynamic_model.preprocessing.electrode_set
            self.stim_electrodes = stim_ele_set
            #selected_channels = self.dynamic_model.preprocessing.selected_channels
            # if self.selected_electrodes is None:
            #     self.selected_electrodes = []
            #     for i in range(len(selected_channels)):
            #         if selected_channels[i] > 0:
            #             if i < 6:
            #                 channel = str(1) + str(i+2)
            #             elif i > 53:
            #                 channel = str(8) + str(i - 52)
            #             else:
            #                 channel = str((i+3)//8) + str((i+3)%8)
            #             self.selected_electrodes.append(channel)

    #设置调控reference（array）
    def set_reference(self, ref):
        self.mpc_.set_reference(np.ones(6).T*ref)

    def set_stimulation_type(self, stim_types):
        if stim_types == 'random':
            self.timer15ms.stop()
            self.random_stim = True
            self.recording.clear_buffer_before_starting_timer()
            self.timer15ms.start(self.time_step)
        else:
            self.random_stim = False
            self.timer15ms.stop()
            self.recording.clear_buffer_before_starting_timer()
            self.timer15ms.start(self.time_step)  # 10ms 定时器
        self.cur_time = time.time()
        self.recording.set_spike_save_thread(1000 // self.time_step, 60)
        print(self.random_stim)

    def spike_detection_in_interest_electrodes(self, data, electrodes):
        spikes = np.zeros((len(electrodes), 1))
        for i in range(len(electrodes)):
            self.spike_detection.set_data(data[electrodes[i], :], self.recording.constant)
            self.spike_detection.detection()
            spikes[i] = self.spike_detection.get_spike_num()
        return spikes

    def select_data_from_electrodes(self):

        return self.sample_data_

    def set_stimulation(self):
        if self.random_stim:  # randomly initialize the stimulate parameter
            stimulate_parames = np.random.randn(self.dynamic_model_input_dimention, 1)

            # self.stimulation.set_sti_signal(stimulate_parames)
        else:  # set optimal parameter
            stimulate_parames = np.zeros((self.dynamic_model_input_dimention, 1))
            # self.stimulation.set_sti_signal(stimulate_parames)
        return stimulate_parames

    def read_data_from_device(self):
        # print('npdata shape: ', self.recording.read_numpy_data.shape)
        # if self.testRecording<20:
        #     ori_data=np.reshape(self.recording.read_numpy_data,
        #                     (int(self.recording.read_numpy_data.shape[0]/self.recording.frame_ret),self.recording.frame_ret), "F")[:60,:]
        #     self.testRecording +=1
        #     return None

        ori_data = self.recording.read_numpy_data.copy()
        ori_data = np.reshape(ori_data,
                                (77, ori_data.shape[0] // 77), "F")[:60, :]

        self.store_data = np.append(self.store_data, ori_data, axis=1)
        # self.iter_count+=1

    def generate_spikes(self):
        electrode_key = self.recording.channel_map
        return None

    def reset_sample_data(self):
        del self.sample_data_
        self.sample_data_ = np.zeros((len(self.electrodes), 150))

    def reset_input_data(self):
        for i in self.input_data:
            self.input_data[i].clear()
            self.input_data[i] = []

    def set_optimal_result_from_optimization_framework(self, generate_optimal_signal):
        # self.optimal_stim_signal=np.array(optimal_signal)
        self.return_an_optimal_stim_signal = True

        u, self.input_amplitudes, self.input_durations, self.amplitudes, self.frequencys = self.optim_.get_optimal_control()
        
        print('frequency:', self.frequencys)
        print('amplitude:', self.amplitudes)
        print('input_amplitudes:', self.input_amplitudes)
        print('input_duration:', self.input_durations)
        print(np.array([self.frequencys[0][0], self.amplitudes[0][0]]).shape)

        # self.stimulation.set_and_send_optim_stimulation(input_amplitudes, input_durations)
        self.external_input_signal = np.append(self.external_input_signal, np.array(
            [self.frequencys[0][0], self.amplitudes[0][0]]), axis=1)

        self.return_an_optimal_stim_signal = False
        self.optimal_input_signal = np.append(self.optimal_input_signal, u, axis=1)

    def set_optimal_result_from_mpc(self, generate_optimal_signal):
        # self.optimal_stim_signal=np.array(optimal_signal)
        self.return_an_optimal_stim_signal = True

        u, self.input_amplitudes, self.input_durations, self.amplitudes, self.frequencys,loss = self.mpc_.get_optimal_control()
        # print('frequency:', self.frequencys)
        # print('amplitude:', self.amplitudes)
        # print('input_amplitudes:', self.input_amplitudes)
        # print('input_duration:', self.input_durations)

        # self.stimulation.set_and_send_optim_stimulation(input_amplitudes, input_durations)
        self.external_input_signal = np.append(self.external_input_signal, np.array(
            [self.frequencys[0][0], self.amplitudes[0][0]]), axis=1)

        self.return_an_optimal_stim_signal = False
        self.optimal_input_signal = np.append(self.optimal_input_signal, u, axis=1)

    def random_stimulate(self):
        input_amplitudes = None
        input_durations = None
        amplitudes = None
        frequencys = None
        self.mpc_.obtain_random_stim(self.dynamic_model_input_dimention, 1)
        u, self.input_amplitudes, self.input_durations, self.amplitudes, self.frequencys,loss = self.mpc_.get_optimal_control()
        # if len(self.sample_data_[0])<200:
        # self.stimulation.set_and_send_optim_stimulation(input_amplitudes, input_durations)
        print('frequency:', self.frequencys)
        print('amplitude:', self.amplitudes)
        print('input_duration:', self.input_durations)
        # print(np.array([frequencys[0][0],amplitudes[0][0],frequencys[1][0],amplitudes[1][0]]).shape)
        self.external_input_signal = np.append(self.external_input_signal, np.array([self.frequencys[0][0],
                                                                                    self.amplitudes[0][0]]), axis=1)
        # return self.input_amplitudes, self.input_durations, self.amplitudes, self.frequencys

    def optimal_stimulate(self):
        input_amplitudes = None
        input_durations = None
        amplitudes = None
        frequencys = None
        # 这里只负责设置初值。
        ##################################
        # optimal control
        ##################################
        print(self.random_stim,len(self.sample_data_[0]),self.dynamic_model.model.data_len + 2)
        if (len(self.sample_data_[0]) >  2) and self.random_stim == False:
            # start=time.time()
            # 得到hidden_init的代码再调整一下，现在写得有点绕
            selected_sample_data = self.select_data_from_electrodes()
            # selected_sample_data
            # # data_ = np.reshape(np.random.uniform(-1,1,(self.dynamic_model.model.data_len,self.dynamic_model.model.input_dim)),#selected_sample_data[:,-self.dynamic_model.model.data_len:].T,
            # #                     (1,self.dynamic_model.model.data_len,self.dynamic_model.model.input_dim)).astype('float32') # 最后几步用于初始化hidden state
            # data_ = np.reshape(selected_sample_data[:,-1:].T,
            #                     (1,1,self.dynamic_model.model.input_dim)).astype('float32') # 最后几步用于初始化hidden state)
            # input_ = self.external_input_signal[:,-(self.dynamic_model.model.data_len):-1]
            
            previous_state = torch.from_numpy(np.reshape(selected_sample_data[:,-1:].T, # time * channel
                                (1,1,self.dynamic_model.model.input_dim))) # 最后几步用于初始化hidden state
            # initial hidden state
            hidden_init = torch.permute(self.dynamic_model.model.encode(previous_state.type(torch.float32)), (1, 0, 2))

            self.mpc_.set_initial_state(np.reshape(hidden_init.detach().numpy(),(self.dynamic_model.model.hidden_dim,1)))
            # self.mpc_.set_initial_state(np.reshape(hidden_init.detach().numpy(),
            #                                        (self.dynamic_model.model.hidden_dim,1),),
            #                             control_strategy=self.control_strategy)
        else:
            self.mpc_.init_stim(num_stim_channels=None,num_stim=1)
            u, self.input_amplitudes, self.input_durations, self.amplitudes, self.frequencys,loss = self.mpc_.get_optimal_control()
            self.external_input_signal=np.append(self.external_input_signal,np.array([frequencys[0][0],
                                                                                    amplitudes[0][0],
                                                                                    frequencys[1][0],
                                                                                    amplitudes[1][0]]),axis=1)
        return self.input_amplitudes,self.input_durations, self.amplitudes, self.frequencys



    def run(self,control_strategy='Tracking'):
        '''
        :param self:
        :param control_strategy:控制策略 'Tracking', 'Activating', 'Inhibiting'
        :param ori_data: 获取的数据60记录通道原始数据
        :return: 刺激参数
        '''
        if self.random_stim:
            # random stim
            self.random_stimulate()
        elif self.random_stim == False:
            # optimal stim
            self.control_strategy = control_strategy
            self.optimal_stimulate()
            # a=0
        print('stimulation_parameters:')
        print(self.amplitudes)
        print(self.frequencys)
        # store each time step spike data into the self-defined buffer (sample_data)
        # self.read_data_from_device()
        # self.store_data = np.append(self.store_data, ori_data, axis=1)
        # detect spikes from original data
        # self.sample_data_ = np.append(self.sample_data_,
        #                                 self.spike_detection_in_interest_electrodes(self.store_data, self.electrodes),
        #                                 axis=1)

        # self.data_length += 1
        # self.sample_data_ori = np.append(self.sample_data_ori, self.store_data, axis=1)
        # # empty storage
        # self.store_data = np.empty((60, 0))
        return self.input_amplitudes,self.input_durations, self.amplitudes, self.frequencys


    def set_input_spike_data(self, spikes):
        print('data:',spikes.shape,self.sample_data_.shape)
        if self.sample_data_.shape[1]>0:
            self.sample_data_ = np.append(self.sample_data_, 0.8*self.sample_data_[:,-1:]+0.2*spikes, axis=1)
        else:
            self.sample_data_ = np.append(self.sample_data_, spikes, axis=1)
        # if len(spikes) == 60:
        #     self.sample_data_ = np.append(self.sample_data_, 0.2*self.sample_data_[:,-1]+0.8*array_spk.T, axis=1)
        #     pass   # TODO 根据电极筛选数据
        # else:
        #     array_spk = np.array(spikes)
        #     self.sample_data_ = array_spk.T
        

# import sys
# import time, copy
# import PyQt5
# from PyQt5.QtWidgets import QApplication, QWidget
# from PyQt5.QtGui import QIcon
# from PyQt5.QtGui import *
# from PyQt5.QtCore import *
# from PyQt5.QtWidgets import *
# from PyQt5.QtChart import *

# import numpy as np
# # from src.recording import Recording
# # from src.stimulation import Stimulation
# # from src.spike_detection import SpikeDetection
# # from src.dynamic_model.DynamicModel import DynamicModel
# from src.dynamic_model.Trainer import ModelTrainer
# import torch
# from src.dynamic_model.mpc_setup import my_mpc
# from src.dynamic_model.optim_setup import optim_setup




# class NeuroDynamicModel(QThread):
#     def __init__(self, ckpt_json, ckpt_path, recording):
#         '''

#         :param ckpt_json: 模型对应json文件
#         :param ckpt_path: 模型参数存储位置
#         :param stim_electrodes: 刺激电极组，目前是两组四个电极 eg：['26', '27', '52', '53']
#          :param rec_electrodes: 记录电极 eg：['14', '32', '37', '62', '67']
#         '''
#         super().__init__()
#         # super(QThread,self).__init__()
#         self.time_step = 250
#         self.expected_time_horizon = 250

#         self.store_data = np.empty((60, 0))  # 拼接每10ms的数据
#         # self.iter_count = 0
#         self.data_length = 0

#         self.stim_reset = True
#         self.random_stim = False

#         self.stim_electrodes = None #stim_electrodes#['26', '27', '52', '53']  # [32, 33, 19, 26]
#         self.selected_electrodes = None #rec_electrodes#['14', '32', '37', '62', '67']

        
#         self.dynamic_model = None
#         self.init_dynamic_model(ckpt_json, ckpt_path)

#         self.electrodes = self.selected_electrodes#[i for i in range(5)]  # 设置感兴趣的记录电极
#         self.sample_data_ = np.empty((len(self.electrodes), 0))  # 每一个时间窗的数据做完spike detection，把FR存进来
#         self.sample_data_smooth = np.empty((len(self.electrodes), 0))

#         self.sample_data_ori = np.empty((len(self.electrodes), 0))
#         # network_params=None
#         network_params = {'Wrec': self.dynamic_model.model.rnn_cell.get_w_rec().detach().numpy(),
#                           # Network Connectivity
#                           'Wrec_bias': self.dynamic_model.model.rnn_cell.bias_rec.detach().numpy(),
#                           # State Transition Bias
#                           'B': self.dynamic_model.model.rnn_cell.B.detach().numpy(),  # 输入矩阵的Weight
#                           'B_bias': self.dynamic_model.model.rnn_cell.B_bias.detach().numpy(),  # 输入矩阵的bias
#                           'C': self.dynamic_model.model.rnn_cell.C.detach().numpy(),  # 输出矩阵的Weight
#                           'C_bias': self.dynamic_model.model.rnn_cell.C_bias.detach().numpy(),  # 输出矩阵的bias
#                           }
#         # self.mpc_ = my_mpc(network_params)  # 需要把网络的参数传进去
#         # # self.mpc_.set_reference(np.array([10,10,10,10,10]).T)
#         # self.mpc_._signal.connect(self.set_optimal_result_from_mpc)
#         # self.mpc_.start()
#         # self.return_an_optimal_stim_signal = False

#         self.optim_ = optim_setup(network_params)
#         self.optim_.emit_optimal_input_.connect(self.set_optimal_result_from_optimization_framework)
#         self.optim_.start()

#         self.testRecording = 0
#         self.dynamic_model_input_dimention = self.dynamic_model.model.ext_input_dim
#         self.external_input_signal = np.zeros((self.dynamic_model_input_dimention, self.dynamic_model.model.data_len))
#         self.optimal_input_signal = np.empty((self.dynamic_model_input_dimention, 0))


#         # 不需要读取数据
#         # self.recording = Recording(my_timer=self.time_step)
#         # self.stimulation = Stimulation(select_electrodes=self.stim_electrodes)
#         # self.stimulation.set_recording(self.recording)
#         self.recording = recording

#         # self.stimulation2=Stimulation()
#         # self.stimulation2.set_recording(self.recording)

#         # self.spike_detection = SpikeDetection()
#         # self.spike_detection.start()

#         self.input_data = copy.deepcopy(self.recording.channel_map)
#         for i in self.input_data:
#             self.input_data[i] = []

#         # start qtimer
#         self.timer15ms = QTimer()
#         # self.timer15ms.timeout.connect(self.record_and_stim)
#         # self.timer15ms.start(10) # 10ms 定时器
#         # end all initialization


#     def init_dynamic_model(self, ckpt_json, ckpt_path):
#         if self.dynamic_model is None:
#             # self.dynamic_model = DynamicModel(optimize=True)
#             self.dynamic_model = ModelTrainer(simple_params=ckpt_json, json_file_type=True)
#             self.dynamic_model.load_model(checkpoint_path=ckpt_path)
#             # 加载默认模型
#             # self.dynamic_model.set_init_model()
#             print("Load dynamic model successed!...")

#             stim_ele_set = self.dynamic_model.preprocessing.electrode_set
#             self.stim_electrodes = stim_ele_set
#             selected_channels = self.dynamic_model.preprocessing.selected_channels
#             if self.selected_electrodes is None:
#                 self.selected_electrodes = []
#                 for i in range(len(selected_channels)):
#                     if selected_channels[i] > 0:
#                         if i < 6:
#                             channel = str(1) + str(i+2)
#                         elif i > 53:
#                             channel = str(8) + str(i - 52)
#                         else:
#                             channel = str((i+3)//8) + str((i+3)%8)
#                         self.selected_electrodes.append(channel)

#     #设置调控reference（array）
#     def set_reference(self, ref):
#         self.mpc_.set_reference(np.ones(len(self.selected_electrodes)).T*ref)

#     def set_stimulation_type(self, stim_types):
#         if stim_types == 'random':
#             self.timer15ms.stop()
#             self.random_stim = True
#             self.recording.clear_buffer_before_starting_timer()
#             self.timer15ms.start(self.time_step)
#         else:
#             self.random_stim = False
#             self.timer15ms.stop()
#             self.recording.clear_buffer_before_starting_timer()
#             self.timer15ms.start(self.time_step)  # 10ms 定时器
#         self.cur_time = time.time()
#         self.recording.set_spike_save_thread(1000 // self.time_step, 60)
#         print(self.random_stim)

#     def spike_detection_in_interest_electrodes(self, data, electrodes):
#         spikes = np.zeros((len(electrodes), 1))
#         for i in range(len(electrodes)):
#             self.spike_detection.set_data(data[electrodes[i], :], self.recording.constant)
#             self.spike_detection.detection()
#             spikes[i] = self.spike_detection.get_spike_num()
#         return spikes

#     def select_data_from_electrodes(self):
#         # selected_data = np.zeros((len(self.selected_electrodes), self.sample_data_.shape[1]))
#         # for i in range(len(self.selected_electrodes)):
#         #     selected_data[i] = self.sample_data_[self.stimulation.channel_map[self.selected_electrodes[i]]]
#         # return selected_data
#         return self.sample_data_

#     def set_stimulation(self):
#         if self.random_stim:  # randomly initialize the stimulate parameter
#             stimulate_parames = np.random.randn(self.dynamic_model_input_dimention, 1)

#             # self.stimulation.set_sti_signal(stimulate_parames)
#         else:  # set optimal parameter
#             stimulate_parames = np.zeros((self.dynamic_model_input_dimention, 1))
#             # self.stimulation.set_sti_signal(stimulate_parames)
#         return stimulate_parames

#     def read_data_from_device(self):
#         # print('npdata shape: ', self.recording.read_numpy_data.shape)
#         # if self.testRecording<20:
#         #     ori_data=np.reshape(self.recording.read_numpy_data,
#         #                     (int(self.recording.read_numpy_data.shape[0]/self.recording.frame_ret),self.recording.frame_ret), "F")[:60,:]
#         #     self.testRecording +=1
#         #     return None

#         ori_data = self.recording.read_numpy_data.copy()
#         ori_data = np.reshape(ori_data,
#                                 (77, ori_data.shape[0] // 77), "F")[:60, :]

#         self.store_data = np.append(self.store_data, ori_data, axis=1)
#         # self.iter_count+=1

#     def generate_spikes(self):
#         electrode_key = self.recording.channel_map
#         return None

#     def reset_sample_data(self):
#         del self.sample_data_
#         self.sample_data_ = np.empty((len(self.electrodes), 0))

#     def reset_input_data(self):
#         for i in self.input_data:
#             self.input_data[i].clear()
#             self.input_data[i] = []

#     def set_optimal_result_from_optimization_framework(self, generate_optimal_signal):
#         # self.optimal_stim_signal=np.array(optimal_signal)
#         self.return_an_optimal_stim_signal = True

#         u, input_amplitudes, input_durations, amplitudes, frequencys = self.optim_.get_optimal_control()

#         print('frequency:', frequencys)
#         print('amplitude:', amplitudes)
#         print('input_amplitudes:', input_amplitudes)
#         print('input_duration:', input_durations)
#         print(np.array([frequencys[0][0], amplitudes[0][0], frequencys[1][0], amplitudes[1][0]]).shape)

#         self.stimulation.set_and_send_optim_stimulation(input_amplitudes, input_durations)
#         self.external_input_signal = np.append(self.external_input_signal, np.array(
#             [frequencys[0][0], amplitudes[0][0], frequencys[1][0], amplitudes[1][0]]), axis=1)

#         self.return_an_optimal_stim_signal = False
#         self.optimal_input_signal = np.append(self.optimal_input_signal, u, axis=1)

#     def set_optimal_result_from_mpc(self, generate_optimal_signal):
#         # self.optimal_stim_signal=np.array(optimal_signal)
#         self.return_an_optimal_stim_signal = True

#         u, input_amplitudes, input_durations, amplitudes, frequencys = self.mpc_.get_optimal_control()


#         self.stimulation.set_and_send_optim_stimulation(input_amplitudes, input_durations)
#         self.external_input_signal = np.append(self.external_input_signal, np.array(
#             [frequencys[0][0], amplitudes[0][0], frequencys[1][0], amplitudes[1][0]]), axis=1)

#         self.return_an_optimal_stim_signal = False
#         self.optimal_input_signal = np.append(self.optimal_input_signal, u, axis=1)

#     def random_stimulate(self):
#         input_amplitudes = None
#         input_durations = None
#         amplitudes = None
#         frequencys = None
#         self.mpc_.obtain_random_stim(self.dynamic_model_input_dimention, 1)
#         u, input_amplitudes, input_durations, amplitudes, frequencys = self.mpc_.get_optimal_control()
#         # if len(self.sample_data_[0])<200:
#         self.stimulation.set_and_send_optim_stimulation(input_amplitudes, input_durations)
#         print('frequency:', frequencys)
#         print('amplitude:', amplitudes)
#         print('input_duration:', input_durations)
#         # print(np.array([frequencys[0][0],amplitudes[0][0],frequencys[1][0],amplitudes[1][0]]).shape)
#         self.external_input_signal = np.append(self.external_input_signal, np.array([frequencys[0][0],
#                                                                                         amplitudes[0][0],
#                                                                                         frequencys[1][0],
#                                                                                         amplitudes[1][0]]), axis=1)
#         return input_amplitudes, input_durations, amplitudes, frequencys

#     def optimal_stimulate(self):
#         input_amplitudes = None
#         input_durations = None
#         amplitudes = None
#         frequencys = None
#         # 这里只负责设置初值。
#         ##################################
#         # optimal control
#         ##################################
#         if (len(self.sample_data_[0]) > self.dynamic_model.model.data_len + 20) and self.random_stim == False:
#             # start=time.time()
#             # 得到hidden_init的代码再调整一下，现在写得有点绕

#             # 提取需要的电极数据
#             selected_sample_data = self.select_data_from_electrodes()
#             data_ = np.reshape(selected_sample_data[:, -self.dynamic_model.model.data_len:].T,
#                                 (1, self.dynamic_model.model.data_len,
#                                 self.dynamic_model.model.input_dim))  # 最后几步用于初始化hidden state
#             input_ = self.external_input_signal[:, -self.dynamic_model.model.data_len + 1:]
#             pre_X = torch.reshape(torch.from_numpy(np.int64(data_)),
#                                     (1, 1, self.dynamic_model.model.data_len * self.dynamic_model.model.input_dim))
#             Ext_IN = torch.reshape(torch.from_numpy(input_),
#                                     (1, 1, (
#                                                 self.dynamic_model.model.data_len - 1) * self.dynamic_model_input_dimention))  # Previous_Input
#             previous_state = torch.cat((pre_X, Ext_IN), dim=2)
#             # initial hidden state
#             hidden_init = torch.permute(self.dynamic_model.model.encode(previous_state.type(torch.float32)),
#                                         (1, 0, 2))
#             self.mpc_.set_initial_state(np.reshape(hidden_init.detach().numpy(),(self.dynamic_model.model.hidden_dim,1)))

#             # self.optim_.set_type_of_optim(hidden_init, 2, 'maximal')

#             # if self.return_an_optimal_stim_signal:
#             # self.stimulation.set_sti_signal(self.optimal_stim_signal)
#             self.mpc_.obtain_optimal_control()

#             u, input_amplitudes, input_durations, amplitudes, frequencys = self.mpc_.get_optimal_control()

#             # print('frequency:', frequencys)
#             # print('amplitude:', amplitudes)
#             # print('input_amplitudes:', input_amplitudes)
#             # print('input_duration:', input_durations)
#             # print(np.array([frequencys[0][0],amplitudes[0][0],frequencys[1][0],amplitudes[1][0]]).shape)

#             # 刺激参数传入刺激器
#             # self.stimulation.set_and_send_optim_stimulation(input_amplitudes,input_durations)
#             # self.external_input_signal=np.append(self.external_input_signal,np.array([frequencys[0][0],amplitudes[0][0],frequencys[1][0],amplitudes[1][0]]),axis=1)

#         return input_amplitudes,input_durations, amplitudes, frequencys

#     def run(self):
#         '''

#         :param self:
#         :param ori_data: 获取的数据60记录通道原始数据
#         :return: 刺激参数
#         '''


#         if self.random_stim:
#             # random stim
#             input_amplitudes,input_durations,amplitudes,frequencys = self.random_stimulate()
#         elif self.random_stim == False:
#             # optimal stim
#             input_amplitudes,input_durations,amplitudes,frequencys = self.optimal_stimulate()
#             # a=0

#         # store each time step spike data into the self-defined buffer (sample_data)
#         # self.read_data_from_device()
#         # self.store_data = np.append(self.store_data, ori_data, axis=1)
#         # detect spikes from original data
#         # self.sample_data_ = np.append(self.sample_data_,
#         #                                 self.spike_detection_in_interest_electrodes(self.store_data, self.electrodes),
#         #                                 axis=1)

#         # self.data_length += 1
#         # self.sample_data_ori = np.append(self.sample_data_ori, self.store_data, axis=1)
#         # # empty storage
#         # self.store_data = np.empty((60, 0))
#         return input_amplitudes,input_durations, amplitudes, frequencys


#     def set_input_spike_data(self, spikes):
#         if len(spikes) == 60:
#             self.sample_data_ = np.append(self.sample_data_, array_spk.T, axis=1)
#             pass   # TODO 根据电极筛选数据
#         else:
#             array_spk = np.array(spikes)
#             self.sample_data_ = array_spk.T
        