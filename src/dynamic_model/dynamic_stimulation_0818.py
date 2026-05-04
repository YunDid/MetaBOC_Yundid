import sys
import time
import PyQt5
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *

import numpy as np
from src.recording import Recording
from src.stimulation import Stimulation
from src.spike_detection import SpikeDetection 
# from src.dynamic_model.DynamicModel import DynamicModel
from src.dynamic_model.Trainer import ModelTrainer
import torch
from mpc_setup import my_mpc
from optim_setup import optim_setup

class dynamic_widget(QWidget):
    def __init__(self):
        super().__init__()
        # super(QThread,self).__init__()
        self.initUI()
        self.time_step = 250
        self.expected_time_horizon = 250
        self.dynamic_model = None
        self.init_dynamic_model()
        
        self.store_data=np.empty((60,0)) # 拼接每10ms的数据
        self.iter_count=0
        self.data_length=0

        self.stim_reset = True

        self.stim_electrodes = ['26','27','52','53']#[32, 33, 19, 26]
        self.selected_electrodes = ['14','32','37','62','67']
        self.electrodes=[i for i in range(60)]#[5,10,15,20] # 设置感兴趣的记录电极
        self.sample_data_=np.empty((len(self.electrodes),0)) # 每一个时间窗的数据做完spike detection，把FR存进来
        self.sample_data_smooth=np.empty((len(self.electrodes),0))
        
        self.sample_data_ori=np.empty((len(self.electrodes),0))
        # network_params=None
        network_params={'Wrec':self.dynamic_model.model.rnn_cell.get_w_rec().detach().numpy(),# Network Connectivity
                        'Wrec_bias':self.dynamic_model.model.rnn_cell.bias_rec.detach().numpy(), # State Transition Bias
                        'B':self.dynamic_model.model.rnn_cell.B.detach().numpy(),# 输入矩阵的Weight
                        'B_bias':self.dynamic_model.model.rnn_cell.B_bias.detach().numpy(),# 输入矩阵的bias
                        'C':self.dynamic_model.model.rnn_cell.C.detach().numpy(),# 输出矩阵的Weight
                        'C_bias':self.dynamic_model.model.rnn_cell.C_bias.detach().numpy(),# 输出矩阵的bias
                        }
        self.mpc_ = my_mpc(network_params)# 需要把网络的参数传进去
        # self.mpc_.set_reference(np.array([10,10,10,10,10]).T)
        self.mpc_.emit_optimal_input_.connect(self.set_optimal_result_from_mpc)
        self.mpc_.start()
        self.return_an_optimal_stim_signal=False

        self.optim_ = optim_setup(network_params)
        self.optim_.emit_optimal_input_.connect(self.set_optimal_result_from_optimization_framework)
        self.mpc_.start()

        self.testRecording = 0
        self.dynamic_model_input_dimention = self.dynamic_model.model.ext_input_dim
        self.external_input_signal=np.zeros((self.dynamic_model_input_dimention,self.dynamic_model.model.data_len))
        self.optimal_input_signal=np.empty((self.dynamic_model_input_dimention,0))
        
        self.recording=Recording(my_timer=self.time_step)
        self.stimulation=Stimulation(select_electrodes=self.stim_electrodes)
        self.stimulation.set_recording(self.recording)

        # self.stimulation2=Stimulation()
        # self.stimulation2.set_recording(self.recording)

        self.spike_detection=SpikeDetection()
        self.spike_detection.start()
        
        self.input_data = self.recording.channel_map
        for i in self.input_data:
            self.input_data[i] = []

        # start qtimer
        self.timer15ms=QTimer()
        self.timer15ms.timeout.connect(self.record_and_stim)
        # self.timer15ms.start(10) # 10ms 定时器
        # end all initialization

    def initUI(self):
        self.setGeometry(300,300,200,200)
        self.setWindowTitle('Test')
        # add two buttom for (1) random stimulation and (2) optimal stimulation
        self.random_stim_bt = QPushButton('Random Stim.', self)
        self.random_stim_bt.move(10, 60)
        self.random_stim_bt.clicked.connect(lambda: self.set_stimulation_type('random'))

        self.optimal_stim_bt = QPushButton('Optimal Stim.', self)
        self.optimal_stim_bt.move(10, 120)
        self.optimal_stim_bt.clicked.connect(lambda: self.set_stimulation_type('optimal'))
        # self.setWindowIcon()
        self.show()

    def init_dynamic_model(self):
        if self.dynamic_model is None:
            # self.dynamic_model = DynamicModel(optimize=True)
            self.dynamic_model = ModelTrainer(json_file_type=True)
            self.dynamic_model.load_model()
            # 加载默认模型
            # self.dynamic_model.set_init_model()
            print("Load dynamic model successed!...")


    def set_stimulation_type(self,stim_types):
        if stim_types=='random':
            self.timer15ms.stop()
            self.random_stim=True
            # self.recording.set_save_state(False)
            self.recording.set_save_path('F:/调试/0818-0-random-test.h5')
            self.recording.set_save_state(True)
            # self.timer15ms.stop()
            self.recording.clear_buffer_before_starting_timer()
            self.timer15ms.start(self.time_step)
        else:
            self.random_stim=False
            self.timer15ms.stop()
            self.recording.set_save_path('F:/调试/0827-3-optimal-test.h5')
            self.recording.set_save_state(True)
            # self.mpc_.start()
            self.recording.clear_buffer_before_starting_timer()
            self.timer15ms.start(self.time_step) # 10ms 定时器
        self.cur_time = time.time()
        self.recording.set_spike_save_thread(1000//self.time_step, 60)
        print(self.random_stim)

    def spike_detection_in_interest_electrodes(self,data,electrodes):
        spikes=np.zeros((len(electrodes),1))
        for i in range(len(electrodes)):
            self.spike_detection.set_data(data[electrodes[i],:],self.recording.constant)
            self.spike_detection.detection()
            spikes[i]=self.spike_detection.get_spike_num()
        return spikes

    def select_data_from_electrodes(self):
        selected_data = np.zeros((len(self.selected_electrodes), self.sample_data_.shape[1]))
        for i in range(len(self.selected_electrodes)):
            selected_data[i] = self.sample_data_[self.stimulation.channel_map[self.selected_electrodes[i]]]
        return selected_data

    def set_stimulation(self):
        if self.random_stim:# randomly initialize the stimulate parameter
            stimulate_parames=np.random.randn(self.dynamic_model_input_dimention,1)
            
            #self.stimulation.set_sti_signal(stimulate_parames)
        else:# set optimal parameter
            stimulate_parames=np.zeros((self.dynamic_model_input_dimention,1))
            #self.stimulation.set_sti_signal(stimulate_parames)
        return stimulate_parames
    
    def read_data_from_device(self):
        # print('npdata shape: ', self.recording.read_numpy_data.shape)
        # if self.testRecording<20:
        #     ori_data=np.reshape(self.recording.read_numpy_data,
        #                     (int(self.recording.read_numpy_data.shape[0]/self.recording.frame_ret),self.recording.frame_ret), "F")[:60,:]
        #     self.testRecording +=1
        #     return None

        ori_data=self.recording.read_numpy_data.copy()
        ori_data=np.reshape(ori_data,
                            (77, ori_data.shape[0]// 77), "F")[:60,:]

        self.store_data=np.append(self.store_data,ori_data,axis=1)
        # self.iter_count+=1

    
    def reset_sample_data(self):
        del self.sample_data_
        self.sample_data_ = np.empty((len(self.electrodes),0))
    
    def reset_input_data(self):
        for i in self.input_data:
            self.input_data[i].clear()
            self.input_data[i] = []

    def set_optimal_result_from_optimization_framework(self,generate_optimal_signal):
        # self.optimal_stim_signal=np.array(optimal_signal)
        self.return_an_optimal_stim_signal=True

        u, input_amplitudes, input_durations, amplitudes, frequencys = self.optim_.get_optimal_control()
                
        print('frequency:', frequencys)
        print('amplitude:', amplitudes)
        print('input_amplitudes:', input_amplitudes)
        print('input_duration:', input_durations)
        print(np.array([frequencys[0][0],amplitudes[0][0],frequencys[1][0],amplitudes[1][0]]).shape)

        self.stimulation.set_and_send_optim_stimulation(input_amplitudes,input_durations)
        self.external_input_signal=np.append(self.external_input_signal,np.array([frequencys[0][0],amplitudes[0][0],frequencys[1][0],amplitudes[1][0]]),axis=1)
        
        self.return_an_optimal_stim_signal=False
        self.optimal_input_signal=np.append(self.optimal_input_signal,u,axis=1)
        

    def set_optimal_result_from_mpc(self,generate_optimal_signal):
        # self.optimal_stim_signal=np.array(optimal_signal)
        self.return_an_optimal_stim_signal=True

        u, input_amplitudes, input_durations, amplitudes, frequencys = self.mpc_.get_optimal_control()
                
        print('frequency:', frequencys)
        print('amplitude:', amplitudes)
        print('input_amplitudes:', input_amplitudes)
        print('input_duration:', input_durations)
        print(np.array([frequencys[0][0],amplitudes[0][0],frequencys[1][0],amplitudes[1][0]]).shape)
        # test 测试单个电极
        # input_amplitudes[0] = [0, 0]
        # input_durations[0] = [125000, 125000]
        # input_amplitudes[1] = [0, 0]
        # input_durations[1] = [125000, 125000]
        # input_amplitudes[1] = [k/10.0 for k in input_amplitudes[1]]

        self.stimulation.set_and_send_optim_stimulation(input_amplitudes,input_durations)
        self.external_input_signal=np.append(self.external_input_signal,np.array([frequencys[0][0],amplitudes[0][0],frequencys[1][0],amplitudes[1][0]]),axis=1)
        
        self.return_an_optimal_stim_signal=False
        self.optimal_input_signal=np.append(self.optimal_input_signal,u,axis=1)

    def random_stimulate(self):
        # amplitudes = [[-103000.0, 103000.0, 0,-103000.0, 103000.0, 0],
        #                 [0, -103000.0, 103000.0, 0,-103000.0, 103000.0]]
        # durations = [[200.0, 200.0, 5000.0,200.0, 200.0, 5000.0],
        #                 [800, 200.0, 200.0,5000, 200.0, 5000.0]] #, 100.0, 100.0, 1000000.0, 100.0, 100.0, 1000000.0, 100.0, 100.0, 1000000.0, 100000.0]
        self.mpc_.obtain_random_stim(self.dynamic_model_input_dimention, 1)
        u, input_amplitudes, input_durations, amplitudes, frequencys = self.mpc_.get_optimal_control()

        # test 测试单个电极
        # input_amplitudes[0] = [0, 0]
        # input_durations[0] = [125000, 125000]
        # input_amplitudes[1] = [0, 0]
        # input_durations[1] = [125000, 125000]

        # if len(self.sample_data_[0])<200:
        self.stimulation.set_and_send_optim_stimulation(input_amplitudes,input_durations)
        print('frequency:', frequencys)
        print('amplitude:', amplitudes)
        print('input_duration:', input_durations)
        # print(np.array([frequencys[0][0],amplitudes[0][0],frequencys[1][0],amplitudes[1][0]]).shape)
        self.external_input_signal=np.append(self.external_input_signal,np.array([frequencys[0][0],
                                                                                amplitudes[0][0],
                                                                                frequencys[1][0],
                                                                                amplitudes[1][0]]),axis=1)

    def optimal_stimulate(self):

        # 这里只负责设置初值。
        ##################################
        # optimal control 
        ##################################
        if (len(self.sample_data_[0]) > self.dynamic_model.model.data_len+20) and self.random_stim==False:
            # start=time.time()
            # 得到hidden_init的代码再调整一下，现在写得有点绕
            
            #提取需要的电极数据
            selected_sample_data = self.select_data_from_electrodes()
            data_ = np.reshape(selected_sample_data[:,-self.dynamic_model.model.data_len:].T,
                                (1,self.dynamic_model.model.data_len,self.dynamic_model.model.input_dim)) # 最后几步用于初始化hidden state
            input_ = self.external_input_signal[:,-self.dynamic_model.model.data_len+1:]
            pre_X = torch.reshape(torch.from_numpy(data_), (1, 1, self.dynamic_model.model.data_len * self.dynamic_model.model.input_dim))
            Ext_IN = torch.reshape(torch.from_numpy(input_),
                                (1, 1, (self.dynamic_model.model.data_len - 1) * self.dynamic_model_input_dimention))  # Previous_Input
            previous_state = torch.cat((pre_X, Ext_IN), dim=2)
            # initial hidden state
            hidden_init = torch.permute(self.dynamic_model.model.encode(previous_state.type(torch.float32)),
                                        (1, 0, 2))
            # self.mpc_.set_initial_state(np.reshape(hidden_init.detach().numpy(),(self.dynamic_model.model.hidden_dim,1)))


            self.optim_.set_type_of_optim(hidden_init,2,'maximal')
        # if self.return_an_optimal_stim_signal:
            # self.stimulation.set_sti_signal(self.optimal_stim_signal)
            # self.mpc_.obtain_optimal_control()
            
            # u, input_amplitudes, input_durations, amplitudes, frequencys = self.mpc_.get_optimal_control()
                
            # print('frequency:', frequencys)
            # print('amplitude:', amplitudes)
            # print('input_amplitudes:', input_amplitudes)
            # print('input_duration:', input_durations)
            # print(np.array([frequencys[0][0],amplitudes[0][0],frequencys[1][0],amplitudes[1][0]]).shape)
            # # test 测试单个电极
            # # input_amplitudes[0] = [0, 0]
            # # input_durations[0] = [125000, 125000]
            # # input_amplitudes[1] = [0, 0]
            # # input_durations[1] = [125000, 125000]
            # # input_amplitudes[1] = [k/10.0 for k in input_amplitudes[1]]

            # self.stimulation.set_and_send_optim_stimulation(input_amplitudes,input_durations)
            # self.external_input_signal=np.append(self.external_input_signal,np.array([frequencys[0][0],amplitudes[0][0],frequencys[1][0],amplitudes[1][0]]),axis=1)
            
            # self.return_an_optimal_stim_signal=False
            # self.optimal_input_signal=np.append(self.optimal_input_signal,u,axis=1)
    
    def record_and_stim(self):   
        # 10ms 定时器响应函数，读数据+刺激
        self.timer15ms.stop()

        if self.random_stim:
            # random stim
            self.random_stimulate()
        elif self.random_stim==False:
            # optimal stim
            self.optimal_stimulate()
        past_time = self.cur_time
        self.cur_time = time.time()
        print('Interval time between stimulation: %.3f'%(self.cur_time-past_time))
        self.timer15ms.start(self.time_step)

        # store each time step spike data into the self-defined buffer (sample_data)
        self.read_data_from_device()
        # detect spikes from original data
        self.sample_data_=np.append(self.sample_data_,
                                    self.spike_detection_in_interest_electrodes(self.store_data,self.electrodes),axis=1)
        
        self.data_length+=1
        self.sample_data_ori=np.append(self.sample_data_ori,
                                            self.store_data,axis=1)
        # empty storage
        self.store_data=np.empty((60,0))
        # end = time.time()
        # print('save data, time consumed:',(end-start)*1000)

                # time.sleep(sum(duration)/1000000 +.5)
                
            # end = time.time()
            # print('calculate time consumed:',(end-start)*1000)
                # self.mpc_.start()

                # optimal_u=self.mpc_.obtain_optimal_control(np.reshape(hidden_init.detach().numpy(),(self.dynamic_model.model.hidden_dim,1)))# return 4*1 matrix
                # print('------------------------------------------')
                # print(optimal_u)

                # # # data smooth
                # # self.sample_data_smooth=None

                # ele_keys = self.recording.channel_map
                # for index in range(len(self.sample_data_[0])):
                #     for i in ele_keys:
                #         # 将对应channel data传入
                #         self.input_data[i].append(0.0)
                # input_data = self.input_data
                # self.dynamic_model_sti(input_data)
                # self.reset_input_data

                # self.reset_sample_data()
                # # end = time.time()
                # # print('optimal control, time consumed:',(end-start)*1000)
                
        # 如果是随机刺激，就采集到300s*4个数据时进行数据保存
        # 
        if self.data_length==30*4: # and self.random_stim==True:
            self.timer15ms.stop()
            from scipy.io import savemat
            self.recording.set_spike_data(self.sample_data_)
            savemat('random_stimulation_and_recording.mat',{'Output':self.sample_data_,
                                                            # 'Ori':self.sample_data_ori,
                                                            'Input':self.external_input_signal})
            print('save_data')
        # self.timer15ms.start(self.time_step)
        
        # end = time.time()
        # print('record_and_stim, time consumed:',(end-start)*1000)

    # 计时器定时启动动力学模型的刺激
    def dynamic_model_sti(self, spikes):
        #获取当前时间的sti和recording
        # cur_spikes = copy.copy(spikes)    # 保存了电极的key and spikes

        # 据此获取刺激信号，并转化为需要的格式
        # self.para.sti_para["stimulate_name"]

        # t1 = time.time()
        # for i in range(10):
        #     self.dynamic_model.set_record_data(cur_spikes)
        #     amplitude, duration = self.dynamic_model.run()
        # print("dynamic model processing time", (time.time() - t1)*0.01, "s")

        self.dynamic_model.set_record_data(spikes)
        amplitude, duration = self.dynamic_model.run()
        

        # 进行下一次刺激
        self.is_record_data = False
        self.recording.reset_target_time_process()


        # 刺激信号传入刺激器，进行下一步的刺激
        # amplitude = [100, 0, -100, 0]    # mV
        # duration = [200, 200, 200, 10000] # μs
        self.stimulation.update_amplitude_duration(amplitude, duration)

        self.stimulation.update_stimulation_one_time(['21'])

        # self.timer.start(1000)    # 每1s更新一次,可修改
            
