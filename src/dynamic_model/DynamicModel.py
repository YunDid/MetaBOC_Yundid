import numpy as np

from src.dynamic_model.Trainer import *
from src.dynamic_model.OptimizeU import generate_stim
from src.dynamic_model.DataPreprocessing import *

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *


class DynamicModel(QThread):
# class DynamicModel():
    USE_CUDA = False
        
    sample_fre = 4  # discretization time step in per seconds
    stim_size = 4
    # 2 channels
    selected_channels_2 = np.array([0., 0., 0., 0., 0., 0.,
                                    0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0., 1., 0., 0., 0., 0., 0.,
                                    0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0., 0., 0., 0., 0., 0., 0.,
                                    0., 0., 0., 0., 1., 0., 0., 0.,
                                    0., 0., 0., 0., 0., 0.])
    selected_channels = selected_channels_2
    num_recording_channels = int(np.sum(selected_channels))
    data_Sample_Size = 200000
    stim_channels = 4
    hidden = 15
    tp_len = 10
    time_length = 5
    model_type = 'GRU'
    ar = 1
    activate_fun = 'ReLU'
    final_activate_fun = 'Softplus'
    load_data_params = {
        'Num_Cortical': num_recording_channels,
        'Tp': tp_len,
        'Sample_Size': data_Sample_Size,
        'data_length': 200,  # Hyperparameter
        'with_input': True,
        'data_type': 'GroundTruth',  # 'GroundTruth'  '1dB'  '5dB'  '10dB'  '15dB'  '20dB'
        'input_index': 'Seizure',  # Seizure  NonSeizure  Both
        'ext_input_dim': stim_channels,  # input_index= Seizure  1; input_index= NonSeizure  1; input_index= Both  2;
        'AR_order': 3
    }

    model_params = {
        'Tp': load_data_params['Tp'],
        'IN_DIM': load_data_params['Num_Cortical'],
        'HIDDEN_DIM': hidden,
        'LATENT_DIM': hidden,  # Hyperparameter for the hidden dimension of GRU
        # symbol for loss selection
        'X_recon': 100,
        'Y_recon': 1,
        'Tp_recon': 1,
        'Linear_Loss': 200,
        'data_length': time_length,
        # Hyperparameter Previous Steps for Initializing the Hidden State
        'with_input': load_data_params['with_input'],
        'ext_input_dim': load_data_params['ext_input_dim'],
        'AR_order': ar,
        'StateDependent': False,
        'Conv': False,
        'activation': activate_fun,
        'final_activation': final_activate_fun,
        'RNN_Type': model_type,
        'L_factors': 0.05,
        'device': 'cuda:0'
    }

    def __init__(self, rec_data=None, stim_data=None, optimize=False, Y_ref=[],sample_fre=sample_fre, stim_size=stim_size,
                 selected_channels=selected_channels, data_Sample_Size=data_Sample_Size, stim_channels=stim_channels,
                 hidden=hidden, tp_len=tp_len, time_length=time_length, model_type=model_type, ar=ar,
                 activate_fun=activate_fun, final_activate_fun=final_activate_fun, load_data_params=load_data_params,
                 model_params=model_params,
                 USE_CUDA = False,
                 initial_checkpoint='./src/dynamic_model/checkpoints/GRU/2022-12-29_set7-FT_hidden_15_time_length_5_Tp_10_4Hz_smth_2channels_day_29.ckpt'):
        QThread.__init__(self, parent=None)

        self.selected_channels = selected_channels
        #######################输入输出格式转化################################
        self.rec_data = np.asarray(self.screenData(rec_data, selected_channels)) if rec_data!=None else None
        self.stim_data = stim_data
        ####################################################################
        self.sample_fre = sample_fre
        self.stim_size = stim_size
        self.data_Sample_Size = data_Sample_Size
        self.stim_channels = stim_channels
        self.hidden = hidden
        self.tp_len = tp_len
        self.time_length = time_length
        self.model_type = model_type
        self.ar = ar
        self.activate_fun = activate_fun
        self.final_activate_fun = final_activate_fun
        self.load_data_params = load_data_params
        self.model_params = model_params
        self.USE_CUDA = USE_CUDA
        self.initial_checkpoint = None

        self.recording_data = None
        self.model = Latent_Model(self.model_params)

        if optimize:
            self.initial_checkpoint = initial_checkpoint
            # U = self.optimizeOutput()
            # amplitude, duration = self.testOutput_1(U)
            # return amplitude, duration

    def load_Trainer(self, unique_name='set7', day='27', test_sample_size=30, epochs=100, resultDirPath='./result/',
                     modelDirPath='./checkpoints/', saveTestFig=True, save_prediction=False, initial_checkpoint=None):
        recording_data = self.rec_data
        stim_data = self.stim_data
        # print(stim_data.shape)

        # 与数据相匹配
        # recording_channels = channels
        # stim_channels = num_stim_channels #
        # sample_fre = 50
        channels = int(np.sum(self.selected_channels))
        smth = 'smth_%dchannels' % (channels)

        processing(recording_data=recording_data,
                   stim_data=stim_data,
                   load_data_params=self.load_data_params,
                   model_params=self.model_params,
                   unique_name=unique_name,
                   sample_fre=self.sample_fre,
                   smth=smth,
                   test_sample_size=test_sample_size,
                   resultDirPath=resultDirPath,
                   modelDirPath=modelDirPath,
                   day=day,
                   epochs=epochs,
                   saveTestFig=saveTestFig,
                   save_prediction=save_prediction,
                   initial_checkpoint=initial_checkpoint)
    
    def set_record_data(self, rec_data):
        self.recording_data = rec_data

    def set_stim_data(self,stim_data):
        self.stim_data = stim_data

    def set_init_model(self, initial_checkpoint='./checkpoints/GRU/2022-12-29_set7-FT_hidden_15_time_length_5_Tp_10_4Hz_smth_2channels_day_29.ckpt'):
        if os.path.exists(initial_checkpoint):
            print("Loading checkpoint...")
            if self.USE_CUDA:
                self.model = self.model.load_from_checkpoint(checkpoint_path=initial_checkpoint, params=self.model_params).to('cuda:0')
            else:
                self.model = self.model.load_from_checkpoint(checkpoint_path=initial_checkpoint, params=self.model_params)
        else:
            # 应该返回报错
            print('Address Error! ')

    def run(self, Y_ref=np.array([5,5]), Pred_Horizon=tp_len, Control_Horizon=tp_len, minimize_hz=4,
                       iters=100, min_input=[0, 0.1, 0, 0.1], max_input=[20, 0.5, 20, 0.5]):
        recording_data = np.asarray(self.screenData(self.recording_data, self.selected_channels))
        stim_data = self.stim_data
        print(recording_data.shape)

        U = generate_stim(previous_state=recording_data,
                        model=self.model,
                        #   model_params=self.model_params,
                        #   checkpoint=self.initial_checkpoint,
                          N_trials=1,
                          Y_ref=Y_ref,  # np.array([1,1,1,1,1.5,1.5,1.5,1.5,1.5]),
                          sample_fre=self.sample_fre,
                          iters=iters,
                          Pred_Horizon=Pred_Horizon,
                          Control_Horizon=Control_Horizon,
                          min_input=min_input,
                          max_input=max_input,
                          minimize_hz=minimize_hz,
                          index_=4)
        amplitude, duration = self.testOutput_1(U)
        return amplitude, duration

    @staticmethod
    def dir2np(rec_data):
        data = np.zeros([len(rec_data), len(rec_data['12'])])
        electrode = -1
        for row in range(1, 9):
            for col in range(1, 9):
                str_read_key = str(row) + str(col)
                key_name = [k for k in rec_data.keys() if str_read_key in k]

                # 当对应电极存在神经响应记录时，滑动窗计算spike count
                if len(key_name) > 0:
                    #                     print(spikes_sequence[key_name[0]].shape)
                    #                     spike_data_=np.array(spikes_sequence[key_name[0]][0,:])#h5 file
                    # spike_data_ = np.array(spikes_sequence[key_name[0]][:, 0])  # loadmat type
                    # print(spike_data_)
                    data[electrode] = rec_data[key_name[-1]]
                    electrode += 1
                # 当没有记录时，排除四角电极（11，18，81，88），其余填空
                else:
                    if (row == 1 or row == 8) and (col == 1 or col == 8):
                        continue
                    else:
                        electrode += 1
        return data


    def screenData(self, rec_data, mask):
        data = self.dir2np(rec_data)
        # print('data_shape:',len(data),'*',len(data[0]))
        num_selected_channels = int(np.sum(mask))
        screened_data = np.zeros([num_selected_channels, len(data[0])])
        index_ = 0
        for i in range(len(data)):
            if mask[i] > 0:
                screened_data[index_] = data[i]
                index_ += 1
        # print('screened_data_shape:', len(screened_data), '*', len(screened_data[0]))
        return screened_data

    @staticmethod
    def testOutput_1(U):
        # 刺激参数（amplitude,frequency）转化为输入参数（amplitude，duration）
        stim_1_freq = U[0, :]
        stim_1_amp = U[1, :] * 1000
        amplitude = []
        duration = []
        duration_time = 200
        print('stim_1_freq shape: ',stim_1_freq.shape)
        for i in range(stim_1_freq.shape[0]):
            if stim_1_freq[i] >=4:
                isi = np.round(1000 / np.round(stim_1_freq[i]))
                times = np.trunc(250 / (np.round(1000 / np.round(stim_1_freq[i]))))
                amp = np.round(stim_1_amp[i])
                time_rest = 250 - np.round(1000 / np.round(stim_1_freq[ i])) * np.trunc(
                    250 / (np.round(1000 / np.round(stim_1_freq[i]))))
                if time_rest < .5:
                    time_rest = isi
                else:
                    times = times + 1
                for j in range(int(times)):
                    # 设置刺激幅值（μV）和刺激时间（μs）
                    amplitude.append(amp*1000)
                    duration.append(duration_time)
                    # 设置刺激间隔（μs）
                    amplitude.append(0)
                    duration.append(isi*1000)
            else:
                time_rest = 250
            # 相邻刺激间隔（μs）
            amplitude.append(0)
            duration.append(time_rest*1000)
        return amplitude, duration


    # def testOutput_1(U):
    #     stim_1_freq = U[0, :]
    #     stim_1_amp = U[1, :] * 1000
    #     amplitude = []
    #     duration = []
    #     print('stim_1_freq shape: ',stim_1_freq.shape)
    #     for i in range(stim_1_freq.shape[0]):
    #         if stim_1_freq[i] >=4:
    #             isi = np.round(1000 / np.round(stim_1_freq[i]))
    #             times = np.trunc(250 / (np.round(1000 / np.round(stim_1_freq[i]))))
    #             amp = np.round(stim_1_amp[i])
    #             time_rest = 250 - np.round(1000 / np.round(stim_1_freq[ i])) * np.trunc(
    #                 250 / (np.round(1000 / np.round(stim_1_freq[i]))))
    #             if time_rest < .5:
    #                 time_rest = isi
    #             else:
    #                 times = times + 1
    #             for j in range(int(times)):
    #                 amplitude.append(amp)
    #                 duration.append(isi)
    #         else:
    #             time_rest = 250
    #         amplitude.append(0)
    #         duration.append(time_rest)
    #     return amplitude, duration

    

# import numpy as np

# from src.dynamic_model.Trainer import *
# from src.dynamic_model.OptimizeU import generate_stim
# from src.dynamic_model.DataPreprocessing import *

# from PyQt5.QtGui import *
# from PyQt5.QtCore import *
# from PyQt5.QtWidgets import *
# from PyQt5.QtChart import *


# class DynamicModel(QThread):
# # class DynamicModel():
#     USE_CUDA = False
        
#     sample_fre = 4  # discretization time step in per seconds
#     stim_size = 4
#     # 2 channels
#     selected_channels_2 = np.array([0., 0., 0., 0., 0., 0.,
#                                     0., 0., 0., 0., 0., 0., 0., 0.,
#                                     0., 0., 1., 0., 0., 0., 0., 0.,
#                                     0., 0., 0., 0., 0., 0., 0., 0.,
#                                     0., 0., 0., 0., 0., 0., 0., 0.,
#                                     0., 0., 0., 0., 0., 0., 0., 0.,
#                                     0., 0., 0., 0., 1., 0., 0., 0.,
#                                     0., 0., 0., 0., 0., 0.])
#     selected_channels = selected_channels_2
#     num_recording_channels = int(np.sum(selected_channels))
#     data_Sample_Size = 200000
#     stim_channels = 4
#     hidden = 15
#     tp_len = 10
#     time_length = 5
#     model_type = 'GRU'
#     ar = 1
#     activate_fun = 'ReLU'
#     final_activate_fun = 'Softplus'
#     load_data_params = {
#         'Num_Cortical': num_recording_channels,
#         'Tp': tp_len,
#         'Sample_Size': data_Sample_Size,
#         'data_length': 200,  # Hyperparameter
#         'with_input': True,
#         'data_type': 'GroundTruth',  # 'GroundTruth'  '1dB'  '5dB'  '10dB'  '15dB'  '20dB'
#         'input_index': 'Seizure',  # Seizure  NonSeizure  Both
#         'ext_input_dim': stim_channels,  # input_index= Seizure  1; input_index= NonSeizure  1; input_index= Both  2;
#         'AR_order': 3
#     }

#     model_params = {
#         'Tp': load_data_params['Tp'],
#         'IN_DIM': load_data_params['Num_Cortical'],
#         'HIDDEN_DIM': hidden,
#         'LATENT_DIM': hidden,  # Hyperparameter for the hidden dimension of GRU
#         # symbol for loss selection
#         'X_recon': 100,
#         'Y_recon': 1,
#         'Tp_recon': 1,
#         'Linear_Loss': 200,
#         'data_length': time_length,
#         # Hyperparameter Previous Steps for Initializing the Hidden State
#         'with_input': load_data_params['with_input'],
#         'ext_input_dim': load_data_params['ext_input_dim'],
#         'AR_order': ar,
#         'StateDependent': False,
#         'Conv': False,
#         'activation': activate_fun,
#         'final_activation': final_activate_fun,
#         'RNN_Type': model_type,
#         'L_factors': 0.05,
#         'device': 'cuda:0'
#     }

#     def __init__(self, rec_data=None, stim_data=None, optimize=False, Y_ref=[],sample_fre=sample_fre, stim_size=stim_size,
#                  selected_channels=selected_channels, data_Sample_Size=data_Sample_Size, stim_channels=stim_channels,
#                  hidden=hidden, tp_len=tp_len, time_length=time_length, model_type=model_type, ar=ar,
#                  activate_fun=activate_fun, final_activate_fun=final_activate_fun, load_data_params=load_data_params,
#                  model_params=model_params,
#                  USE_CUDA = False,
#                  initial_checkpoint='./src/dynamic_model/checkpoints/GRU/2022-12-29_set7-FT_hidden_15_time_length_5_Tp_10_4Hz_smth_2channels_day_29.ckpt'):
#         QThread.__init__(self, parent=None)

#         self.selected_channels = selected_channels
#         #######################输入输出格式转化################################
#         self.rec_data = np.asarray(self.screenData(rec_data, selected_channels)) if rec_data!=None else None
#         self.stim_data = stim_data
#         ####################################################################
#         self.sample_fre = sample_fre
#         self.stim_size = stim_size
#         self.data_Sample_Size = data_Sample_Size
#         self.stim_channels = stim_channels
#         self.hidden = hidden
#         self.tp_len = tp_len
#         self.time_length = time_length
#         self.model_type = model_type
#         self.ar = ar
#         self.activate_fun = activate_fun
#         self.final_activate_fun = final_activate_fun
#         self.load_data_params = load_data_params
#         self.model_params = model_params
#         self.USE_CUDA = USE_CUDA
#         self.initial_checkpoint = None

#         self.recording_data = None
#         self.model = Latent_Model(self.model_params)

#         if optimize:
#             self.initial_checkpoint = initial_checkpoint
#             # U = self.optimizeOutput()
#             # amplitude, duration = self.testOutput_1(U)
#             # return amplitude, duration

#     def load_Trainer(self, unique_name='set7', day='27', test_sample_size=30, epochs=100, resultDirPath='./result/',
#                      modelDirPath='./checkpoints/', saveTestFig=True, save_prediction=False, initial_checkpoint=None):
#         recording_data = self.rec_data
#         stim_data = self.stim_data
#         # print(stim_data.shape)

#         # 与数据相匹配
#         # recording_channels = channels
#         # stim_channels = num_stim_channels #
#         # sample_fre = 50
#         channels = int(np.sum(self.selected_channels))
#         smth = 'smth_%dchannels' % (channels)

#         processing(recording_data=recording_data,
#                    stim_data=stim_data,
#                    load_data_params=self.load_data_params,
#                    model_params=self.model_params,
#                    unique_name=unique_name,
#                    sample_fre=self.sample_fre,
#                    smth=smth,
#                    test_sample_size=test_sample_size,
#                    resultDirPath=resultDirPath,
#                    modelDirPath=modelDirPath,
#                    day=day,
#                    epochs=epochs,
#                    saveTestFig=saveTestFig,
#                    save_prediction=save_prediction,
#                    initial_checkpoint=initial_checkpoint)
    
#     def set_record_data(self, rec_data):
#         self.recording_data = rec_data

#     def set_stim_data(self,stim_data):
#         self.stim_data = stim_data

#     def set_init_model(self, initial_checkpoint='./checkpoints/GRU/2022-12-29_set7-FT_hidden_15_time_length_5_Tp_10_4Hz_smth_2channels_day_29.ckpt'):
#         if os.path.exists(initial_checkpoint):
#             print("Loading checkpoint...")
#             if self.USE_CUDA:
#                 self.model = self.model.load_from_checkpoint(checkpoint_path=initial_checkpoint, params=self.model_params).to('cuda:0')
#             else:
#                 self.model = self.model.load_from_checkpoint(checkpoint_path=initial_checkpoint, params=self.model_params)
#         else:
#             # 应该返回报错
#             print('Address Error! ')

#     def run(self, Y_ref=np.array([5,5]), Pred_Horizon=tp_len, Control_Horizon=tp_len, minimize_hz=4,
#                        iters=100, min_input=[0, 0.1, 0, 0.1], max_input=[20, 0.5, 20, 0.5]):
#         recording_data = np.asarray(self.screenData(self.recording_data, self.selected_channels))
#         stim_data = self.stim_data
#         print(recording_data.shape)

#         U = generate_stim(previous_state=recording_data,
#                         model=self.model,
#                         #   model_params=self.model_params,
#                         #   checkpoint=self.initial_checkpoint,
#                           N_trials=1,
#                           Y_ref=Y_ref,  # np.array([1,1,1,1,1.5,1.5,1.5,1.5,1.5]),
#                           sample_fre=self.sample_fre,
#                           iters=iters,
#                           Pred_Horizon=Pred_Horizon,
#                           Control_Horizon=Control_Horizon,
#                           min_input=min_input,
#                           max_input=max_input,
#                           minimize_hz=minimize_hz,
#                           index_=4)
#         amplitude, duration = self.testOutput_1(U)
#         return amplitude, duration

#     @staticmethod
#     def dir2np(rec_data):
#         data = np.zeros([len(rec_data), len(rec_data['12'])])
#         electrode = -1
#         for row in range(1, 9):
#             for col in range(1, 9):
#                 str_read_key = str(row) + str(col)
#                 key_name = [k for k in rec_data.keys() if str_read_key in k]

#                 # 当对应电极存在神经响应记录时，滑动窗计算spike count
#                 if len(key_name) > 0:
#                     #                     print(spikes_sequence[key_name[0]].shape)
#                     #                     spike_data_=np.array(spikes_sequence[key_name[0]][0,:])#h5 file
#                     # spike_data_ = np.array(spikes_sequence[key_name[0]][:, 0])  # loadmat type
#                     # print(spike_data_)
#                     data[electrode] = rec_data[key_name[-1]]
#                     electrode += 1
#                 # 当没有记录时，排除四角电极（11，18，81，88），其余填空
#                 else:
#                     if (row == 1 or row == 8) and (col == 1 or col == 8):
#                         continue
#                     else:
#                         electrode += 1
#         return data


#     def screenData(self, rec_data, mask):
#         data = self.dir2np(rec_data)
#         # print('data_shape:',len(data),'*',len(data[0]))
#         num_selected_channels = int(np.sum(mask))
#         screened_data = np.zeros([num_selected_channels, len(data[0])])
#         index_ = 0
#         for i in range(len(data)):
#             if mask[i] > 0:
#                 screened_data[index_] = data[i]
#                 index_ += 1
#         # print('screened_data_shape:', len(screened_data), '*', len(screened_data[0]))
#         return screened_data

#     @staticmethod
#     def testOutput_1(U):
#         # 刺激参数（amplitude,frequency）转化为输入参数（amplitude，duration）
#         stim_1_freq = U[0, :]
#         stim_1_amp = U[1, :] * 1000
#         amplitude = []
#         duration = []
#         duration_time = 200
#         print('stim_1_freq shape: ',stim_1_freq.shape)
#         for i in range(stim_1_freq.shape[0]):
#             if stim_1_freq[i] >=4:
#                 isi = np.round(1000 / np.round(stim_1_freq[i]))
#                 times = np.trunc(250 / (np.round(1000 / np.round(stim_1_freq[i]))))
#                 amp = np.round(stim_1_amp[i])
#                 time_rest = 250 - np.round(1000 / np.round(stim_1_freq[ i])) * np.trunc(
#                     250 / (np.round(1000 / np.round(stim_1_freq[i]))))
#                 if time_rest < .5:
#                     time_rest = isi
#                 else:
#                     times = times + 1
#                 for j in range(int(times)):
#                     # 设置刺激幅值（μV）和刺激时间（μs）
#                     amplitude.append(amp*1000)
#                     duration.append(duration_time)
#                     # 设置刺激间隔（μs）
#                     amplitude.append(0)
#                     duration.append(isi*1000)
#             else:
#                 time_rest = 250
#             # 相邻刺激间隔（μs）
#             amplitude.append(0)
#             duration.append(time_rest*1000)
#         return amplitude, duration


#     # def testOutput_1(U):
#     #     stim_1_freq = U[0, :]
#     #     stim_1_amp = U[1, :] * 1000
#     #     amplitude = []
#     #     duration = []
#     #     print('stim_1_freq shape: ',stim_1_freq.shape)
#     #     for i in range(stim_1_freq.shape[0]):
#     #         if stim_1_freq[i] >=4:
#     #             isi = np.round(1000 / np.round(stim_1_freq[i]))
#     #             times = np.trunc(250 / (np.round(1000 / np.round(stim_1_freq[i]))))
#     #             amp = np.round(stim_1_amp[i])
#     #             time_rest = 250 - np.round(1000 / np.round(stim_1_freq[ i])) * np.trunc(
#     #                 250 / (np.round(1000 / np.round(stim_1_freq[i]))))
#     #             if time_rest < .5:
#     #                 time_rest = isi
#     #             else:
#     #                 times = times + 1
#     #             for j in range(int(times)):
#     #                 amplitude.append(amp)
#     #                 duration.append(isi)
#     #         else:
#     #             time_rest = 250
#     #         amplitude.append(0)
#     #         duration.append(time_rest)
#     #     return amplitude, duration

    
