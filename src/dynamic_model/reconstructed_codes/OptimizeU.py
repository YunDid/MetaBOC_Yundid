import random

from DataSample import Input_Data
from Models import GRU,RNN,LSTM

import torch
import scipy.io as scio
from torch.utils.data import DataLoader
import numpy as np
from pytorch_lightning import Trainer
import numpy as np

def MPC_optimization(model, Pred_Horizon, Control_Horizon, Y_ref, previous_state, iters=10, min_input=-10,
                     max_input=10):
    # opt_U = torch.nn.parameter.Parameter(
    #     -10000 * torch.randn((1, Control_Horizon, model.ext_input_dim), dtype=torch.float32), requires_grad=True)
    opt_U = torch.nn.parameter.Parameter(torch.randn((1, Control_Horizon, model.ext_input_dim), dtype=torch.float32), requires_grad=True)
    # 迭代优化
    for steps in range(iters):
        loss = 0

        hidden_init = torch.permute(model.hidden_initial_nn(previous_state), (1, 0, 2)).to('cpu:0')  # Batch X 1 X H_dim -> 1 X Batch X H_dim
        for i in range(Pred_Horizon):

            inputs = torch.zeros((1, 1, model.ext_input_dim))
            if i < Control_Horizon:
                # print(opt_U[0,0,:].size())
                for k in range(opt_U[0,0,:].size()[0]):
                    inputs[:,:,k] = torch.clamp(opt_U[:, i:i + 1, k], min_input, max_input[k])

                # 构造tc step的输入惩罚
                loss += 0.001 * torch.matmul(opt_U[0, i:i + 1, :].to('cpu:0'),torch.transpose(opt_U[0, i:i + 1, :].to('cpu:0'), 0, 1))

                # print('input',torch.matmul(opt_U[0, i:i + 1, :].to('cuda:1'),torch.transpose(opt_U[0, i:i + 1, :].to('cuda:1'), 0, 1)))
                # print('imput',loss)
            # else:
            #     inputs = torch.zeros((1, 1, model.ext_input_dim))

            # model prediction
            output, hidden_init = model.gru(inputs.to('cpu:0'), hidden_init)

            Tp_prediction = model.decode(torch.permute(hidden_init, (1, 0, 2)))

            deltaY = Y_ref.to('cpu:0') - Tp_prediction[0, :, :]

            # 构造tp step的预测误差
            loss += torch.matmul(deltaY, torch.transpose(deltaY, 0, 1)) * 10
            # print('error', torch.matmul(deltaY, torch.transpose(deltaY, 0, 1)))
        optimizer = torch.optim.Adam({opt_U}, lr=0.001)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

        # 把u限制在合理的范围内
        with torch.no_grad():
            for k in range(opt_U[0,0,:].size()[0]):
                opt_U[:,:,k].clamp_(min_input, max_input[k])
    return opt_U

def generate_stim(previous_state,gru_model_params,checkpoint,N_trials,amplitude,Pred_Horizon=20,Control_Horizon=10,iters=1000,max_input=[5, 1, 5, 1]):
    # previous_state channels*T
    model = GRU.load_from_checkpoint(checkpoint_path=checkpoint, params=gru_model_params).to('cpu:0')
    opt_Us = np.zeros((N_trials, Control_Horizon, model.ext_input_dim))
    pre_X = torch.zeros((1, 1, model.data_len * model.input_dim))
    # len = len(previous_state)
    # pre_X += torch.reshape(torch.from_numpy(previous_state[len - model.data_len:, :].astype(np.float32)),
    #                        (1, 1, model.data_len * model.input_dim))

    # len = len(previous_state.T)
    # pre_X += torch.reshape(torch.from_numpy(previous_state.T[len-model.data_len:,:].astype(np.float32)),(1, 1, model.data_len * model.input_dim))
    for i in range(N_trials):
        # random_index = np.random.randint(0, 1000)
        length = len(previous_state.T)
        print(previous_state.shape)
        pre_X = torch.reshape(torch.from_numpy(previous_state.T[length - model.data_len:, :].astype(np.float32)),
                               (1, 1, model.data_len * model.input_dim))
        print(pre_X.shape)

        Ext_IN = torch.zeros((1, 1, (model.data_len - 1) * model.ext_input_dim))  # Previous_Input
        previous_state = torch.cat((pre_X, Ext_IN), dim=2).to('cpu:0')  # ?
        # print(previous_state.shape)

        current_input = Ext_IN
        # deltaU=torch.nn.parameter.Parameter(torch.zeros((1,Control_Horizon,model.ext_input_dim),dtype=torch.float32),requires_grad=True)
        Y_ref = torch.ones((1, model.input_dim)) * amplitude
        opt_U = MPC_optimization(model, Pred_Horizon, Control_Horizon, Y_ref, previous_state, iters=iters, min_input=0,
                                 max_input=max_input)
        opt_Us[i, :, :] = opt_U.detach().cpu().numpy()

    average_U = np.mean(opt_Us, axis=0)
    print(np.round(average_U, 3))

    average_U_downsample = np.zeros([Control_Horizon // 10, 2])
    for i in range(Control_Horizon // 10):
        average_U_downsample[i, 0] = np.mean(average_U[i * 10:(i + 1) * 10, 0])
        average_U_downsample[i, 1] = np.mean(average_U[i * 10:(i + 1) * 10, 1])

    return average_U.T
#
# load_data_params = {
#                     'file_path': r'/mnt/database6/Organoid/codes/baseline/data/firing_rate_data6-1107_50Hz_smth_PSTH_5channels_newStimET.mat',
#                     'Num_Cortical': 5,
#                     'Tp': 10,
#                     'Sample_Size': 90000,
#                     'data_length': 100,  # Hyperparameter
#                     'with_input': True,
#                     'data_type': 'GroundTruth',  # 'GroundTruth'  '1dB'  '5dB'  '10dB'  '15dB'  '20dB'
#                     'input_index': 'Seizure',  # Seizure  NonSeizure  Both
#                     'ext_input_dim': 4,  # input_index= Seizure  1; input_index= NonSeizure  1; input_index= Both  2;
#                     'AR_order': 3
#                 }
# Data_Path = load_data_params['file_path']
# recording_data = scio.loadmat(Data_Path)['Cortex']
# stim_data = scio.loadmat(Data_Path)['Inputs']
# jr_data=Input_Data(load_data_params,recording_data,stim_data)
#
# gru_model_params={
#         'Tp':load_data_params['Tp'],
#         'IN_DIM':load_data_params['Num_Cortical'],
#         'HIDDEN_DIM':30,
#         'LATENT_DIM':30, # Hyperparameter for the hidden dimension of GRU
#         # symbol for loss selection
#         'X_recon':100,
#         'Y_recon':1,
#         'Tp_recon':1,
#         'Linear_Loss':200,
#         'data_length':10,  # Hyperparameter Previous Steps for Initializing the Hidden State
#         'with_input':load_data_params['with_input'],
#         'ext_input_dim':load_data_params['ext_input_dim'],
#         'AR_order':1,
#         'StateDependent':False,
#         'Conv':False,
#         'device':'cuda:0'
#     }
# # checkpoint="/mnt/database6/Organoid/codes/baseline/gru/gru_1103data6-1103-smth_PSTH_5channels_2stimET-testX10_hidden_30_latent_30_time_length_10_Tp_10_100Hz_smth_PSTH_5channels_2stimET.ckpt"
# # checkpoint = "/mnt/database6/Organoid/codes/baseline/gru/gru_1103data6-1103_AN-smth_PSTH_5channels_2stimET-test_hidden_30_latent_30_time_length_10_Tp_10_100Hz_smth_PSTH_5channels_2stimET.ckpt"
# checkpoint = "/mnt/database6/Organoid/codes/baseline/gru/gru_1107data6-1107-smth_PSTH_5channels_newStimET-test_hidden_30_latent_30_time_length_10_Tp_10_50Hz_smth_PSTH_5channels_newStimET.ckpt"
# model=GRU.load_from_checkpoint(checkpoint_path=checkpoint,params=gru_model_params).to('cuda:1')
#
# Pred_Horizon = 20
# Control_Horizon = 10
#
# # 10 groups
# N_trials=1
# amplitude = 0.2
# opt_Us=np.zeros((N_trials,Control_Horizon,model.ext_input_dim))
# pre_X = torch.zeros((1, 1, model.data_len * model.input_dim))
# for i in range(10):
#     random.seed(i)
#     random_index = np.random.randint(0, 1000)
#     pre_X += torch.reshape(torch.from_numpy(jr_data.X[random_index:random_index + model.data_len, :].astype(np.float32)),
#                           (1, 1, model.data_len * model.input_dim))
# # pre_X /= 10
# for i in range(N_trials):
#     random_index = np.random.randint(0,1000)
#     pre_X = torch.reshape(torch.from_numpy(jr_data.X[random_index:random_index+model.data_len, :].astype(np.float32)), (1, 1, model.data_len * model.input_dim))
#     print(pre_X.shape)
#
#     Ext_IN = torch.zeros((1, 1, (model.data_len - 1) * model.ext_input_dim))  # Previous_Input
#     previous_state = torch.cat((pre_X, Ext_IN), dim=2).to('cuda:1')     #?
#     # print(previous_state.shape)
#
#     current_input = Ext_IN
#     # deltaU=torch.nn.parameter.Parameter(torch.zeros((1,Control_Horizon,model.ext_input_dim),dtype=torch.float32),requires_grad=True)
#     Y_ref = torch.ones((1, model.input_dim))*amplitude
#     opt_U = MPC_optimization(model, Pred_Horizon, Control_Horizon, Y_ref, previous_state, iters=1000, min_input=0,
#                              max_input=[5,1,5,1])
#     opt_Us[i,:,:]=opt_U.detach().cpu().numpy()
#
# average_U=np.mean(opt_Us,axis=0)
# print(np.round(average_U,3))
#
# average_U_downsample = np.zeros([Control_Horizon//10,2])
# for i in range(Control_Horizon//10):
#     average_U_downsample[i,0] = np.mean(average_U[i*10:(i+1)*10,0])
#     average_U_downsample[i, 1] = np.mean(average_U[i * 10:(i + 1) * 10, 1])
#
# #