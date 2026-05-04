import random

from src.dynamic_model.DataSample import Input_Data
from src.dynamic_model.Models import Latent_Model

import torch
import scipy.io as scio
from torch.utils.data import DataLoader
import numpy as np
from pytorch_lightning import Trainer
import numpy as np


from scipy.optimize import LinearConstraint, minimize

mpc_pred_len = 10

def objective_function(deltaU,H,f):
    # print(deltaU.shape,H.shape,f.shape)
    return np.reshape(0.5*np.dot(np.dot(np.transpose(deltaU),H),deltaU)+np.dot(np.transpose(deltaU),f),1)[0]

def Hessian(deltaU,H,f):
    return H

def Jacobian(deltaU,H,f):
    return np.dot(H,deltaU)+f

#print(B_matrix)
# def MPC(koop_mat,B,delta_state,cur_state,u_prev,R_s):
    

import numpy as np

def MPC_optimization(model, Pred_Horizon, Control_Horizon, Y_ref, previous_state, delta_state=None, u_prev=None, iters=10, min_input=-10,
                     max_input=10,minimize_hz=4):

    if not model.RNN_Type=='Koopman':
        # opt_U = torch.nn.parameter.Parameter(
        #     -10000 * torch.randn((1, Control_Horizon, model.ext_input_dim), dtype=torch.float32), requires_grad=True)
        init_input_U=np.zeros((1,Control_Horizon, model.ext_input_dim))
        init_input_U[:,:,0]=1.0*np.random.randint(4, 20, Control_Horizon)
        init_input_U[:,:,2]=1.0*np.random.randint(4, 20, Control_Horizon)

        init_input_U[:,:,1]=np.random.normal(5, 2, Control_Horizon)
        init_input_U[:,:,3]=np.random.normal(5, 2, Control_Horizon)


        opt_U = torch.nn.parameter.Parameter(torch.from_numpy(init_input_U), requires_grad=True)

        # 迭代优化
        for steps in range(iters):
            loss = 0

            hidden_init = torch.permute(model.encode(previous_state), (1, 0, 2)).to('cuda:0')  # Batch X 1 X H_dim -> 1 X Batch X H_dim
            for i in range(Pred_Horizon):

                inputs = torch.zeros((1, 1, model.ext_input_dim))
                if i < Control_Horizon:
                    # print(opt_U[0,0,:].size())
                    for k in range(opt_U[0,0,:].size()[0]):
                        inputs[:,:,k] = torch.clamp(opt_U[:, i:i + 1, k], min_input[k], max_input[k])
                        if k==0 and k==2:
                            if inputs[:,:,k]<minimize_hz:
                                inputs[:,:,k]=0

                    # 构造tc step的输入惩罚
                    # loss += 0.0000000000000000001 * torch.matmul(opt_U[0, i:i + 1, :].to('cuda:0'),torch.transpose(opt_U[0, i:i + 1, :].to('cuda:0'), 0, 1))

                    # print('input',torch.matmul(opt_U[0, i:i + 1, :].to('cuda:0'),torch.transpose(opt_U[0, i:i + 1, :].to('cuda:0'), 0, 1)))
                    # print('imput',loss)
                # else:
                #     inputs = torch.zeros((1, 1, model.ext_input_dim))

                # model prediction
                if model.RNN_Type=='LSTM':
                    output, (hidden_init, hidden_init) = model.rnn_cell(inputs.to('cuda:0'), (hidden_init, hidden_init))
                elif model.RNN_Type=='EI-RNN':
                    output, hidden_init = model.rnn_cell(torch.permute(inputs.to('cuda:0'), (1, 0, 2)), hidden_init)
                else:
                    output, hidden_init = model.rnn_cell(inputs.to('cuda:0'), hidden_init)
                # output, hidden_init = model.gru(inputs.to('cuda:0'), hidden_init)

                # output = None
                if model.RNN_Type!='EI-RNN':
                    Tp_prediction = model.decode(torch.permute(hidden_init, (1, 0, 2)))
                else:
                    Tp_prediction = torch.permute(output, (1, 0, 2))

                deltaY = Y_ref.to('cuda:0') - Tp_prediction[0, :, :]

                # 构造tp step的预测误差
                loss += torch.matmul(deltaY, torch.transpose(deltaY, 0, 1))
                # print('error', torch.matmul(deltaY, torch.transpose(deltaY, 0, 1)))
            optimizer = torch.optim.Adam({opt_U}, lr=0.01)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        
            # 把u限制在合理的范围内
            with torch.no_grad():
                # print(opt_U.shape)
                for k in range(opt_U[0,0,:].size()[0]):
                    opt_U[:,:,k].clamp_(min_input[k], max_input[k])

            # print(opt_U)
                for i in range(opt_U.shape[0]):
                    for j in range(opt_U.shape[1]):
                        for m in [0,2]:
                            if opt_U[i,j,m] < minimize_hz:
                                opt_U[i,j,m]=0

            if (steps%100==0):
                print('Iters:',steps,' Loss:',loss.detach().cpu().numpy()[0,0])#,'opt_u:',opt_U)

    else:
        ############################################
        ##############   Koopman   #################
        ############################################

        init_input_U=np.zeros((1,Control_Horizon, model.ext_input_dim))
        init_input_U[:,:,0]=1.0*np.random.randint(4, 20, Control_Horizon)
        init_input_U[:,:,2]=1.0*np.random.randint(4, 20, Control_Horizon)

        init_input_U[:,:,1]=np.random.normal(5, 2, Control_Horizon)
        init_input_U[:,:,3]=np.random.normal(5, 2, Control_Horizon)

        opt_U = torch.nn.parameter.Parameter(torch.from_numpy(init_input_U), requires_grad=True)

        # 迭代优化
        for steps in range(iters):
            loss = 0
            basis_num = model.latent_dim*model.ar_order
            # print(previous_state.shape)
            latent_new=torch.reshape(previous_state,(1,1,basis_num))

            for i in range(Pred_Horizon):

                inputs = torch.zeros((1, 1, model.ext_input_dim))
                if i < Control_Horizon:
                    # print(opt_U[0,0,:].size())
                    for k in range(opt_U[0,0,:].size()[0]):
                        inputs[:,:,k] = torch.clamp(opt_U[:, i:i + 1, k], min_input[k], max_input[k])
                        if k==0 and k==2:
                            if inputs[:,:,k]<minimize_hz:
                                inputs[:,:,k]=0

                    # 构造tc step的输入惩罚
                    # loss += 0.0000000000000000001 * torch.matmul(opt_U[0, i:i + 1, :].to('cuda:0'),torch.transpose(opt_U[0, i:i + 1, :].to('cuda:0'), 0, 1))


                # model prediction
                Aug_Lat_Input=torch.permute(torch.cat((latent_new.to('cuda:0'),inputs.to('cuda:0')),dim=2),(0,2,1))
                # print(Aug_Lat_Input)
                # print(model.K_B)
                latent_new=torch.permute(torch.bmm(model.K_B,Aug_Lat_Input.type(torch.float32)),(0,2,1))
                # Transform back into the original state space
                Tp_prediction = model.decode(latent_new[:,:,-1*model.latent_dim:])

                deltaY = Y_ref.to('cuda:0') - Tp_prediction[0, :, :]

                # 构造tp step的预测误差
                loss += torch.matmul(deltaY, torch.transpose(deltaY, 0, 1))
                # print('error', torch.matmul(deltaY, torch.transpose(deltaY, 0, 1)))
            optimizer = torch.optim.Adam({opt_U}, lr=0.1)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        


            with torch.no_grad():
                # print(opt_U.shape)
                for k in range(opt_U[0,0,:].size()[0]):
                    opt_U[:,:,k].clamp_(min_input[k], max_input[k])

            # print(opt_U)
                for i in range(opt_U.shape[0]):
                    for j in range(opt_U.shape[1]):
                        for m in [0,2]:
                            if opt_U[i,j,m] < minimize_hz:
                                opt_U[i,j,m]=0

            if (steps%100==0):
                print('Iters:',steps,' Loss:',loss.detach().cpu().numpy()[0,0])#,'opt_u:',opt_U)


        ########################################
        ######### Traditional Method############
        ########################################
        # # Init Preparation
        # basis_num = model.latent_dim*model.ar_order
        # C_matrix = np.zeros((basis_num,2*basis_num))
        # C_matrix[:,basis_num:2*basis_num]=np.eye(basis_num)
        # # MPC coding
        # # Linear Model
        # koop_mat=model.K_B[0,:basis_num,:basis_num].detach().cpu().numpy()
        # B=model.K_B[0,:basis_num,basis_num:].detach().cpu().numpy()
        
        # A_matrix = np.zeros((2*basis_num,2*basis_num))
        # A_matrix[:basis_num,:basis_num]=koop_mat #np.transpose()
        # A_matrix[basis_num:2*basis_num,:basis_num]=koop_mat #np.transpose()
        # A_matrix[basis_num:2*basis_num,basis_num:2*basis_num]=np.eye(basis_num)
        # A_size = 2*basis_num

        # B_matrix = np.zeros((2*basis_num,model.ext_input_dim))
        # B_matrix[:basis_num,:]=B
        # B_matrix[basis_num:,:]=B
        
        # F = np.zeros(shape=(mpc_pred_len*basis_num,A_size))
        # Phi = np.zeros(shape=(mpc_pred_len*basis_num,mpc_pred_len*model.ext_input_dim))
        
        # x_k = np.zeros((2*basis_num,1))
        # x_k[:basis_num,:]=delta_state  # Latent x 1
        # x_k[basis_num:2*basis_num,:]=np.reshape(previous_state.detach().cpu().numpy(),(basis_num,1)) # Latent x 1

        # for j in range(mpc_pred_len):
        #     # Construct F matrix
        #     if j == 0:
        #         F[0:basis_num,:]=np.dot(C_matrix,A_matrix)
        #     else:
        #         F[j*basis_num:(j+1)*basis_num,:]=np.dot(F[(j-1)*basis_num:j*basis_num,:],A_matrix)

        #     # Construct Phi matrix
        #     matrix = B_matrix
        #     for k in range(j,-1,-1):
        #         Phi[j*basis_num:(j+1)*basis_num,k*model.ext_input_dim:(k+1)*model.ext_input_dim] = np.dot(C_matrix,matrix)
        #         matrix = np.dot(A_matrix,matrix)

        # #Y_ref=torch.from_numpy(np.array([5,1,3]))
        # Y_ref_=torch.reshape(Y_ref.repeat(mpc_pred_len,1),(1,mpc_pred_len,Y_ref.shape[0]))

        # R_s = model.encode(Y_ref_.float().to('cuda:0')).to('cpu').detach().numpy()
        # R_s = np.reshape(R_s,(mpc_pred_len*basis_num,1))
        
        # # MPC Without Constraint
        # A_term = np.dot(np.transpose(Phi),Phi)+0.1*np.eye(model.ext_input_dim *mpc_pred_len)
        # B_term = np.dot(np.transpose(Phi),R_s-np.dot(F,x_k))

        
        # #mpc with constraint
        # C1=np.ones(model.ext_input_dim * mpc_pred_len)
        # # C2=np.tril(np.ones((mpc_pred_len,mpc_pred_len)))
        # C2T=np.zeros((model.ext_input_dim * mpc_pred_len,model.ext_input_dim * mpc_pred_len))
        # for i in range(mpc_pred_len):
        #     for j in range(i,mpc_pred_len):
        #         C2T[i*model.ext_input_dim:(i+1)*model.ext_input_dim,j*model.ext_input_dim:(j+1)*model.ext_input_dim]=np.eye(model.ext_input_dim)
        # C2=C2T.T
        
        # M=np.zeros((4*model.ext_input_dim*mpc_pred_len,mpc_pred_len*model.ext_input_dim))
        # M[:(mpc_pred_len*model.ext_input_dim),:]=-1*np.eye(mpc_pred_len*model.ext_input_dim)
        # M[(mpc_pred_len*model.ext_input_dim):2*(mpc_pred_len*model.ext_input_dim),:]=np.eye((mpc_pred_len*model.ext_input_dim))
        # M[2*(mpc_pred_len*model.ext_input_dim):3*(mpc_pred_len*model.ext_input_dim),:]=-1*C2
        # M[3*(mpc_pred_len*model.ext_input_dim):4*(mpc_pred_len*model.ext_input_dim),:]=C2
        
        # N=np.zeros(4*model.ext_input_dim*mpc_pred_len)
        # min_delta_input=[-2,-1,-2,-1]
        # max_delta_input=[2,1,2,1]
        # N[:(mpc_pred_len*model.ext_input_dim)]=-np.array(min_delta_input).repeat(mpc_pred_len) #5.0*np.ones((mpc_pred_len*model.ext_input_dim)) #deltaU min -10
        # N[(mpc_pred_len*model.ext_input_dim):2*(mpc_pred_len*model.ext_input_dim)]=np.array(max_delta_input).repeat(mpc_pred_len) #0.2*np.ones((mpc_pred_len*model.ext_input_dim)) #deltaU max 0.1

        # # print(C1.shape,u_prev.repeat(mpc_pred_len).shape)
        # N[2*(mpc_pred_len*model.ext_input_dim):3*(mpc_pred_len*model.ext_input_dim)]=-np.array(min_input).repeat(mpc_pred_len)+u_prev.repeat(mpc_pred_len) #28.0*np.ones((mpc_pred_len*model.ext_input_dim))+u_prev.repeat(mpc_pred_len) #U min -25
        # N[3*(mpc_pred_len*model.ext_input_dim):4*(mpc_pred_len*model.ext_input_dim)]=np.array(max_input).repeat(mpc_pred_len)-u_prev.repeat(mpc_pred_len) #10*np.ones((mpc_pred_len*model.ext_input_dim))-u_prev.repeat(mpc_pred_len) #U max
        

        # linear_constraint = LinearConstraint(M.tolist(),(-np.inf*np.ones(4*mpc_pred_len*model.ext_input_dim)).tolist(),N.tolist())
        
        # init_input_U=np.zeros((1,Control_Horizon, model.ext_input_dim))
        # init_input_U[:,:,0]=1.0*np.random.randint(-2, 2, Control_Horizon)
        # init_input_U[:,:,2]=1.0*np.random.randint(-2, 2, Control_Horizon)

        # init_input_U[:,:,1]=np.random.normal(0, 2, Control_Horizon)
        # init_input_U[:,:,3]=np.random.normal(0, 2,Control_Horizon)
        # u0 = np.reshape(init_input_U,(Control_Horizon*model.ext_input_dim))
        # # print(u0)
        # import time
        # start = time.time()
        # opt_result = minimize(objective_function,u0,method='trust-constr',
        #                         jac=Jacobian,hess=Hessian,
        #                         constraints=linear_constraint,
        #                         args=(A_term,np.reshape(B_term,mpc_pred_len*model.ext_input_dim)),tol=1e-10)
        # deltaU = opt_result.x 
        # print(deltaU)
        # end_time=time.time()
        # print('time consumed:',end_time-start)
        # # return deltaU+, end_time-start
        # opt_U=np.zeros((1,Control_Horizon, model.ext_input_dim))
        # opt_U[:,0,:]=deltaU[:model.ext_input_dim]+u_prev
        


        # for m in [0,2]:
        #     if opt_U[0,0,m]<4:
        #         opt_U[0,0,m]=0

        # for i in range(1,mpc_pred_len):
        #     opt_U[0,i,:]=opt_U[0,(i-1),:]+deltaU[i*(model.ext_input_dim):(i+1)*model.ext_input_dim]
        #     for m in [0,2]:
        #         if opt_U[0,i,m]<4:
        #             opt_U[0,i,m]=0

    return opt_U

def generate_stim(previous_state,
                    model_params,
                    stim_series,
                    checkpoint,
                    N_trials,
                    Y_ref,                    # the desired output of each observation channel
                    sample_fre,
                    output_fre = 10,          
                    Pred_Horizon=20,          # prediction horizon
                    Control_Horizon=10,       # Control horizon
                    iters=1000,               # the number of iteration during optimization
                    min_input=[0,0.3,0,0.3],
                    max_input=[5, 1, 5, 1],   # input constraints
                    minimize_hz=4,
                    index_=1                  # run_index
                    ):
    # previous_state channels*T
    model = Latent_Model.load_from_checkpoint(checkpoint_path=checkpoint, params=model_params).to('cuda:0')
    opt_Us = np.zeros((N_trials, Control_Horizon, model.ext_input_dim))
    pre_X = torch.zeros((1, 1, model.data_len * model.input_dim))

    for i in range(N_trials):
        delta_state=None
        u_prev=None
        length = len(previous_state.T)
        print('previous_state:',previous_state.shape)
        if not model.RNN_Type=='Koopman':
            
            pre_X = torch.reshape(torch.from_numpy(previous_state.T[length - model.data_len:, :].astype(np.float32)),
                                (1, 1, model.data_len * model.input_dim))
            print(pre_X.shape)

            Ext_IN = torch.zeros((1, 1, (model.data_len - 1) * model.ext_input_dim))  # Previous_Input
            previous_state = torch.cat((pre_X, Ext_IN), dim=2).to('cuda:0')  # ?

        else:
            ################
            ## Koopman #####
            ################
            length = stim_series.shape[0]-20
            previous_state_ = previous_state[:,:(length+1)]
            last_state =  torch.reshape(torch.from_numpy(previous_state[:,-1].astype(np.float32)),(1, 1, model.input_dim)).to('cuda:0')

            Previous_X = torch.reshape(torch.from_numpy(previous_state_.T[length - model.data_len-1:length-1, :].astype(np.float32)),
                                            (1, model.data_len, model.input_dim)).to('cuda:0')
            Previous_Y = torch.reshape(torch.from_numpy(previous_state_.T[length - model.data_len:length, :].astype(np.float32)),
                                            (1, model.data_len, model.input_dim)).to('cuda:0')

            koop_latent_X=model.encode(Previous_X.type(torch.float32))
            koop_latent_Y=model.encode(Previous_Y.type(torch.float32))

            
            Previous_Input=torch.zeros([1,model.data_len,model.ext_input_dim])
            Previous_Input[0,:,:]=torch.from_numpy(stim_series)[(length-model.data_len):length,:]
            Aug_Lat_Input=torch.permute(torch.cat((koop_latent_X,Previous_Input.to('cuda:0')),dim=2),(0,2,1))  # [Latent_X; Ext_in]  batch*len*(latent_dim*ar_order+ext_in_dim)
            model.K_B=torch.bmm(torch.permute(koop_latent_Y.type(torch.float32),(0,2,1)),model.batch_pinv(Aug_Lat_Input.type(torch.float32),model.L_factors)) #   (latent_dim*ar_order+ext_in_dim) * latent_dim


            delta_state=torch.permute(koop_latent_Y[0,(model.data_len-1):,model.latent_dim*(model.ar_order-1):]-koop_latent_Y[0,(model.data_len-2):(model.data_len-1),model.latent_dim*(model.ar_order-1):],(1,0)).to('cpu').detach().numpy()
            cur_state=torch.permute(koop_latent_Y[0,(model.data_len-1):,model.latent_dim*(model.ar_order-1):],(1,0)).to('cpu').detach().numpy()
            
            previous_state=model.encode(last_state.type(torch.float32))
            u_prev=np.array(stim_series[-1,:]) #需要设定
            print(u_prev)
        

        opt_U = MPC_optimization(model, Pred_Horizon, Control_Horizon, torch.from_numpy(Y_ref), previous_state, delta_state, u_prev, iters=iters, min_input=min_input,
                                max_input=max_input,minimize_hz=minimize_hz)
        if type(opt_U) is np.ndarray:
            opt_Us[i, :, :] = opt_U                     
        else:
            opt_Us[i, :, :] = opt_U.detach().cpu().numpy()

    average_U = np.mean(opt_Us, axis=0)
    # print(np.round(average_U, 2))
    print(average_U)
    from datetime import datetime
    dateNum = datetime.now().strftime("%m-%d-%H-%M-%S")  # 跨天训练可能会保存到两个文件夹

    scio.savemat(r'./data/%s_Sampling_%dHz_Stim_Parameter.mat' % (dateNum,sample_fre),
                    {'reference':Y_ref*3,
                     'stim1_Freq': np.transpose(average_U)[0,:],
                     'stim1_Amp': np.transpose(average_U)[1,:]*1000,
                     'stim2_Freq': np.transpose(average_U)[2,:],
                     'stim2_Amp': np.transpose(average_U)[3,:]*1000})

    return average_U.T

# def generate_stim(previous_state,
#                     model_params,
#                     checkpoint,
#                     N_trials,
#                     Y_ref,                    # the desired output of each observation channel
#                     sample_fre,
#                     output_fre = 10,          
#                     Pred_Horizon=20,          # prediction horizon
#                     Control_Horizon=10,       # Control horizon
#                     iters=1000,               # the number of iteration during optimization
#                     max_input=[5, 1, 5, 1],   # input constraints
#                     index_=1                  # run_index
#                     ):
#     # previous_state channels*T
#     model = Latent_Model.load_from_checkpoint(checkpoint_path=checkpoint, params=model_params).to('cuda:0')
#     opt_Us = np.zeros((N_trials, Control_Horizon, model.ext_input_dim))
#     pre_X = torch.zeros((1, 1, model.data_len * model.input_dim))
#     # len = len(previous_state)
#     # pre_X += torch.reshape(torch.from_numpy(previous_state[len - model.data_len:, :].astype(np.float32)),
#     #                        (1, 1, model.data_len * model.input_dim))

#     # len = len(previous_state.T)
#     # pre_X += torch.reshape(torch.from_numpy(previous_state.T[len-model.data_len:,:].astype(np.float32)),(1, 1, model.data_len * model.input_dim))
#     for i in range(N_trials):
#         # random_index = np.random.randint(0, 1000)
#         length = len(previous_state.T)
#         # print(previous_state.shape)
#         # print(model.data_len,model.input_dim)
#         pre_X = torch.reshape(torch.from_numpy(previous_state.T[length - model.data_len:, :].astype(np.float32)),
#                                (1, 1, model.data_len * model.input_dim))
#         print(pre_X.shape)

#         Ext_IN = torch.zeros((1, 1, (model.data_len - 1) * model.ext_input_dim))  # Previous_Input
#         previous_state = torch.cat((pre_X, Ext_IN), dim=2).to('cuda:0')  # ?
#         # print(previous_state.shape)

#         current_input = Ext_IN
#         # deltaU=torch.nn.parameter.Parameter(torch.zeros((1,Control_Horizon,model.ext_input_dim),dtype=torch.float32),requires_grad=True)
#         # Y_ref = torch.ones((1, model.input_dim)) * amplitude
#         opt_U = MPC_optimization(model, Pred_Horizon, Control_Horizon, torch.from_numpy(Y_ref), previous_state, iters=iters, min_input=.1,
#                                  max_input=max_input)
#         opt_Us[i, :, :] = opt_U.detach().cpu().numpy()

#     average_U = np.mean(opt_Us, axis=0)
#     print(np.round(average_U, 2))

#     # down_sample_rate = sample_fre // output_fre
#     # average_U_downsample = np.zeros([Control_Horizon // down_sample_rate, 2])
#     # for i in range(Control_Horizon // down_sample_rate):
#     #     average_U_downsample[i, 0] = np.mean(average_U[i * down_sample_rate:(i + 1) * down_sample_rate, 0])
#     #     average_U_downsample[i, 1] = np.mean(average_U[i * down_sample_rate:(i + 1) * down_sample_rate, 1])

#     # for i in range(2):
#     #     scio.savemat(r'./data/1220_Ref%d_%dHz_stimAmplitude%d.mat'%(amplitude,sample_fre,i+1),
#     #                  {'stimAmplitude':np.transpose(average_U)[i,:]*1000})
    
#     from datetime import datetime
#     dateNum = datetime.now().strftime("%m-%d-%H:%M")  # 跨天训练可能会保存到两个文件夹

#     scio.savemat(r'./data/%s_Sampling_50Hz_Stim_Parameter.mat' % (dateNum),
#                     {'reference':Y_ref,
#                      'stim1_Freq': np.transpose(average_U)[0,:],
#                      'stim1_Amp': np.transpose(average_U)[1,:]*1000,
#                      'stim2_Freq': np.transpose(average_U)[2,:],
#                      'stim2_Amp': np.transpose(average_U)[3,:]*1000})

#     return average_U.T















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
# model_params={
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
# model=GRU.load_from_checkpoint(checkpoint_path=checkpoint,params=model_params).to('cuda:0')
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
#     previous_state = torch.cat((pre_X, Ext_IN), dim=2).to('cuda:0')     #?
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