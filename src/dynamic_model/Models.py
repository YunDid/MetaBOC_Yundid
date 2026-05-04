from os import X_OK
from tkinter.tix import Y_REGION
from turtle import forward
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from src.dynamic_model.utils import EI_Sparse_Mask_


import math
class EI_RNN(nn.Module):
    def __init__(self, 
                 input_size,    #  
                 hidden_size,   # hidden dimension
                 output_size,   # observe dimension
                 fr_type=False, # transform neural state to neural firing rate by r(t)=relu(x(t))
                 ei_ratio=4,  # the ratio between the size of excitatory neurons and inhibitory neurons.
                 sparsity=0,
                 e_clusters=2,
                 i_clusters=1,
                 inter_or_intra='Inter',
                 output_layers=2,
                 with_Tanh=True, 
                 device='cpu'):
        super(EI_RNN, self).__init__()

        self.fr_type=fr_type

        self.hidden_size = hidden_size
        self.u = nn.Linear(input_size, hidden_size)
        
        self.B = nn.Parameter(torch.empty((hidden_size,input_size)),requires_grad=True)
        self.B_bias = nn.Parameter(torch.empty((hidden_size,1)),requires_grad=True)
        
        nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.B)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.B_bias, -bound, bound)
        
        self.device=device
        probability=1.0*ei_ratio/(ei_ratio+1)
        
        self.C1 = nn.Parameter(torch.empty((int(hidden_size/2),hidden_size)),requires_grad=True)
        self.C1_bias = nn.Parameter(torch.empty((int(hidden_size/2),1)),requires_grad=True)
        self.W_rec_fix=EI_Sparse_Mask_(hidden_dim=hidden_size,e_clusters=e_clusters,i_clusters=i_clusters,sparsity=sparsity,ei_ratio=4,module_type=inter_or_intra).to(torch.device(device))
        
        self.C2 = nn.Parameter(torch.empty((hidden_size,hidden_size)),requires_grad=True)
        self.C2_bias = nn.Parameter(torch.empty((hidden_size,1)),requires_grad=True)
        
        ones_tensor=torch.ones(int(hidden_size*probability),device=device)
        minus_ones_tensor=-1*torch.ones(hidden_size-int(hidden_size*probability),device=device)
        self.ei_constraints=torch.diag(torch.cat((ones_tensor,minus_ones_tensor)))

        self.w_rec = nn.Parameter(torch.empty((hidden_size,hidden_size)),requires_grad=True)
        self.bias_rec = nn.Parameter(torch.empty((hidden_size,1)),requires_grad=True)

        nn.init.kaiming_uniform_(self.w_rec, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_rec)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias_rec, -bound, bound)
        
        self.w_rec_diag0=torch.ones((hidden_size,hidden_size),device=device)
        # self.w_rec_diag0.fill_diagonal_(0)
        
        self.output_layers=output_layers

        self.with_Tanh=with_Tanh

        # if self.output_layers==2:
        #     self.v1 = nn.Linear(hidden_size, int(hidden_size/2))
        #     self.v2 = nn.Linear(int(hidden_size/2), output_size)
        # else:
        # self.v2 = nn.Linear(hidden_size, output_size)
        self.C = nn.Parameter(torch.empty((output_size,hidden_size)),requires_grad=True)
        self.C_bias = nn.Parameter(torch.empty((output_size,1)),requires_grad=True)
        
        nn.init.kaiming_uniform_(self.C, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.C)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.C_bias, -bound, bound)
            
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.softplus=nn.Softplus()
        self.softmax = nn.LogSoftmax(dim=1)

    def get_w_rec_fix(self):
        return self.W_rec_fix.detach().cpu().numpy()
        
    def set_w_rec_fix(self,w_rec_fix):
        self.W_rec_fix=torch.from_numpy(w_rec_fix).to(torch.device(self.device)).type(torch.float32)
        
    def get_w_rec(self):
        return torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints)))
        
    def forward(self, inputs, hidden):
        u_x = self.u(inputs)
        # print(u_x.shape)
        # print(inputs.shape,hidden.shape)
        hidden_ = torch.permute(hidden,(1,2,0))
        inputs_ = torch.permute(inputs,(1,2,0))
        # print(inputs_.shape,hidden_.shape)
        # print(torch.matmul(self.B,inputs_).shape)
        u_x=torch.matmul(self.B,inputs_)+self.B_bias
        # print(u_x.shape)
        if self.fr_type:
            #h2h= torch.matmul(self.relu(hidden),torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints)).T))+self.bias_rec
            h2h= torch.matmul(torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints))),self.relu(hidden_))+self.bias_rec
        else:
            #h2h = torch.matmul(hidden,torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints)).T))+self.bias_rec
            h2h= torch.matmul(torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints))),hidden_)+self.bias_rec
        if self.with_Tanh:
            hidden_new = self.tanh(h2h + u_x)
        else:
            hidden_new = h2h + u_x
            
        #hidden=hidden_new # self.dt*hidden_new+self.minusdt*hidden
        # output=None
        # if self.output_layers==1:
        # print(self.C.shape,hidden_new.shape)
        output = self.softplus(torch.matmul(self.C,self.relu(hidden_new))+self.C_bias)
        # print(output.shape,hidden_new.shape)
        # else:
        #     output=self.softplus(self.v2(self.relu(self.v1(self.relu(hidden)))))# 2-layer outputs
        return torch.permute(output,(2,0,1)), torch.permute(hidden_new,(2,0,1))


class Latent_Model(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.Tp = params['Tp']  # Tp step Prediction
        self.input_dim = int(params['IN_DIM'])
        self.hidden_dim = int(params['HIDDEN_DIM'])  # Mid Dimension
        self.latent_dim = int(params['LATENT_DIM'])  # Approximate High Dimension
        self.X_recon = params['X_recon']
        self.Y_recon = params['Y_recon']
        self.Tp_recon = params['Tp_recon']
        self.Linear_Loss = params['Linear_Loss']
        self.data_len = params['data_length']
        self.with_input = params['with_input']  # With external input or not
        self.StateDependent = params['StateDependent']  # The input matrix is a constant or state dependent?
        self.L_factors = params['L_factors']
        self.ar_order = params['AR_order']
        self.ext_input_dim = params['ext_input_dim']
        self.run_device = params['device']
        self.l2_reg=params['l2_norm']
        self.learning_rate = params['learning_rate']
        # self.weight_decay = params['weight_decay']
        
        if params['activation']=='ReLU':
            self.activate=nn.ReLU()
        elif  params['activation']=='Tanh':
            self.activate=nn.Tanh()
        elif params['activation']=='Softplus':
            self.activate=nn.Softplus()
            
        self.l1_reg = params['l1_reg']

        self.sparsity=params['sparsity']
            
        self.e_clusters=params['e_clusters']
        self.i_clusters=params['i_clusters']
        self.inter_or_intra=params['inter_or_intra']
        
        if params['final_activation']=='ReLU':
            self.final_activate=nn.ReLU()
        elif params['final_activation']=='Softplus':
            self.final_activate=nn.Softplus()

        self.rnn_cell=None
        self.RNN_Type=params['RNN_Type']

        # Koopman Operator
        self.K = None
        self.K_inv = None
        self.B = None
        self.K_B = None

        self.iters = 0

        # self.encode = nn.Sequential(
        #     torch.nn.Linear(self.data_len * (self.input_dim) + (self.data_len - 1) * (self.ext_input_dim),
        #                     self.latent_dim),
        #     self.activate,
        # )

        self.encode = nn.Sequential(
                    torch.nn.Linear(self.input_dim,self.latent_dim),
                    # self.activate,
                    torch.nn.Tanh(),
                    torch.nn.Linear(self.latent_dim,self.latent_dim),
                    # self.activate,
                    # torch.nn.Tanh(),
                    # torch.nn.Linear(self.latent_dim,self.latent_dim),
                )
        print('input size:',self.data_len * (self.input_dim) + (self.data_len - 1) * (self.ext_input_dim),' ',self.latent_dim)


        if not self.RNN_Type=='EI-RNN':
            self.decode = nn.Sequential(
                torch.nn.Linear(self.latent_dim, int(self.latent_dim / 2)),
                self.activate,
                torch.nn.Linear(int(self.latent_dim / 2), self.input_dim),
                self.final_activate
            )


        if self.RNN_Type=='GRU':
            self.rnn_cell = torch.nn.GRU(input_size=self.ext_input_dim, hidden_size=self.latent_dim, num_layers=1,
                                    batch_first=True)
        elif self.RNN_Type=='RNN':
            self.rnn_cell = torch.nn.RNN(input_size=self.ext_input_dim, hidden_size=self.latent_dim, num_layers=1,
                                    batch_first=True)
        elif self.RNN_Type=='LSTM':
            self.rnn_cell = torch.nn.LSTM(input_size=self.ext_input_dim, hidden_size=self.latent_dim, num_layers=1,
                                    batch_first=True)
        elif self.RNN_Type=='EI-RNN':
            self.ei_ratio=params['EI_ratio']
            self.rnn_cell=EI_RNN(input_size=self.ext_input_dim, 
                                 hidden_size=self.latent_dim, 
                                 output_size=self.input_dim, 
                                 output_layers=1,
                                 fr_type=True, 
                                 ei_ratio=self.ei_ratio,
                                 sparsity=self.sparsity,
                                 e_clusters=self.e_clusters,
                                 i_clusters=self.i_clusters,
                                 inter_or_intra=self.inter_or_intra,
                                 with_Tanh=params['with_Tanh'],
                                 device=self.run_device)
            print('run_device:',self.run_device)
            
        elif self.RNN_Type=='Koopman':
            self.encode = nn.Sequential(
                                        nn.Linear(self.input_dim,self.hidden_dim),
                                        self.activate,
                                        nn.Linear(self.hidden_dim,self.hidden_dim),
                                        self.activate,
                                        nn.Linear(self.hidden_dim,self.hidden_dim),
                                        self.activate,
                                        nn.Linear(self.hidden_dim,self.latent_dim)
                                    )
            self.decode = nn.Sequential(
                                        torch.nn.Linear(self.latent_dim, self.hidden_dim),
                                        self.activate,
                                        torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                                        self.activate,
                                        torch.nn.Linear(self.hidden_dim, self.hidden_dim),
                                        self.activate,
                                        torch.nn.Linear(self.hidden_dim, self.input_dim),
                                        self.final_activate
                                    )

    # @staticmethod
    def batch_pinv(self,x, I_factor,device):
        """
        :param x: B x N x D (N > D)
        :param I_factor:
        :return:
        """
        B, N, D = x.size()
        if N < D:
            x = torch.transpose(x, 1, 2)
            N, D = D, N
            trans = True
        else:
            trans = False

        x_t = torch.transpose(x, 1, 2)

        # use_gpu = torch.cuda.is_available()
        I = torch.eye(D)[None, :, :].repeat(B, 1, 1).to(torch.device(device))
        # if use_gpu:
        #     I = I.to('cuda:0')

        x_pinv = torch.bmm(
            torch.inverse(torch.bmm(x_t, x) + I_factor * I),
            x_t
        )

        if trans:
            x_pinv = torch.transpose(x_pinv, 1, 2)

        return x_pinv

        

    def training_step(self, input):
        loss=0
        
        if not self.RNN_Type == 'Koopman':
            # print(X.shape,ext_in.shape)
            X, ext_in, start_index = input  # Batch x (Length+Tp+ar) x origin_dim
            batch, length, ori_dim = X.size()

            # Data Init Position
            init_pos = length - self.Tp - self.data_len

            X_ = torch.reshape(X[:, init_pos:init_pos + self.data_len, :], (batch, 1, self.data_len * self.input_dim))
            Ext_IN = torch.reshape(ext_in[:, :(self.data_len - 1), :],
                                (batch, 1, (self.data_len - 1) * self.ext_input_dim))  # Previous_Input
            previous_state = torch.cat((X_, Ext_IN), dim=2)
        
            hidden_init = torch.permute(self.encode(previous_state),
                                        (1, 0, 2))  # Batch X 1 X H_dim -> 1 X Batch X H_dim

            Tp_prediction = torch.zeros((batch, self.Tp, self.input_dim), device=self.run_device)
            output=None
            for i in range(self.Tp):
                inputs = ext_in[:, (self.data_len - 1 + i):(self.data_len + i), :]
                if self.RNN_Type=='LSTM':
                    output, (hidden_init, hidden_init) = self.rnn_cell(inputs, (hidden_init, hidden_init))
                elif self.RNN_Type=='EI-RNN':
                    output, hidden_init = self.rnn_cell(torch.permute(inputs,(1,0,2)), hidden_init)
                else:
                    output, hidden_init = self.rnn_cell(inputs, hidden_init)

                if self.RNN_Type=='EI-RNN':
                    Tp_prediction[:, i:i + 1, :] = torch.permute(output,(1, 0, 2))
                else:
                    Tp_prediction[:, i:i + 1, :] = (self.decode(torch.permute(hidden_init, (1, 0, 2))))
            loss_fun = torch.nn.MSELoss(reduction='mean')
            loss = loss_fun(X[:, (init_pos + self.data_len):(init_pos + self.data_len + self.Tp), :], Tp_prediction)

            if self.RNN_Type=='EI-RNN':
                loss+=self.l2_reg*(torch.norm(self.rnn_cell.get_w_rec(),2)**2+torch.norm(self.rnn_cell.bias_rec,2)**2)
                
                # if self.l1_reg:
                #     loss+=self.lambda_penalty*(torch.norm(self.rnn_cell.get_w_rec(),1)+torch.norm(self.rnn_cell.bias_rec,1))
        else:

            X,ext_in,start_index = input  # Batch x (Length+Tp+ar) x origin_dim
            #print(X.shape,ext_in.shape)
            batch,length,ori_dim=X.size()
        
            
            ############################
            ### Latent Augmentation ####
            ############################
            latent_X=torch.zeros((batch,self.data_len,self.latent_dim*self.ar_order),dtype=torch.float32,device=self.run_device)
            latent_Y=torch.zeros((batch,self.data_len,self.latent_dim*self.ar_order),dtype=torch.float32,device=self.run_device)

            
            for i in range(self.ar_order):
                latent_X[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,i:(i+self.data_len),:])
                latent_Y[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,(i+1):(i+1+self.data_len),:])
                #latent_Z[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,(i+self.Tp):(i+self.Tp+self.data_len),:])

            X_=X[:,(self.ar_order-1):(self.ar_order-1+self.data_len),:] # Current_time=self.ar_order-1
            Y_=X[:,(self.ar_order):(self.ar_order+self.data_len),:]
            
            # Input data
            Ext_IN=ext_in[:,(self.ar_order-1):(self.ar_order-1+self.data_len),:] # Current_Input

            recon_X=self.decode(latent_X[:,:,((self.ar_order-1)*self.latent_dim):]) # Xk
            #print(X_.shape,recon_X.shape)
            OneStep_Prediction=latent_X
            TpStep_Prediciton=latent_X
            
            """
            :Latent_X: Batch x length x latent_dim
            :Aug_Lat_Input: Batch x augment_dim x length
            """
            # augmentation.
            Aug_Lat_Input=torch.permute(torch.cat((latent_X,Ext_IN),dim=2),(0,2,1))  # [Latent_X; Ext_in]  batch*len*(latent_dim*ar_order+ext_in_dim)
            
            
            """
            :self.K_B: batch x latent_dim x augment_dim
            """
            # update K and B
            self.K_B=torch.bmm(torch.permute(latent_Y,(0,2,1)),self.batch_pinv(Aug_Lat_Input,self.L_factors,self.run_device)) #   (latent_dim*ar_order+ext_in_dim) * latent_dim
            
            self.K=self.K_B[:,:,:self.latent_dim*self.ar_order]
            self.B=self.K_B[:,:,self.latent_dim*self.ar_order:]

            # Loss Definition
            loss_fun=torch.nn.MSELoss(reduction='mean')
            # Reconstruction loss
            loss_X_recon=loss_fun(X_,recon_X)                  # Train Encoder and Decoder
            loss_Linear=0
            loss_Prediction=0
            lambda_weight=1

            # Tp Step Prediction Equation
            OneStep_Prediction = latent_Y[:,-1:,:] # Current State is the last step of latent_Y
            for tp in range(self.Tp):
                Aug_Lat_Input=torch.permute(torch.cat((OneStep_Prediction,ext_in[:,(self.data_len+self.ar_order-1+tp):(self.data_len+self.ar_order+tp),:]),dim=2),(0,2,1))   # [Latent_X; Ext_in]  batch*len*(latent_dim*ar_order+ext_in_dim)
                OneStep_Prediction=torch.permute(torch.bmm(self.K_B,Aug_Lat_Input),(0,2,1))
                Z_=X[:,(self.ar_order+self.data_len+tp):(self.ar_order+self.data_len+tp+1),:] # Original of Next State
                TpStep_Recon = self.decode(OneStep_Prediction[:,:,((self.ar_order-1)*self.latent_dim):])  # Latent representation of Next State and transform back into original space.
                
                # prediction reconstruction loss
                loss_Prediction+=pow(lambda_weight,tp)*loss_fun(Z_,TpStep_Recon)            # Train Encoder and Decoder and Koopman_Operator
            
            loss=self.X_recon*loss_X_recon+self.Tp_recon*loss_Prediction

        return loss

    def test_step(self, input, index):

        loss=0
        
        if not self.RNN_Type == 'Koopman':
            # print(X.shape,ext_in.shape)
            X, ext_in, start_index = input  # Batch x (Length+Tp+ar) x origin_dim
            batch, length, ori_dim = X.size()

            # Data Init Position
            init_pos = length - self.Tp - self.data_len

            X_ = torch.reshape(X[:, init_pos:init_pos + self.data_len, :], (batch, 1, self.data_len * self.input_dim))
            Ext_IN = torch.reshape(ext_in[:, :(self.data_len - 1), :],
                                (batch, 1, (self.data_len - 1) * self.ext_input_dim))  # Previous_Input
            previous_state = torch.cat((X_, Ext_IN), dim=2)
        
            hidden_init = torch.permute(self.encode(previous_state),
                                        (1, 0, 2))  # Batch X 1 X H_dim -> 1 X Batch X H_dim

            Tp_prediction = torch.zeros((batch, self.Tp, self.input_dim), device=self.run_device)
            output=None
            for i in range(self.Tp):
                inputs = ext_in[:, (self.data_len - 1 + i):(self.data_len + i), :]
                if self.RNN_Type=='LSTM':
                    output, (hidden_init, hidden_init) = self.rnn_cell(inputs, (hidden_init, hidden_init))
                elif self.RNN_Type=='EI-RNN':
                    output, hidden_init = self.rnn_cell(torch.permute(inputs,(1,0,2)), hidden_init)
                else:
                    output, hidden_init = self.rnn_cell(inputs, hidden_init)

                if self.RNN_Type=='EI-RNN':
                    Tp_prediction[:, i:i + 1, :] = torch.permute(output,(1, 0, 2))
                else:
                # print(hidden_init.shape)
                    Tp_prediction[:, i:i + 1, :] = torch.permute(output,(1, 0, 2))# (self.decode(torch.permute(hidden_init, (1, 0, 2))))
            loss_fun = torch.nn.MSELoss(reduction='mean')
            # loss_X_recon=loss_fun(X_,recon_X)
            loss = loss_fun(X[:, (init_pos + self.data_len):(init_pos + self.data_len + self.Tp), :], Tp_prediction)
        else:

            X,ext_in,start_index = input  # Batch x (Length+Tp+ar) x origin_dim
            #print(X.shape,ext_in.shape)
            batch,length,ori_dim=X.size()
        
            
            ############################
            ### Latent Augmentation ####
            ############################
            latent_X=torch.zeros((batch,self.data_len,self.latent_dim*self.ar_order),dtype=torch.float32,device=self.run_device)
            latent_Y=torch.zeros((batch,self.data_len,self.latent_dim*self.ar_order),dtype=torch.float32,device=self.run_device)

            
            for i in range(self.ar_order):
                latent_X[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,i:(i+self.data_len),:])
                latent_Y[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,(i+1):(i+1+self.data_len),:])
                #latent_Z[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,(i+self.Tp):(i+self.Tp+self.data_len),:])

            X_=X[:,(self.ar_order-1):(self.ar_order-1+self.data_len),:] # Current_time=self.ar_order-1
            Y_=X[:,(self.ar_order):(self.ar_order+self.data_len),:]
            
            # Input data
            Ext_IN=ext_in[:,(self.ar_order-1):(self.ar_order-1+self.data_len),:] # Current_Input

            recon_X=self.decode(latent_X[:,:,((self.ar_order-1)*self.latent_dim):]) # Xk
            #print(X_.shape,recon_X.shape)
            OneStep_Prediction=latent_X
            TpStep_Prediciton=latent_X
            
            """
            :Latent_X: Batch x length x latent_dim
            :Aug_Lat_Input: Batch x augment_dim x length
            """
            # augmentation.
            Aug_Lat_Input=torch.permute(torch.cat((latent_X,Ext_IN),dim=2),(0,2,1))  # [Latent_X; Ext_in]  batch*len*(latent_dim*ar_order+ext_in_dim)
            
            
            """
            :self.K_B: batch x latent_dim x augment_dim
            """
            # update K and B
            self.K_B=torch.bmm(torch.permute(latent_Y,(0,2,1)),self.batch_pinv(Aug_Lat_Input,self.L_factors)) #   (latent_dim*ar_order+ext_in_dim) * latent_dim
            
            self.K=self.K_B[:,:,:self.latent_dim*self.ar_order]
            self.B=self.K_B[:,:,self.latent_dim*self.ar_order:]

            # Loss Definition
            loss_fun=torch.nn.MSELoss(reduction='mean')
            # Reconstruction loss
            loss_X_recon=loss_fun(X_,recon_X)                  # Train Encoder and Decoder
            loss_Linear=0
            loss_Prediction=0
            lambda_weight=1

            #if self.with_input:
            # Tp Step Prediction Equation
            OneStep_Prediction = latent_Y[:,-1:,:] # Current State is the last step of latent_Y
            for tp in range(self.Tp):
                Aug_Lat_Input=torch.permute(torch.cat((OneStep_Prediction,ext_in[:,(self.data_len+self.ar_order-1+tp):(self.data_len+self.ar_order+tp),:]),dim=2),(0,2,1))   # [Latent_X; Ext_in]  batch*len*(latent_dim*ar_order+ext_in_dim)
                OneStep_Prediction=torch.permute(torch.bmm(self.K_B,Aug_Lat_Input),(0,2,1))
                Z_=X[:,(self.ar_order+self.data_len+tp):(self.ar_order+self.data_len+tp+1),:] # Original of Next State
                TpStep_Recon = self.decode(OneStep_Prediction[:,:,((self.ar_order-1)*self.latent_dim):])  # Latent representation of Next State and transform back into original space.
                
                # prediction reconstruction loss
                loss_Prediction+=pow(lambda_weight,tp)*loss_fun(Z_,TpStep_Recon)            # Train Encoder and Decoder and Koopman_Operator
            
            print('X_recon_error:',loss_X_recon,'Tp_error:',loss_Prediction)
            loss=self.X_recon*loss_X_recon+self.Tp_recon*loss_Prediction

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)#,weight_decay=self.weight_decay)





# from os import X_OK
# from tkinter.tix import Y_REGION
# from turtle import forward
# import torch
# from torch.nn import functional as F
# from torch import nn
# from pytorch_lightning import LightningModule
# # from pytorch_lightning.core.lightning import LightningModule
# from src.dynamic_model.utils import EI_Sparse_Mask_


# import math
# class EI_RNN(nn.Module):
#     def __init__(self, 
#                  input_size,    #  
#                  hidden_size,   # hidden dimension
#                  output_size,   # observe dimension
#                  fr_type=False, # transform neural state to neural firing rate by r(t)=relu(x(t))
#                  ei_ratio=4,  # the ratio between the size of excitatory neurons and inhibitory neurons.
#                  sparsity=0,
#                  e_clusters=2,
#                  i_clusters=1,
#                  inter_or_intra='Inter',
#                  output_layers=2,
#                  with_Tanh=True, 
#                  device='cpu'):
#         super(EI_RNN, self).__init__()

#         self.fr_type=fr_type

#         self.hidden_size = hidden_size
#         self.u = nn.Linear(input_size, hidden_size)
        
#         self.B = nn.Parameter(torch.empty((hidden_size,input_size)),requires_grad=True)
#         self.B_bias = nn.Parameter(torch.empty((hidden_size,1)),requires_grad=True)
        
#         nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.B)
#         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#         nn.init.uniform_(self.B_bias, -bound, bound)
        
#         self.device=device
#         probability=1.0*ei_ratio/(ei_ratio+1)
        
        
#         self.W_rec_fix=EI_Sparse_Mask_(hidden_dim=hidden_size,e_clusters=e_clusters,i_clusters=i_clusters,sparsity=sparsity,ei_ratio=4,module_type=inter_or_intra).to(torch.device(device))
        
#         ones_tensor=torch.ones(int(hidden_size*probability),device=device)
#         minus_ones_tensor=-1*torch.ones(hidden_size-int(hidden_size*probability),device=device)
#         self.ei_constraints=torch.diag(torch.cat((ones_tensor,minus_ones_tensor)))

#         self.w_rec = nn.Parameter(torch.empty((hidden_size,hidden_size)),requires_grad=True)
#         self.bias_rec = nn.Parameter(torch.empty((hidden_size,1)),requires_grad=True)

#         nn.init.kaiming_uniform_(self.w_rec, a=math.sqrt(5))
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_rec)
#         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#         nn.init.uniform_(self.bias_rec, -bound, bound)
        
#         self.w_rec_diag0=torch.ones((hidden_size,hidden_size),device=device)
#         self.w_rec_diag0.fill_diagonal_(0)
        
#         self.output_layers=output_layers

#         self.with_Tanh=with_Tanh

#         # if self.output_layers==2:
#         #     self.v1 = nn.Linear(hidden_size, int(hidden_size/2))
#         #     self.v2 = nn.Linear(int(hidden_size/2), output_size)
#         # else:
#         # self.v2 = nn.Linear(hidden_size, output_size)
#         self.C = nn.Parameter(torch.empty((output_size,hidden_size)),requires_grad=True)
#         self.C_bias = nn.Parameter(torch.empty((output_size,1)),requires_grad=True)
        
#         nn.init.kaiming_uniform_(self.C, a=math.sqrt(5))
#         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.C)
#         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
#         nn.init.uniform_(self.C_bias, -bound, bound)
            
#         self.tanh = nn.Tanh()
#         self.relu = nn.ReLU()
#         self.softplus=nn.Softplus()
#         self.softmax = nn.LogSoftmax(dim=1)

#     def get_w_rec_fix(self):
#         return self.W_rec_fix.detach().cpu().numpy()
        
#     def set_w_rec_fix(self,w_rec_fix):
#         self.W_rec_fix=torch.from_numpy(w_rec_fix).to(torch.device(self.device)).type(torch.float32)
        
#     def get_w_rec(self):
#         return torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints)))
        
#     def forward(self, inputs, hidden):
#         u_x = self.u(inputs)
#         # print(u_x.shape)
#         # print(inputs.shape,hidden.shape)
#         hidden_ = torch.permute(hidden,(1,2,0))
#         inputs_ = torch.permute(inputs,(1,2,0))
#         # print(inputs_.shape,hidden_.shape)
#         # print(torch.matmul(self.B,inputs_).shape)
#         u_x=torch.matmul(self.B,inputs_)+self.B_bias
#         # print(u_x.shape)
#         if self.fr_type:
#             #h2h= torch.matmul(self.relu(hidden),torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints)).T))+self.bias_rec
#             h2h= torch.matmul(torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints))),self.relu(hidden_))+self.bias_rec
#         else:
#             #h2h = torch.matmul(hidden,torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints)).T))+self.bias_rec
#             h2h= torch.matmul(torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints))),hidden_)+self.bias_rec
#         if self.with_Tanh:
#             hidden_new = self.tanh(h2h + u_x)
#         else:
#             hidden_new = h2h + u_x
            
#         #hidden=hidden_new # self.dt*hidden_new+self.minusdt*hidden
#         # output=None
#         # if self.output_layers==1:
#         # print(self.C.shape,hidden_new.shape)
#         output = self.softplus(torch.matmul(self.C,self.relu(hidden_new))+self.C_bias)
#         # print(output.shape,hidden_new.shape)
#         # else:
#         #     output=self.softplus(self.v2(self.relu(self.v1(self.relu(hidden)))))# 2-layer outputs
#         return torch.permute(output,(2,0,1)), torch.permute(hidden_new,(2,0,1))


# class Latent_Model(LightningModule):
#     def __init__(self, params):
#         super().__init__()

#         self.Tp = params['Tp']  # Tp step Prediction
#         self.input_dim = int(params['IN_DIM'])
#         self.hidden_dim = int(params['HIDDEN_DIM'])  # Mid Dimension
#         self.latent_dim = int(params['LATENT_DIM'])  # Approximate High Dimension
#         self.X_recon = params['X_recon']
#         self.Y_recon = params['Y_recon']
#         self.Tp_recon = params['Tp_recon']
#         self.Linear_Loss = params['Linear_Loss']
#         self.data_len = params['data_length']
#         self.with_input = params['with_input']  # With external input or not
#         self.StateDependent = params['StateDependent']  # The input matrix is a constant or state dependent?
#         self.L_factors = params['L_factors']
#         self.ar_order = params['AR_order']
#         self.ext_input_dim = params['ext_input_dim']
#         self.run_device = params['device']
#         self.l2_reg=params['l2_norm']
#         self.learning_rate = params['learning_rate']
#         # self.weight_decay = params['weight_decay']
        
#         if params['activation']=='ReLU':
#             self.activate=nn.ReLU()
#         elif  params['activation']=='Tanh':
#             self.activate=nn.Tanh()
#         elif params['activation']=='Softplus':
#             self.activate=nn.Softplus()
            
#         self.l1_reg = params['l1_reg']

#         self.sparsity=params['sparsity']
            
#         self.e_clusters=params['e_clusters']
#         self.i_clusters=params['i_clusters']
#         self.inter_or_intra=params['inter_or_intra']
        
#         if params['final_activation']=='ReLU':
#             self.final_activate=nn.ReLU()
#         elif params['final_activation']=='Softplus':
#             self.final_activate=nn.Softplus()

#         self.rnn_cell=None
#         self.RNN_Type=params['RNN_Type']

#         # Koopman Operator
#         self.K = None
#         self.K_inv = None
#         self.B = None
#         self.K_B = None

#         self.iters = 0

#         self.encode = nn.Sequential(
#             torch.nn.Linear(self.data_len * (self.input_dim) + (self.data_len - 1) * (self.ext_input_dim),
#                             self.latent_dim),
#             self.activate,
#         )
#         print('input size:',self.data_len * (self.input_dim) + (self.data_len - 1) * (self.ext_input_dim),' ',self.latent_dim)


#         if not self.RNN_Type=='EI-RNN':
#             self.decode = nn.Sequential(
#                 torch.nn.Linear(self.latent_dim, int(self.latent_dim / 2)),
#                 self.activate,
#                 torch.nn.Linear(int(self.latent_dim / 2), self.input_dim),
#                 self.final_activate
#             )


#         if self.RNN_Type=='GRU':
#             self.rnn_cell = torch.nn.GRU(input_size=self.ext_input_dim, hidden_size=self.latent_dim, num_layers=1,
#                                     batch_first=True)
#         elif self.RNN_Type=='RNN':
#             self.rnn_cell = torch.nn.RNN(input_size=self.ext_input_dim, hidden_size=self.latent_dim, num_layers=1,
#                                     batch_first=True)
#         elif self.RNN_Type=='LSTM':
#             self.rnn_cell = torch.nn.LSTM(input_size=self.ext_input_dim, hidden_size=self.latent_dim, num_layers=1,
#                                     batch_first=True)
#         elif self.RNN_Type=='EI-RNN':
#             self.ei_ratio=params['EI_ratio']
#             self.rnn_cell=EI_RNN(input_size=self.ext_input_dim, 
#                                  hidden_size=self.latent_dim, 
#                                  output_size=self.input_dim, 
#                                  output_layers=1,
#                                  fr_type=True, 
#                                  ei_ratio=self.ei_ratio,
#                                  sparsity=self.sparsity,
#                                  e_clusters=self.e_clusters,
#                                  i_clusters=self.i_clusters,
#                                  inter_or_intra=self.inter_or_intra,
#                                  with_Tanh=params['with_Tanh'],
#                                  device=self.run_device)
#             print('run_device:',self.run_device)
            
#         elif self.RNN_Type=='Koopman':
#             self.encode = nn.Sequential(
#                                         nn.Linear(self.input_dim,self.hidden_dim),
#                                         self.activate,
#                                         nn.Linear(self.hidden_dim,self.hidden_dim),
#                                         self.activate,
#                                         nn.Linear(self.hidden_dim,self.hidden_dim),
#                                         self.activate,
#                                         nn.Linear(self.hidden_dim,self.latent_dim)
#                                     )
#             self.decode = nn.Sequential(
#                                         torch.nn.Linear(self.latent_dim, self.hidden_dim),
#                                         self.activate,
#                                         torch.nn.Linear(self.hidden_dim, self.hidden_dim),
#                                         self.activate,
#                                         torch.nn.Linear(self.hidden_dim, self.hidden_dim),
#                                         self.activate,
#                                         torch.nn.Linear(self.hidden_dim, self.input_dim),
#                                         self.final_activate
#                                     )

#     # @staticmethod
#     def batch_pinv(self,x, I_factor,device):
#         """
#         :param x: B x N x D (N > D)
#         :param I_factor:
#         :return:
#         """
#         B, N, D = x.size()
#         if N < D:
#             x = torch.transpose(x, 1, 2)
#             N, D = D, N
#             trans = True
#         else:
#             trans = False

#         x_t = torch.transpose(x, 1, 2)

#         # use_gpu = torch.cuda.is_available()
#         I = torch.eye(D)[None, :, :].repeat(B, 1, 1).to(torch.device(device))
#         # if use_gpu:
#         #     I = I.to('cuda:0')

#         x_pinv = torch.bmm(
#             torch.inverse(torch.bmm(x_t, x) + I_factor * I),
#             x_t
#         )

#         if trans:
#             x_pinv = torch.transpose(x_pinv, 1, 2)

#         return x_pinv

        

#     def training_step(self, input):
#         loss=0
        
#         if not self.RNN_Type == 'Koopman':
#             # print(X.shape,ext_in.shape)
#             X, ext_in, start_index = input  # Batch x (Length+Tp+ar) x origin_dim
#             batch, length, ori_dim = X.size()

#             # Data Init Position
#             init_pos = length - self.Tp - self.data_len

#             X_ = torch.reshape(X[:, init_pos:init_pos + self.data_len, :], (batch, 1, self.data_len * self.input_dim))
#             Ext_IN = torch.reshape(ext_in[:, :(self.data_len - 1), :],
#                                 (batch, 1, (self.data_len - 1) * self.ext_input_dim))  # Previous_Input
#             previous_state = torch.cat((X_, Ext_IN), dim=2)
        
#             hidden_init = torch.permute(self.encode(previous_state),
#                                         (1, 0, 2))  # Batch X 1 X H_dim -> 1 X Batch X H_dim

#             Tp_prediction = torch.zeros((batch, self.Tp, self.input_dim), device=self.run_device)
#             output=None
#             for i in range(self.Tp):
#                 inputs = ext_in[:, (self.data_len - 1 + i):(self.data_len + i), :]
#                 if self.RNN_Type=='LSTM':
#                     output, (hidden_init, hidden_init) = self.rnn_cell(inputs, (hidden_init, hidden_init))
#                 elif self.RNN_Type=='EI-RNN':
#                     output, hidden_init = self.rnn_cell(torch.permute(inputs,(1,0,2)), hidden_init)
#                 else:
#                     output, hidden_init = self.rnn_cell(inputs, hidden_init)

#                 if self.RNN_Type=='EI-RNN':
#                     Tp_prediction[:, i:i + 1, :] = torch.permute(output,(1, 0, 2))
#                 else:
#                     Tp_prediction[:, i:i + 1, :] = (self.decode(torch.permute(hidden_init, (1, 0, 2))))
#             loss_fun = torch.nn.MSELoss(reduction='mean')
#             loss = loss_fun(X[:, (init_pos + self.data_len):(init_pos + self.data_len + self.Tp), :], Tp_prediction)

#             if self.RNN_Type=='EI-RNN':
#                 loss+=self.l2_reg*(torch.norm(self.rnn_cell.get_w_rec(),2)**2+torch.norm(self.rnn_cell.bias_rec,2)**2)
                
#                 # if self.l1_reg:
#                 #     loss+=self.lambda_penalty*(torch.norm(self.rnn_cell.get_w_rec(),1)+torch.norm(self.rnn_cell.bias_rec,1))
#         else:

#             X,ext_in,start_index = input  # Batch x (Length+Tp+ar) x origin_dim
#             #print(X.shape,ext_in.shape)
#             batch,length,ori_dim=X.size()
        
            
#             ############################
#             ### Latent Augmentation ####
#             ############################
#             latent_X=torch.zeros((batch,self.data_len,self.latent_dim*self.ar_order),dtype=torch.float32,device=self.run_device)
#             latent_Y=torch.zeros((batch,self.data_len,self.latent_dim*self.ar_order),dtype=torch.float32,device=self.run_device)

            
#             for i in range(self.ar_order):
#                 latent_X[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,i:(i+self.data_len),:])
#                 latent_Y[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,(i+1):(i+1+self.data_len),:])
#                 #latent_Z[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,(i+self.Tp):(i+self.Tp+self.data_len),:])

#             X_=X[:,(self.ar_order-1):(self.ar_order-1+self.data_len),:] # Current_time=self.ar_order-1
#             Y_=X[:,(self.ar_order):(self.ar_order+self.data_len),:]
            
#             # Input data
#             Ext_IN=ext_in[:,(self.ar_order-1):(self.ar_order-1+self.data_len),:] # Current_Input

#             recon_X=self.decode(latent_X[:,:,((self.ar_order-1)*self.latent_dim):]) # Xk
#             #print(X_.shape,recon_X.shape)
#             OneStep_Prediction=latent_X
#             TpStep_Prediciton=latent_X
            
#             """
#             :Latent_X: Batch x length x latent_dim
#             :Aug_Lat_Input: Batch x augment_dim x length
#             """
#             # augmentation.
#             Aug_Lat_Input=torch.permute(torch.cat((latent_X,Ext_IN),dim=2),(0,2,1))  # [Latent_X; Ext_in]  batch*len*(latent_dim*ar_order+ext_in_dim)
            
            
#             """
#             :self.K_B: batch x latent_dim x augment_dim
#             """
#             # update K and B
#             self.K_B=torch.bmm(torch.permute(latent_Y,(0,2,1)),self.batch_pinv(Aug_Lat_Input,self.L_factors,self.run_device)) #   (latent_dim*ar_order+ext_in_dim) * latent_dim
            
#             self.K=self.K_B[:,:,:self.latent_dim*self.ar_order]
#             self.B=self.K_B[:,:,self.latent_dim*self.ar_order:]

#             # Loss Definition
#             loss_fun=torch.nn.MSELoss(reduction='mean')
#             # Reconstruction loss
#             loss_X_recon=loss_fun(X_,recon_X)                  # Train Encoder and Decoder
#             loss_Linear=0
#             loss_Prediction=0
#             lambda_weight=1

#             # Tp Step Prediction Equation
#             OneStep_Prediction = latent_Y[:,-1:,:] # Current State is the last step of latent_Y
#             for tp in range(self.Tp):
#                 Aug_Lat_Input=torch.permute(torch.cat((OneStep_Prediction,ext_in[:,(self.data_len+self.ar_order-1+tp):(self.data_len+self.ar_order+tp),:]),dim=2),(0,2,1))   # [Latent_X; Ext_in]  batch*len*(latent_dim*ar_order+ext_in_dim)
#                 OneStep_Prediction=torch.permute(torch.bmm(self.K_B,Aug_Lat_Input),(0,2,1))
#                 Z_=X[:,(self.ar_order+self.data_len+tp):(self.ar_order+self.data_len+tp+1),:] # Original of Next State
#                 TpStep_Recon = self.decode(OneStep_Prediction[:,:,((self.ar_order-1)*self.latent_dim):])  # Latent representation of Next State and transform back into original space.
                
#                 # prediction reconstruction loss
#                 loss_Prediction+=pow(lambda_weight,tp)*loss_fun(Z_,TpStep_Recon)            # Train Encoder and Decoder and Koopman_Operator
            
#             loss=self.X_recon*loss_X_recon+self.Tp_recon*loss_Prediction

#         return loss

#     def test_step(self, input, index):

#         loss=0
        
#         if not self.RNN_Type == 'Koopman':
#             # print(X.shape,ext_in.shape)
#             X, ext_in, start_index = input  # Batch x (Length+Tp+ar) x origin_dim
#             batch, length, ori_dim = X.size()

#             # Data Init Position
#             init_pos = length - self.Tp - self.data_len

#             X_ = torch.reshape(X[:, init_pos:init_pos + self.data_len, :], (batch, 1, self.data_len * self.input_dim))
#             Ext_IN = torch.reshape(ext_in[:, :(self.data_len - 1), :],
#                                 (batch, 1, (self.data_len - 1) * self.ext_input_dim))  # Previous_Input
#             previous_state = torch.cat((X_, Ext_IN), dim=2)
        
#             hidden_init = torch.permute(self.encode(previous_state),
#                                         (1, 0, 2))  # Batch X 1 X H_dim -> 1 X Batch X H_dim

#             Tp_prediction = torch.zeros((batch, self.Tp, self.input_dim), device=self.run_device)
#             output=None
#             for i in range(self.Tp):
#                 inputs = ext_in[:, (self.data_len - 1 + i):(self.data_len + i), :]
#                 if self.RNN_Type=='LSTM':
#                     output, (hidden_init, hidden_init) = self.rnn_cell(inputs, (hidden_init, hidden_init))
#                 elif self.RNN_Type=='EI-RNN':
#                     output, hidden_init = self.rnn_cell(torch.permute(inputs,(1,0,2)), hidden_init)
#                 else:
#                     output, hidden_init = self.rnn_cell(inputs, hidden_init)

#                 if self.RNN_Type=='EI-RNN':
#                     Tp_prediction[:, i:i + 1, :] = torch.permute(output,(1, 0, 2))
#                 else:
#                 # print(hidden_init.shape)
#                     Tp_prediction[:, i:i + 1, :] = torch.permute(output,(1, 0, 2))# (self.decode(torch.permute(hidden_init, (1, 0, 2))))
#             loss_fun = torch.nn.MSELoss(reduction='mean')
#             # loss_X_recon=loss_fun(X_,recon_X)
#             loss = loss_fun(X[:, (init_pos + self.data_len):(init_pos + self.data_len + self.Tp), :], Tp_prediction)
#         else:

#             X,ext_in,start_index = input  # Batch x (Length+Tp+ar) x origin_dim
#             #print(X.shape,ext_in.shape)
#             batch,length,ori_dim=X.size()
        
            
#             ############################
#             ### Latent Augmentation ####
#             ############################
#             latent_X=torch.zeros((batch,self.data_len,self.latent_dim*self.ar_order),dtype=torch.float32,device=self.run_device)
#             latent_Y=torch.zeros((batch,self.data_len,self.latent_dim*self.ar_order),dtype=torch.float32,device=self.run_device)

            
#             for i in range(self.ar_order):
#                 latent_X[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,i:(i+self.data_len),:])
#                 latent_Y[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,(i+1):(i+1+self.data_len),:])
#                 #latent_Z[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,(i+self.Tp):(i+self.Tp+self.data_len),:])

#             X_=X[:,(self.ar_order-1):(self.ar_order-1+self.data_len),:] # Current_time=self.ar_order-1
#             Y_=X[:,(self.ar_order):(self.ar_order+self.data_len),:]
            
#             # Input data
#             Ext_IN=ext_in[:,(self.ar_order-1):(self.ar_order-1+self.data_len),:] # Current_Input

#             recon_X=self.decode(latent_X[:,:,((self.ar_order-1)*self.latent_dim):]) # Xk
#             #print(X_.shape,recon_X.shape)
#             OneStep_Prediction=latent_X
#             TpStep_Prediciton=latent_X
            
#             """
#             :Latent_X: Batch x length x latent_dim
#             :Aug_Lat_Input: Batch x augment_dim x length
#             """
#             # augmentation.
#             Aug_Lat_Input=torch.permute(torch.cat((latent_X,Ext_IN),dim=2),(0,2,1))  # [Latent_X; Ext_in]  batch*len*(latent_dim*ar_order+ext_in_dim)
            
            
#             """
#             :self.K_B: batch x latent_dim x augment_dim
#             """
#             # update K and B
#             self.K_B=torch.bmm(torch.permute(latent_Y,(0,2,1)),self.batch_pinv(Aug_Lat_Input,self.L_factors)) #   (latent_dim*ar_order+ext_in_dim) * latent_dim
            
#             self.K=self.K_B[:,:,:self.latent_dim*self.ar_order]
#             self.B=self.K_B[:,:,self.latent_dim*self.ar_order:]

#             # Loss Definition
#             loss_fun=torch.nn.MSELoss(reduction='mean')
#             # Reconstruction loss
#             loss_X_recon=loss_fun(X_,recon_X)                  # Train Encoder and Decoder
#             loss_Linear=0
#             loss_Prediction=0
#             lambda_weight=1

#             #if self.with_input:
#             # Tp Step Prediction Equation
#             OneStep_Prediction = latent_Y[:,-1:,:] # Current State is the last step of latent_Y
#             for tp in range(self.Tp):
#                 Aug_Lat_Input=torch.permute(torch.cat((OneStep_Prediction,ext_in[:,(self.data_len+self.ar_order-1+tp):(self.data_len+self.ar_order+tp),:]),dim=2),(0,2,1))   # [Latent_X; Ext_in]  batch*len*(latent_dim*ar_order+ext_in_dim)
#                 OneStep_Prediction=torch.permute(torch.bmm(self.K_B,Aug_Lat_Input),(0,2,1))
#                 Z_=X[:,(self.ar_order+self.data_len+tp):(self.ar_order+self.data_len+tp+1),:] # Original of Next State
#                 TpStep_Recon = self.decode(OneStep_Prediction[:,:,((self.ar_order-1)*self.latent_dim):])  # Latent representation of Next State and transform back into original space.
                
#                 # prediction reconstruction loss
#                 loss_Prediction+=pow(lambda_weight,tp)*loss_fun(Z_,TpStep_Recon)            # Train Encoder and Decoder and Koopman_Operator
            
#             print('X_recon_error:',loss_X_recon,'Tp_error:',loss_Prediction)
#             loss=self.X_recon*loss_X_recon+self.Tp_recon*loss_Prediction

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.learning_rate)#,weight_decay=self.weight_decay)



# # from os import X_OK
# # from tkinter.tix import Y_REGION
# # from turtle import forward
# # import torch
# # from torch.nn import functional as F
# # from torch import nn
# # from pytorch_lightning import LightningModule
# # # from pytorch_lightning.core.lightning import LightningModule
# # from src.dynamic_model.utils import EI_Sparse_Mask_


# # import math
# # class EI_RNN(nn.Module):
# #     def __init__(self, 
# #                  input_size,    #  
# #                  hidden_size,   # hidden dimension
# #                  output_size,   # observe dimension
# #                  fr_type=False, # transform neural state to neural firing rate by r(t)=relu(x(t))
# #                  ei_ratio=4,  # the ratio between the size of excitatory neurons and inhibitory neurons.
# #                  sparsity=0,
# #                  e_clusters=2,
# #                  i_clusters=1,
# #                  inter_or_intra='Inter',
# #                  output_layers=2,
# #                  with_Tanh=True, 
# #                  device='cpu'):
# #         super(EI_RNN, self).__init__()

# #         self.fr_type=fr_type

# #         self.hidden_size = hidden_size
# #         self.u = nn.Linear(input_size, hidden_size)
        
# #         self.B = nn.Parameter(torch.empty((hidden_size,input_size)),requires_grad=True)
# #         self.B_bias = nn.Parameter(torch.empty((hidden_size,1)),requires_grad=True)
        
# #         nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
# #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.B)
# #         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
# #         nn.init.uniform_(self.B_bias, -bound, bound)
        
# #         self.device=device
# #         probability=1.0*ei_ratio/(ei_ratio+1)
        
        
# #         self.W_rec_fix=EI_Sparse_Mask_(hidden_dim=hidden_size,e_clusters=e_clusters,i_clusters=i_clusters,sparsity=sparsity,ei_ratio=4,module_type=inter_or_intra).to(torch.device(device))
        
# #         ones_tensor=torch.ones(int(hidden_size*probability),device=device)
# #         minus_ones_tensor=-1*torch.ones(hidden_size-int(hidden_size*probability),device=device)
# #         self.ei_constraints=torch.diag(torch.cat((ones_tensor,minus_ones_tensor)))

# #         self.w_rec = nn.Parameter(torch.empty((hidden_size,hidden_size)),requires_grad=True)
# #         self.bias_rec = nn.Parameter(torch.empty((hidden_size,1)),requires_grad=True)

# #         nn.init.kaiming_uniform_(self.w_rec, a=math.sqrt(5))
# #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_rec)
# #         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
# #         nn.init.uniform_(self.bias_rec, -bound, bound)
        
# #         self.w_rec_diag0=torch.ones((hidden_size,hidden_size),device=device)
# #         self.w_rec_diag0.fill_diagonal_(0)
        
# #         self.output_layers=output_layers

# #         self.with_Tanh=with_Tanh

# #         # if self.output_layers==2:
# #         #     self.v1 = nn.Linear(hidden_size, int(hidden_size/2))
# #         #     self.v2 = nn.Linear(int(hidden_size/2), output_size)
# #         # else:
# #         # self.v2 = nn.Linear(hidden_size, output_size)
# #         self.C = nn.Parameter(torch.empty((output_size,hidden_size)),requires_grad=True)
# #         self.C_bias = nn.Parameter(torch.empty((output_size,1)),requires_grad=True)
        
# #         nn.init.kaiming_uniform_(self.C, a=math.sqrt(5))
# #         fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.C)
# #         bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
# #         nn.init.uniform_(self.C_bias, -bound, bound)
            
# #         self.tanh = nn.Tanh()
# #         self.relu = nn.ReLU()
# #         self.softplus=nn.Softplus()
# #         self.softmax = nn.LogSoftmax(dim=1)

# #     def get_w_rec_fix(self):
# #         return self.W_rec_fix.detach().cpu().numpy()
        
# #     def set_w_rec_fix(self,w_rec_fix):
# #         self.W_rec_fix=torch.from_numpy(w_rec_fix).to(torch.device(self.device)).type(torch.float32)
        
# #     def get_w_rec(self):
# #         return torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints)))
        
# #     def forward(self, inputs, hidden):
# #         u_x = self.u(inputs)
# #         # print(u_x.shape)
# #         # print(inputs.shape,hidden.shape)
# #         hidden_ = torch.permute(hidden,(1,2,0))
# #         inputs_ = torch.permute(inputs,(1,2,0))
# #         # print(inputs_.shape,hidden_.shape)
# #         # print(torch.matmul(self.B,inputs_).shape)
# #         u_x=torch.matmul(self.B,inputs_)+self.B_bias
# #         # print(u_x.shape)
# #         if self.fr_type:
# #             #h2h= torch.matmul(self.relu(hidden),torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints)).T))+self.bias_rec
# #             h2h= torch.matmul(torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints))),self.relu(hidden_))+self.bias_rec
# #         else:
# #             #h2h = torch.matmul(hidden,torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints)).T))+self.bias_rec
# #             h2h= torch.matmul(torch.multiply(self.w_rec_diag0,torch.multiply(self.W_rec_fix,torch.matmul(self.relu(self.w_rec),self.ei_constraints))),hidden_)+self.bias_rec
# #         if self.with_Tanh:
# #             hidden_new = self.tanh(h2h + u_x)
# #         else:
# #             hidden_new = h2h + u_x
            
# #         #hidden=hidden_new # self.dt*hidden_new+self.minusdt*hidden
# #         # output=None
# #         # if self.output_layers==1:
# #         # print(self.C.shape,hidden_new.shape)
# #         output = self.softplus(torch.matmul(self.C,self.relu(hidden_new))+self.C_bias)
# #         # print(output.shape,hidden_new.shape)
# #         # else:
# #         #     output=self.softplus(self.v2(self.relu(self.v1(self.relu(hidden)))))# 2-layer outputs
# #         return torch.permute(output,(2,0,1)), torch.permute(hidden_new,(2,0,1))


# # class Latent_Model(LightningModule):
# #     def __init__(self, params):
# #         super().__init__()

# #         self.Tp = params['Tp']  # Tp step Prediction
# #         self.input_dim = int(params['IN_DIM'])
# #         self.hidden_dim = int(params['HIDDEN_DIM'])  # Mid Dimension
# #         self.latent_dim = int(params['LATENT_DIM'])  # Approximate High Dimension
# #         self.X_recon = params['X_recon']
# #         self.Y_recon = params['Y_recon']
# #         self.Tp_recon = params['Tp_recon']
# #         self.Linear_Loss = params['Linear_Loss']
# #         self.data_len = params['data_length']
# #         self.with_input = params['with_input']  # With external input or not
# #         self.StateDependent = params['StateDependent']  # The input matrix is a constant or state dependent?
# #         self.L_factors = params['L_factors']
# #         self.ar_order = params['AR_order']
# #         self.ext_input_dim = params['ext_input_dim']
# #         self.run_device = params['device']
# #         self.l2_reg=params['l2_norm']
# #         self.learning_rate = params['learning_rate']
# #         # self.weight_decay = params['weight_decay']
        
# #         if params['activation']=='ReLU':
# #             self.activate=nn.ReLU()
# #         elif  params['activation']=='Tanh':
# #             self.activate=nn.Tanh()
# #         elif params['activation']=='Softplus':
# #             self.activate=nn.Softplus()
            
# #         self.l1_reg = params['l1_reg']

# #         self.sparsity=params['sparsity']
            
# #         self.e_clusters=params['e_clusters']
# #         self.i_clusters=params['i_clusters']
# #         self.inter_or_intra=params['inter_or_intra']
        
# #         if params['final_activation']=='ReLU':
# #             self.final_activate=nn.ReLU()
# #         elif params['final_activation']=='Softplus':
# #             self.final_activate=nn.Softplus()

# #         self.rnn_cell=None
# #         self.RNN_Type=params['RNN_Type']

# #         # Koopman Operator
# #         self.K = None
# #         self.K_inv = None
# #         self.B = None
# #         self.K_B = None

# #         self.iters = 0

# #         self.encode = nn.Sequential(
# #             torch.nn.Linear(self.data_len * (self.input_dim) + (self.data_len - 1) * (self.ext_input_dim),
# #                             self.latent_dim),
# #             self.activate,
# #         )
# #         print('input size:',self.data_len * (self.input_dim) + (self.data_len - 1) * (self.ext_input_dim),' ',self.latent_dim)


# #         if not self.RNN_Type=='EI-RNN':
# #             self.decode = nn.Sequential(
# #                 torch.nn.Linear(self.latent_dim, int(self.latent_dim / 2)),
# #                 self.activate,
# #                 torch.nn.Linear(int(self.latent_dim / 2), self.input_dim),
# #                 self.final_activate
# #             )


# #         if self.RNN_Type=='GRU':
# #             self.rnn_cell = torch.nn.GRU(input_size=self.ext_input_dim, hidden_size=self.latent_dim, num_layers=1,
# #                                     batch_first=True)
# #         elif self.RNN_Type=='RNN':
# #             self.rnn_cell = torch.nn.RNN(input_size=self.ext_input_dim, hidden_size=self.latent_dim, num_layers=1,
# #                                     batch_first=True)
# #         elif self.RNN_Type=='LSTM':
# #             self.rnn_cell = torch.nn.LSTM(input_size=self.ext_input_dim, hidden_size=self.latent_dim, num_layers=1,
# #                                     batch_first=True)
# #         elif self.RNN_Type=='EI-RNN':
# #             self.ei_ratio=params['EI_ratio']
# #             self.rnn_cell=EI_RNN(input_size=self.ext_input_dim, 
# #                                  hidden_size=self.latent_dim, 
# #                                  output_size=self.input_dim, 
# #                                  output_layers=1,
# #                                  fr_type=True, 
# #                                  ei_ratio=self.ei_ratio,
# #                                  sparsity=self.sparsity,
# #                                  e_clusters=self.e_clusters,
# #                                  i_clusters=self.i_clusters,
# #                                  inter_or_intra=self.inter_or_intra,
# #                                  with_Tanh=params['with_Tanh'],
# #                                  device=self.run_device)
# #             print('run_device:',self.run_device)
            
# #         elif self.RNN_Type=='Koopman':
# #             self.encode = nn.Sequential(
# #                                         nn.Linear(self.input_dim,self.hidden_dim),
# #                                         self.activate,
# #                                         nn.Linear(self.hidden_dim,self.hidden_dim),
# #                                         self.activate,
# #                                         nn.Linear(self.hidden_dim,self.hidden_dim),
# #                                         self.activate,
# #                                         nn.Linear(self.hidden_dim,self.latent_dim)
# #                                     )
# #             self.decode = nn.Sequential(
# #                                         torch.nn.Linear(self.latent_dim, self.hidden_dim),
# #                                         self.activate,
# #                                         torch.nn.Linear(self.hidden_dim, self.hidden_dim),
# #                                         self.activate,
# #                                         torch.nn.Linear(self.hidden_dim, self.hidden_dim),
# #                                         self.activate,
# #                                         torch.nn.Linear(self.hidden_dim, self.input_dim),
# #                                         self.final_activate
# #                                     )

# #     # @staticmethod
# #     def batch_pinv(self,x, I_factor,device):
# #         """
# #         :param x: B x N x D (N > D)
# #         :param I_factor:
# #         :return:
# #         """
# #         B, N, D = x.size()
# #         if N < D:
# #             x = torch.transpose(x, 1, 2)
# #             N, D = D, N
# #             trans = True
# #         else:
# #             trans = False

# #         x_t = torch.transpose(x, 1, 2)

# #         # use_gpu = torch.cuda.is_available()
# #         I = torch.eye(D)[None, :, :].repeat(B, 1, 1).to(torch.device(device))
# #         # if use_gpu:
# #         #     I = I.to('cuda:0')

# #         x_pinv = torch.bmm(
# #             torch.inverse(torch.bmm(x_t, x) + I_factor * I),
# #             x_t
# #         )

# #         if trans:
# #             x_pinv = torch.transpose(x_pinv, 1, 2)

# #         return x_pinv

        

# #     def training_step(self, input):
# #         loss=0
        
# #         if not self.RNN_Type == 'Koopman':
# #             # print(X.shape,ext_in.shape)
# #             X, ext_in, start_index = input  # Batch x (Length+Tp+ar) x origin_dim
# #             batch, length, ori_dim = X.size()

# #             # Data Init Position
# #             init_pos = length - self.Tp - self.data_len

# #             X_ = torch.reshape(X[:, init_pos:init_pos + self.data_len, :], (batch, 1, self.data_len * self.input_dim))
# #             Ext_IN = torch.reshape(ext_in[:, :(self.data_len - 1), :],
# #                                 (batch, 1, (self.data_len - 1) * self.ext_input_dim))  # Previous_Input
# #             previous_state = torch.cat((X_, Ext_IN), dim=2)
        
# #             hidden_init = torch.permute(self.encode(previous_state),
# #                                         (1, 0, 2))  # Batch X 1 X H_dim -> 1 X Batch X H_dim

# #             Tp_prediction = torch.zeros((batch, self.Tp, self.input_dim), device=self.run_device)
# #             output=None
# #             for i in range(self.Tp):
# #                 inputs = ext_in[:, (self.data_len - 1 + i):(self.data_len + i), :]
# #                 if self.RNN_Type=='LSTM':
# #                     output, (hidden_init, hidden_init) = self.rnn_cell(inputs, (hidden_init, hidden_init))
# #                 elif self.RNN_Type=='EI-RNN':
# #                     output, hidden_init = self.rnn_cell(torch.permute(inputs,(1,0,2)), hidden_init)
# #                 else:
# #                     output, hidden_init = self.rnn_cell(inputs, hidden_init)

# #                 if self.RNN_Type=='EI-RNN':
# #                     Tp_prediction[:, i:i + 1, :] = torch.permute(output,(1, 0, 2))
# #                 else:
# #                     Tp_prediction[:, i:i + 1, :] = (self.decode(torch.permute(hidden_init, (1, 0, 2))))
# #             loss_fun = torch.nn.MSELoss(reduction='mean')
# #             loss = loss_fun(X[:, (init_pos + self.data_len):(init_pos + self.data_len + self.Tp), :], Tp_prediction)

# #             if self.RNN_Type=='EI-RNN':
# #                 loss+=self.l2_reg*(torch.norm(self.rnn_cell.get_w_rec(),2)**2+torch.norm(self.rnn_cell.bias_rec,2)**2)
                
# #                 # if self.l1_reg:
# #                 #     loss+=self.lambda_penalty*(torch.norm(self.rnn_cell.get_w_rec(),1)+torch.norm(self.rnn_cell.bias_rec,1))
# #         else:

# #             X,ext_in,start_index = input  # Batch x (Length+Tp+ar) x origin_dim
# #             #print(X.shape,ext_in.shape)
# #             batch,length,ori_dim=X.size()
        
            
# #             ############################
# #             ### Latent Augmentation ####
# #             ############################
# #             latent_X=torch.zeros((batch,self.data_len,self.latent_dim*self.ar_order),dtype=torch.float32,device=self.run_device)
# #             latent_Y=torch.zeros((batch,self.data_len,self.latent_dim*self.ar_order),dtype=torch.float32,device=self.run_device)

            
# #             for i in range(self.ar_order):
# #                 latent_X[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,i:(i+self.data_len),:])
# #                 latent_Y[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,(i+1):(i+1+self.data_len),:])
# #                 #latent_Z[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,(i+self.Tp):(i+self.Tp+self.data_len),:])

# #             X_=X[:,(self.ar_order-1):(self.ar_order-1+self.data_len),:] # Current_time=self.ar_order-1
# #             Y_=X[:,(self.ar_order):(self.ar_order+self.data_len),:]
            
# #             # Input data
# #             Ext_IN=ext_in[:,(self.ar_order-1):(self.ar_order-1+self.data_len),:] # Current_Input

# #             recon_X=self.decode(latent_X[:,:,((self.ar_order-1)*self.latent_dim):]) # Xk
# #             #print(X_.shape,recon_X.shape)
# #             OneStep_Prediction=latent_X
# #             TpStep_Prediciton=latent_X
            
# #             """
# #             :Latent_X: Batch x length x latent_dim
# #             :Aug_Lat_Input: Batch x augment_dim x length
# #             """
# #             # augmentation.
# #             Aug_Lat_Input=torch.permute(torch.cat((latent_X,Ext_IN),dim=2),(0,2,1))  # [Latent_X; Ext_in]  batch*len*(latent_dim*ar_order+ext_in_dim)
            
            
# #             """
# #             :self.K_B: batch x latent_dim x augment_dim
# #             """
# #             # update K and B
# #             self.K_B=torch.bmm(torch.permute(latent_Y,(0,2,1)),self.batch_pinv(Aug_Lat_Input,self.L_factors,self.run_device)) #   (latent_dim*ar_order+ext_in_dim) * latent_dim
            
# #             self.K=self.K_B[:,:,:self.latent_dim*self.ar_order]
# #             self.B=self.K_B[:,:,self.latent_dim*self.ar_order:]

# #             # Loss Definition
# #             loss_fun=torch.nn.MSELoss(reduction='mean')
# #             # Reconstruction loss
# #             loss_X_recon=loss_fun(X_,recon_X)                  # Train Encoder and Decoder
# #             loss_Linear=0
# #             loss_Prediction=0
# #             lambda_weight=1

# #             # Tp Step Prediction Equation
# #             OneStep_Prediction = latent_Y[:,-1:,:] # Current State is the last step of latent_Y
# #             for tp in range(self.Tp):
# #                 Aug_Lat_Input=torch.permute(torch.cat((OneStep_Prediction,ext_in[:,(self.data_len+self.ar_order-1+tp):(self.data_len+self.ar_order+tp),:]),dim=2),(0,2,1))   # [Latent_X; Ext_in]  batch*len*(latent_dim*ar_order+ext_in_dim)
# #                 OneStep_Prediction=torch.permute(torch.bmm(self.K_B,Aug_Lat_Input),(0,2,1))
# #                 Z_=X[:,(self.ar_order+self.data_len+tp):(self.ar_order+self.data_len+tp+1),:] # Original of Next State
# #                 TpStep_Recon = self.decode(OneStep_Prediction[:,:,((self.ar_order-1)*self.latent_dim):])  # Latent representation of Next State and transform back into original space.
                
# #                 # prediction reconstruction loss
# #                 loss_Prediction+=pow(lambda_weight,tp)*loss_fun(Z_,TpStep_Recon)            # Train Encoder and Decoder and Koopman_Operator
            
# #             loss=self.X_recon*loss_X_recon+self.Tp_recon*loss_Prediction

# #         return loss

# #     def test_step(self, input, index):

# #         loss=0
        
# #         if not self.RNN_Type == 'Koopman':
# #             # print(X.shape,ext_in.shape)
# #             X, ext_in, start_index = input  # Batch x (Length+Tp+ar) x origin_dim
# #             batch, length, ori_dim = X.size()

# #             # Data Init Position
# #             init_pos = length - self.Tp - self.data_len

# #             X_ = torch.reshape(X[:, init_pos:init_pos + self.data_len, :], (batch, 1, self.data_len * self.input_dim))
# #             Ext_IN = torch.reshape(ext_in[:, :(self.data_len - 1), :],
# #                                 (batch, 1, (self.data_len - 1) * self.ext_input_dim))  # Previous_Input
# #             previous_state = torch.cat((X_, Ext_IN), dim=2)
        
# #             hidden_init = torch.permute(self.encode(previous_state),
# #                                         (1, 0, 2))  # Batch X 1 X H_dim -> 1 X Batch X H_dim

# #             Tp_prediction = torch.zeros((batch, self.Tp, self.input_dim), device=self.run_device)
# #             output=None
# #             for i in range(self.Tp):
# #                 inputs = ext_in[:, (self.data_len - 1 + i):(self.data_len + i), :]
# #                 if self.RNN_Type=='LSTM':
# #                     output, (hidden_init, hidden_init) = self.rnn_cell(inputs, (hidden_init, hidden_init))
# #                 elif self.RNN_Type=='EI-RNN':
# #                     output, hidden_init = self.rnn_cell(torch.permute(inputs,(1,0,2)), hidden_init)
# #                 else:
# #                     output, hidden_init = self.rnn_cell(inputs, hidden_init)

# #                 if self.RNN_Type=='EI-RNN':
# #                     Tp_prediction[:, i:i + 1, :] = torch.permute(output,(1, 0, 2))
# #                 else:
# #                 # print(hidden_init.shape)
# #                     Tp_prediction[:, i:i + 1, :] = torch.permute(output,(1, 0, 2))# (self.decode(torch.permute(hidden_init, (1, 0, 2))))
# #             loss_fun = torch.nn.MSELoss(reduction='mean')
# #             # loss_X_recon=loss_fun(X_,recon_X)
# #             loss = loss_fun(X[:, (init_pos + self.data_len):(init_pos + self.data_len + self.Tp), :], Tp_prediction)
# #         else:

# #             X,ext_in,start_index = input  # Batch x (Length+Tp+ar) x origin_dim
# #             #print(X.shape,ext_in.shape)
# #             batch,length,ori_dim=X.size()
        
            
# #             ############################
# #             ### Latent Augmentation ####
# #             ############################
# #             latent_X=torch.zeros((batch,self.data_len,self.latent_dim*self.ar_order),dtype=torch.float32,device=self.run_device)
# #             latent_Y=torch.zeros((batch,self.data_len,self.latent_dim*self.ar_order),dtype=torch.float32,device=self.run_device)

            
# #             for i in range(self.ar_order):
# #                 latent_X[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,i:(i+self.data_len),:])
# #                 latent_Y[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,(i+1):(i+1+self.data_len),:])
# #                 #latent_Z[:,:,i*self.latent_dim:(i+1)*self.latent_dim]=self.encode(X[:,(i+self.Tp):(i+self.Tp+self.data_len),:])

# #             X_=X[:,(self.ar_order-1):(self.ar_order-1+self.data_len),:] # Current_time=self.ar_order-1
# #             Y_=X[:,(self.ar_order):(self.ar_order+self.data_len),:]
            
# #             # Input data
# #             Ext_IN=ext_in[:,(self.ar_order-1):(self.ar_order-1+self.data_len),:] # Current_Input

# #             recon_X=self.decode(latent_X[:,:,((self.ar_order-1)*self.latent_dim):]) # Xk
# #             #print(X_.shape,recon_X.shape)
# #             OneStep_Prediction=latent_X
# #             TpStep_Prediciton=latent_X
            
# #             """
# #             :Latent_X: Batch x length x latent_dim
# #             :Aug_Lat_Input: Batch x augment_dim x length
# #             """
# #             # augmentation.
# #             Aug_Lat_Input=torch.permute(torch.cat((latent_X,Ext_IN),dim=2),(0,2,1))  # [Latent_X; Ext_in]  batch*len*(latent_dim*ar_order+ext_in_dim)
            
            
# #             """
# #             :self.K_B: batch x latent_dim x augment_dim
# #             """
# #             # update K and B
# #             self.K_B=torch.bmm(torch.permute(latent_Y,(0,2,1)),self.batch_pinv(Aug_Lat_Input,self.L_factors)) #   (latent_dim*ar_order+ext_in_dim) * latent_dim
            
# #             self.K=self.K_B[:,:,:self.latent_dim*self.ar_order]
# #             self.B=self.K_B[:,:,self.latent_dim*self.ar_order:]

# #             # Loss Definition
# #             loss_fun=torch.nn.MSELoss(reduction='mean')
# #             # Reconstruction loss
# #             loss_X_recon=loss_fun(X_,recon_X)                  # Train Encoder and Decoder
# #             loss_Linear=0
# #             loss_Prediction=0
# #             lambda_weight=1

# #             #if self.with_input:
# #             # Tp Step Prediction Equation
# #             OneStep_Prediction = latent_Y[:,-1:,:] # Current State is the last step of latent_Y
# #             for tp in range(self.Tp):
# #                 Aug_Lat_Input=torch.permute(torch.cat((OneStep_Prediction,ext_in[:,(self.data_len+self.ar_order-1+tp):(self.data_len+self.ar_order+tp),:]),dim=2),(0,2,1))   # [Latent_X; Ext_in]  batch*len*(latent_dim*ar_order+ext_in_dim)
# #                 OneStep_Prediction=torch.permute(torch.bmm(self.K_B,Aug_Lat_Input),(0,2,1))
# #                 Z_=X[:,(self.ar_order+self.data_len+tp):(self.ar_order+self.data_len+tp+1),:] # Original of Next State
# #                 TpStep_Recon = self.decode(OneStep_Prediction[:,:,((self.ar_order-1)*self.latent_dim):])  # Latent representation of Next State and transform back into original space.
                
# #                 # prediction reconstruction loss
# #                 loss_Prediction+=pow(lambda_weight,tp)*loss_fun(Z_,TpStep_Recon)            # Train Encoder and Decoder and Koopman_Operator
            
# #             print('X_recon_error:',loss_X_recon,'Tp_error:',loss_Prediction)
# #             loss=self.X_recon*loss_X_recon+self.Tp_recon*loss_Prediction

# #     def configure_optimizers(self):
# #         return torch.optim.Adam(self.parameters(), lr=self.learning_rate)#,weight_decay=self.weight_decay)


