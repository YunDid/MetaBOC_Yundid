import numpy as np
import sys
from casadi import *
# import do_mpc
import PyQt5
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *
import time
import torch
class optim_setup(QThread):
    emit_optimal_input_ = pyqtSignal(bool)   # 发送bool变量表示已经生成了最优刺激信号
    def __init__(self,model_params=None,predict_length=5):
        # super().__init__()
        QThread.__init__(self, parent=None)
        print('thread_id: ', QThread().currentThreadId())
        
        self.hidden_size=50
        self.input_size= model_params['B'].shape[1]
        self.output_size=model_params['C'].shape[0]
        self.lasttime = time.time()
        self.predict_length=predict_length
        self.relu=torch.nn.ReLU()
        self.softmax=torch.nn.Softmax()
        self.num_channels=5

        self.optim_type=None
        self.optim_channel=None
        self.init_x=None
        self.start_optimize=False
        # self.set_type_of_optim(torch.zeros((self.hidden_size,1)),1,'maximal')
        #self.start_optimize=True
        
        self.max_input = torch.tensor([20.0,5.0,20.0,5.0])
        self.min_input = torch.tensor([0.0,0.0,0.0,0.0])
        if model_params==None:
            # load model parameter from checkpoint
            self.Wrec=np.abs(np.random.randn(self.hidden_size,self.hidden_size))
            self.Wrec_bias=np.abs(np.random.randn(self.hidden_size,1))

            self.B = np.abs(np.random.randn(self.hidden_size,self.input_size))
            self.B_bias= np.abs(np.random.randn(self.hidden_size,1))

            self.C=np.abs(np.random.randn(self.output_size,self.hidden_size))
            self.C_bias=np.abs(np.random.randn(self.output_size,1))
        else:
            self.Wrec=torch.from_numpy(model_params['Wrec']).type(torch.float32)
            self.Wrec_bias=torch.from_numpy(model_params['Wrec_bias']).type(torch.float32)

            self.B = torch.from_numpy(model_params['B']).type(torch.float32)
            self.B_bias= torch.from_numpy(model_params['B_bias']).type(torch.float32)

            self.C=torch.from_numpy(model_params['C']).type(torch.float32)
            self.C_bias=torch.from_numpy(model_params['C_bias']).type(torch.float32)


    def set_type_of_optim(self,init_x,optim_channel,optim_type='softmax'):
        self.optim_type=optim_type
        self.optim_channel=optim_channel
        self.init_x=init_x
        self.start_optimize=True

    def __del__(self):
        self.wait()


    def run(self):
        while True:
            if self.start_optimize:
                # self.start_optimize=False
                self.lasttime= time.time()
                # self._signal.emit(list(self.obtain_optimal_control()))
                self.obtain_optimal_control()
                cost_time = time.time() - self.lasttime
                # print('cost_time:',cost_time*1000)
            time.sleep(0.05)


    def obtain_optimal_control(self):
        print('--------obtan optimal control--------')
        init_input_U=np.zeros((self.predict_length, self.input_size))
        init_input_U[:,0]=1.0*np.random.randint(4, 20, self.predict_length)
        init_input_U[:,2]=1.0*np.random.randint(4, 20, self.predict_length)

        init_input_U[:,1]=np.random.normal(5, 2, self.predict_length)
        init_input_U[:,3]=np.random.normal(5, 2, self.predict_length)

        opt_U = torch.nn.parameter.Parameter(torch.from_numpy(init_input_U.astype(np.float32)),requires_grad=True)
        current_x=self.init_x.type(torch.float32)[0].transpose(0,1)

        print(current_x.shape)
        if self.optim_type=='maximal':
            for epoch in range(20):
                loss=0
                for i in range(self.predict_length):
                    input=torch.reshape(torch.clamp(opt_U[i,:].clone(),self.min_input,self.max_input),(self.input_size,1))
                    # print(self.B,input)
                    # print(input.shape,self.B.shape,self.Wrec.shape,self.B_bias.shape,self.Wrec_bias.shape,self.C.shape,self.C_bias.shape)
                    current_x= torch.matmul(self.Wrec,self.relu(current_x)) + self.Wrec_bias + torch.matmul(self.B,input)+ self.B_bias
                    current_y=torch.log(1+torch.exp(torch.matmul(self.C,self.relu(current_x))+self.C_bias))
                    # print(current_y.shape)
                    loss+=-current_y[self.optim_channel,0] # 某个通道的最大值 最小
                optimizer=torch.optim.Adam({opt_U},lr=0.1)
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                with torch.no_grad():
                    for i in range(self.predict_length):
                        opt_U[i,:].clamp_(self.min_input, self.max_input)
                #if epoch%5==0:
                # print('Iters:',epoch,' Loss:',loss.detach().cpu().numpy())#,'opt_u:',opt_U)

        elif self.optim_type=='softmax':
            current_x=self.init_x[0].transpose(0,1)
            for epoch in range(20):
                optimizer=torch.optim.Adam({opt_U},lr=0.1)
                loss_func=torch.nn.CrossEntropyLoss()
                
                loss=0
                for i in range(self.predict_length):
                    input=torch.reshape(torch.clamp(opt_U[i,:].clone(),self.min_input,self.max_input),(self.input_size,1))

                    current_x= torch.matmul(self.Wrec,self.relu(current_x)) + self.Wrec_bias + torch.matmul(self.B,input)+ self.B_bias
                    current_y=torch.log(1+torch.exp(torch.matmul(self.C,self.relu(current_x))+self.C_bias))

                    loss+=loss_func(self.softmax(current_y),
                                    torch.nn.functional.one_hot(torch.tensor(self.optim_channel-1),self.num_channels))#.float())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    for i in range(self.predict_length):
                        opt_U[i,:].clamp_(self.min_input, self.max_input)
                #if epoch%5==0:
                # print('Iters:',epoch,' Loss:',loss.detach().cpu().numpy()[0,0])#,'opt_u:',opt_U)

        # send u
        self.u = opt_U.detach().numpy()[:,0:1]
        num_stim_channels = len(self.u) // 2
        self.input_amplitudes = {}
        self.input_durations = {}
        self.amplitudes = {}
        self.frequencys = {}
        for i in range(num_stim_channels):
            self.input_amplitudes[i], self.input_durations[i] = self.testOutput_1(self.u[i*2:(i+1)*2,:])
            self.frequencys[i] = self.u[2*i].reshape(1,1)
            self.amplitudes[i] = self.u[2*i+1].reshape(1,1)
        # self.emit_optimal_input_.emit(True)

        
    def obtain_random_stim(self, num_stim_channels, num_stim):
        self.input_amplitudes = {}
        self.input_durations = {}
        self.amplitudes = {}
        self.frequencys = {}
        for i in range(num_stim_channels//2):
            self.amplitudes[i] = np.random.uniform(low=0, high=9,size=num_stim).reshape(1,num_stim)
            self.frequencys[i] = np.random.uniform(low=0, high=20,size=num_stim).reshape(1,num_stim)
            # self.frequencys[i]=frequency
            print(np.concatenate((self.frequencys[i],self.amplitudes[i]),axis=0).shape)
            self.input_amplitudes[i] , self.input_durations[i] = self.testOutput_1(np.concatenate((self.frequencys[i],self.amplitudes[i]),axis=0))
            self.u[2*i]=self.frequencys[i]
            self.u[2*i+1]=self.amplitudes[i]
    
    def get_optimal_control(self):
        return self.u, self.input_amplitudes, self.input_durations, self.amplitudes, self.frequencys

    @staticmethod
    def testOutput_1(U):
        # 刺激参数（amplitude,frequency）转化为输入参数（amplitude，duration）
        stim_1_freq = U[0, :] + 4
        stim_1_amp = U[1, :] * 100 + 100 # 应该是100
        amplitude = []
        duration = []
        duration_time = 200
        # print('stim_1_freq shape: ',stim_1_freq.shape)
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
                for j in range(int(times)-1):
                    # 设置刺激幅值（μV）和刺激时间（μs）
                    amplitude.append(-amp*1000)
                    duration.append(duration_time/2)
                    amplitude.append(amp*1000)
                    duration.append(duration_time/2)
                    # 设置刺激间隔（μs）
                    amplitude.append(0)
                    duration.append(isi*1000)
                amplitude.append(-amp*1000)
                duration.append(duration_time/2)
                amplitude.append(amp*1000)
                duration.append(duration_time/2)
                time_rest = 250 - sum(duration)/1000
                time_rest = time_rest if time_rest >=0 else 0
            else:
                time_rest = 250
            # 相邻刺激间隔（μs）
            
            # amplitude.append(0)
            # duration.append(np.round(time_rest*1000))
        return amplitude, duration