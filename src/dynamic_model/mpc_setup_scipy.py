import numpy as np
import sys
from casadi import *
import do_mpc
import PyQt5
from PyQt5.QtCore import *
from multiprocessing import Pool
from scipy.optimize import minimize, LinearConstraint
import time
class my_mpc(QThread):
    emit_optimal_input_ = pyqtSignal(bool)   # 发送bool变量表示已经生成了最优刺激信号
    def __init__(self,model_params=None,predict_length=6,control_length=5,ref=None,control_strategy='Tracking',control_channel=None,control_type='closed_loop'):
        # super().__init__()
        QThread.__init__(self, parent=None)
        print('thread_id: ', QThread().currentThreadId())
        
        self.hidden_size=50
        self.input_size= model_params['B'].shape[1]
        self.output_size=model_params['C'].shape[0]
        self.lasttime = time.time()
        print('input_dim:%d  system_dim:%d'%(self.input_size,self.output_size))
        model_type = 'discrete' # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(model_type)
        self.u = np.zeros(shape=(self.input_size,1))
        self.optimal_loss=0
        self.input_amplitudes = {}
        self.input_durations = {}
        self.amplitudes = {}
        self.frequencys = {}
        for i in range(2):
            self.input_amplitudes[i], self.input_durations[i] = self.testOutput_1(self.u[i*2:(i+1)*2,:])
            self.frequencys[i] = self.u[2*i].reshape(1,-1)
            self.amplitudes[i] = self.u[2*i+1].reshape(1,-1)
        # self._x = self.model.set_variable(var_type='_x', var_name='x', shape=(self.hidden_size,1))
        # self._u = self.model.set_variable(var_type='_u', var_name='u', shape=(self.input_size,1))
        if model_params==None:
            # load model parameter from checkpoint
            self.Wrec=np.abs(np.random.randn(self.hidden_size,self.hidden_size))
            self.Wrec_bias=np.abs(np.random.randn(self.hidden_size,1))

            self.B = np.abs(np.random.randn(self.hidden_size,self.input_size))
            self.B_bias= np.abs(np.random.randn(self.hidden_size,1))

            self.C=np.abs(np.random.randn(self.output_size,self.hidden_size))
            self.C_bias=np.abs(np.random.randn(self.output_size,1))
        else:
            self.Wrec=model_params['Wrec']
            self.Wrec_bias=model_params['Wrec_bias']

            self.B = model_params['B']
            self.B_bias= model_params['B_bias']

            self.C=model_params['C']
            self.C_bias=model_params['C_bias']

        self.predict_length = predict_length
        self.control_length= control_length

        self.max_us = np.array([15.0,2.0,15.0,2.0])  # 改动 [12.0,9.0,12.0,9.0] 9 代表900mV 15代表频率 数值需要先×100 后×1000 代表最终数值
        self.min_us = np.array([0,0.5,0,0.5]) # 0.1mV - 900mV 100 - 900000  10 - 
        # self.min_us = np.array([[0.0],[0.0],[0.0],[0.0]]).T
        # self.set_input_bounds(min_us,max_us)

        self.control_strategy=control_strategy #'Tracking'
        self.control_channel=control_channel
        self.control_type = control_type
        self.ref = np.reshape(ref,(self.output_size,1))
        self.u0 = np.zeros((self.control_length,4))
        self.start_optimize=False

    def objective(self,u):
        x = self.init_x.copy()
        
        u = u.reshape((self.control_length,-1))
        cost = 0
        if self.control_strategy=='Tracking':
            for t in range(self.predict_length):
                if t<self.control_length:
                    x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(u[t:t+1,:].T) + self.B_bias
                else:
                    x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
                y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))
                cost += np.sum((y - self.ref)**2) #/self.predict_length/4 #  + np.sum(u[t]**2)
                #print(y.T,self.ref.T)
                # print(self.ref,y)
        elif self.control_strategy=='Activating':
            for t in range(self.predict_length):
                if t<self.control_length:
                    x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(u[t:t+1,:].T) + self.B_bias
                else:
                    x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
                y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))
                cost += -(y[self.control_channel,0]**2)

        elif self.control_strategy=='Inhibiting':
            for t in range(self.predict_length):
                if t<self.control_length:
                    x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(u[t:t+1,:].T) + self.B_bias
                else:
                    x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
                y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))
                cost += (y[self.control_channel,0]**2)

        elif self.control_strategy=='Activating_channel':
            for t in range(self.predict_length):
                if t<self.control_length:
                    x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(u[t:t+1,:].T) + self.B_bias
                else:
                    x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
                y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))
                sum_exp = np.sum(np.exp(y))
                cost += - np.exp(y[self.control_channel,0])/sum_exp

        elif self.control_strategy=='Inhibiting_channel':
            for t in range(self.predict_length):
                if t<self.control_length:
                    x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(u[t:t+1,:].T) + self.B_bias
                else:
                    x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
                y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))
                sum_exp = np.sum(np.exp(-y))
                cost += - np.exp(-y[self.control_channel,0])/sum_exp

        # cost += 0.0001*np.sum(u**2)
        return cost
    
    def constraint(self,u):
        u = u.reshape((self.control_length,-1))
        constraints = []
        for t in range(self.control_length):
            # print(u[t], self.max_us)
            # for i in range(4):
            #     constraints.append(self.max_us[i]-u[t,i])
            #     constraints.append(u[t,i])
                # constraints.append({'type': 'ineq', 'fun': lambda u, t=t, i=i: self.max_us[i] - u[t, i]})
                # # Constraint: u[t, i] >= 0 (assuming a lower limit of zero)
                # constraints.append({'type': 'ineq', 'fun': lambda u, t=t, i=i: u[t, i]})
            constraints.extend(self.max_us-u[t])
            constraints.extend(u[t]-self.min_us) #-self.min_us)
        return constraints

    def __del__(self):
        self.wait()



    def set_initial_state(self,init_x):
        self.init_x=init_x
        self.start_optimize=True

    def run(self):
        while True:
            if self.start_optimize:
                self.start_optimize=False
                self.lasttime= time.time()
                # self._signal.emit(list(self.obtain_optimal_control()))
                self.obtain_optimal_control()
                cost_time = time.time() - self.lasttime
                # print('cost_time:',cost_time*1000)
            time.sleep(0.02)

    # def optimize_(self,init_x0):
    #     constraints = [{'type': 'ineq', 'fun': self.constraint}]
    #     return minimize(self.objective, 
    #                 np.reshape(self.u0,-1), 
    #                 method='Powell', #'COBYLA', #'Nelder-Mead', #'Powell', #'Powell',#  'SLSQP', #'trust-constr',  # 
    #                 constraints=constraints, 
    #                 options={'maxiter': 5})

    def obtain_optimal_control(self,epochs=20):
        # e = np.ones([self.model.n_x,1])
        # x0 = np.random.uniform(-3*e,3*e)
        # for k in range(50):
        import time
        start = time.time()
        
        
        # given an initial state and output the optimal input

        # initial_u0 = []
        # for i in range(3):
        #     init_input_U=np.zeros((self.predict_length, self.input_size))
        #     init_input_U[:,0]=np.random.uniform(5,10,self.predict_length)#1.0*np.random.randint(4, 20, self.predict_length)
        #     init_input_U[:,2]=np.random.uniform(5,10,self.predict_length)#1.0*np.random.randint(4, 20, self.predict_length)

        #     init_input_U[:,1]=np.random.uniform(3,6,self.predict_length)#np.random.normal(5, 2, self.predict_length)
        #     init_input_U[:,3]=np.random.uniform(3,6,self.predict_length)#np.random.normal(5, 2, self.predict_length)

        #     self.u0=init_input_U
        #     initial_u0.append(init_input_U)
        # from multiprocessing import Pool
        # with Pool(processes=4) as pool:
        #     results = pool.map(self.optimize_, initial_u0)

        # best_result = min(results, key=lambda x: x.fun)
        # self.u0 = best_result.x.reshape((self.predict_length,4))# 'trust-constr' #
        
        init_input_U=np.zeros((self.control_length, self.input_size))
        init_input_U[:,0]=np.random.uniform(5,10,self.control_length)#1.0*np.random.randint(4, 20, self.predict_length)
        init_input_U[:,2]=np.random.uniform(5,10,self.control_length)#1.0*np.random.randint(4, 20, self.predict_length)

        init_input_U[:,1]=np.random.uniform(3,6,self.control_length)#np.random.normal(5, 2, self.predict_length)
        init_input_U[:,3]=np.random.uniform(3,6,self.control_length)#np.random.normal(5, 2, self.predict_length)

        self.u0=init_input_U

        constraints = [{'type': 'ineq', 'fun': self.constraint}]
        # print(constraints)
        result = minimize(self.objective, 
                          np.reshape(self.u0,-1), 
                          method='COBYLA',  # 'L-BFGS-B', #'trust-constr', #'Powell',# 'Nelder-Mead',#'COBYLA', #'Powell', # 'SLSQP', #
                          constraints=constraints,#self.constraint(self.u0), 
                          options={'maxiter': 100})
        self.u0 = np.abs(result.x.reshape((self.control_length,4))) # 'trust-constr' #

        print("Optimal cost:%2f \n"%result.fun)
        #self.optimal_loss = result.fun
        # print("Optimal u0:", np.around(np.abs(self.u0),2))
        #print("Optimization result:", result)
        # print('time_consumed:', (end-start)*1000,'ms' )

        if self.control_type=='closed_loop':
            self.u = self.u0[:1,:].T
        elif self.control_type=='open_loop':
            self.u = self.u0[:,:].T

        # if self.control_strategy=='Tracking':
        #     x = self.init_x.copy()
        #     for t in range(self.predict_length):
        #         if t<self.control_length:
        #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(self.u0[t:t+1,:].T) + self.B_bias
        #         else:
        #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
        #         y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))
        #         print('step:',t+1)
        #         print('simulated_y:%.2f  %.2f  %.2f  %.2f'%(y[0,0],y[1,0],y[2,0],y[3,0]))
        #         print('  desired_y:%.2f  %.2f  %.2f  %.2f'%(self.ref[0,0],self.ref[1,0],self.ref[2,0],self.ref[3,0]))
        #         print('\n')
        # elif self.control_strategy=='Activating' or self.control_strategy=='Inhibiting':
        #     x = self.init_x.copy()
        #     for t in range(self.predict_length):
        #         if t<self.control_length:
        #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(self.u0[t:t+1,:].T) + self.B_bias
        #         else:
        #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
        #         y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))
        #         print('step:',t+1)
        #         print('simulated_y:%.2f  %.2f  %.2f  %.2f'%(y[0,0],y[1,0],y[2,0],y[3,0]))
        #         # print('  desired_y:%.2f  %.2f  %.2f  %.2f'%(self.ref[0,0],self.ref[1,0],self.ref[2,0],self.ref[3,0]))
        #         print('\n')
        # elif self.control_strategy=='Activating_channel':
        #     x = self.init_x.copy()
        #     for t in range(self.predict_length):
        #         if t<self.control_length:
        #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(self.u0[t:t+1,:].T) + self.B_bias
        #         else:
        #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
        #         y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))

        #         sum_exp = np.sum(np.exp(y))
        #         print('step:',t+1)
        #         print('simulated_y:%.2f  %.2f  %.2f  %.2f'%(np.exp(y[0,0])/sum_exp,np.exp(y[1,0])/sum_exp,np.exp(y[2,0])/sum_exp,np.exp(y[3,0])/sum_exp))
        #         # print('  desired_y:%.2f  %.2f  %.2f  %.2f'%(self.ref[0,0],self.ref[1,0],self.ref[2,0],self.ref[3,0]))
        #         print('\n')
        # elif self.control_strategy=='Inhibiting_channel':
        #     x = self.init_x.copy()
        #     for t in range(self.predict_length):
        #         if t<self.control_length:
        #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(self.u0[t:t+1,:].T) + self.B_bias
        #         else:
        #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
        #         y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))

        #         sum_exp = np.sum(np.exp(-y))
        #         print('step:',t+1)
        #         print('simulated_y:%.2f  %.2f  %.2f  %.2f'%(np.exp(-y[0,0])/sum_exp,np.exp(-y[1,0])/sum_exp,np.exp(-y[2,0])/sum_exp,np.exp(-y[3,0])/sum_exp))
        #         # print('  desired_y:%.2f  %.2f  %.2f  %.2f'%(self.ref[0,0],self.ref[1,0],self.ref[2,0],self.ref[3,0]))
        #         print('\n')
        
        # print(self.u.shape)
        num_stim_channels = 2
        self.input_amplitudes = {}
        self.input_durations = {}
        self.amplitudes = {}
        self.frequencys = {}
        for i in range(num_stim_channels):
            self.input_amplitudes[i], self.input_durations[i] = self.testOutput_1(self.u[i*2:(i+1)*2,:])
            self.frequencys[i] = self.u[2*i].reshape(1,-1)
            self.amplitudes[i] = self.u[2*i+1].reshape(1,-1)
        # return u0
        end = time.time()
        print('time_consumed:',(end-start)*1000,' ms')
        self.optimal_loss = result.fun
        self.emit_optimal_input_.emit(True)
        # print(u0)
        
    def init_stim(self, num_stim_channels, num_stim):
        self.input_amplitudes = {}
        self.input_durations = {}
        self.amplitudes = {}
        self.frequencys = {}
        for i in range(2):
            self.amplitudes[i] = np.zeros((1,num_stim))
            self.frequencys[i] = np.zeros((1,num_stim))

            self.input_amplitudes[i] , self.input_durations[i] = self.testOutput_1(np.concatenate((self.frequencys[i],self.amplitudes[i]),axis=0))
            self.u[2*i]=self.frequencys[i]
            self.u[2*i+1]=self.amplitudes[i]
    

    def obtain_random_stim(self, num_stim_channels, num_stim,stim_channel=0):
        self.input_amplitudes = {}
        self.input_durations = {}
        self.amplitudes = {}
        self.frequencys = {}
        for i in range(2):
            # if stim_channel
            # 初始化随机刺激参数
            self.amplitudes[i] = np.random.uniform(low=0.5, high=2.0,size=num_stim).reshape(1,num_stim)
            self.frequencys[i] = np.random.randint(low=0, high=15,size=num_stim).reshape(1,num_stim)  # 改动high=12

            probability = np.random.uniform(low=0, high=1,size=1)
            if probability>0.75:
                self.amplitudes[i] = np.zeros((1,num_stim))
                self.frequencys[i] = np.zeros((1,num_stim))
            
            # if self.frequencys[i]<4:
            #     self.frequencys[i]=0
            # probability = np.random.uniform(low=0, high=1,size=1)
            # if probability>0.9:
            #     self.frequencys[i] = np.zeros((1,num_stim))
            # self.frequencys[i]=frequency
            # print(np.concatenate((self.frequencys[i],self.amplitudes[i]),axis=0).shape)
            self.amplitudes[stim_channel] =np.zeros((1,num_stim))
            self.frequencys[stim_channel]=np.zeros((1,num_stim))

            self.input_amplitudes[i] , self.input_durations[i] = self.testOutput_1(np.concatenate((self.frequencys[i],self.amplitudes[i]),axis=0))
            self.u[2*i]=self.frequencys[i]
            self.u[2*i+1]=self.amplitudes[i]
    def set_reference(self,ref):
        self.ref = np.reshape(ref,(self.output_size,1)) 
    def get_optimal_control(self):
        return self.u, self.input_amplitudes, self.input_durations, self.amplitudes, self.frequencys, self.optimal_loss

    @staticmethod
    def testOutput_1(U):
        # 刺激参数（amplitude,frequency）转化为输入参数（amplitude，duration）
        stim_1_freq = U[0, :]# + 4
        stim_1_amp = U[1, :] * 100 #+ 100 # 应该是100
        amplitude = []
        duration = []
        duration_time = 400
        # print('stim_1_freq shape: ',stim_1_freq.shape)
        for i in range(stim_1_freq.shape[0]):
            if stim_1_freq[i] >=4:
                isi = np.round(1000 / np.round(stim_1_freq[i]))
                times = np.trunc(250 / (np.round(1000 / np.round(stim_1_freq[i]))))
                amp = np.round(stim_1_amp[i])
                # time_rest = 250 - np.round(1000 / np.round(stim_1_freq[ i])) * np.trunc(250 / (np.round(1000 / np.round(stim_1_freq[i]))))
                # if time_rest < .5:
                #     time_rest = isi
                # else:
                #     times = times
                for j in range(int(times)):
                    # 设置刺激幅值（μV）和刺激时间（μs）
                    amplitude.append(-amp*1000)
                    duration.append(duration_time/2)
                    amplitude.append(amp*1000)
                    duration.append(duration_time/2)
                    # 设置刺激间隔（μs）
                    amplitude.append(0)
                    duration.append(isi*1000-duration_time)
                
                # time_rest = 250 - sum(duration)/1000
                # time_rest = time_rest if time_rest >=0 else 0
                # print(isi,' times:',times)
                if 250*1000-times*isi*1000-duration_time>=0:
                    amplitude.append(-amp*1000)
                    duration.append(duration_time/2)
                    amplitude.append(amp*1000)
                    duration.append(duration_time/2)
                    amplitude.append(0)
                    duration.append(250*1000-times*isi*1000-duration_time)
                else:
                    amplitude.append(0)
                    duration.append(250*1000-times*isi*1000)
            else:
                
                time_rest = 250
            # 相邻刺激间隔（μs）
            
            # amplitude.append(0)
            # duration.append(np.round(time_rest*1000))
        return amplitude, duration

# import numpy as np
# import sys
# from casadi import *
# import do_mpc
# import PyQt5
# from PyQt5.QtCore import *
# from multiprocessing import Pool
# from scipy.optimize import minimize, LinearConstraint
# import time
# class my_mpc(QThread):
#     emit_optimal_input_ = pyqtSignal(bool)   # 发送bool变量表示已经生成了最优刺激信号
#     def __init__(self,model_params=None,predict_length=6,control_length=5,ref=None,control_strategy='Tracking',control_channel=None,control_type='closed_loop'):
#         # super().__init__()
#         QThread.__init__(self, parent=None)
#         print('thread_id: ', QThread().currentThreadId())
        
#         self.hidden_size=50
#         self.input_size= model_params['B'].shape[1]
#         self.output_size=model_params['C'].shape[0]
#         self.lasttime = time.time()
#         print('input_dim:%d  system_dim:%d'%(self.input_size,self.output_size))
#         model_type = 'discrete' # either 'discrete' or 'continuous'
#         self.model = do_mpc.model.Model(model_type)
#         self.u = np.empty(shape=(self.input_size,1))

#         # self._x = self.model.set_variable(var_type='_x', var_name='x', shape=(self.hidden_size,1))
#         # self._u = self.model.set_variable(var_type='_u', var_name='u', shape=(self.input_size,1))
#         if model_params==None:
#             # load model parameter from checkpoint
#             self.Wrec=np.abs(np.random.randn(self.hidden_size,self.hidden_size))
#             self.Wrec_bias=np.abs(np.random.randn(self.hidden_size,1))

#             self.B = np.abs(np.random.randn(self.hidden_size,self.input_size))
#             self.B_bias= np.abs(np.random.randn(self.hidden_size,1))

#             self.C=np.abs(np.random.randn(self.output_size,self.hidden_size))
#             self.C_bias=np.abs(np.random.randn(self.output_size,1))
#         else:
#             self.Wrec=model_params['Wrec']
#             self.Wrec_bias=model_params['Wrec_bias']

#             self.B = model_params['B']
#             self.B_bias= model_params['B_bias']

#             self.C=model_params['C']
#             self.C_bias=model_params['C_bias']

#         self.predict_length = predict_length
#         self.control_length= control_length

#         self.max_us = np.array([15.0,9.0])
#         self.min_us = np.array([0,0.001])
#         # self.min_us = np.array([[0.0],[0.0],[0.0],[0.0]]).T
#         # self.set_input_bounds(min_us,max_us)

#         self.control_strategy=control_strategy #'Tracking'
#         self.control_channel=control_channel
#         self.control_type = control_type
#         self.ref = np.reshape(ref,(self.output_size,1))
#         self.u0 = np.zeros((self.control_length,self.output_size))
#         self.start_optimize=False

#     def set_reference(self,ref):
#         self.ref = np.reshape(ref,(self.output_size,1))
        
#     def objective(self,u):
#         x = self.init_x.copy()
        
#         u = u.reshape((self.control_length,-1))
#         cost = 0
#         if self.control_strategy=='Tracking':
#             for t in range(self.predict_length):
#                 if t<self.control_length:
#                     x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(u[t:t+1,:].T) + self.B_bias
#                 else:
#                     x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
#                 y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))
#                 cost += np.sum((y - self.ref)**2) #/self.predict_length/4 #  + np.sum(u[t]**2)
#                 #print(y.T,self.ref.T)
#                 # print(self.ref,y)
#         elif self.control_strategy=='Activating':
#             for t in range(self.predict_length):
#                 if t<self.control_length:
#                     x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(u[t:t+1,:].T) + self.B_bias
#                 else:
#                     x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
#                 y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))
#                 cost += -np.sum(y[:,0]**2)

#         elif self.control_strategy=='Inhibiting':
#             for t in range(self.predict_length):
#                 if t<self.control_length:
#                     x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(u[t:t+1,:].T) + self.B_bias
#                 else:
#                     x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
#                 y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))
#                 cost += (0.9**t)*np.sum(y[:,0]**2)

#         elif self.control_strategy=='Activating_channel':
#             for t in range(self.predict_length):
#                 if t<self.control_length:
#                     x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(u[t:t+1,:].T) + self.B_bias
#                 else:
#                     x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
#                 y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))
#                 sum_exp = np.sum(np.exp(y))
#                 cost += - (0.9**t)*exp(y[self.control_channel,0])/sum_exp

#         elif self.control_strategy=='Inhibiting_channel':
#             for t in range(self.predict_length):
#                 if t<self.control_length:
#                     x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(u[t:t+1,:].T) + self.B_bias
#                 else:
#                     x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
#                 y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))
#                 sum_exp = np.sum(np.exp(-y))
#                 cost += - (0.9**t)*np.exp(-y[self.control_channel,0])/sum_exp

#         # cost += 0.0001*np.sum(u**2)
#         return cost
    
#     def constraint(self,u):
#         u = u.reshape((self.control_length,-1))
#         constraints = []
#         for t in range(self.control_length):
#             # print(u[t], self.max_us)
#             # for i in range(4):
#             #     constraints.append(self.max_us[i]-u[t,i])
#             #     constraints.append(u[t,i])
#                 # constraints.append({'type': 'ineq', 'fun': lambda u, t=t, i=i: self.max_us[i] - u[t, i]})
#                 # # Constraint: u[t, i] >= 0 (assuming a lower limit of zero)
#                 # constraints.append({'type': 'ineq', 'fun': lambda u, t=t, i=i: u[t, i]})
#             constraints.extend(self.max_us-u[t])
#             constraints.extend(u[t]-self.min_us) #-self.min_us)
#         return constraints

#     def __del__(self):
#         self.wait()



#     def set_initial_state(self,init_x,control_strategy='Tracking'):
#         self.control_strategy = control_strategy
#         self.init_x=init_x
#         self.start_optimize=True

#     def run(self):
#         while True:
#             if self.start_optimize:
#                 self.start_optimize=False
#                 self.lasttime= time.time()
#                 # self._signal.emit(list(self.obtain_optimal_control()))
#                 self.obtain_optimal_control()
#                 cost_time = time.time() - self.lasttime
#                 print('cost_time of optimization:%.3f ms'%(cost_time*1000))
#             time.sleep(0.02)

#     # def optimize_(self,init_x0):
#     #     constraints = [{'type': 'ineq', 'fun': self.constraint}]
#     #     return minimize(self.objective, 
#     #                 np.reshape(self.u0,-1), 
#     #                 method='Powell', #'COBYLA', #'Nelder-Mead', #'Powell', #'Powell',#  'SLSQP', #'trust-constr',  # 
#     #                 constraints=constraints, 
#     #                 options={'maxiter': 5})

#     def obtain_optimal_control(self,epochs=20):
#         # e = np.ones([self.model.n_x,1])
#         # x0 = np.random.uniform(-3*e,3*e)
#         # for k in range(50):
#         import time
#         start = time.time()
        
        
#         # given an initial state and output the optimal input

#         # initial_u0 = []
#         # for i in range(3):
#         #     init_input_U=np.zeros((self.predict_length, self.input_size))
#         #     init_input_U[:,0]=np.random.uniform(5,10,self.predict_length)#1.0*np.random.randint(4, 20, self.predict_length)
#         #     init_input_U[:,2]=np.random.uniform(5,10,self.predict_length)#1.0*np.random.randint(4, 20, self.predict_length)

#         #     init_input_U[:,1]=np.random.uniform(3,6,self.predict_length)#np.random.normal(5, 2, self.predict_length)
#         #     init_input_U[:,3]=np.random.uniform(3,6,self.predict_length)#np.random.normal(5, 2, self.predict_length)

#         #     self.u0=init_input_U
#         #     initial_u0.append(init_input_U)
#         # from multiprocessing import Pool
#         # with Pool(processes=4) as pool:
#         #     results = pool.map(self.optimize_, initial_u0)

#         # best_result = min(results, key=lambda x: x.fun)
#         # self.u0 = best_result.x.reshape((self.predict_length,4))# 'trust-constr' #
        
#         init_input_U=np.zeros((self.control_length, self.input_size))
#         init_input_U[:,0]=np.random.uniform(5,10,self.control_length)#1.0*np.random.randint(4, 20, self.predict_length)
#         #init_input_U[:,2]=np.random.uniform(5,10,self.control_length)#1.0*np.random.randint(4, 20, self.predict_length)

#         init_input_U[:,1]=np.random.uniform(3,6,self.control_length)#np.random.normal(5, 2, self.predict_length)
#         #init_input_U[:,3]=np.random.uniform(3,6,self.control_length)#np.random.normal(5, 2, self.predict_length)

#         self.u0=init_input_U

#         constraints = [{'type': 'ineq', 'fun': self.constraint}]
#         # print(constraints)
#         result = minimize(self.objective, 
#                           np.reshape(self.u0,-1), 
#                           method='COBYLA',  # 'L-BFGS-B', #'trust-constr', #'Powell',# 'Nelder-Mead',#'COBYLA', #'Powell', # 'SLSQP', #
#                           constraints=constraints,#self.constraint(self.u0), 
#                           options={'maxiter': 200})
#         self.u0 = np.abs(result.x.reshape((self.control_length,self.input_size)))# 'trust-constr' #

#         # print("Optimal cost:", result.fun)
#         print("Optimal u0:", np.around(np.abs(self.u0),2))
#         #print("Optimization result:", result)
#         # print('time_consumed:', (end-start)*1000,'ms' )

#         if self.control_type=='closed_loop':
#             self.u = self.u0[:1,:].T
#         elif self.control_type=='open_loop':
#             self.u = self.u0[:,:].T

#         # if self.control_strategy=='Tracking':
#         #     x = self.init_x.copy()
#         #     for t in range(self.predict_length):
#         #         if t<self.control_length:
#         #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(self.u0[t:t+1,:].T) + self.B_bias
#         #         else:
#         #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
#         #         y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))
#         #         print('step:',t+1)
#         #         print('simulated_y:%.2f  %.2f  %.2f  %.2f'%(y[0,0],y[1,0],y[2,0],y[3,0]))
#         #         print('  desired_y:%.2f  %.2f  %.2f  %.2f'%(self.ref[0,0],self.ref[1,0],self.ref[2,0],self.ref[3,0]))
#         #         print('\n')
#         # elif self.control_strategy=='Activating' or self.control_strategy=='Inhibiting':
#         #     x = self.init_x.copy()
#         #     for t in range(self.predict_length):
#         #         if t<self.control_length:
#         #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(self.u0[t:t+1,:].T) + self.B_bias
#         #         else:
#         #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
#         #         y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))
#         #         print('step:',t+1)
#         #         print('simulated_y:%.2f  %.2f  %.2f  %.2f'%(y[0,0],y[1,0],y[2,0],y[3,0]))
#         #         # print('  desired_y:%.2f  %.2f  %.2f  %.2f'%(self.ref[0,0],self.ref[1,0],self.ref[2,0],self.ref[3,0]))
#         #         print('\n')
#         # elif self.control_strategy=='Activating_channel':
#         #     x = self.init_x.copy()
#         #     for t in range(self.predict_length):
#         #         if t<self.control_length:
#         #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(self.u0[t:t+1,:].T) + self.B_bias
#         #         else:
#         #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
#         #         y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))

#         #         sum_exp = np.sum(np.exp(y))
#                 # print('step:',t+1)
#                 # print('simulated_y:%.2f  %.2f  %.2f  %.2f'%(exp(y[0,0])/sum_exp,exp(y[1,0])/sum_exp,exp(y[2,0])/sum_exp,exp(y[3,0])/sum_exp))
#                 # # print('  desired_y:%.2f  %.2f  %.2f  %.2f'%(self.ref[0,0],self.ref[1,0],self.ref[2,0],self.ref[3,0]))
#                 # print('\n')
#         # elif self.control_strategy=='Inhibiting_channel':
#         #     x = self.init_x.copy()
#         #     for t in range(self.predict_length):
#         #         if t<self.control_length:
#         #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B@(self.u0[t:t+1,:].T) + self.B_bias
#         #         else:
#         #             x = self.Wrec @ np.maximum(x,0) + self.Wrec_bias + self.B_bias
#         #         y = np.log(1+np.exp(self.C@ np.maximum(x,0)+self.C_bias))

#         #         sum_exp = np.sum(np.exp(-y))
#         #         print('step:',t+1)
#         #         print('simulated_y:%.2f  %.2f  %.2f  %.2f'%(exp(-y[0,0])/sum_exp,exp(-y[1,0])/sum_exp,exp(-y[2,0])/sum_exp,exp(-y[3,0])/sum_exp))
#         #         # print('  desired_y:%.2f  %.2f  %.2f  %.2f'%(self.ref[0,0],self.ref[1,0],self.ref[2,0],self.ref[3,0]))
#         #         print('\n')
        
#         # print(self.u.shape)
#         # num_stim_channels = 1
#         self.input_amplitudes = {}
#         self.input_durations = {}
#         self.amplitudes = {}
#         self.frequencys = {}
#         for i in range(int(self.input_size/2)):
#             self.input_amplitudes[i], self.input_durations[i] = self.testOutput_1(self.u[i*2:(i+1)*2,:])
#             self.frequencys[i] = self.u[2*i].reshape(1,-1)
#             self.amplitudes[i] = self.u[2*i+1].reshape(1,-1)
#         # return u0
#         end = time.time()
#         # print('time_consumed:',(end-start)*1000,' ms')
#         self.emit_optimal_input_.emit(True)
#         # print(u0)
        
#     def init_stim(self, num_stim_channels, num_stim):
#         self.input_amplitudes = {}
#         self.input_durations = {}
#         self.amplitudes = {}
#         self.frequencys = {}
#         for i in range(int(self.input_size/2)):
#             self.amplitudes[i] = np.zeros((1,num_stim))
#             self.frequencys[i] = np.zeros((1,num_stim))

#             self.input_amplitudes[i] , self.input_durations[i] = self.testOutput_1(np.concatenate((self.frequencys[i],self.amplitudes[i]),axis=0))
#             self.u[2*i]=self.frequencys[i]
#             self.u[2*i+1]=self.amplitudes[i]
    

#     def obtain_random_stim(self, num_stim_channels, num_stim):
#         self.input_amplitudes = {}
#         self.input_durations = {}
#         self.amplitudes = {}
#         self.frequencys = {}
#         for i in range(int(self.input_size/2)):
#             self.amplitudes[i] = np.random.uniform(low=0, high=9,size=num_stim).reshape(1,num_stim)
#             self.frequencys[i] = np.random.uniform(low=4, high=12,size=num_stim).reshape(1,num_stim)

#             probability = np.random.uniform(low=0, high=1,size=1)
#             if probability>0.75:
#                 self.amplitudes[i] = np.zeros((1,num_stim))
#                 self.frequencys[i] = np.zeros((1,num_stim))
#             # probability = np.random.uniform(low=0, high=1,size=1)
#             # if probability>0.9:
#             #     self.frequencys[i] = np.zeros((1,num_stim))
#             # self.frequencys[i]=frequency
#             # print(np.concatenate((self.frequencys[i],self.amplitudes[i]),axis=0).shape)
#             self.input_amplitudes[i] , self.input_durations[i] = self.testOutput_1(np.concatenate((self.frequencys[i],self.amplitudes[i]),axis=0))
#             self.u[2*i]=self.frequencys[i]
#             self.u[2*i+1]=self.amplitudes[i]
    
#     def get_optimal_control(self):
#         return self.u, self.input_amplitudes, self.input_durations, self.amplitudes, self.frequencys

#     @staticmethod
#     def testOutput_1(U):
#         # 刺激参数（amplitude,frequency）转化为输入参数（amplitude，duration）
#         stim_1_freq = U[0, :]# + 4
#         stim_1_amp = U[1, :] * 100 #+ 100 # 应该是100
#         amplitude = []
#         duration = []
#         duration_time = 400
#         # print('stim_1_freq shape: ',stim_1_freq.shape)
#         for i in range(stim_1_freq.shape[0]):
#             if stim_1_freq[i] >=4:
#                 isi = np.round(1000 / np.round(stim_1_freq[i]))
#                 times = np.trunc(250 / (np.round(1000 / np.round(stim_1_freq[i]))))
#                 amp = np.round(stim_1_amp[i])
#                 # time_rest = 250 - np.round(1000 / np.round(stim_1_freq[ i])) * np.trunc(250 / (np.round(1000 / np.round(stim_1_freq[i]))))
#                 # if time_rest < .5:
#                 #     time_rest = isi
#                 # else:
#                 #     times = times
#                 for j in range(int(times)):
#                     # 设置刺激幅值（μV）和刺激时间（μs）
#                     amplitude.append(-amp*1000)
#                     duration.append(duration_time/2)
#                     amplitude.append(amp*1000)
#                     duration.append(duration_time/2)
#                     # 设置刺激间隔（μs）
#                     amplitude.append(0)
#                     duration.append(isi*1000-duration_time)
                
#                 # time_rest = 250 - sum(duration)/1000
#                 # time_rest = time_rest if time_rest >=0 else 0
#                 # print(isi,' times:',times)
#                 if 250*1000-times*isi*1000-duration_time>=0:
#                     amplitude.append(-amp*1000)
#                     duration.append(duration_time/2)
#                     amplitude.append(amp*1000)
#                     duration.append(duration_time/2)
#                     amplitude.append(0)
#                     duration.append(250*1000-times*isi*1000-duration_time)
#                 else:
#                     amplitude.append(0)
#                     duration.append(250*1000-times*isi*1000)
#             else:
                
#                 time_rest = 250
#             # 相邻刺激间隔（μs）
            
#             # amplitude.append(0)
#             # duration.append(np.round(time_rest*1000))
#         return amplitude, duration