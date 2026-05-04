import numpy as np
import sys
from casadi import *
import do_mpc
import PyQt5
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QIcon
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *
import time


class my_mpc(QThread):
    _signal = pyqtSignal(list)
    def __init__(self,model_params=None):
        # super().__init__()
        QThread.__init__(self, parent=None)
        print('thread_id: ', QThread().currentThreadId())
        
        self.hidden_size=50
        self.input_size= model_params['B'].shape[1]
        self.output_size=model_params['C'].shape[0]
        self.lasttime = time.time()

        model_type = 'discrete' # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(model_type)
        self.u = np.empty(shape=(4,1))
        self.x = np.empty(shape=(self.hidden_size,1))

        self._x = self.model.set_variable(var_type='_x', var_name='x', shape=(self.hidden_size,1))
        self._u = self.model.set_variable(var_type='_u', var_name='u', shape=(self.input_size,1))
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

        
        # define model with network parameters
        self.h_next = (self.Wrec @ fmax(self._x,0)) + self.Wrec_bias + self.B@self._u + self.B_bias

        self.model.set_rhs('x', self.h_next)

        # Build the model
        self.model.setup()

        self.mpc = do_mpc.controller.MPC(self.model)


        setup_mpc = {
            'n_robust': 0,
            'n_horizon': 2,
            't_step': 0.25,
            'state_discretization': 'discrete',
            'store_full_solution':True,
            # Use MA27 linear solver in ipopt for faster calculations:
            'nlpsol_opts': {'ipopt.linear_solver': 'mumps', # 还没拿到MA27的license,'MA27'
                            'ipopt.max_iter':1,
                            'ipopt.print_level':0, 
                            'ipopt.sb': 'yes', 
                            'print_time':0} 
        }

        self.mpc.set_param(**setup_mpc)

        max_us = np.array([[20.0],[5.0]])#,[20.0],[5.0]])
        min_us = np.array([[4.0],[1.0]])#,[4.0],[1.0]])
        self.set_input_bounds(min_us,max_us)

        # design_ref=np.abs(np.random.randn(self.output_size,1))
        # self.set_reference(design_ref)
        
        self.start_optimize=False
        self.set_reference(np.array([1,1,1,1,1,1]).T*3)
        
        self.init_x=np.abs(np.random.randn(self.hidden_size,1))
        for i in range(5):
            self.obtain_optimal_control()

        # self.set_input_bounds(min_us,max_us)
        # self.mpc.flags['setup'] = False
        # self.set_reference(np.array([1,1,1,1,1,1]).T*3)

    def set_reference(self,ref):

        # objective function
        # _x = model.x
        #design_ref=np.abs(np.random.randn(self.output_size,1))
        mterm = norm_2(log(1+exp(self.C@fmax(self._x,0)+self.C_bias))-ref) # terminal cost
        lterm = norm_2(log(1+exp(self.C@fmax(self._x,0)+self.C_bias))-ref) # terminal cost
        

        # stage cost
        while True:
            if not self.start_optimize:
                self.mpc.flags['setup'] = False
                self.mpc.set_objective(mterm=mterm, lterm=lterm)
                break
            else:
                print('Can not set reference!')

        self.mpc.set_rterm(u=1e-3) # input penalty
        self.mpc.setup()
        self.start_optimize=False

    def __del__(self):
        self.wait()

    def set_input_bounds(self,min_us,max_us):
        # bound of inputs
        self.mpc.bounds['lower','_u','u'] = min_us
        self.mpc.bounds['upper','_u','u'] = max_us


    def set_initial_state(self,init_x):
        self.init_x=init_x
        self.start_optimize=True

    def run(self):
        while True:
            if self.start_optimize:
                self.lasttime= time.time()
                # self._signal.emit(list(self.obtain_optimal_control()))
                self.obtain_optimal_control()
                self.start_optimize=False
                cost_time = time.time() - self.lasttime
                print('cost_time:',cost_time*1000)
            time.sleep(0.05)


    def obtain_optimal_control(self):
        # e = np.ones([self.model.n_x,1])
        # x0 = np.random.uniform(-3*e,3*e)
        # for k in range(50):
        import time
        start = time.time()
        # given an initial state and output the optimal input
        u0 = self.mpc.make_step(self.init_x)
        x = self.mpc.data._x[0]
        # end = time.time()
        # print(u0,'\n')
        # print('\n')
        # print('time_consumed:', (end-start)*1000,'ms' )
        self.u = u0
        self.x = x
        num_stim_channels = len(self.u) // 2
        self.input_amplitudes = {}
        self.input_durations = {}
        self.amplitudes = {}
        self.frequencys = {}
        for i in range(num_stim_channels):
            self.input_amplitudes[i], self.input_durations[i] = self.testOutput_1(self.u[i*2:(i+1)*2,:])
            self.frequencys[i] = self.u[2*i].reshape(1,1)
            self.amplitudes[i] = self.u[2*i+1].reshape(1,1)
        # return u0
        end = time.time()
        print('Obtain optimal control:',u0)
        # print('time_consumed:',(end-start)*1000,' ms')

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