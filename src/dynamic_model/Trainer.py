from src.dynamic_model.DataSample import Input_Data
# from test_common_data_for_visulation.DeepKoopman import HighOrderDeepKoopman
import torch
import os
import json
import numpy as np
import scipy.io as scio
from datetime import datetime
from scipy.io import savemat, loadmat
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
# from sklearn.metrics import median_absolute_error
from scipy.stats import ttest_1samp
from src.dynamic_model.Models import Latent_Model
from src.dynamic_model.DataPreprocessing import Preprocessing
# from src.dynamic_model.DataPreprocessing import read_data_params_from_jason
from src.dynamic_model.OptimizeU import MPC_optimization
from src.dynamic_model.utils import PCA_dim_reduction, FA_dim_reduction, LLE_dim_reduction


# batchsize = 200
# gpu = 0
# epochs = 50
# ar_type = [1, 2]
# prediction_len = 500


# tp_prediction_type = [10]

run_device = 'cuda:0'
ckpt_json = 'src/dynamic_model/checkpoints/test-0108.json'
# ckpt_json = 'src/dynamic_model/checkpoints/test-0822.json'
# ckpt_json = r'C:\Users\tju\Desktop\test_close_loop_stimulation\src\dynamic_model\checkpoints\random_test\EI-RNN\03-14-zhichao\init_training\Set7_01-05\_4channels_4Hz_kern_sd10\hidden_100_time_length_3_tp_10\L_factors_0.05_EI_ratio_4_l1_reg_False_learning_rate_0.001_with_Tanh_False_sparsity_0.75_e_clusters_2_i_clusters_1_inter_or_intra_Inter_l2_norm_0.001.json'
# ckpt_path = 'src/dynamic_model/checkpoints/random_test/EI-RNN/03-14-zhichao/init_training/Set7_01-05/_4channels_4Hz_kern_sd10/hidden_100_time_length_3_tp_10/L_factors_0.05_EI_ratio_4_l1_reg_False_learning_rate_0.001_with_Tanh_False_sparsity_0.75_e_clusters_2_i_clusters_1_inter_or_intra_Inter_l2_norm_0.001.ckpt'
ckpt_path = 'src\dynamic_model\checkpoints\random_test\EI-RNN\01-08\fine_tuning\Set13_01-08\_5channels_4Hz_kern_sd10\hidden_50_time_length_3_tp_10\L_factors_0.05_EI_ratio_4_l1_reg_False_learning_rate_0.001_with_Tanh_False_sparsity_1_e_clusters_2_i_clusters_1_inter_or_intra_Inter_l2_norm_0.001.json'

optimize_params = {
        'N_trials': 1,
        'Control_Horizon': 10,
        'Pred_Horizon': 10,
        'iters': 2000,
        # 'Y_ref': np.array(ref) / 3,
        'min_input': [4, 0.0, 4, 0.0],
        'max_input': [20, 0.9, 20, 0.9],
        'minimize_hz': 4
    }

######################################################################################
#########################   加载/训练/FineTune模型   ################################## 
######################################################################################

def get_suffix(special_params):
    suffix = ''
    for k in special_params.keys():
        if k != 'wrec_fix':
            suffix += '%s_%s_' % (k, str(special_params[k]))
    return suffix


def renameFileSuffix(file, suffix):
    filename = os.path.split(file)
    filesion = os.path.splitext(filename[1])
    newfile = filename[0] + '/' + filesion[0] + suffix
    return newfile


class ModelTrainer:
    '''
    *1. 模型训练：数据读取，模型训练，保存ckpt，保存预测结果，保存统计结果，保存模型可视化结果
    *2. 模型微调
    *3. 指定片段预测，input-baseline & output-baseline test
    *4. 输出优化刺激序列

    simple_params = {
                        'json_dir': None
                        'Tp': tp_len,  # 模型可预测数据的步长
                        'Sample_Size': data_Sample_Size,   # 样本数量
                        'data_length': time_length,  # Hyperparameter  # 初始化隐状态的长度
                        'HIDDEN_DIM': hidden,  # 隐状态维度
                        'activation': activate_fun, # 普通网络的激活函数
                        'final_activation': final_activate_fun, # 最后一层网络的激活函数
                        'RNN_Type': model_type, # RNN的类型，包括 GRU,LSTM,RNN,EI-RNN,Koopman
                        'l1_reg': False, # 是否使用L1正则化惩罚模型的参数 (目前都置为False)
                        'with_Tanh': False, # 是否限定EI-RNN模型中的隐状态在-1到1之间 (目前都置为False)
                        'train_day': , # 设置训练的日期
                        'init_ckpt_path': None, # 设置初始化模型参数的路径
                        'test_sample_size': , # 测试样本的数量
                        'prediction_len': ,  # 总共预测的长度，以Tp为单位逐渐叠加
                        'cross_validation': , # 是否交叉验证，True or False
                        'num_folder':  # 采用交叉验证时，使用多少fold
                        'special_params':  # EI-RNN模型的模型参数，主要包括稀疏度，EI-ratio，E-cluster，I-cluster，cluster类型等
                    }

    '''

    def __init__(self, simple_params=ckpt_json, preprocessing=None, json_file_type=False):
        self.simple_params = simple_params 
        if isinstance(simple_params, str):
            json_file_type = True
        elif isinstance(simple_params, dict):
            json_file_type = False

        try:
            if json_file_type:
                self.simple_params = self.read_params_from_json()
        except:
            print('Error in reading model params...')
            print(self.simple_params)
        else:
            print('Successfully loading model params!')

        # 数据预处理
        # self.preprocessing = None
        if preprocessing is None or isinstance(preprocessing, str):
            self.preprocessing = Preprocessing(self.simple_params['json_dir'], json_file_type=True)
        else:
            self.preprocessing = preprocessing
        self.testPreprocessing = Preprocessing(self.simple_params['test_json_dir'],
                                               json_file_type=True) if 'test_json_dir' in self.simple_params else None
        self.load_data_params = None
        self.model_params = None

        self.number_recording_channels = 6 #int(np.sum(self.preprocessing.selected_channels))
        self.Tp = self.simple_params['Tp']
        self.Sample_Size = self.simple_params['Sample_Size']
        self.data_length = self.simple_params['data_length']
        self.ext_input_dim = 4 #self.preprocessing.stim_channels * 2

        self.hidden_dimension = self.simple_params['HIDDEN_DIM']
        self.activation = self.simple_params['activation']
        self.final_activation = self.simple_params['final_activation']
        self.RNN_Type = self.simple_params['RNN_Type']
        self.L1_reg = self.simple_params['l1_reg']
        self.with_Tanh = self.simple_params['with_Tanh']
        self.training_day = self.simple_params['train_day']
        self.init_checkpoint_path = self.simple_params['init_ckpt_path']
        self.test_sample_size = self.simple_params['test_sample_size']
        self.prediction_length = self.simple_params['prediction_len']
        self.cross_validation = self.simple_params['cross_validation']
        self.number_folders = self.simple_params['num_folder'] if self.cross_validation else 1
        self.special_params = self.simple_params['special_params']
        self.extra_params = self.simple_params['extra_params']
        self.checkpoint_save_dir = None
        self.result_save_dir = None
        self.model = None
        self.statistic_result = {}
        self.complete_params()
        self.epochs = self.simple_params['epochs']

    def read_params_from_json(self):
        # print('Loading model params from json...')
        if not os.path.exists(self.simple_params):
            print('model params json file not found!')
            print(self.simple_params)
        f = open(self.simple_params, 'r', encoding='UTF-8')
        content = f.read()
        params = json.loads(content)
        f.close()
        if 'wrec_fix' in params['extra_params']:
            params['extra_params']['wrec_fix'] = np.array(params['extra_params']['wrec_fix'])
        return params

    def get_dir(self, init_dir):
        save_dir = init_dir
        save_dir += 'cross_validation/' if self.cross_validation else 'random_test/'
        save_dir += '%s/%s/' % (self.RNN_Type, self.training_day)
        save_dir += 'fine_tuning/' if self.init_checkpoint_path is not None else 'init_training/'
        # save_dir += '%s_%s_%dchannels_%dHz_kern_sd%d/' % (
        #     '29', self.preprocessing.day, self.number_recording_channels,
        #     self.preprocessing.data_sampling_frequency, self.preprocessing.smth_kern_sd)
        save_dir += 'hidden_%d_time_length_%d_tp_%d/' % (
            self.hidden_dimension, self.data_length, self.Tp)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        return save_dir

    def get_checkpoint_dir(self):
        self.checkpoint_save_dir = self.get_dir('./checkpoints/')

    def get_result_dir(self):
        self.result_save_dir = self.get_dir('./result_summary/')

    def set_testPreprocessing(self, path):
        self.testPreprocessing = Preprocessing(path, json_file_type=True)

    def complete_params(self):
        self.load_data_params = {
            'Num_Cortical': self.number_recording_channels,
            'Tp': self.Tp,
            'Sample_Size': self.Sample_Size,
            'data_length': self.data_length,  # Hyperparameter
            'with_input': True,
            'data_type': 'GroundTruth',  # 'GroundTruth'  '1dB'  '5dB'  '10dB'  '15dB'  '20dB'
            'input_index': 'Seizure',  # Seizure  NonSeizure  Both
            'ext_input_dim': self.ext_input_dim,
            # input_index= Seizure  1; input_index= NonSeizure  1; input_index= Both  2;
            'AR_order': 1
        }
        self.model_params = {
            'Tp': self.Tp,
            'IN_DIM': self.number_recording_channels,
            'HIDDEN_DIM': self.hidden_dimension,
            'LATENT_DIM': self.hidden_dimension,  # Hyperparameter for the hidden dimension of GRU
            # symbol for loss selection
            'X_recon': 500,
            'Y_recon': 1,
            'Tp_recon': 20,
            'Linear_Loss': 100,
            'data_length': self.data_length,
            # Hyperparameter Previous Steps for Initializing the Hidden State
            'with_input': True,
            'ext_input_dim': self.ext_input_dim,
            'AR_order': 1,
            'StateDependent': False,
            'Conv': False,
            'activation': self.activation,
            'final_activation': self.final_activation,
            'RNN_Type': self.RNN_Type,
            'L_factors': 0.05,
            'device': 'cpu',
            # 'EI_ratio': self.EI_ratio,
            'l1_reg': self.L1_reg,
            'with_Tanh': self.with_Tanh
        }

        if self.special_params is not None:
            for k in self.special_params.keys():
                self.model_params[k] = self.special_params[k]
        if self.extra_params is not None:
            for k in self.extra_params.keys():
                self.model_params[k] = self.extra_params[k]

    def save_params2json(self, params, path):
        if self.RNN_Type == 'EI-RNN' and 'wrec_fix' in params['extra_params']:
            params['extra_params']['wrec_fix'] = self.model.rnn_cell.get_w_rec_fix().tolist()

        save = json.dumps(params, ensure_ascii=False, indent=4)
        f = open(path + '.json', 'w', encoding='utf-8')
        f.write(save)
        f.close()

    def load_model(self, checkpoint_path=ckpt_path):
        self.model = Latent_Model(self.model_params)
        # print(checkpoint_path)
        if os.path.exists(checkpoint_path):
            print("Loading checkpoint...")
            print(checkpoint_path)
            # self.model = self.model.load_state_dict(checkpoint_path)
            print(self.model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))))
            if self.RNN_Type == 'EI-RNN':
                jsonFileName = renameFileSuffix(checkpoint_path, suffix='.json')
                a = ModelTrainer(jsonFileName, json_file_type=True)
                if 'wrec_fix' in a.model_params and a.model_params['wrec_fix'] is not None:
                    print("Refreshing W_rec mask...")
                    # self.model.rnn_cell.set_w_rec_fix(a.model_params['wrec_fix'])
                    self.model_params['wrec_fix'] = a.model_params['wrec_fix']
        if self.RNN_Type == 'EI-RNN' and ('wrec_fix' in self.model_params) and self.model_params[
            'wrec_fix'] is not None:
            print("Loading W_rec mask...")
            # print(type(self.model_params['wrec_fix']))
            self.model.rnn_cell.set_w_rec_fix(self.model_params['wrec_fix'])

    def training(self):
        self.get_checkpoint_dir()
        self.get_result_dir()

        self.statistic_result = {
            'R2': np.zeros((self.number_folders, self.test_sample_size)),
            'mse': np.zeros((self.number_folders, self.test_sample_size)),
            'EV': np.zeros((self.number_folders, self.test_sample_size)),
            'R2_by_channel': np.zeros((self.number_folders, self.test_sample_size, self.number_recording_channels)),
            'EV_by_channel': np.zeros((self.number_folders, self.test_sample_size, self.number_recording_channels)),
            'mse_by_channel': np.zeros((self.number_folders, self.test_sample_size, self.number_recording_channels)),
        }
        input_data = Input_Data(self.load_data_params, self.preprocessing.recording_data,
                                self.preprocessing.stimulating_data)
        if self.cross_validation:
            train_dataloader_dir, start_indexes_dir = data_partition_with_cross_validation(input_data, num_trails=len(
                self.preprocessing.recording_file_sequence_keys), num_of_folder=self.number_folders, batchsize=200)
        else:
            train_dataloader_dir, start_indexes_dir = data_partition(input_data, batchsize=200)

        suffix = get_suffix(self.special_params)
        for i in range(self.number_folders):
            if self.cross_validation:
                ckpt_suffix = suffix + 'cross_validation_%d.ckpt' % (i + 1)
                train_dataloader = train_dataloader_dir[i]
                start_indexs = start_indexes_dir[i]
            else:
                ckpt_suffix = suffix[:-1] + '.ckpt'
                train_dataloader = train_dataloader_dir
                start_indexs = start_indexes_dir
            print('ckpt: ', self.checkpoint_save_dir + ckpt_suffix)
            if self.init_checkpoint_path is not None:
                self.load_model(self.init_checkpoint_path)
            else:
                self.load_model(self.checkpoint_save_dir + ckpt_suffix)

            if os.path.exists(self.checkpoint_save_dir + ckpt_suffix):
                print(self.checkpoint_save_dir + ckpt_suffix + ' Exists')
                continue

            # print(self.checkpoint_save_dir + ckpt_suffix)
            trainer = Trainer(max_epochs=self.epochs, gpus=[0])
            trainer.fit(self.model, train_dataloader)
            trainer.save_checkpoint(self.checkpoint_save_dir + ckpt_suffix)
            self.save_params2json(params=self.simple_params, path=self.checkpoint_save_dir + ckpt_suffix[:-5])

            self.testing(i, input_data, start_indexs, prediction_len=500)

            if self.RNN_Type == 'EI-RNN':
                self.save_Wrec(suffix + 'cross_validation_%d' % (i + 1))
        self.get_mean_of_statistic_result()
        result_suffix = suffix + '%dfolder_cross_validation.mat' if self.cross_validation else suffix + '.mat'
        savemat(self.result_save_dir + result_suffix, self.statistic_result)
        print('------------------End of Training-----------------')

    def testing(self, index_folder, input_data, start_indexs, prediction_len=500):
        #   待加入功能
        #   1.加入std
        #   2.不同测试情况
        prediction_result = np.zeros([self.test_sample_size, self.prediction_length, self.number_recording_channels])
        for run_index in range(self.test_sample_size):
            start_index = start_indexs[run_index]
            GroundTruth = input_data.X[start_index:(start_index + prediction_len), :]
            pred_gru = model_prediction(self.model.to('cuda:0'), self.load_data_params, input_data, start_index,
                                        model_param=None, model_type=self.RNN_Type)
            prediction_result[run_index, :, :] = pred_gru[0, :, :]
            self.statistic_result['R2'][index_folder, run_index] = r2_score(GroundTruth[:, :], pred_gru[0, :, :])
            self.statistic_result['mse'][index_folder, run_index] = mean_squared_error(GroundTruth[:, :],
                                                                                       pred_gru[0, :, :])
            self.statistic_result['EV'][index_folder, run_index] = explained_variance_score(GroundTruth[:, :],
                                                                                            pred_gru[0, :, :])
            for c in range(self.number_recording_channels):
                self.statistic_result['R2_by_channel'][index_folder, run_index, c] = r2_score(GroundTruth[:, c],
                                                                                              pred_gru[0, :, c])
                self.statistic_result['mse_by_channel'][index_folder, run_index, c] = mean_squared_error(
                    GroundTruth[:, c], pred_gru[0, :, c])
                self.statistic_result['EV_by_channel'][index_folder, run_index, c] = explained_variance_score(
                    GroundTruth[:, c], pred_gru[0, :, c])
        if self.cross_validation:
            print('------------------------%dfolder--------------------' % (index_folder + 1))

        print("R2:", np.mean(self.statistic_result['R2'][index_folder]), "_std:",
              np.std(self.statistic_result['R2'][index_folder]), "\n")
        print("MSE:", np.mean(self.statistic_result['mse'][index_folder]), "_std:",
              np.std(self.statistic_result['mse'][index_folder]), "\n")  # ,
        print("EV:", np.mean(self.statistic_result['EV'][index_folder]), "_std:",
              np.std(self.statistic_result['EV'][index_folder]), "\n")

    def cross_day_testing(self):
        self.get_checkpoint_dir()
        self.get_result_dir()

        self.statistic_result = {
            'R2': np.zeros((self.number_folders, self.test_sample_size)),
            'mse': np.zeros((self.number_folders, self.test_sample_size)),
            'EV': np.zeros((self.number_folders, self.test_sample_size)),
            'R2_by_channel': np.zeros((self.number_folders, self.test_sample_size, self.number_recording_channels)),
            'EV_by_channel': np.zeros((self.number_folders, self.test_sample_size, self.number_recording_channels)),
            'mse_by_channel': np.zeros((self.number_folders, self.test_sample_size, self.number_recording_channels)),
        }
        input_data = Input_Data(self.load_data_params, self.testPreprocessing.recording_data,
                                self.testPreprocessing.stimulating_data)
        if self.cross_validation:
            train_dataloader_dir, start_indexes_dir = data_partition_with_cross_validation(input_data, num_trails=len(
                self.testPreprocessing.recording_file_sequence_keys), num_of_folder=self.number_folders, batchsize=200)
        else:
            train_dataloader_dir, start_indexes_dir = data_partition(input_data, batchsize=200)

        suffix = get_suffix(self.special_params)
        print('------------------Start Cross-day testing-----------------')
        for index_folder in range(self.number_folders):
            if self.cross_validation:
                ckpt_suffix = suffix + 'cross_validation_%d.ckpt' % (index_folder + 1)
                start_indexs = start_indexes_dir[index_folder]
            else:
                ckpt_suffix = suffix[:-1] + '.ckpt'
                start_indexs = start_indexes_dir
            print('ckpt: ', self.checkpoint_save_dir + ckpt_suffix)
            if self.init_checkpoint_path is not None:
                self.load_model(self.init_checkpoint_path)
            else:
                self.load_model(self.checkpoint_save_dir + ckpt_suffix)

            # input_datas = self.separate_avg_testing_data()、
            print("\n sparsity:" + str(self.model_params['sparsity']) \
                  + "\n e_clusters_" + str(self.model_params['e_clusters']) \
                  + "\n i_clusters_" + str(self.model_params['i_clusters']) \
                  + '\n ' + self.model_params['inter_or_intra'])

            self.testing(index_folder, input_data, start_indexs, prediction_len=500)

            # if self.RNN_Type == 'EI-RNN':
            #     self.save_Wrec(suffix + 'cross_validation_%d' % (index_folder + 1))
        self.get_mean_of_statistic_result()
        suffix += '_cross_day_%s_' % self.testPreprocessing.day[:-1]
        result_suffix = suffix + '%dfolder_cross_validation.mat' if self.cross_validation else suffix + '.mat'
        savemat(self.result_save_dir + result_suffix, self.statistic_result)
        print('save_result: ', self.result_save_dir + result_suffix)
        print('------------------End of Cross-day testing-----------------')

    def avg_testing(self, start_index, prediction_len, max_y=80, max_x=200):
        import seaborn as sns
        self.get_checkpoint_dir()
        save_dir = self.get_dir('./data_analysis/statistic_fig/avg_testing/')

        data_path = self.testPreprocessing.address + 'FR_data.mat'
        df = mat2pd(file=data_path, channels=self.number_recording_channels,
                    trail_num=len(self.testPreprocessing.recording_file_sequence_keys))

        avg_recording_data = self.testPreprocessing.avg_recording_data()
        avg_stimulating_data1, avg_stimulating_data2 = self.testPreprocessing.avg_stimulating_data()

        input_data = Input_Data(self.load_data_params, avg_recording_data, avg_stimulating_data1)

        suffix = get_suffix(self.special_params)
        print('------------------Start average testing-----------------')
        for index_folder in range(self.number_folders):
            ckpt_suffix = suffix[:-1] + '.ckpt'
            # start_indexs = start_index
            print('ckpt: ', self.checkpoint_save_dir + ckpt_suffix)
            self.load_model(self.checkpoint_save_dir + ckpt_suffix)

            # input_datas = self.separate_avg_testing_data()、
            print("\n sparsity:" + str(self.model_params['sparsity']) \
                  + "\n e_clusters_" + str(self.model_params['e_clusters']) \
                  + "\n i_clusters_" + str(self.model_params['i_clusters']) \
                  + '\n ' + self.model_params['inter_or_intra'])

            # self.testing(index_folder, input_data, start_indexs, prediction_len=500)
            GroundTruth = input_data.X[start_index:(start_index + prediction_len), :]
            pred_gru = model_prediction(self.model.to('cuda:0'), self.load_data_params, input_data, start_index,
                                        model_param=None, model_type=self.RNN_Type, prediction_len=prediction_len)
            prediction_result = np.zeros([prediction_len, self.number_recording_channels])
            prediction_result[:, :] = pred_gru[0, :, :] * 3
            pre_data = prediction_result.transpose()
            sample_fre = self.testPreprocessing.data_sampling_frequency
            times = np.linspace(start_index / sample_fre, (start_index + prediction_len) / sample_fre, prediction_len)
            data_pre = np.column_stack((np.transpose(pre_data), times.flatten(order='F')))
            columns = []
            for i in range(self.number_recording_channels):
                columns.append('channel_%d' % (i + 1))
            columns.append('time')

            df_gt = pd.DataFrame(np.column_stack((GroundTruth * 3, times.flatten(order='F'))), columns=columns)
            df_pre = pd.DataFrame(data_pre, columns=columns)

            # 画每个通道的结果图
            for i in range(self.number_recording_channels):
                R2 = r2_score(GroundTruth[:, i], pred_gru[0, :, i])
                EV = explained_variance_score(GroundTruth[:, i], pred_gru[0, :, i])

                plt.figure(figsize=(4, 3), dpi=300)
                axes = plt.axes()
                axes.spines['top'].set_visible(False)
                axes.spines['right'].set_visible(False)
                save_figure_file = 'channel ' + str(i + 1)
                plt.ylim(0, max_y)
                plt.xlim(5, max_x)
                plt.title('R2 = %.2f\n' % R2 + 'EV = %.2f\n' % EV + save_figure_file)
                plt.ylabel('Firing rate')
                plt.xlabel('Time(s)')
                # sns.relplot(x='time',y='channel_%d'%(i+1),data=df,kind='line',ci='sd')
                sns.lineplot(x='time', y='channel_%d' % (i + 1), data=df, ci='sd',
                             color='k')  # , label='avg firing rate')
                # sns.lineplot(x='time', y='channel_%d' % (i + 1), data=df_gt, ci='sd', color='k')
                sns.lineplot(x='time', y='channel_%d' % (i + 1), data=df_pre, ci='sd',
                             color='orange')  # , label='predicton firing rate')
                # plt.legend(fontsize='large', title_fontsize="20", loc=2, bbox_to_anchor=(1, 1))
                save_dir_next = save_dir + '/%s_sparsity=%s/' % (
                    self.model_params['inter_or_intra'], str(self.model_params['sparsity']))
                if not os.path.isdir(save_dir_next):
                    os.makedirs(save_dir_next)
                save_fig = save_dir_next + 'avg_testing_channel' + str(i) + '.png'
                plt.tight_layout()
                plt.savefig(save_fig)
                plt.close()

            # if self.RNN_Type == 'EI-RNN':
            #     self.save_Wrec(suffix + 'cross_validation_%d' % (index_folder + 1))
        # self.get_mean_of_statistic_result()
        # suffix += '_cross_day_%s_' % self.testPreprocessing.day[:-1]
        # result_suffix = suffix + '%dfolder_cross_validation.mat' if self.cross_validation else suffix + '.mat'
        # savemat(self.result_save_dir + result_suffix, self.statistic_result)
        # print('save_result: ', self.result_save_dir + result_suffix)
        print('------------------End of average testing-----------------')
        return None

    def input_baseline_test(self, start_index, prediction_len, run_times=1000):
        # input-baseline test
        import seaborn as sns
        self.get_checkpoint_dir()
        save_dir = self.get_dir('./data_analysis/statistic_fig/baseline_test/')

        data_path = self.testPreprocessing.address + 'FR_data.mat'
        df = mat2pd(file=data_path, channels=self.number_recording_channels,
                    trail_num=len(self.testPreprocessing.recording_file_sequence_keys))

        avg_recording_data = self.testPreprocessing.avg_recording_data()
        avg_stimulating_data1, avg_stimulating_data2 = self.testPreprocessing.avg_stimulating_data()

        input_data = Input_Data(self.load_data_params, avg_recording_data, avg_stimulating_data1)

        suffix = get_suffix(self.special_params)
        print('------------------Start input baseline test-----------------')
        for index_folder in range(self.number_folders):
            ckpt_suffix = suffix[:-1] + '.ckpt'
            # start_indexs = start_index
            print('ckpt: ', self.checkpoint_save_dir + ckpt_suffix)
            self.load_model(self.checkpoint_save_dir + ckpt_suffix)

            # input_datas = self.separate_avg_testing_data()、
            print("\n sparsity:" + str(self.model_params['sparsity']) \
                  + "\n e_clusters_" + str(self.model_params['e_clusters']) \
                  + "\n i_clusters_" + str(self.model_params['i_clusters']) \
                  + '\n ' + self.model_params['inter_or_intra'])

            # self.testing(index_folder, input_data, start_indexs, prediction_len=500)
            GroundTruth = input_data.X[start_index:(start_index + prediction_len), :]
            pred_gru = model_prediction(self.model.to('cuda:0'), self.load_data_params, input_data, start_index,
                                        model_param=None, model_type=self.RNN_Type, prediction_len=prediction_len)
            prediction_result = np.zeros([prediction_len, self.number_recording_channels])
            R2 = r2_score(GroundTruth[:, :], pred_gru[0, :, :])
            EV = explained_variance_score(GroundTruth[:, :], pred_gru[0, :, :])

            input_baseline_test_R2 = np.zeros(run_times)
            input_baseline_test_EV = np.zeros(run_times)
            for i in range(run_times):
                length = self.testPreprocessing.data_sampling_frequency * self.testPreprocessing.time_length
                random_stim_data = np.zeros((4, length))
                random_stim_data[0] = np.random.randint(0, 20, length)
                random_stim_data[1] = np.random.randint(0, 9, length)
                random_stim_data[2] = np.random.randint(0, 20, length)
                random_stim_data[3] = np.random.randint(0, 9, length)
                random_input_data = Input_Data(self.load_data_params, avg_recording_data, random_stim_data)
                input_baseline_pred_gru = model_prediction(self.model.to('cuda:0'), self.load_data_params,
                                                           random_input_data, start_index, model_param=None,
                                                           model_type=self.RNN_Type, prediction_len=prediction_len)
                input_baseline_test_R2[i] = r2_score(GroundTruth[:, :], input_baseline_pred_gru[0, :, :])
                input_baseline_test_EV[i] = explained_variance_score(GroundTruth[:, :],
                                                                     input_baseline_pred_gru[0, :, :])
            # 画100次input-baseline结果的统计直方图
            metrics = ['R2', 'EV']
            GTs = {'R2': R2, 'EV': EV}
            baselines = {'R2': input_baseline_test_R2, 'EV': input_baseline_test_EV}
            for metric in metrics:
                color_red = sns.color_palette("Set1")[0]
                # plt.figure(figsize=(3, 4))
                plt.figure(figsize=(4, 3))
                axes = plt.axes()
                axes.spines['top'].set_visible(False)
                axes.spines['right'].set_visible(False)
                plt.ylabel('Probability')
                plt.xlabel('Prediction accuracy(%s)' % metric)
                # plt.xlim(-0.2,0.2)
                # 手动归一化
                # normed_input_baseline_test_R2 = input_baseline_test_R2 - np.mean(input_baseline_test_R2)
                # normed_input_baseline_test_EV = input_baseline_test_EV - np.mean(input_baseline_test_EV)
                # normed_R2 = R2 - np.mean(input_baseline_test_R2)
                # normes_EV = EV - np.mean(input_baseline_test_EV)
                weights = np.ones_like(baselines[metric]) / float(len(baselines[metric]))
                plt.hist(baselines[metric], bins=20, color='gray', weights=weights)  # 返回值元组
                plt.axvline(x=GTs[metric], color='orange')  # , label='reference firing rate')
                plt.tight_layout()

                p_val = ttest_1samp(baselines[metric], GTs[metric]).pvalue
                if p_val == 0.0:
                    p_str = "-log10($\mathit{:}$)>25".format('{p}')
                elif p_val < 0.05:
                    p_str = '$\mathit{:}$ = {:0.0e}'.format('{p}', p_val)
                else:
                    p_str = "$\mathit{:}$ = {:.3f}".format('{p}', p_val)

                textstr = '{:}'.format(p_str)
                axes.text(R2 - (np.abs(R2) * 0.0025), axes.get_ylim()[1], textstr,
                          horizontalalignment='left', verticalalignment='top',
                          rotation=0, c='k')
                save_dir_next = save_dir + '/%s_sparsity=%s/' % (
                    self.model_params['inter_or_intra'], str(self.model_params['sparsity']))
                if not os.path.isdir(save_dir_next):
                    os.makedirs(save_dir_next)
                save_fig = save_dir_next + 'input_baseline_test_%s.png' % metric
                plt.tight_layout()
                plt.savefig(save_fig)
                plt.close()

    def output_baseline_test(self, start_index, prediction_len, run_times=1000):
        # output-baseline test
        import seaborn as sns
        self.get_checkpoint_dir()
        save_dir = self.get_dir('./data_analysis/statistic_fig/baseline_test/')

        data_path = self.testPreprocessing.address + 'FR_data.mat'
        df = mat2pd(file=data_path, channels=self.number_recording_channels,
                    trail_num=len(self.testPreprocessing.recording_file_sequence_keys))

        avg_recording_data = self.testPreprocessing.avg_recording_data()
        avg_stimulating_data1, avg_stimulating_data2 = self.testPreprocessing.avg_stimulating_data()

        input_data = Input_Data(self.load_data_params, avg_recording_data, avg_stimulating_data1)

        suffix = get_suffix(self.special_params)
        print('------------------Start output baseline test-----------------')
        for index_folder in range(self.number_folders):
            ckpt_suffix = suffix[:-1] + '.ckpt'
            # start_indexs = start_index
            print('ckpt: ', self.checkpoint_save_dir + ckpt_suffix)
            self.load_model(self.checkpoint_save_dir + ckpt_suffix)

            # input_datas = self.separate_avg_testing_data()、
            print("\n sparsity:" + str(self.model_params['sparsity']) + "\n e_clusters_" + str(
                self.model_params['e_clusters']) + "\n i_clusters_" + str(self.model_params['i_clusters']) + '\n ' +
                  self.model_params['inter_or_intra'])

            # self.testing(index_folder, input_data, start_indexs, prediction_len=500)
            GroundTruth = input_data.X[start_index:(start_index + prediction_len), :]
            pred_gru = model_prediction(self.model.to('cuda:0'), self.load_data_params, input_data, start_index,
                                        model_param=None, model_type=self.RNN_Type, prediction_len=prediction_len)
            prediction_result = np.zeros([prediction_len, self.number_recording_channels])
            R2 = r2_score(GroundTruth[:, :], pred_gru[0, :, :])
            EV = explained_variance_score(GroundTruth[:, :], pred_gru[0, :, :])

            output_baseline_test_R2 = np.zeros(run_times)
            output_baseline_test_EV = np.zeros(run_times)
            for i in range(run_times):
                shuffled_rec_data = avg_recording_data
                for j in range(self.number_recording_channels):
                    np.random.shuffle(shuffled_rec_data[j])
                random_input_data = Input_Data(self.load_data_params, shuffled_rec_data, avg_stimulating_data1)
                output_baseline_pred_gru = model_prediction(self.model.to('cuda:0'), self.load_data_params,
                                                            random_input_data, start_index, model_param=None,
                                                            model_type=self.RNN_Type, prediction_len=prediction_len)
                output_baseline_test_R2[i] = r2_score(GroundTruth[:, :], output_baseline_pred_gru[0, :, :])
                output_baseline_test_EV[i] = explained_variance_score(GroundTruth[:, :],
                                                                      output_baseline_pred_gru[0, :, :])
            # 画100次output-baseline结果的统计直方图
            metrics = ['R2', 'EV']
            GTs = {'R2': R2, 'EV': EV}
            baselines = {'R2': output_baseline_test_R2, 'EV': output_baseline_test_EV}
            for metric in metrics:
                # plt.figure(figsize=(3, 4))
                plt.figure(figsize=(4, 3))
                axes = plt.axes()
                axes.spines['top'].set_visible(False)
                axes.spines['right'].set_visible(False)
                plt.ylabel('Probability')
                plt.xlabel('Prediction accuracy(%s)' % metric)
                # plt.xlim(-0.2,0.2)
                # 手动归一化
                # normed_input_baseline_test_R2 = input_baseline_test_R2 - np.mean(input_baseline_test_R2)
                # normed_input_baseline_test_EV = input_baseline_test_EV - np.mean(input_baseline_test_EV)
                # normed_R2 = R2 - np.mean(input_baseline_test_R2)
                # normes_EV = EV - np.mean(input_baseline_test_EV)
                weights = np.ones_like(baselines[metric]) / float(len(baselines[metric]))
                plt.hist(baselines[metric], bins=20, color='gray', weights=weights)  # 返回值元组
                plt.axvline(x=GTs[metric], color='orange')  # , label='reference firing rate')
                plt.tight_layout()

                p_val = ttest_1samp(baselines[metric], GTs[metric]).pvalue
                if p_val == 0.0:
                    p_str = "-log10($\mathit{:}$)>25".format('{p}')
                elif p_val < 0.05:
                    p_str = '$\mathit{:}$ = {:0.0e}'.format('{p}', p_val)
                else:
                    p_str = "$\mathit{:}$ = {:.3f}".format('{p}', p_val)

                textstr = '{:}'.format(p_str)
                axes.text(R2 - (np.abs(R2) * 0.0025), axes.get_ylim()[1], textstr,
                          horizontalalignment='left', verticalalignment='top',
                          rotation=0, c='k')
                save_dir_next = save_dir + '/%s_sparsity=%s/' % (
                    self.model_params['inter_or_intra'], str(self.model_params['sparsity']))
                if not os.path.isdir(save_dir_next):
                    os.makedirs(save_dir_next)
                save_fig = save_dir_next + 'output_baseline_test_%s.png' % metric
                plt.tight_layout()
                plt.savefig(save_fig)
                plt.close()

    def separate_avg_testing_data(self):
        trail_num = len(self.testPreprocessing.recording_file_sequence_keys)
        num_stim_type = len(self.testPreprocessing.stimulating_file_sequence_keys)
        time_slices = self.testPreprocessing.data_sampling_frequency * self.testPreprocessing.time_length
        trail_avg_data = np.zeros([num_stim_type, self.number_recording_channels, time_slices])
        stim_data = np.zeros([num_stim_type, self.testPreprocessing.stim_channels * 2, time_slices])
        counter = np.zeros(num_stim_type)
        input_datas = {}
        for i in range(trail_num):
            stim_type = int(i % num_stim_type)
            counter[stim_type] += 1
            trail_avg_data[stim_type] += self.testPreprocessing.recording_data[:, i * time_slices:(i + 1) * time_slices]
        for stim_type in range(num_stim_type):
            trail_avg_data[stim_type] /= counter[stim_type]
            stim_data[stim_type] = self.testPreprocessing.stimulating_data[:,
                                   stim_type * time_slices:(stim_type + 1) * time_slices]
            # input_datas[stim_data] = Input_Data(self.load_data_params, trail_avg[stim_type], input[stim_type])
            input_datas[stim_data] = Input_Data(self.load_data_params, trail_avg_data[stim_type], input[stim_type])

        # for stim_type in range(num_stim_type):
        #     input_data = Input_Data(self.load_data_params, trail_avg[stim_type], input[stim_type])
        #     GroundTruth = input_data.X[start_index:(start_index + prediction_len), :]
        #     pred_gru = model_prediction(model.to('cuda:0'), load_data_params, input_data, start_index, model_param=None,
        #                                 model_type=self.RNN_Type, prediction_len=prediction_len)
        #
        # times = np.linspace(0, self.preprocessing.time_length, time_slices)
        # time_arraies = {}
        # for stim_type in range(num_stim_type):
        #     time_arraies[stim_type] = times
        #     for i in range(int(counter[stim_type]-1)):
        #         time_arraies[stim_type]
        return input_datas

    def dynamics_analysis(self, generate_length, start_index=0):
        self.get_checkpoint_dir()
        save_dir = self.get_dir('./data_analysis/dynamics_analysis/')

        print('------------------Start dynamic analysis-----------------')
        suffix = get_suffix(self.special_params)
        for index_folder in range(self.number_folders):
            ckpt_suffix = suffix[:-1] + '.ckpt'
            # start_indexs = start_index
            print('ckpt: ', self.checkpoint_save_dir + ckpt_suffix)
            self.load_model(self.checkpoint_save_dir + ckpt_suffix)
            self.model.to('cuda:0')
            ###################################################################
            ## generate 1000 trials with random hidden_init and zero inputs ###
            ###################################################################
            # if start_index_==0:
            # self.load_data_params['Tp'] = generate_length
            tanh_func = torch.nn.Tanh()
            trials = 20
            functional_dynamics = np.zeros((self.model_params['HIDDEN_DIM'], trials * generate_length))
            for trial in range(trials):
                print('functional_dynamics trails: ', trial, end='\r')
                # torch.normal(mean=0.5, std=torch.arange(1., model_params['HIDDEN_DIM']))
                hidden_init = tanh_func(torch.randn((1, 1, self.model_params['HIDDEN_DIM'])))  #
                for t_step in range(generate_length):
                    # inputs = Inputs[:, (time + model.data_len - 1 + i):(time + model.data_len + i), :]

                    inputs = torch.zeros((1, 1, 4))
                    output, hidden_init = self.model.rnn_cell(inputs.type(torch.float32).to('cuda:0'),
                                                         hidden_init.type(torch.float32).to('cuda:0'))
                    functional_dynamics[:, trial * generate_length + t_step] = np.reshape(
                        hidden_init.detach().cpu().numpy(), (self.model_params['HIDDEN_DIM']))
            print('\n')
            #####################################################################################
            ## generate XX trials with random hidden_init and all possible one-step inputs ######
            #####################################################################################
            trials = 20
            range_functional_dynamics = np.zeros((self.model_params['HIDDEN_DIM'], trials * (10 * 10 * 10 * 10)))
            range_functional_dynamics_output = np.zeros((self.model_params['IN_DIM'], trials * (10 * 10 * 10 * 10)))
            _, original_signal = self.random_genarate(generate_length)
            for trial in range(trials):
                print('range_functional_dynamics trails: ', trial, end='\r')
                #torch.normal(mean=0.5, std=torch.arange(1., model_params['HIDDEN_DIM']))
                hidden_init=torch.reshape(torch.from_numpy(original_signal[trial+start_index,:]),(1,1,self.model_params['HIDDEN_DIM']))#tanh_func(torch.randn((1,1,model_params['HIDDEN_DIM']))) #
                index=0
                for in_1 in [0]+[i for i in range(4,21,2)]:# Freq
                    # for in_2 in [0]+[1+0.01*i for i in range(801)]: #from 1 to 9 #Amp
                    for in_2 in [0]+[1+1*i for i in range(9)]: # Freq
                        for in_3 in [0]+[i for i in range(4,21,2)]: # Amp
                            # for in_4 in [0]+[1+0.01*i for i in range(801)]: #from 1 to 9
                            for in_4 in [0]+[1+1*i for i in range(9)]:#[1+0.5*i for i in range(17)]:
                                inputs=torch.from_numpy(np.array([[[in_1,in_2,in_3,in_4]]]))
                                # inputs=torch.zeros((1,1,4))
                                output, hidden_result = self.model.rnn_cell(inputs.type(torch.float32).to('cuda:0'), hidden_init.type(torch.float32).to('cuda:0'))
                                range_functional_dynamics[:,trial*(10*10*10*10)+index]=np.reshape(hidden_result.detach().cpu().numpy(),(self.model_params['HIDDEN_DIM']))
                                range_functional_dynamics_output[:,trial*(10*10*10*10)+index]=np.reshape(output.detach().cpu().numpy(),(self.model_params['IN_DIM']))
                                index+=1
            print('\n')
            save_dir = self.get_dir('./data_analysis/dynamics_analysis/')
            save_dir_next = save_dir + '/%s_sparsity=%s/' % (
                self.model_params['inter_or_intra'], str(self.model_params['sparsity']))
            if not os.path.isdir(save_dir_next):
                os.makedirs(save_dir_next)
            savemat(save_dir_next + 'functional_dynamics.mat',
                    {'functional_dynamics': functional_dynamics,
                     'range_functional_dynamics': range_functional_dynamics,
                     'range_output': range_functional_dynamics_output,
                     'original_signal': original_signal})

        return functional_dynamics, range_functional_dynamics, original_signal

    def draw_PCA_FA(self, functional_dynamics, range_functional_dynamics, original_signal, generate_length=5000, draw_start_index=1000):
        save_dir = self.get_dir('./data_analysis/dynamics_analysis/')
        save_dir_next = save_dir + '/%s_sparsity=%s/' % (
            self.model_params['inter_or_intra'], str(self.model_params['sparsity']))
        if not os.path.isdir(save_dir_next):
            os.makedirs(save_dir_next)
        # savemat(save_dir_next + 'functional_dynamics.mat',
        #         {'functional_dynamics': functional_dynamics,
        #          'range_functional_dynamics': range_functional_dynamics,
        #          'original_signal': original_signal})
        from sklearn.decomposition import PCA
        pca = PCA(n_components=10, svd_solver='arpack')
        print(functional_dynamics.shape)
        proj_zero_inputs_signal = pca.fit_transform(functional_dynamics.T)
        proj_with_inputs = pca.transform(range_functional_dynamics.T)

        print(pca.explained_variance_ratio_)
        print(proj_with_inputs.shape)

        print(proj_zero_inputs_signal.shape)
        proj = pca.transform(original_signal)

        plt.figure(figsize=(3, 3))
        plt.plot(np.linspace(1, 10, 10), pca.explained_variance_ratio_)
        plt.savefig(save_dir_next+'PCA_EV-ratio.png')
        plt.close()

        fig = plt.figure(figsize=(5, 5))
        for i in range(20):
            plt.plot(proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 0],
                     proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 1])

        # for i in range(10000):
        plt.scatter(proj[:, 0],
                    proj[:, 1])

        plt.xlabel('$PC 1$', fontsize=15)
        plt.ylabel('$PC 2$', fontsize=15)
        plt.tight_layout()
        plt.savefig(save_dir_next+'PC-trajectory-2D.png')
        plt.close()

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection='3d')

        for i in range(20):
            plt.plot(proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 0],
                     proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 1],
                     proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 2])
            # plt.plot(proj_zero_inputs_signal[2000:,0],
            #             proj_zero_inputs_signal[2000:,1],
            #             proj_zero_inputs_signal[2000:,2])

        # for i in range(200):
        ax.scatter(proj[:, 0],
                   proj[:, 1],
                   proj[:, 2], color='grey')

        ax.set_xlabel('$PC 1$', fontsize=15)
        ax.set_ylabel('$PC 2$', fontsize=15)
        ax.set_zlabel('$PC 3$', fontsize=15)
        plt.tight_layout()
        plt.savefig(save_dir_next+'PC-trajectory-3D.png')
        plt.close()

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection='3d')

        for i in range(20):
            plt.plot(proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 0],
                     proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 1],
                     proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 2])
            # plt.plot(proj_zero_inputs_signal[2000:,0],
            #             proj_zero_inputs_signal[2000:,1],
            #             proj_zero_inputs_signal[2000:,2])

        # for i in range(200):
        # ax.scatter(proj[:, 0],
        #            proj[:, 1],
        #            proj[:, 2], color='grey')

        ax.set_xlabel('$PC 1$', fontsize=15)
        ax.set_ylabel('$PC 2$', fontsize=15)
        ax.set_zlabel('$PC 3$', fontsize=15)
        plt.tight_layout()
        plt.savefig(save_dir_next + 'PC-trajectory-3D-without-random-points.png')
        plt.close()


        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection='3d')
        for i in range(20):
            plt.plot(proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 0],
                     proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 1],
                     proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 2])
        ax.scatter(proj_with_inputs[:, 0],
                   proj_with_inputs[:, 1],
                   proj_with_inputs[:, 2], color='green')

        ax.set_xlabel('$FA 1$', fontsize=15)
        ax.set_ylabel('$FA 2$', fontsize=15)
        ax.set_zlabel('$FA 3$', fontsize=15)

        plt.title('Functional Projection with Factor Analysis')
        plt.show()
        plt.savefig(save_dir_next+'FA-trajectory-3D.png')
        plt.close()
        return None

    def random_genarate(self, prediction_len=5000):
        import random
        input_data = Input_Data(self.load_data_params, self.preprocessing.recording_data,
                                self.preprocessing.stimulating_data)
        random_sample_size = 10000 if len(input_data.indexs) > 10000 else len(input_data.indexs)
        draw_list = random.sample(range(len(input_data.indexs)), random_sample_size)
        # print('draw_list shape: ', len(draw_list))
        trial_prediction_result = np.zeros([len(draw_list), self.model_params['Tp'], self.model.hidden_dim])
        trial_prediction_result2 = np.zeros([len(draw_list) * self.model_params['Tp'], self.model.hidden_dim])
        for run_index in range(len(draw_list)):
            if (run_index+1) % 100 == 0:
                print('run_index:', run_index, end='\r')
            start_index = input_data.indexs[draw_list[run_index]]

            GroundTruth = input_data.X[start_index:(start_index + prediction_len), :]

            # if GroundTruth.shape[0]<prediction_len:
            #     continue
            pred_hidden = model_prediction_hidden(self.model.to('cuda:0'), self.load_data_params, input_data, start_index,
                                                  model_param=None,
                                                  model_type=self.model_params['RNN_Type'],
                                                  prediction_len=self.load_data_params['Tp'])
            trial_prediction_result[run_index, :, :] = pred_hidden[0, :, :]
            trial_prediction_result2[(run_index * self.model_params['Tp']):((run_index + 1) * self.model_params['Tp']), :] = pred_hidden[0, :, :]
        print('\n')
        return trial_prediction_result, trial_prediction_result2

    def save_Wrec(self, suffix):
        w_rec_save_dir = self.get_dir('./data_analysis/Wrec/')
        Wrec = self.model.rnn_cell.get_w_rec().detach().cpu().numpy()
        w_rec_mat = w_rec_save_dir + suffix + '_Wrec.mat'
        savemat(w_rec_mat, {'Wrec': Wrec})

        # 保存Wrec
        plt.figure(figsize=(10, 10))
        plt.imshow(Wrec, cmap='bwr', vmin=-np.max(np.abs(Wrec)), vmax=np.max(np.abs(Wrec)))
        plt.colorbar()
        w_rec_save = w_rec_save_dir + suffix + '_Wrec.png'
        plt.savefig(w_rec_save, dpi=200)
        plt.close()

        w, v = np.linalg.eig(Wrec)
        # 保存 Wrec 特征值分布
        w_real = np.real(w)
        w_imag = np.imag(w)
        plt.figure(figsize=(5, 5))
        x_ = []
        y_ = []
        for i in range(1001):
            x_.append(np.sin(i / 1000 * 2 * np.pi))
            y_.append(np.cos(i / 1000 * 2 * np.pi))
        plt.scatter(w_real, w_imag)
        plt.plot(x_, y_)
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        w_rec_eig_save = w_rec_save_dir + suffix + '_eig_distribution.png'
        plt.savefig(w_rec_eig_save, dpi=200)
        plt.close()

    def get_mean_of_statistic_result(self):
        for k in self.statistic_result.keys():
            self.statistic_result[k] = np.mean(self.statistic_result[k], axis=1)

    def generate_optimized_stim_sequence(self, data_json, params, recording_data=None, stimulating_data=None):
        preprocess = Preprocessing(data_json, json_file_type=True) if data_json is not None else None
        recording_data = preprocess.recording_data if recording_data is None else self.screenData(recording_data)
        stim_data = preprocess.stimulating_data if stimulating_data is None and preprocess is not None else stimulating_data

        optimize_params_ = optimize_params
        for key in params:
            optimize_params_[key] = params[key]
        # recording_data, stim_data = readSingleRecData(file_path, time_len, sample_fre, selected_channels, day,
        #                                               sample_fre,
        #                                               fileType=fileType, kern_sd_ms=100,
        #                                               saveDir='/mnt/database6/Organoid/codes/baseline/reconstructed_codes/data/')
        # print(recording_data.shape)
        self.get_checkpoint_dir()
        self.get_result_dir()
        suffix = get_suffix(self.special_params)
        ckpt_suffix = suffix[:-1] + '.ckpt'
        if self.init_checkpoint_path is not None:
            self.load_model(self.init_checkpoint_path)
        else:
            self.load_model(self.checkpoint_save_dir + ckpt_suffix)

        previous_state = recording_data / 3

        # previous_state channels*T
        # model = Latent_Model.load_from_checkpoint(checkpoint_path=checkpoint, params=model_params).to('cuda:0')
        self.model = self.model.to('cuda:0')
        opt_Us = np.zeros((optimize_params['N_trials'], optimize_params['Control_Horizon'], self.model.ext_input_dim))
        pre_X = torch.zeros((1, 1, self.model.data_len * self.model.input_dim))

        for i in range(optimize_params['N_trials']):
            delta_state = None
            u_prev = None
            length = len(previous_state.T)
            # print('previous_state:', previous_state.shape)
            if not self.model.RNN_Type == 'Koopman':
                # print('Not Koopman!')
                pre_X = torch.reshape(
                    torch.from_numpy(previous_state.T[length - self.model.data_len:, :].astype(np.float32)),
                    (1, 1, self.model.data_len * self.model.input_dim))
                # print(pre_X.shape)

                Ext_IN = torch.zeros((1, 1, (self.model.data_len - 1) * self.model.ext_input_dim))  # Previous_Input
                previous_state = torch.cat((pre_X, Ext_IN), dim=2).to('cuda:0')  # ?

            opt_U = MPC_optimization(self.model, optimize_params['Pred_Horizon'], optimize_params['Control_Horizon'],
                                     torch.from_numpy(optimize_params['Y_ref']), previous_state,
                                     delta_state, u_prev, iters=optimize_params['iters'], min_input=optimize_params['min_input'],
                                     max_input=optimize_params['max_input'], minimize_hz=optimize_params['minimize_hz'])
            if type(opt_U) is np.ndarray:
                opt_Us[i, :, :] = opt_U
            else:
                opt_Us[i, :, :] = opt_U.detach().cpu().numpy()

        average_U = np.mean(opt_Us, axis=0)
        # print(np.round(average_U, 2))
        print(np.around(average_U, 3))
        from datetime import datetime
        dateNum = datetime.now().strftime("%m-%d-%H-%M-%S")  # 跨天训练可能会保存到两个文件夹

        ref_str = ''
        for i in optimize_params['Y_ref']:
            ref_str += '_' + str(int(i * 3))
        print('T-ref: ', ref_str[1:])

        # 不规范
        scio.savemat(r'./data/optimized_stim/%s_Sampling_%dHz_Stim_Parameter_kern_sd_%d_Ref%s.mat' % (
            dateNum, 4, self.preprocessing.smth_kern_sd, ref_str),
                     {'reference': optimize_params['Y_ref'] * 3,
                      'stim1_Freq': np.transpose(average_U)[0, :],
                      'stim1_Amp': np.transpose(average_U)[1, :] * 1000,
                      'stim2_Freq': np.transpose(average_U)[2, :],
                      'stim2_Amp': np.transpose(average_U)[3, :] * 1000})

        amplitude, duration = self.testOutput_1(average_U.T)
        return amplitude, duration

    @staticmethod
    def testOutput_1(U):
        stim_1_freq = U[0, :]
        stim_1_amp = U[1, :] * 1000
        amplitude = []
        duration = []
        print('stim_1_freq shape: ', stim_1_freq.shape)
        for i in range(stim_1_freq.shape[0]):
            isi = np.round(1000 / np.round(stim_1_freq[i]))
            times = np.trunc(250 / (np.round(1000 / np.round(stim_1_freq[i]))))
            amp = np.round(stim_1_amp[i])
            time_rest = 250 - np.round(1000 / np.round(stim_1_freq[i])) * np.trunc(
                250 / (np.round(1000 / np.round(stim_1_freq[i]))))
            if time_rest < .5:
                time_rest = isi
            else:
                times = times + 1
            for j in range(int(times)):
                amplitude.append(amp)
                duration.append(isi)
            amplitude.append(0)
            duration.append(time_rest)
        return amplitude, duration

    def generate_trajectory_without_input(self, generate_length=5000, trials=20):
        ###################################################################
        ## generate 1000 trials with random hidden_init and zero inputs ###
        ###################################################################
        self.get_checkpoint_dir()
        suffix = get_suffix(self.special_params)
        ckpt_suffix = suffix[:-1] + '.ckpt'
        if self.init_checkpoint_path is not None:
            self.load_model(self.init_checkpoint_path)
        else:
            self.load_model(self.checkpoint_save_dir + ckpt_suffix)

        self.model.to('cuda:0')

        self.load_data_params['Tp'] = generate_length
        tanh_func = torch.nn.Tanh()

        functional_dynamics = np.zeros((self.model_params['HIDDEN_DIM'], trials * self.load_data_params['Tp']))
        for trial in range(trials):
            # torch.normal(mean=0.5, std=torch.arange(1., model_params['HIDDEN_DIM']))
            hidden_init = tanh_func(torch.randn((1, 1, self.model_params['HIDDEN_DIM'])))  #
            for t_step in range(self.load_data_params['Tp']):
                # inputs = Inputs[:, (time + model.data_len - 1 + i):(time + model.data_len + i), :]

                inputs = torch.zeros((1, 1, 4))
                output, hidden_init = self.model.rnn_cell(inputs.type(torch.float32).to('cuda:0'),
                                                          hidden_init.type(torch.float32).to('cuda:0'))
                functional_dynamics[:, trial * self.load_data_params['Tp'] + t_step] = np.reshape(
                    hidden_init.detach().cpu().numpy(), (self.model_params['HIDDEN_DIM']))
        return functional_dynamics

    def low_dim_dynamics(self, data, dim_red_method='PCA'):
        dim_red_ = None
        if dim_red_method == 'PCA':
            dim_red_ = PCA_dim_reduction(data)
        elif dim_red_method == 'FA':
            dim_red_ = FA_dim_reduction(data)
        elif dim_red_method == 'LLE':
            dim_red_ = LLE_dim_reduction(data)
        return dim_red_.transform(data)

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

    def screenData(self, rec_data):
        data = self.dir2np(rec_data)
        # print('data_shape:',len(data),'*',len(data[0]))
        num_selected_channels = self.number_recording_channels
        screened_data = np.zeros([num_selected_channels, len(data[0])])
        index_ = 0
        for i in range(len(data[0])):
            if self.preprocessing.selected_channels[i] > 0:
                screened_data[index_] = data[i]
                index_ += 1
        # print('screened_data_shape:', len(screened_data), '*', len(screened_data[0]))
        return screened_data


def load_model(train_dataloader, model_params, checkpoint, conLearning=False, epochs=200, ReTrain=True):
    model = Latent_Model(model_params)
    if os.path.exists(checkpoint) and (not ReTrain):
        print("Loading checkpoint...")
        model = model.load_from_checkpoint(checkpoint_path=checkpoint, params=model_params)
        if conLearning != False:
            trainer = Trainer(max_epochs=epochs, gpus=[int(model_params['device'].split(':')[1])])
            trainer.fit(model, train_dataloader)
            # trainer.test(model_gru, validation_dataloader)
            trainer.save_checkpoint(conLearning)

            # if model_params['device'] == 'cuda:0':
            #     trainer = Trainer(max_epochs=epochs, gpus=[0])
            #     trainer.fit(model, train_dataloader)
            #     # trainer.test(model_gru, validation_dataloader)
            #     trainer.save_checkpoint(conLearning)
            # else:
            #     trainer = Trainer(max_epochs=epochs, gpus=[1])
            #     trainer.fit(model, train_dataloader)
            #     # trainer.test(model_gru, validation_dataloader)
            #     trainer.save_checkpoint(conLearning)
    else:
        print("Trainning...")
        trainer = Trainer(max_epochs=epochs, gpus=[int(model_params['device'].split(':')[1])])
        trainer.fit(model, train_dataloader)
        trainer.test(model, train_dataloader)
        trainer.save_checkpoint(checkpoint)
        # if model_params['device'] == 'cuda:0':
        #     trainer = Trainer(max_epochs=epochs, gpus=[0])
        #     trainer.fit(model, train_dataloader)
        #     trainer.test(model, train_dataloader)
        #     trainer.save_checkpoint(checkpoint)
        # else:
        #     trainer = Trainer(max_epochs=epochs, gpus=[1])
        #     trainer.fit(model, train_dataloader)
        #     trainer.test(model, train_dataloader)
        #     trainer.save_checkpoint(checkpoint)

    return model_params, model


def model_prediction(model, load_data_params, data, start_index, model_param, model_type='koopman', prediction_len=500):
    Data = torch.reshape(torch.from_numpy(
        data.X[start_index - model.ar_order - model.data_len:(start_index + prediction_len + load_data_params['Tp']),
        :].astype(
            'float32')),
        (1, prediction_len + model.ar_order + model.data_len + load_data_params['Tp'],
         load_data_params['Num_Cortical'])).type(
        torch.DoubleTensor)
    Inputs = torch.reshape(torch.from_numpy(
        data.ext_input[
        start_index - model.ar_order - model.data_len:(start_index + prediction_len + load_data_params['Tp']),
        :].astype('float32')),
        (1, prediction_len + model.ar_order + model.data_len + load_data_params['Tp'], model.ext_input_dim)).type(
        torch.DoubleTensor)

    PredictionState = np.zeros((1, prediction_len, model.input_dim), dtype='float32')
    if not model.RNN_Type == 'Koopman':
        for time in range(0, prediction_len, load_data_params['Tp']):

            pre_X = torch.reshape(Data[:, time:(model.data_len + time), :], (1, 1, model.data_len * model.input_dim))
            Ext_IN = torch.reshape(Inputs[:, time:(time + model.data_len - 1), :],
                                   (1, 1, (model.data_len - 1) * model.ext_input_dim))  # Previous_Input
            previous_state = torch.cat((pre_X, Ext_IN), dim=2)

            hidden_init = torch.permute(model.encode(previous_state.type(torch.float32).to('cuda:0')),
                                        (1, 0, 2))  # Batch X 1 X H_dim -> 1 X Batch X H_dim

            for i in range(load_data_params['Tp']):
                inputs = Inputs[:, (time + model.data_len - 1 + i):(time + model.data_len + i), :]

                if model_type == 'LSTM':
                    output, (hidden_init, hidden_init) = model.rnn_cell(inputs.type(torch.float32).to('cuda:0'), (
                        hidden_init.type(torch.float32).to('cuda:0'), hidden_init.type(torch.float32).to('cuda:0')))
                else:
                    output, hidden_init = model.rnn_cell(inputs.type(torch.float32).to('cuda:0'),
                                                         hidden_init.type(torch.float32).to('cuda:0'))

                if model.RNN_Type == 'EI-RNN':
                    PredictionState[:, time + i:time + i + 1, :] = torch.permute(output, (1, 0, 2)).to(
                        'cpu').detach().numpy()
                else:
                    PredictionState[:, time + i:time + i + 1, :] = model.decode(
                        torch.permute(hidden_init.type(torch.float32).to('cuda:0'), (1, 0, 2))).to(
                        'cpu').detach().numpy()

    else:
        for time in range(0, prediction_len, load_data_params['Tp']):
            koop_latent_X = torch.zeros((1, model.data_len, model.latent_dim * model.ar_order), dtype=torch.float32,
                                        device='cuda:0')
            koop_latent_Y = torch.zeros((1, model.data_len, model.latent_dim * model.ar_order), dtype=torch.float32,
                                        device='cuda:0')

            for i in range(model.ar_order):
                Previous_X = Data[:, time + i:(time + i + model.data_len), :].type(torch.float32).to(
                    'cuda:0')  # data_len+ar_order
                Previous_Y = Data[:, time + i + 1:(time + i + 1 + model.data_len), :].type(torch.float32).to('cuda:0')

                koop_latent_X[:, :, i * model.latent_dim:(i + 1) * model.latent_dim] = model.encode(Previous_X)
                koop_latent_Y[:, :, i * model.latent_dim:(i + 1) * model.latent_dim] = model.encode(Previous_Y)

            latent_new = koop_latent_Y[:, (model.data_len - 1):model.data_len, :]
            Previous_Input = Inputs[:, (time + model.ar_order - 1):(time + model.ar_order - 1 + model.data_len), :].to(
                'cuda:0')  # Current_Input

            Aug_Lat_Input = torch.permute(torch.cat((koop_latent_X, Previous_Input), dim=2),
                                          (0, 2, 1))  # [Latent_X; Ext_in]  batch*len*(latent_dim*ar_order+ext_in_dim)
            model.K_B = torch.bmm(torch.permute(koop_latent_Y.type(torch.float32), (0, 2, 1)),
                                  model.batch_pinv(Aug_Lat_Input.type(torch.float32),
                                                   model.L_factors))  # (latent_dim*ar_order+ext_in_dim) * latent_dim

            for i in range(load_data_params['Tp']):
                # The current input
                current_input = Inputs[:,
                                (time + model.data_len + model.ar_order - 1):(time + model.data_len + model.ar_order),
                                :].to('cuda:0')

                # Augmenting of the latent state and current input
                Aug_Lat_Input = torch.permute(torch.cat((latent_new, current_input), dim=2), (
                    0, 2, 1))  # [Latent_X; Ext_in]  batch*len*(latent_dim*ar_order+ext_in_dim)
                # Update the latent state
                latent_new = torch.permute(torch.bmm(model.K_B, Aug_Lat_Input.type(torch.float32)), (0, 2, 1))
                # Transform back into the original state space
                PredictionState[:, time + i, :] = model.decode(latent_new[:, :, -1 * model.latent_dim:]).to(
                    'cpu').detach().numpy()

    return PredictionState


def data_partition(input_data, batchsize=50, training_ratio=0.8):
    r'''
    :param input_data: Input_Data
    :param batchsize: training batch size
    :param training_ratio: ratio of training data in all samples
    :return: train_dataloader, start_indexs for prediction and test
    '''
    dataset_size = len(input_data)
    # print(input_data.ext_input.shape)
    indices = list(range(dataset_size))
    split = int(np.floor(training_ratio * dataset_size))
    train_indices, val_indices = indices[:split], indices[split:]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_dataloader = torch.utils.data.DataLoader(input_data, batch_size=batchsize,
                                                   sampler=train_sampler, num_workers=4)
    validation_dataloader = torch.utils.data.DataLoader(input_data, batch_size=batchsize,
                                                        sampler=valid_sampler)

    X, ext_in, start_indexs = next(iter(validation_dataloader))
    return train_dataloader, start_indexs


def data_partition_with_cross_validation(input_data, num_trails, batchsize=50, training_ratio=0.8, num_of_folder=5):
    '''

    :param num_trails:
    :param input_data:
    :param batchsize:
    :param training_ratio:
    :param num_of_folder: N-folder cross validation
    :return: train_dataloader list and start_indexs list for prediction and test
    '''
    dataset_size = len(input_data)
    # print(input_data.ext_input.shape)
    indices = list(range(dataset_size))

    train_dataloader_dir = {}
    start_indexs_dir = {}

    for i in range(num_of_folder):
        train_indices = []
        val_indices = []
        for j in range(num_trails):
            i_min = int(np.floor(j * dataset_size / num_trails))
            i_max = int(np.floor((j + 1) * dataset_size / num_trails))
            split_small = int(np.floor(i_min + i / num_of_folder * dataset_size / num_trails))
            split_large = int(np.floor(i_min + (i + 1) / num_of_folder * dataset_size / num_trails))
            train_indices += indices[i_min:split_small] + indices[split_large:i_max]
            val_indices += indices[split_small:split_large]
        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_dataloader = torch.utils.data.DataLoader(input_data, batch_size=batchsize,
                                                       sampler=train_sampler, num_workers=4)
        validation_dataloader = torch.utils.data.DataLoader(input_data, batch_size=batchsize,
                                                            sampler=valid_sampler)

        X, ext_in, start_indexs = next(iter(validation_dataloader))
        train_dataloader_dir[i] = train_dataloader
        start_indexs_dir[i] = start_indexs

    return train_dataloader_dir, start_indexs_dir


def virtualize_test_result(GroundTruth, pred_gru, run_index, num_recording_channels, sample_fre, model_type, latent, ar,
                           time_length,
                           resultDirPath, dataNum, load_data_params, ratio=10):
    fs = 100
    fs_legend = 12
    if run_index % ratio == 0:
        plt.figure(figsize=(500, 500))

        for i in range(num_recording_channels):
            if i < 6:
                ax = plt.subplot(8, 8, i + 2)
            elif i > 53:
                ax = plt.subplot(8, 8, i + 4)
            else:
                ax = plt.subplot(8, 8, i + 3)
            # ax = plt.subplot(2, 8, 2)
            plt.title(model_type, fontsize=fs, fontname='Arial')
            ax.plot(GroundTruth[:, i], '-', linewidth=5)
            ax.plot(pred_gru[0, :, i], '-.', linewidth=5)
            plt.xticks([0, 200, 400, 600, 800, 1000], fontsize=fs)
            plt.yticks([0, 2, 4, 6, 8, 10], fontsize=fs)
            plt.ylabel('Channel %d ' % i, fontname='Arial', loc='center', fontsize=fs)
            # plt.xlabel('Time Steps',fontname='Arial',loc='center',fontsize=15)
            ax.legend(labels=['GT', '%s_Pred' % model_type], loc='upper center', ncol=3,
                      fancybox=True,
                      shadow=True, fontsize=fs)

        save_fig = '%s/%s/%s_%dHz_%dChannels/Latent_' % (
            resultDirPath, model_type, dataNum, sample_fre, num_recording_channels) + str(
            latent) + '_AR_' + str(ar) + '_Time_Length_' + str(
            time_length) + '_Tp_Length_' + str(load_data_params['Tp']) + '_Test_' + str(
            run_index + 1) + '.pdf'
        plt.savefig(save_fig, dpi=400)
        plt.close()


def mat2pd(
        file=r'Z:/Organoid/codes/baseline/reconstructed_codes/data/FR_Set7_day01-14_4Hz_smth_4channels_kern_std_100.mat',
        channels=4, trail_num=40, sample_fre=4, time_len=335):
    response_data = np.array(loadmat(file)['Cortex'])  # 读40个trails的数据 (4,1340)
    print(response_data.shape)
    times = np.linspace(0, time_len, time_len * sample_fre)
    time_array = times
    for i in range(trail_num - 1):
        time_array = np.column_stack((time_array, times))
    # time平铺成一列和data一一对应
    time_array = time_array.flatten(order='F')  # (800,)
    print(time_array.shape)
    data = np.column_stack((np.transpose(response_data), time_array))
    columns = []
    for i in range(channels):
        columns.append('channel_%d' % (i + 1))
    columns.append('time')
    df = pd.DataFrame(data, columns=columns)
    return df


def model_prediction_hidden(model, load_data_params, data, start_index, model_param, model_type='koopman',
                            prediction_len=10):
    Data = torch.reshape(torch.from_numpy(
        data.X[start_index - model.ar_order - model.data_len:(start_index + prediction_len + load_data_params['Tp']),
        :].astype(
            'float32')),
        (1, prediction_len + model.ar_order + model.data_len + load_data_params['Tp'],
         load_data_params['Num_Cortical'])).type(
        torch.DoubleTensor)
    Inputs = torch.reshape(torch.from_numpy(
        data.ext_input[
        start_index - model.ar_order - model.data_len:(start_index + prediction_len + load_data_params['Tp']),
        :].astype('float32')),
        (1, prediction_len + model.ar_order + model.data_len + load_data_params['Tp'], model.ext_input_dim)).type(
        torch.DoubleTensor)

    PredictionHiddenState = np.zeros((1, prediction_len, model.hidden_dim), dtype='float32')
    if not model.RNN_Type == 'Koopman':
        for time in range(0, prediction_len, load_data_params['Tp']):

            pre_X = torch.reshape(Data[:, time:(model.data_len + time), :], (1, 1, model.data_len * model.input_dim))
            Ext_IN = torch.reshape(Inputs[:, time:(time + model.data_len - 1), :],
                                   (1, 1, (model.data_len - 1) * model.ext_input_dim))  # Previous_Input
            previous_state = torch.cat((pre_X, Ext_IN), dim=2)

            hidden_init = torch.permute(model.encode(previous_state.type(torch.float32).to('cuda:0')),
                                        (1, 0, 2))  # Batch X 1 X H_dim -> 1 X Batch X H_dim

            for i in range(load_data_params['Tp']):
                inputs = Inputs[:, (time + model.data_len - 1 + i):(time + model.data_len + i), :]

                output, hidden_init = model.rnn_cell(inputs.type(torch.float32).to('cuda:0'),
                                                     hidden_init.type(torch.float32).to('cuda:0'))

                PredictionHiddenState[:, time + i:time + i + 1, :] = torch.permute(hidden_init, (1, 0, 2)).to(
                    'cpu').detach().numpy()

    return PredictionHiddenState
