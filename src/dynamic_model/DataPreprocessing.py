import os
import matplotlib.pyplot as plt
import scipy.io as scio
import numpy as np
import h5py
import scipy.signal as signal
import math
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from scipy.io import savemat


def read_data_params_from_jason(json_dir):
    f = open(json_dir, 'r', encoding='UTF-8')
    content = f.read()
    params = json.loads(content)
    f.close()
    params['selected_channels'] = np.array(params['selected_channels'])
    return params


def coordinates_translation(coordinates, type):
    coordinate1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                   28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                   53, 54, 55, 56, 57, 58, 59]
    coordinate2 = [[1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7],
                   [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7], [2, 8],
                   [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7], [3, 8],
                   [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
                   [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7], [5, 8],
                   [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7], [6, 8],
                   [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7], [7, 8],
                   [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7]]

    translation = {}
    for i in range(len(coordinates)):
        if type == 1:
            # 0-60 to (1,2)-(8,7)
            translation[i] = coordinate2[coordinate1.index(coordinates[i])]
        else:
            # 0-60 to (1,2)-(8,7)
            translation[i] = coordinate1[coordinate2.index(coordinates[i])]
    return translation


class Preprocessing:
    def __init__(self, params, json_file_type=False, time_len_PSTH=40, threshold=.5):
        '''
        1.根据输入参数批量读取原始数据（.mat文件），预处理后保存为模型训练可用文件
        2.根据输入json文件读取params，响应及刺激数据
        *3.缺指定文件读取 & ref文件读取
        :param params: 数据相关参数
        '''
        self.params = None
        self.data_root_dir = None  # 数据根目录
        self.stim_dir = None  # 刺激文件在根目录相对路径
        self.set = None  # 属于第几批实验数据
        self.day = None  # 实验数据获取日期
        self.subdir = None  # 实验类型，如“系统辨识”（子文件夹目录）
        self.recording_file_sequence_keys = None  # 记录文件关键字序列
        self.stimulating_file_sequence_keys = None  # 刺激序列关键字序列
        self.data_sampling_frequency = None  # 目标采样频率（Hz）
        self.recording_channels = None  # 记录电极总通道数
        self.stim_channels = None  # 刺激电极类别数目
        self.smth_kern_sd = None  # 平滑核方差
        self.time_length = None  # 每个trail记录时间
        self.Ref = None  # 没有刺激的自放电数据
        self.stim_type = None  # 转化刺激类型（1，2，3）
        self.electrode_set = None  # 刺激电极位点
        self.filename = None
        self.PSTHs = None
        self.selected_channels = None  # 通道筛选mask
        # 默认保存在 data/ 文件夹下
        self.address = None
        self.recording_data = None  # 神经响应数据
        self.stimulating_data = None  # 刺激数据
        self.threshold = threshold
        self.time_len_PSTH = time_len_PSTH
        if isinstance(params, str):
            json_file_type = True
        elif isinstance(params, dict):
            json_file_type = False
        if not json_file_type:
            self.params = params
            self.load_features(json_file_type)
            # 读取数据
            if self.Ref != 'stim':
                self.recording_data = self.read_multi_trails_recording_data()
                print('Successfully loading Recording Data!', self.recording_data.shape, np.mean(self.recording_data))
            self.stimulating_data = None if self.Ref == 'Ref' or self.Ref is True else self.read_multi_trails_stimulating_data()

            self.resize_and_smooth_recording_data()
            # 通道选择 & 保存
            self.save_data(self.selected_channels)
            self.save_params2json(self.params)
        else:
            # 从json文件读取参数
            try:
                self.params = read_data_params_from_jason(params)
                self.address = params[:-9]
            except:
                print('Error in reading data params...')
                print(params)
            else:
                print('Successfully loading data params!')
            self.load_features(json_file_type)

            # if self.Ref != 'stim':
            #     print('loading data from: ', self.address , 'FR_data.mat')
            #     self.recording_data = scio.loadmat(self.address + 'FR_data.mat')['Cortex']
            # if self.Ref != 'Ref' and self.Ref != True:
            #     print('Ref:', self.Ref)
            #     self.stimulating_data = scio.loadmat(self.address + 'FR_data.mat')['Inputs']

        # print(read_params_from_jason(self.address + 'data.json'))

    def load_features(self, json_file_type):
        params = self.params
        # print(params)
        # self.data_root_dir = None #params['dataPath']  # 数据根目录
        # self.stim_dir = None #params['stimPath']  # 刺激文件在根目录相对路径
        # self.set = params['set']  # 属于第几批实验数据
        # self.day = params['day']  # 实验数据获取日期
        # self.subdir = params['type']  # 实验类型，如“系统辨识”（子文件夹目录）
        # self.recording_file_sequence_keys = params['rec_seq']  # 记录文件关键字序列
        # self.stimulating_file_sequence_keys = params['stim_seq']  # 刺激序列关键字序列
        # self.data_sampling_frequency = params['sample_fre']  # 目标采样频率（Hz）
        # self.recording_channels = params['rec_channels']  # 记录电极总通道数
        # self.stim_channels = params['stim_channels']  # 刺激电极类别数目
        # self.smth_kern_sd = params['kern_sd']  # 平滑核方差
        # self.time_length = params['recording_time_len']  # 每个trail记录时间
        # self.Ref = params['ref']  # 没有刺激的自放电数据
        # self.stim_type = params['stim_type']  # 转化刺激类型（1，2，3）
        # self.electrode_set = params['electrode_set']  # 刺激电极位点
        # self.selected_channels = params['selected_channels']  # 通道筛选mask# 计算PSTHs并保存
        # if not json_file_type and (self.Ref is None or self.Ref is False):
        #     print(self.data_root_dir)
        #     self.calPSTH(time_len_PSTH=self.time_len_PSTH)
        #     self.drawPSTHs(time_len_PSTH=self.time_len_PSTH, threshold=self.threshold)
        #     if self.selected_channels is None:
        #         self.channelSelection(threshold=self.threshold)

        # 默认保存在 data/ 文件夹下
        # self.address = './data/%s/%s/%dchannels/%dHz/kern_sd_%d/' % (
        #     self.set, self.day, int(np.sum(self.selected_channels)), self.data_sampling_frequency, self.smth_kern_sd) if self.address is None else self.address
        # # print('address: ',self.address)
        # if not os.path.isdir(self.address):
        #     os.makedirs(self.address)
        # if self.Ref is not None and self.Ref is not False:
        #     if 'filename' not in self.params or self.params['filename'] is None:
        #         self.filename = self.recording_file_sequence_keys[0]
        #     else:
        #         self.filename = self.params['filename']
        #     self.address += self.filename + '_'

        self.recording_data = None  # 神经响应数据
        self.stimulating_data = None  # 刺激数据

    def screen_recording_data(self, selected_channels):
        num_selected_channels = int(np.sum(selected_channels))
        index_ = 0
        screened_rec_data = np.zeros([num_selected_channels, len(self.recording_data[0])])
        for i in range(60):
            if selected_channels[i] > 0:
                screened_rec_data[index_] = self.recording_data[i]
                index_ += 1
        return screened_rec_data

    def save_params2json(self, params):
        params['selected_channels'] = params['selected_channels'].tolist() if params[
                                                                                  'selected_channels'] is not None else self.selected_channels.tolist()
        save = json.dumps(params, ensure_ascii=False, indent=2)
        address = self.address + 'data.json'  # if not self.Ref else self.address + '%s_data.json' % self.filename
        f = open(address, 'w', encoding='utf-8')
        f.write(save)
        f.close()

    def save_data(self, selected_channels=None):
        data = {}
        print(self.Ref)
        if self.Ref != 'stim':
            if selected_channels is not None:
                screened_rec_data = self.screen_recording_data(selected_channels)
            else:
                screened_rec_data = self.recording_data
            data['Cortex'] = screened_rec_data
            #data['Inputs']=np.zeros((50,1))
        if self.Ref != 'Ref':
            data['Inputs'] = self.stimulating_data
            
        # spon data时这里跑步过去。现在先这样改
        if data.__contains__('Inputs'):
            if data['Inputs'] is None:
                data['Inputs']=np.zeros((1,1))
            
        print(data)
        scio.savemat(self.address + 'FR_data.mat', data)

    def read_multi_trails_recording_data(self):
        day = self.day
        seq = self.recording_file_sequence_keys
        channels = self.recording_channels
        time_len = self.time_length
        sample_freq = self.data_sampling_frequency
        rec_spike_count = np.zeros([1, len(seq), channels, time_len * sample_freq])
        for j in range(len(seq)):
            stim_ind = seq[j]
            # 找到包含记录数据的记录文件
            # print('file path:',self.data_root_dir + str(day) + '/' + self.subdir)
            # print('Stim index:',stim_ind)
            filePath = findRecFiles_1((self.data_root_dir + str(day) + '/' + self.subdir), stim_ind)
            # print('return file path:',filePath)
            # 读取单个记录文件
            print(filePath)
            spikes_sequence, fileType = readFile(filePath)
            electrode = -1
            for row in range(1, 9):
                for col in range(1, 9):
                    # str_read_key = 'AnSt_Label_E_00159_' + str(row) + str(col) + '_ID_'
                    str_read_key = 'AnSt_Label_D_00138_' + str(row) + str(col) + '_ID_'
                    key_name = [k for k, v in spikes_sequence.items() if str_read_key in k]
                    # 当对应电极存在神经响应记录时，滑动窗计算spike count
                    if len(key_name) > 0:
                        spike_data_ = getSpikeSequence(spikes_sequence, key_name, fileType)
                        electrode += 1
                        for stim_time_index in range(time_len * sample_freq):
                            stim_time_cur = stim_time_index / sample_freq
                            intervel = 1 / sample_freq
                            stim_time_lower = stim_time_cur - intervel / 2
                            stim_time_upper = stim_time_cur + intervel / 2
                            rec_spike_count[0, j, electrode, stim_time_index] = np.sum(
                                np.all([spike_data_ > stim_time_lower, spike_data_ < stim_time_upper],
                                       axis=0) == True)
                    # 当没有记录时，排除四角电极（11，18，81，88），其余填空
                    else:
                        if (row == 1 or row == 8) and (col == 1 or col == 8):
                            continue
                        else:
                            electrode += 1
        return rec_spike_count

    # def read_sigle_trail_recording_data(self):

    def read_multi_trails_stimulating_data(self):
        '''
        有时间都整理一下，目前先用之前的代码
        :return: 没有平滑的刺激采样数据
        '''
        if self.Ref is 'stim':
            print('Reading optimized stim...')
            return readOptimizedStimulatingData(self.stim_dir, self.stimulating_file_sequence_keys, self.data_root_dir,
                                                [str(self.day)], self.recording_file_sequence_keys, self.time_length,
                                                self.data_sampling_frequency)
        else:
            if self.stim_type == 1:
                return readStimulatingData(self.stim_dir, self.stimulating_file_sequence_keys, self.data_root_dir,
                                           self.subdir, [str(self.day)], self.recording_file_sequence_keys,
                                           self.time_length, self.data_sampling_frequency)
            elif self.stim_type == 2:
                return readStimulatingData2(self.stim_dir, self.stimulating_file_sequence_keys, self.data_root_dir,
                                            self.subdir, [str(self.day)], self.recording_file_sequence_keys,
                                            self.time_length, self.data_sampling_frequency)
            elif self.stim_type == 3:
                return readStimulatingData3(self.stim_dir, self.stimulating_file_sequence_keys, self.data_root_dir,
                                            self.subdir, [str(self.day)], self.recording_file_sequence_keys,
                                            self.time_length, self.data_sampling_frequency)
            else:
                return None

    def resize_and_smooth_recording_data(self):
        firing_rate_per_channel = self.recording_data * self.data_sampling_frequency if self.recording_data is not None else None
        resized_stim_inputs = None
        stim_inputs_V = self.stimulating_data  # / 1000
        for i in [1, 3]:
            if not self.Ref:
                stim_inputs_V[:, :, i, :] = self.stimulating_data[:, :, i, :] * 10  # 幅值整理由0.x V 变为 0.x*10  V/10
        resized_stim_spike_count = resizeData(firing_rate_per_channel)
        resized_stim_inputs = resizeData(stim_inputs_V)
        # resized_stim_inputs = None if self.Ref is 'Ref' or self.Ref else resizeData(stim_inputs_V)

        # kern_sd_ms = 100
        kern_sd = int(round(self.smth_kern_sd))
        window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
        window /= np.sum(window)
        filt = lambda x: np.convolve(x, window, 'same')
        if resized_stim_spike_count is not None:
            self.recording_data = np.apply_along_axis(filt, 1, resized_stim_spike_count)
        self.stimulating_data = resized_stim_inputs

    def calPSTH(self, time_len_PSTH, start_=10, time_bin=10):
        PSTHs = np.zeros([1, len(self.recording_file_sequence_keys), 60])

        stim_path_0 = self.data_root_dir + self.stim_dir[0]
        stim_time_0 = np.array(scio.loadmat(stim_path_0)['stim1'][:, 0])
        stim_path_1 = self.data_root_dir + self.stim_dir[1]
        stim_time_1 = np.array(scio.loadmat(stim_path_1)['stim2'][:, 0])

        stim_time_indexs = calStimSet(self.data_root_dir, [self.day], self.recording_file_sequence_keys,
                                      self.stimulating_file_sequence_keys, self.subdir, self.electrode_set)

        day = self.day
        for j in range(len(self.recording_file_sequence_keys)):
            stim_ind = self.recording_file_sequence_keys[j]
            # 找到包含stim_ind的记录文件
            filePath = findRecFiles_1((self.data_root_dir + day + self.subdir), stim_ind)
            # 读取单个记录文件
            # print('return file path:',filePath)
            spikes_sequence, fileType = readFile(filePath)
            electrode = -1
            for row in range(1, 9):
                for col in range(1, 9):
                    # str_read_key = 'AnSt_Label_E_00159_' + str(row) + str(col) + '_ID_'
                    str_read_key = 'AnSt_Label_D_00138_' + str(row) + str(col) + '_ID_'
                    # print(spikes_sequence.keys())
                    key_name = [k for k, v in spikes_sequence.items() if str_read_key in k]
                    # 当对应电极存在神经响应记录时，滑动窗计算spike count
                    if len(key_name) > 0:
                        # spike_data_ = np.array(spikes_sequence[key_name[0]][:, 0])
                        spike_data_ = getSpikeSequence(spikes_sequence, key_name, fileType)
                        electrode += 1
                        stim_index = stim_time_indexs[0, j, row - 1, col - 1]
                        if stim_index == 0:
                            stim_time_ = stim_time_0
                        elif stim_index == 1:
                            stim_time_ = stim_time_1
                        else:
                            print('Error: invalid stim_time_index!')
                            continue
                        psth, aera = simplePSTH(stim_time_, spike_data_, start_, time_len_PSTH, time_bin)
                        PSTHs[0, j, electrode] = aera

                    # 当没有记录时，排除四角电极（11，18，81，88），其余填空
                    else:
                        if (row == 1 or row == 8) and (col == 1 or col == 8):
                            continue
                        else:
                            electrode += 1
        self.PSTHs = PSTHs
        # return PSTHs

    def channelSelection(self, threshold=.5):
        self.selected_channels, x = channelSelection(self.PSTHs, threshold=threshold)
        #     只对第七批数据
        #     self.selected_channels[12] = 0
        #     self.selected_channels[20] = 0
        #     self.selected_channels[23] = 0
        #     self.selected_channels[31] = 0
        for i in range(len(self.electrode_set)):
            for j in range(len(self.electrode_set[0])):
                index = coordinates_translation([self.electrode_set[i][j]], type=2)
                self.selected_channels[index[0]] = 0

    def avg_recording_data(self):
        data_len = self.data_sampling_frequency*self.time_length
        avg_rec_data = np.zeros([int(np.sum(self.selected_channels)), data_len])
        for i in range(len(self.recording_file_sequence_keys)):
            avg_rec_data += self.recording_data[:, i * data_len:(i + 1) * data_len]
        avg_rec_data /= len(self.recording_file_sequence_keys)
        return avg_rec_data

    def avg_stimulating_data(self):
        data_len = self.data_sampling_frequency * self.time_length
        return self.stimulating_data[:, 0:data_len], self.stimulating_data[:, data_len: 2 * data_len]


    def drawPSTHs(self, time_len_PSTH, threshold):
        directory = './data_analysis/statistic_fig/PSTHs/%s_%s/' % (self.set, self.day)
        if not os.path.isdir(directory):
            os.makedirs(directory)
        selected_channels, x = channelSelection(self.PSTHs, threshold=threshold)
        figAddr_PSTH = directory + 'timeLen_10_%d.png' % (time_len_PSTH + 10)
        figAddr_selectedChannels = directory + 'threshold%.1f_selectedChannels.png' % threshold
        drawHeatmap(np.mean(np.mean(self.PSTHs, axis=1), axis=0), 'PSTH area', save=True, saveAdd=figAddr_PSTH)
        drawHeatmap(selected_channels, 'selected channels', save=True, saveAdd=figAddr_selectedChannels)

    def drawTrailAvg(self, ref=None, start_time=2, end_time=4.5, plot_line=False):
        from scipy.io import loadmat
        if isinstance(ref, str):
            ref = np.array(loadmat(ref)['reference'])

        num_trails = len(self.recording_file_sequence_keys)
        channels = int(np.sum(self.selected_channels))

        num_slices = self.time_length * self.data_sampling_frequency
        times = np.linspace(0, self.time_length, num_slices)
        time_array = times
        for i in range(num_trails - 1):
            time_array = np.column_stack((time_array, times))
        # time平铺成一列和data一一对应
        time_array = time_array.flatten(order='F')  # (800,)
        # print(self.recording_data.shape)          (channels, time_slices)
        data = np.column_stack((np.transpose(self.recording_data), time_array))
        columns = []
        for i in range(channels):
            columns.append('channel_%d' % (i + 1))
        columns.append('time')
        df = pd.DataFrame(data, columns=columns)
        #       channel_1  channel_2  channel_3  channel_4  channel_5      time
        #    0
        #   ...
        #   287

        # y_ref = np.array(ref)
        # y_max = np.max(self.recording_data)

        for i in range(channels):
            plt.figure(figsize=(5, 4))
            plt.legend(loc='upper right')
            if ref is not None:
                plt.fill_between(np.linspace(start_time, end_time, 10), 0, 50, color='lightgray')  # 刺激区间
                plt.axhline(y=ref[i], color='gray', ls='--')
                plt.axhline(y=ref[i], xmin=start_time / self.time_length, xmax=end_time / self.time_length,
                            color='black', label='reference firing rate')
            save_figure_file = 'channel ' + str(i + 1)
            plt.ylim(0, 50)
            plt.title(save_figure_file)
            plt.ylabel('Firing rate')
            plt.xlabel('Time(s)')
            # sns.relplot(x='time',y='channel_%d'%(i+1),data=df,kind='line',ci='sd')
            sns.lineplot(x='time', y='channel_%d' % (i + 1), data=df, ci='sd', label='test firing rate')
            dir_path = 'data_analysis/statistic_fig/optimized_records/' + '%s/%s/%dchannels/%dHz/kern_sd_%d/' % (
                self.set, self.day, int(np.sum(self.selected_channels)), self.data_sampling_frequency,
                self.smth_kern_sd)
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
            plt.savefig(dir_path + self.filename + save_figure_file + '.png', dpi=300)
            plt.close()

        if plot_line:
            trail_avg_values = np.zeros([channels, num_trails])
            trail_avg_resposne_values = np.zeros([channels, num_trails])
            for i in range(num_trails):
                for j in range(channels):
                    trail_avg_values[j, i] = np.mean(self.recording_data[j, i * num_slices:(i + 1) * num_slices])
                    start_index = int(i * num_slices + start_time * self.data_sampling_frequency)
                    end_index = int(i * num_slices + end_time * self.data_sampling_frequency)
                    trail_avg_resposne_values[j, i] = np.mean(self.recording_data[j, start_index: end_index])
            trail_number_column = np.linspace(1, num_trails, num_trails)
            trail_avg_data = np.column_stack(
                (np.transpose(trail_avg_values), np.transpose(trail_avg_resposne_values), trail_number_column))
            # trail_avg_data = np.column_stack((trail_avg_data, trail_number_column)
            columns = []
            for i in range(channels):
                columns.append('channel_%d_avg' % (i + 1))
            for i in range(channels):
                columns.append('channel_%d_response_avg' % (i + 1))
            columns.append('trail number')
            trail_avg_df = pd.DataFrame(trail_avg_data, columns=columns)

            for i in range(channels):
                plt.figure(figsize=(5, 4))
                plt.legend(loc='upper right')
                save_figure_file = 'channel ' + str(i + 1)
                plt.ylim(0, 50)
                plt.title(save_figure_file)
                plt.ylabel('Firing rate')
                plt.xlabel('Number of trails')
                plt.axhline(y=ref[i], color='gray', ls='--', label='reference')
                sns.lineplot(x='trail number', y='channel_%d_avg' % (i + 1), data=trail_avg_df, label='average FR')
                sns.lineplot(x='trail number', y='channel_%d_response_avg' % (i + 1), data=trail_avg_df,
                             label='average response FR')
                dir_path = 'data_analysis/statistic_fig/optimized_records/' + '%s/%s/%dchannels/%dHz/kern_sd_%d/' % (
                    self.set, self.day, int(np.sum(self.selected_channels)), self.data_sampling_frequency,
                    self.smth_kern_sd)
                if not os.path.isdir(dir_path):
                    os.makedirs(dir_path)
                plt.savefig(dir_path + self.filename + save_figure_file + '_lineplot.png', dpi=300)
                plt.close()
                
    def draw_observed_signal(self,observe_type='Spon'):
        plt.figure(figsize=(20,4))
        data=self.screen_recording_data(self.selected_channels)
        plt.plot(data[0,:1000])
        plt.plot(data[1,:1000])
        self.address = './data/%s/%s/%dchannels/%dHz/kern_sd_%d/' % (
            self.set, self.day, int(np.sum(self.selected_channels)), self.data_sampling_frequency, self.smth_kern_sd)
        plt.legend(['channel 1','channel 2'])
        plt.savefig(self.address +observe_type+ '_observed_data_visualization.png', dpi=300)
        plt.close()

def matrix(data, mask=None):
    matrix = np.zeros([8, 8])
    mask_resized = np.zeros([8, 8], dtype=bool)
    #     index = 0
    if mask is not None:
        for i in range(60):
            if i < 6:
                ax = i + 1
            elif i > 53:
                ax = i + 3
            else:
                ax = i + 2
            row = ax // 8
            col = ax % 8
            #             data_ = 0
            #             if mask[i]:
            #                 data_ = data[index]
            #                 index += 1
            #             matrix[row][col] = data_
            matrix[row][col] = data[i]
            mask_resized[row][col] = not mask[i]
    else:
        for i in range(60):
            if i < 6:
                ax = i + 1
            elif i > 53:
                ax = i + 3
            else:
                ax = i + 2
            row = ax // 8
            col = ax % 8
            matrix[row][col] = data[i]
    #     print(matrix)
    #     print(mask)
    return matrix, mask_resized


def drawHeatmap(data0, metric, save=False, saveAdd='', mask=None, vmax=None, vmin=None):
    fig = plt.figure(figsize=(5, 5))
    # print(data0)
    data, mask_resized = matrix(data0, mask)
    sns.heatmap(data, square=True, fmt='.2f', annot=True, mask=mask_resized, vmax=vmax, vmin=vmin)
    #     plt.colorbar(im)
    plt.title(metric)
    #     fig.tight_layout()  #自动调整子图参数,使之填充整个图像区域。
    if save:
        f = plt.gcf()
        f.savefig(saveAdd)
        #         plt.show()
        f.clear()
    plt.close()


def calDistanse(coordinates, row, col):
    totalDistance = 0
    for i in range(len(coordinates)):
        distance_ = math.sqrt((coordinates[i][0] - row) ** 2 + (coordinates[i][1] - col) ** 2)
        totalDistance += distance_
    return totalDistance / len(coordinates)


def calSets(seq, row, col):
    sets_num = len(seq)
    nodes_num = len(seq[0])
    #     print("%d sets with %d nodes in each set."%(sets_num,nodes_num))
    distances = np.zeros([sets_num])
    for i in range(sets_num):
        distances[i] = calDistanse(seq[i], row, col)
    #     print('Node[%d,%d] belongs to set %d'%(row,col,np.argmin(distances)))
    for i in range(sets_num):
        if [row, col] in seq[i]:
            #             print('[%d,%d]is in sets_%d'%(row,col,i))
            return i
    return np.argmin(distances)


def resizeData(data):
    if data is None:
        return None
    channels = data.shape[2]
    length = data.shape[0] * data.shape[1] * data.shape[3]
    resizedData = np.zeros([channels, length])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            resizedData[:, (i * data.shape[1] + j) * data.shape[3]:(i * data.shape[1] + j + 1) * data.shape[3]] = \
                data[i][j]
    return resizedData


def PSTH(stim_time, spike_time, start, time_len, bin_len, show=False, title='', save=''):
    bin_num = time_len // bin_len
    psth = np.zeros([bin_num])
    for i in range(len(stim_time)):
        start_time_ = stim_time[i] + start / 1000
        for j in range(bin_num):
            stim_time_lower = start_time_ + j * bin_len / 1000
            stim_time_upper = start_time_ + (j + 1) * bin_len / 1000
            psth[j] += np.sum(np.all([spike_time > stim_time_lower, spike_time < stim_time_upper], axis=0) == True)
    psth = psth / (len(stim_time) * bin_len / 1000)
    #     print(psth)
    aera = np.sum(psth) * bin_len / 1000
    #     print('area: %.3f'%aera)
    plt.bar(np.arange(0, time_len, bin_len), psth, width=10)
    plt.title(title)
    if show:
        plt.show()
    if save != '':
        f = plt.gcf()
        f.savefig(save)
    plt.close()
    return psth, aera


def simplePSTH(stim_time, spike_time, start, time_len, bin_len):
    bin_num = time_len // bin_len
    psth = np.zeros([bin_num])
    aera = 0
    for i in range(len(stim_time)):
        start_time_ = stim_time[i] + start / 1000
        stim_time_lower = start_time_
        stim_time_upper = start_time_ + time_len / 1000
        aera += np.sum(np.all([spike_time > stim_time_lower, spike_time < stim_time_upper], axis=0) == True)
    #     psth = psth/(len(stim_time)*bin_len/1000)
    #     print(psth)
    aera = aera / len(stim_time)
    return psth, aera


def findRecFiles_1(difPath, partOfFileName):
    # print(difPath)
    # print(partOfFileName)
    filePath = ''
    for filewalks in os.walk(difPath):
        # print('filewalks:',filewalks)
        for files in filewalks[2]:
            # print(files)
            if partOfFileName in files and partOfFileName + "0" not in files:
                filePath = os.path.join(filewalks[0], files)
                # print(filePath)
    return filePath


def readFiles(filePath, fileType=2):
    print(filePath)
    if fileType == 1:
        spikes_sequence = h5py.File(filePath)
    elif fileType == 2:
        spikes_sequence = scio.loadmat(filePath)
    else:
        spikes_sequence = ''
        print("Invalid index!")
    return spikes_sequence


def readFile(filePath):
    fileType = 2
    try:
        spikes_sequence = scio.loadmat(filePath)

    except:
        spikes_sequence = h5py.File(filePath)
        fileType = 1
    return spikes_sequence, fileType


def getSpikeSequence(spikes_sequence, key_name, fileType=2):
    if fileType == 1:
        spike_data_ = np.array(spikes_sequence[key_name[0]][0, :])  # h5 file
    elif fileType == 2:
        spike_data_ = np.array(spikes_sequence[key_name[0]][:, 0])  # loadmat type
    else:
        spikes_sequence = ''
        print("Invalid index!")
    return spike_data_


def readSingleRecData(file, time_len, sample_freq, selected_channels, day, sampe_fre, fileType=2, kern_sd_ms=100,
                      saveDir='./data/'):
    # 读取单个记录文件
    before_spike_count = np.zeros([60, time_len * sample_freq])
    # file = r"Z:\Organoid\科技部数据0710\课题四数据\第六批数据\11-3闭环10HzMatData\after-spon3min.mat"
    # spikes_sequence=h5py.File(file)
    spikes_sequence = readFiles(file, fileType)
    electrode = -1
    for row in range(1, 9):
        for col in range(1, 9):
            # str_read_key = 'AnSt_Label_E_00159_' + str(row) + str(col) + '_ID_'
            str_read_key = 'AnSt_Label_D_00138_' + str(row) + str(col) + '_ID_'
            key_name = [k for k, v in spikes_sequence.items() if str_read_key in k]

            # 当对应电极存在神经响应记录时，滑动窗计算spike count
            if len(key_name) > 0:
                #             spike_data_=np.array(spikes_sequence[key_name[0]][0,:])
                spike_data_ = np.array(spikes_sequence[key_name[0]][:, 0])
                electrode += 1

                for stim_time_index in range(time_len * sample_freq):
                    stim_time_cur = stim_time_index / sample_freq
                    intervel = 1 / sample_freq
                    stim_time_lower = stim_time_cur - intervel / 2
                    stim_time_upper = stim_time_cur + intervel / 2
                    before_spike_count[electrode, stim_time_index] = np.sum(
                        np.all([spike_data_ > stim_time_lower, spike_data_ < stim_time_upper], axis=0) == True)
            #                         data_pos+=1

            # 当没有记录时，排除四角电极（11，18，81，88），其余填空
            else:
                if (row == 1 or row == 8) and (col == 1 or col == 8):
                    continue
                else:
                    electrode += 1
    # return before_spike_count
    firing_rate_per_channel = before_spike_count * sample_freq
    stim_inputs = np.zeros([2, time_len * sample_freq])
    kern_sd = int(round(kern_sd_ms / 10))
    window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
    window /= np.sum(window)
    filt = lambda x: np.convolve(x, window, 'same')
    # resized_stim_inputs_smth = np.apply_along_axis(filt, 1, resized_stim_inputs)
    resized_stim_spike_count_smth = np.apply_along_axis(filt, 1, firing_rate_per_channel)
    num_selected_channels = int(np.sum(selected_channels))
    index_ = 0
    screened_rec_data = np.zeros([num_selected_channels, len(resized_stim_spike_count_smth[0])])
    print(screened_rec_data.shape)
    for i in range(60):
        if selected_channels[i] > 0:
            screened_rec_data[index_] = resized_stim_spike_count_smth[i]
            index_ += 1
    scio.savemat(r"%s/FR_SponSpike%s_%dHz_smth_%dchannels.mat" % (saveDir, day, sampe_fre, num_selected_channels),
                 {'Cortex': screened_rec_data, 'Inputs': stim_inputs})
    return screened_rec_data, stim_inputs


def readRecordingData(data_path, rec_type, days, seq, channels, time_len, sample_freq=1, fileType=2):
    rec_spike_count = np.zeros([len(days), len(seq), channels, time_len * sample_freq])
    for i in range(len(days)):
        day = days[i]
        for j in range(len(seq)):
            stim_ind = seq[j]
            # 找到包含记录数据的记录文件
            filePath = findRecFiles_1((data_path + day + rec_type), stim_ind)
            # 读取单个记录文件
            print(filePath)
            ############## 写个方法使用两种数据读取方式 #####################
            #         spikes_sequence=h5py.File(filePath)
            # spikes_sequence = scio.loadmat(filePath)
            import os
            fileType = fileType
            # print(os.path.getsize(filePath))

            # 判断文件大小，并选择哪种type
            # if os.path.getsize(filePath)>=1024*300:
            #     fileType=1
            # else:
            #     fileType=2
            spikes_sequence = readFiles(filePath, fileType)
            electrode = -1
            for row in range(1, 9):
                for col in range(1, 9):
                    # str_read_key = 'AnSt_Label_E_00159_' + str(row) + str(col) + '_ID_'
                    str_read_key = 'AnSt_Label_D_00138_' + str(row) + str(col) + '_ID_'
                    key_name = [k for k, v in spikes_sequence.items() if str_read_key in k]

                    # 当对应电极存在神经响应记录时，滑动窗计算spike count
                    if len(key_name) > 0:
                        #                     print(spikes_sequence[key_name[0]].shape)
                        #                     spike_data_=np.array(spikes_sequence[key_name[0]][0,:])#h5 file
                        # spike_data_ = np.array(spikes_sequence[key_name[0]][:, 0])  # loadmat type
                        # print(spike_data_)
                        spike_data_ = getSpikeSequence(spikes_sequence, key_name, fileType)
                        electrode += 1

                        for stim_time_index in range(time_len * sample_freq):
                            stim_time_cur = stim_time_index / sample_freq
                            intervel = 1 / sample_freq
                            stim_time_lower = stim_time_cur - intervel / 2
                            stim_time_upper = stim_time_cur + intervel / 2
                            rec_spike_count[i, j, electrode, stim_time_index] = np.sum(
                                np.all([spike_data_ > stim_time_lower, spike_data_ < stim_time_upper], axis=0) == True)
                    #                         data_pos+=1

                    # 当没有记录时，排除四角电极（11，18，81，88），其余填空
                    else:
                        if (row == 1 or row == 8) and (col == 1 or col == 8):
                            continue
                        else:
                            electrode += 1
            ##############################################################################
    # 可视化某天某批60通道数据
    # plt.figure(figsize=(50, 120), dpi=80)
    # for j in range(60):
    #     plt.subplot(60, 1, j + 1)
    #     plt.ylim((0, 10))
    #     plt.plot(np.linspace(0.0, time_len, time_len * sample_freq), rec_spike_count[0, 4, j])
    #     #     plt.plot(range(800),firing_rate[j,0],label='seq_'+str(1))
    #     #     plt.plot(range(800),firing_rate[j,3],label='seq_'+str(4))
    #     plt.legend(loc=2)
    # plt.show()
    return rec_spike_count


def readRecordingData_Ref(data_path, ref_type, rec_type, days, seq, channels, time_len, sample_freq=1, fileType=2):
    rec_spike_count = np.zeros([len(days), len(seq), channels, time_len * sample_freq])
    for i in range(len(days)):
        day = days[i]
        for j in range(len(seq)):
            stim_ind = seq[j]
            # 找到包含记录数据的记录文件
            print((data_path + day + 'response_test/' + ref_type))
            filePath = findRecFiles_1((data_path + day + 'response_test/' + ref_type + '/'), stim_ind)
            # 读取单个记录文件
            print(filePath)
            ############## 写个方法使用两种数据读取方式 #####################
            #         spikes_sequence=h5py.File(filePath)
            # spikes_sequence = scio.loadmat(filePath)
            import os
            fileType = fileType
            # print(os.path.getsize(filePath))

            # 判断文件大小，并选择哪种type
            # if os.path.getsize(filePath)>=1024*300:
            #     fileType=1
            # else:
            #     fileType=2
            print(filePath)
            spikes_sequence = readFiles(filePath, fileType)
            electrode = -1
            for row in range(1, 9):
                for col in range(1, 9):
                    # str_read_key = 'AnSt_Label_E_00159_' + str(row) + str(col) + '_ID_'
                    str_read_key = 'AnSt_Label_D_00138_' + str(row) + str(col) + '_ID_'
                    key_name = [k for k, v in spikes_sequence.items() if str_read_key in k]

                    # 当对应电极存在神经响应记录时，滑动窗计算spike count
                    if len(key_name) > 0:
                        #                     print(spikes_sequence[key_name[0]].shape)
                        #                     spike_data_=np.array(spikes_sequence[key_name[0]][0,:])#h5 file
                        # spike_data_ = np.array(spikes_sequence[key_name[0]][:, 0])  # loadmat type
                        # print(spike_data_)
                        spike_data_ = getSpikeSequence(spikes_sequence, key_name, fileType)
                        electrode += 1

                        for stim_time_index in range(time_len * sample_freq):
                            stim_time_cur = stim_time_index / sample_freq
                            intervel = 1 / sample_freq
                            stim_time_lower = stim_time_cur - intervel / 2
                            stim_time_upper = stim_time_cur + intervel / 2
                            rec_spike_count[i, j, electrode, stim_time_index] = np.sum(
                                np.all([spike_data_ > stim_time_lower, spike_data_ < stim_time_upper], axis=0) == True)
                    #                         data_pos+=1

                    # 当没有记录时，排除四角电极（11，18，81，88），其余填空
                    else:
                        if (row == 1 or row == 8) and (col == 1 or col == 8):
                            continue
                        else:
                            electrode += 1

    return rec_spike_count


def readStimulatingData(stimsPath, stim_seq, data_path, rec_type, days, seq, time_len, sample_freq=1):
    stim_inputs = np.zeros([len(days), len(seq), 2, time_len * sample_freq])  # 四个刺激电极16-17（lv），73-74（lan）

    # 读取单次trail的2个刺激类型
    stim_inputs_per_trail = np.zeros([2, time_len * sample_freq])
    for i in range(len(stimsPath)):
        stim_path = data_path + stimsPath[i % 2]
        stim_time = np.array(scio.loadmat(stim_path)['random%d' % (i % 2 + 1)][:, 0])
        stim_amplitude = np.array(scio.loadmat(stim_path)['random%d' % (i % 2 + 1)][:, 1])
        #     print(stim_time.shape)
        for stim_time_index in range(time_len * sample_freq):
            stim_time_cur = stim_time_index / sample_freq
            intervel = 1 / sample_freq
            stim_time_lower = stim_time_cur - intervel / 2
            stim_time_upper = stim_time_cur + intervel / 2
            ks = np.all([stim_time > stim_time_lower, stim_time < stim_time_upper], axis=0) == True
            stim_inputs_per_trail[i, stim_time_index] = np.sum(stim_amplitude[k] for k in range(len(ks)) if ks[k])

    for i in range(len(days)):
        day = days[i]
        for j in range(len(seq)):
            stim_ind = seq[j]

            # 找到包含stim_ind的记录文件
            file = findRecFiles_1(data_path + day + rec_type, stim_ind)

            # 读取单个记录文件
            print(file)
            # 写一个方法来分辨
            # stim_type 0:lan2-lv1,1:lan1-lv2
            # if i == 0:
            #     stim_type = 0 if j < 5 else 1
            # else:
            #     stim_type = [k for k in range(len(stim_seq)) if stim_seq[k] in file][0]
            stim_type = [k for k in range(len(stim_seq)) if stim_seq[k] in file][0]
            y = [0, 1] if stim_type == 0 else [1, 0]
            for x in range(0, 2):
                stim_inputs[i, j, x] = stim_inputs_per_trail[y[x], :]
    return stim_inputs


###########################################################################
############ Read Stimulating Data 生成刺激幅值和刺激频率  ##################
###########################################################################
def readStimulatingData2(stimsPath, stim_seq, data_path, rec_type, days, seq, time_len, sample_freq=1,
                         max_sample_freq=20):
    stim_inputs = np.zeros([len(days), len(seq), 4, time_len * sample_freq])  # 两组刺激电极16-17（lv），73-74（lan）

    # 读取单次trail的2个刺激类型
    # 换类型，每个刺激包含刺激频率序列和幅值序列[0:频率，1:幅值]
    # stim_inputs_per_trail = np.zeros([2,time_len*sample_freq])
    stim_inputs_per_trail = np.zeros([2, 2, time_len * sample_freq])
    delta = 0.01
    for i in range(2):
        stim_path = data_path + stimsPath[i % 2]
        stim_time = np.array(scio.loadmat(stim_path)['stim%d' % (i % 2 + 1)][:, 0])
        print(stim_time)
        stim_amplitude = np.array(scio.loadmat(stim_path)['stim%d' % (i % 2 + 1)][:, 1])
        #     print(stim_time.shape)
        for stim_time_index in range(1, time_len * sample_freq):
            stim_time_cur = stim_time_index / sample_freq
            intervel = 1 / max_sample_freq
            stim_time_lower = stim_time_cur - intervel / 2
            stim_time_upper = stim_time_cur + intervel / 2
            print(stim_time_cur, stim_time_lower, stim_time_upper)
            ks = np.all([stim_time > stim_time_lower, stim_time < stim_time_upper], axis=0) == True
            indexes = np.where(ks)
            #         print(len(indexes[0]))
            # 没有新的刺激，频率与之前相同

            print(stim_time)
            print(indexes[0])
            if len(indexes[0]) == 0:
                stim_inputs_per_trail[i, :, stim_time_index] = stim_inputs_per_trail[i, :, stim_time_index - 1]
            else:
                if len(indexes[0]) < (len(stim_time) - 1):
                    cur_frequency = 1 / (stim_time[indexes[0] + 1] - stim_time[indexes[0]])
                    pre_frequency = stim_inputs_per_trail[i, 0, stim_time_index - 1]
                    pre_amplitude = stim_inputs_per_trail[i, 1, stim_time_index - 1]
                    if pre_frequency != 0 and (cur_frequency < (1 - delta) * pre_frequency or cur_frequency > (
                            1 + delta) * pre_frequency) and pre_amplitude != 0:
                        stim_inputs_per_trail[i, 0, stim_time_index] = 0
                        stim_inputs_per_trail[i, 1, stim_time_index] = 0
                    else:
                        stim_inputs_per_trail[i, 0, stim_time_index] = cur_frequency
                        stim_inputs_per_trail[i, 1, stim_time_index] = stim_amplitude[indexes[0]] / 1000

    for i in range(len(days)):

        day = days[i]

        for j in range(len(seq)):
            stim_ind = seq[j]

            # 找到包含stim_ind的记录文件
            for filewalks in os.walk(data_path + day + rec_type):
                for files in filewalks[2]:
                    if stim_ind in files and stim_ind + "0" not in files:
                        file = os.path.join(filewalks[0], files)
            #                     print(stim_ind,' is in',file)

            # 读取单个记录文件
            print(file)
            # stim_type 0:lan2-lv1,1:lan1-lv2
            if i == 0:
                stim_type = 0 if j < 5 else 1
            else:
                stim_type = [k for k in range(len(stim_seq)) if stim_seq[k] in file][0]

            y = [0, 1] if stim_type == 0 else [1, 0]
            for x in range(0, 2):
                stim_inputs[i, j, x * 2] = stim_inputs_per_trail[y[x], 0, :]
                stim_inputs[i, j, x * 2 + 1] = stim_inputs_per_trail[y[x], 1, :]

    # 可视化刺激序列
    plt.figure(figsize=(50, 20), dpi=80)
    for j in range(4):
        plt.subplot(4, 1, j + 1)
        plt.ylim((0, 5))
        #     plt.plot(np.linspace(0.0,time_len,time_len*sample_freq),stim_inputs[0,6,j])
        plt.plot(np.linspace(0.0, time_len, time_len * sample_freq), stim_inputs_per_trail[j // 2, j % 2])

        #     plt.plot(range(800),firing_rate[j,0],label='seq_'+str(1))
        #     plt.plot(range(800),firing_rate[j,3],label='seq_'+str(4))
        plt.legend(loc=2)
    save_file = 'stimulation_parameter_' + str(sample_freq) + 'hz.png'
    plt.savefig(save_file, dpi=300)
    plt.close()

    return stim_inputs


def readStimulatingData3(stimsPath, stim_seq, data_path, rec_type, days, seq, time_len, sample_freq=1,
                         max_sample_freq=20):
    # import numpy as np
    stim_inputs = np.zeros([len(days), len(seq), 4, time_len * sample_freq])  # 两组刺激电极16-17（lv），73-74（lan）

    # 读取单次trail的2个刺激类型
    # 换类型，每个刺激包含刺激频率序列和幅值序列[0:频率，1:幅值]
    # stim_inputs_per_trail = np.zeros([2,time_len*sample_freq])
    stim_inputs_per_trail = np.zeros([2, 2, time_len * sample_freq])
    delta = 0.01

    # import pandas as pd

    stim_param = pd.read_excel('new_stimulation.xls')

    isi1 = np.array(stim_param['isi'])
    isi2 = np.array(stim_param['isi2'])
    times1 = np.array(stim_param['times'])
    amps1 = np.array(stim_param['amp'])
    times2 = np.array(stim_param['times2'])
    amps2 = np.array(stim_param['amp2'])

    stim_path = data_path + stimsPath[0]
    stim_time = np.array(scio.loadmat(stim_path)['stim1'][:, 0])
    start_time = stim_time[0]
    end_time = stim_time[times1[0] - 1]

    # print(int(np.round(start_time*sample_freq)),int(np.round(end_time*sample_freq)))
    stim_inputs_per_trail[0, 0,
    int(np.round(start_time * sample_freq)):int(np.round(end_time * sample_freq))] = np.round(1000 / isi1[0])

    stim_inputs_per_trail[0, 1, int(np.round(start_time * sample_freq)):int(np.round(end_time * sample_freq))] = amps1[
                                                                                                                     0] / 1000
    print('isi:', isi1[0], 'amp:', amps1[0], 'start_time:', start_time, 'end_time:', end_time)
    for m in range(1, times1.shape[0]):
        size1 = np.sum(times1[:(m)])
        size2 = np.sum(times1[:(m + 1)])
        # print(size1,size2)

        start_time = stim_time[size1]
        end_time = stim_time[size2 - 1]
        # print(size1,size2,start_time,end_time,int(np.round(start_time*sample_freq)),int(np.round(end_time*sample_freq)))
        stim_inputs_per_trail[0, 0,
        int(np.round(start_time * sample_freq)):int(np.round(end_time * sample_freq))] = np.round(1000 / isi1[m])
        stim_inputs_per_trail[0, 1, int(np.round(start_time * sample_freq)):int(np.round(end_time * sample_freq))] = \
            amps1[m] / 1000
        # print('size1:',size1,' size2:',size2)
        # print('isi:',isi1[m],'amp:',amps1[m],'start_time:',start_time,'end_time:',end_time)
    stim_path = data_path + stimsPath[1]
    stim_time = np.array(scio.loadmat(stim_path)['stim2'][:, 0])
    start_time = stim_time[0]
    end_time = stim_time[times2[0] - 1]
    # print(int(np.round(start_time*sample_freq)),int(np.round(end_time*sample_freq)))
    stim_inputs_per_trail[1, 0,
    int(np.round(start_time * sample_freq)):int(np.round(end_time * sample_freq))] = np.round(1000 / isi2[0])
    stim_inputs_per_trail[1, 1, int(np.round(start_time * sample_freq)):int(np.round(end_time * sample_freq))] = amps2[
                                                                                                                     0] / 1000

    for m in range(1, times2.shape[0]):
        size1 = np.sum(times2[:(m)])
        size2 = np.sum(times2[:(m + 1)])
        # print(size1,size2)

        start_time = stim_time[size1]
        end_time = stim_time[size2 - 1]
        # print(start_time,end_time,int(np.round(start_time*sample_freq)),int(np.round(end_time*sample_freq)))
        stim_inputs_per_trail[1, 0,
        int(np.round(start_time * sample_freq)):int(np.round(end_time * sample_freq))] = np.round(1000 / isi2[m])
        stim_inputs_per_trail[1, 1, int(np.round(start_time * sample_freq)):int(np.round(end_time * sample_freq))] = \
            amps2[m] / 1000

    savemat('save_trial.mat', {'stim': stim_inputs_per_trail})

    for i in range(len(days)):

        day = days[i]

        for j in range(len(seq)):
            stim_ind = seq[j]

            # 找到包含stim_ind的记录文件
            for filewalks in os.walk(data_path + day + rec_type):
                for files in filewalks[2]:
                    if stim_ind in files and stim_ind + "0" not in files:
                        file = os.path.join(filewalks[0], files)
            #                     print(stim_ind,' is in',file)

            # 读取单个记录文件
            # print(file)
            # stim_type 0:lan2-lv1,1:lan1-lv2

            # if i==0:
            #     stim_type = 0 if j<5 else 1
            # else:
            #     stim_type = [k for k in range(len(stim_seq)) if stim_seq[k] in file][0]

            # y = [0,1] if stim_type==0 else [1,0]
            # for x in range(0,2):
            #     stim_inputs[i,j,x*2] = stim_inputs_per_trail[y[x],0,:]
            #     stim_inputs[i,j,x*2+1] = stim_inputs_per_trail[y[x],1,:]

            stim_type = [k for k in range(len(stim_seq)) if stim_seq[k] in file][0]

            y = [0, 1] if stim_type == 0 else [1, 0]
            for x in range(0, 2):
                stim_inputs[i, j, x * 2] = stim_inputs_per_trail[y[x], 0, :]
                stim_inputs[i, j, x * 2 + 1] = stim_inputs_per_trail[y[x], 1, :]

    # # 可视化刺激序列
    # plt.figure(figsize=(50, 20), dpi=80)
    # for j in range(4):
    #     plt.subplot(4, 1, j + 1)
    #     # plt.ylim((0,5))
    #     #     plt.plot(np.linspace(0.0,time_len,time_len*sample_freq),stim_inputs[0,6,j])
    #     plt.plot(np.linspace(0.0, time_len, time_len * sample_freq), stim_inputs_per_trail[j // 2, j % 2])
    #
    #     #     plt.plot(range(800),firing_rate[j,0],label='seq_'+str(1))
    #     #     plt.plot(range(800),firing_rate[j,3],label='seq_'+str(4))
    #     plt.legend(loc=2)
    # save_file = 'stimulation_parameter_' + str(sample_freq) + 'hz.png'
    # plt.savefig(save_file, dpi=300)

    return stim_inputs


def readOptimizedStimulatingData(stimsPath, stim_seq, data_path, days, seq, time_len, sample_freq=1, start_time=2):
    # 读取优化刺激
    stim_inputs = np.zeros([len(days), len(seq), 4, time_len * sample_freq])
    stim_inputs_per_trail = np.zeros([2, 2, time_len * sample_freq])
    for i in range(2):
        stim_path = data_path + stimsPath[0]
        stim_frequency = np.array(scio.loadmat(stim_path)['stim%d_Freq' % (i % 2 + 1)])
        stim_amplitude = np.array(scio.loadmat(stim_path)['stim%d_Amp' % (i % 2 + 1)])
        print('stim_frequency shape: ', stim_frequency.shape)

        start_index = start_time * sample_freq
        for stim_time_index in range(len(stim_frequency[0])):
            stim_inputs_per_trail[i, 0, stim_time_index + start_index] = stim_frequency[0, stim_time_index]
            stim_inputs_per_trail[i, 1, stim_time_index + start_index] = stim_amplitude[0, stim_time_index] / 1000

    for i in range(len(days)):
        for j in range(len(seq)):
            y = [0, 1]
            for x in range(0, 2):
                stim_inputs[i, j, x * 2] = stim_inputs_per_trail[y[x], 0, :]
                stim_inputs[i, j, x * 2 + 1] = stim_inputs_per_trail[y[x], 1, :]

    return stim_inputs


def saveDailyFR(rec_spike_count, days, len_seq, channels, time_len, vmax=10, vmin=0):
    # 按通道汇总FR
    avg_FR = np.zeros([len(days), channels])
    for i in range(len(days)):
        for j in range(channels):
            avg_FR[i, j] = np.sum(rec_spike_count[i, :, j, :]) / (len_seq * time_len)

    # print(avg_FR)
    avg_FR_channel = np.zeros([channels])
    for i in range(channels):
        avg_FR_channel[i] = np.mean(avg_FR[:, i])
    #     print("channel %02d: %f"%(i+1,np.mean(avg_FR[:,i])))
    drawHeatmap(avg_FR_channel, 'avg_FR', vmax=vmax, vmin=vmin)
    for i in range(len(days)):
        drawHeatmap(avg_FR[i], 'avg_FR_day%d' % (i + 1), vmax=vmax, vmin=vmin)


def saveTrailFR(rec_spike_count, days, len_seq, channels, time_len, vmax=10, vmin=0):
    # 按通道汇总FR
    avg_FR = np.zeros([len(days), len_seq, channels])
    for i in range(len(days)):
        for k in range(len_seq):
            for j in range(channels):
                avg_FR[i, k, j] = np.sum(rec_spike_count[i, k, j, :]) / time_len

    # print(avg_FR)
    avg_DailyFR_channel = np.zeros([len(days), channels])
    for i in range(len(days)):
        for j in range(channels):
            avg_DailyFR_channel[i, j] = np.mean(avg_FR[i, :, j])
        drawHeatmap(avg_DailyFR_channel[i], 'avg_FR_day%d' % (i + 1), vmax=vmax, vmin=vmin)
        for k in range(len_seq):
            drawHeatmap(avg_FR[i, k], 'avg_FR_day%d_trail%d' % (i + 1, k + 1), vmax=vmax, vmin=vmin)


def calStimSet(data_path, days, rec_seq, stim_seq, rec_type, electrodeSets):
    # 获得每天每个trail中每个通道对应的stim_time number type 1
    stim_time_indexs = np.zeros([len(days), len(rec_seq), 8, 8])
    for i in range(len(days)):
        day = days[i]
        for j in range(len(rec_seq)):
            stim_ind = rec_seq[j]
            # 找到包含stim_ind的记录文件
            print((data_path + day + rec_type))
            filePath = findRecFiles_1((data_path + day + rec_type), stim_ind)
            print(filePath)
            # 读取单个记录文件
            stim_type = [k for k in range(len(stim_seq)) if stim_seq[k] in filePath][0]
            #         print(stim_type)
            y = [0, 1] if stim_type == 0 else [1, 0]
            for row in range(8):
                for col in range(8):
                    stim_time_indexs[i, j, row, col] = y[calSets(electrodeSets, row + 1, col + 1)]
    return stim_time_indexs


def calPSTH_1(data_path, days, rec_seq, rec_type, stimsPath, stim_time_indexs, fileType=2, start_=10, time_len_PSTH=190,
              time_bin=4):
    # 计算每天每个trail各通道的PSTH type 1
    # set_type = 0
    # start_ = 10  # 开始时间 ms
    # time_len_PSTH = 190  # 总时间 ms
    # time_bin = 4  # bin length/ms
    PSTHs = np.zeros([len(days), len(rec_seq), 60])
    stim_path_0 = data_path + stimsPath[0]
    stim_time_0 = np.array(scio.loadmat(stim_path_0)['random%d' % (1)][:, 0])
    stim_path_1 = data_path + stimsPath[1]
    stim_time_1 = np.array(scio.loadmat(stim_path_1)['random%d' % (2)][:, 0])

    for i in range(len(days)):
        day = days[i]
        for j in range(len(rec_seq)):
            stim_ind = rec_seq[j]
            # 找到包含stim_ind的记录文件
            filePath = findRecFiles_1((data_path + day + rec_type), stim_ind)
            # 读取单个记录文件
            print(filePath)
            #         spikes_sequence=h5py.File(file)
            # spikes_sequence = scio.loadmat(filePath)
            spikes_sequence = readFiles(filePath, fileType)
            electrode = -1
            for row in range(1, 9):
                for col in range(1, 9):
                    # str_read_key = 'AnSt_Label_E_00159_' + str(row) + str(col) + '_ID_'
                    str_read_key = 'AnSt_Label_D_00138_' + str(row) + str(col) + '_ID_'
                    key_name = [k for k, v in spikes_sequence.items() if str_read_key in k]
                    # 当对应电极存在神经响应记录时，滑动窗计算spike count
                    if len(key_name) > 0:
                        # spike_data_ = np.array(spikes_sequence[key_name[0]][:, 0])
                        spike_data_ = getSpikeSequence(spikes_sequence, key_name, fileType)
                        electrode += 1
                        stim_index = stim_time_indexs[i, j, row - 1, col - 1]
                        if stim_index == 0:
                            stim_time_ = stim_time_0
                        elif stim_index == 1:
                            stim_time_ = stim_time_1
                        else:
                            print('Error: invalid stim_time_index!')
                            continue
                        psth, aera = simplePSTH(stim_time_, spike_data_, start_, time_len_PSTH, time_bin)
                        PSTHs[i, j, electrode] = aera

                    # 当没有记录时，排除四角电极（11，18，81，88），其余填空
                    else:
                        if (row == 1 or row == 8) and (col == 1 or col == 8):
                            continue
                        else:
                            electrode += 1
    return PSTHs


def channelSelection(PSTHs, threshold=1):
    mean_PSTH = np.mean(np.mean(PSTHs, axis=1), axis=0)
    selected_channels = np.zeros([60])

    for i in range(60):
        selected_channels[i] = 1 if mean_PSTH[i] > threshold else 0
    # drawHeatmap(selected_channels,'selected channels')
    drawHeatmap(selected_channels, 'selected channels')
    drawHeatmap(mean_PSTH, 'PSTH area')
    # print(PSTHs.shape)
    test_mean_path = np.mean(PSTHs, axis=1)
    # print(test_mean_path.shape)
    # print(test_mean_path[1])
    num_days = len(PSTHs)
    daily_selected_channels = np.zeros([num_days, 60])
    for j in range(num_days):
        for i in range(60):
            daily_selected_channels[j, i] = 1 if test_mean_path[j, i] > threshold else 0
    for i in range(num_days):
        drawHeatmap(daily_selected_channels[i], 'Day%d selected channels' % (i + 1))
        drawHeatmap(test_mean_path[i], 'Day%d PSTH area ' % (i + 1))
    return selected_channels, daily_selected_channels


# Not Finished
def channelSelectionByTrail(PSTHs, threshold=1):
    mean_PSTH = np.mean(np.mean(PSTHs, axis=1), axis=0)
    selected_channels = np.zeros([60])
    for i in range(60):
        selected_channels[i] = 1 if mean_PSTH[i] > threshold else 0
    # drawHeatmap(selected_channels,'selected channels')
    drawHeatmap(selected_channels, 'selected channels')
    drawHeatmap(mean_PSTH, 'PSTH area')
    # print(PSTHs.shape)
    test_mean_path = np.mean(PSTHs, axis=1)
    # print(test_mean_path.shape)
    # print(test_mean_path[1])
    num_days = len(PSTHs)
    daily_selected_channels = np.zeros([num_days, 60])
    for j in range(num_days):
        for i in range(60):
            daily_selected_channels[j, i] = 1 if test_mean_path[j, i] > threshold else 0
    for i in range(num_days):
        drawHeatmap(daily_selected_channels[i], 'Day%d selected channels' % (i + 1))
        drawHeatmap(test_mean_path[i], 'Day%d PSTH area ' % (i + 1))
        for j in range(len(PSTHs[0])):
            drawHeatmap(PSTHs[i, j], 'Day%d Trail%d PSTH area' % (i + 1, j + 1))
            drawHeatmap(test_mean_path[i], 'Day%d PSTH area ' % (i + 1))
    return selected_channels, daily_selected_channels


def resizeAndSmooth(rec_spike_count, stim_inputs, sample_freq, kern_sd_ms=100, test_ref=True):
    firing_rate_per_channel = rec_spike_count * sample_freq
    resized_stim_inputs = None
    resized_stim_spike_count = resizeData(firing_rate_per_channel)
    if not test_ref:
        stim_inputs_V = stim_inputs  # / 1000
        for i in [1, 3]:
            stim_inputs_V[:, :, i, :] = stim_inputs[:, :, i, :] * 10  # 幅值整理由0.x V 变为 0.x*10  V/10

        resized_stim_inputs = resizeData(stim_inputs_V)

    # kern_sd_ms = 100
    kern_sd = int(round(kern_sd_ms / 10))
    window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
    window /= np.sum(window)
    filt = lambda x: np.convolve(x, window, 'same')
    # resized_stim_inputs_smth = np.apply_along_axis(filt, 1, resized_stim_inputs)
    resized_rec_spike_count_smth = np.apply_along_axis(filt, 1, resized_stim_spike_count)
    print('input_shape:', resized_stim_inputs.shape)

    plt.figure(figsize=(50, 20), dpi=80)
    for j in range(4):
        plt.subplot(4, 1, j + 1)
        plt.plot(resized_stim_inputs[j, :3000])
        plt.legend(loc=2)
    save_file = 'cur_stimulation_parameter_' + str(sample_freq) + 'hz.png'
    plt.savefig(save_file, dpi=300)
    plt.close()

    return resized_rec_spike_count_smth, resized_stim_inputs


def saveDataByChannel(selected_channels, resized_stim_spike_count_smth, resized_stim_inputs, numOfSet, day, sampe_fre,
                      saveDir='./data/', save_Ref=False, ref_list=None, kern_std=100):
    if (not os.path.isdir(saveDir)):
        os.makedirs(saveDir)
    num_selected_channels = int(np.sum(selected_channels))
    index_ = 0
    screened_rec_data = np.zeros([num_selected_channels, len(resized_stim_spike_count_smth[0])])
    print(screened_rec_data.shape)
    for i in range(60):
        if selected_channels[i] > 0:
            screened_rec_data[index_] = resized_stim_spike_count_smth[i]
            index_ += 1
    if not save_Ref:
        print(r"%s/FR_Set%d_day%s_%dHz_smth_%dchannels_kern_std_%d.mat" % (
            saveDir, numOfSet, day, sampe_fre, num_selected_channels, kern_std))
        print(screened_rec_data.shape)
        print(resized_stim_inputs.shape)
        scio.savemat(r"%s/FR_Set%d_day%s_%dHz_smth_%dchannels_kern_std_%d.mat" % (
            saveDir, numOfSet, day, sampe_fre, num_selected_channels, kern_std),
                     {'Cortex': screened_rec_data, 'Inputs': resized_stim_inputs})
    else:
        print(r"%s/Y_ref_%s_FR_Set%d_day%s_%dHz_smth_%dchannels.mat" % (
            saveDir, ref_list, numOfSet, day, sampe_fre, num_selected_channels))
        scio.savemat(r"%s/Y_ref_%s_FR_Set%d_day%s_%dHz_smth_%dchannels.mat" % (
            saveDir, ref_list, numOfSet, day, sampe_fre, num_selected_channels), {'Cortex': screened_rec_data})
