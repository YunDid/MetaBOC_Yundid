import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from scipy.io import loadmat

from DataPreprocessing import *
# import sys
# sys.path.append('data_analysis')
from data_analysis.drawing import *

from Trainer import ModelTrainer

jsonDataDir = {
    '01-05': '12channels',
    '01-05-2': '13channels',
    '01-08': '7channels',
    '01-09': '9channels',
    '01-09-02': '9channels',
    '01-12': '11channels',
    '01-12-02': '8channels',
    '01-13': '12channels',
    '01-13-02': '8channels',
    '01-14': '8channels',
    '01-14-02': '6channels'
}

selected_channels_4c = np.array([0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 1., 0., 0., 0.,
                                 1., 0., 0., 0., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0., 1., 0.,
                                 0., 0., 0., 1., 0., 0., 0., 0.,
                                 0., 0., 0., 0., 0., 0.])

# set_8
selected_channels_2c = np.array([
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def list2color(listExample):
    string = '#'
    for i in listExample:
        string_ = hex(i)
        index = string_.find('x')
        string += string_[index + 1:] if len(string_[index + 1:]) > 1 else '0' + string_[index + 1:]
    return string


def generateColors(colorA, colorB, num):
    colors = {}
    start = [int(colorA[1:3], 16), int(colorA[3:5], 16), int(colorA[5:7], 16)]
    end = [int(colorB[1:3], 16), int(colorB[3:5], 16), int(colorB[5:7], 16)]
    delta = [(end[i] - start[i]) / (num - 1.0) for i in range(len(start))]

    colors[0] = list2color(start)
    for i in range(1, num):
        colors[i] = list2color([int(start[j] + i * delta[j]) for j in range(3)])
    print(colors)
    return colors


def testPreprocessing():
    # 1.测试数据保存
    # 2.从json文件读取params和数据
    selected_channels_12c = np.array([0., 0., 0., 0., 0., 1.,
                                      0., 0., 1., 0., 0., 0., 0., 1.,
                                      1., 1., 0., 0., 0., 0., 0., 1.,
                                      0., 0., 0., 0., 1., 0., 0., 1.,
                                      1., 0., 0., 0., 0., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0., 1., 0.,
                                      0., 0., 0., 1., 1., 0., 0., 0.,
                                      0., 0., 0., 0., 0., 0.])
    data_params = {
        'dataPath': 'Z:/Organoid/科技部数据0710/课题四数据/第八批数据/',
        # 'dataPath': '/mnt/database6/Organoid/科技部数据0710/课题四数据/第七批数据/',
        'stimPath': ['random_stim1.mat', 'random_stim2.mat'],
        'set': 'Set8',
        'day': '03-10/',
        'type': '系统辨识/',
        'rec_seq': ['1-lv', '2-lv', '3-lv', '4-lv', '5-lv'],  # , '6-lv', '7-lv', '8-lv'],
        'stim_seq': ['lv2-lan1', 'lv1-lan2'],
        'sample_fre': 4,
        'rec_channels': 60,
        'stim_channels': 2,
        'kern_sd': 10,
        'recording_time_len': 335,
        'ref': False,
        'stim_type': 3,
        'electrode_set': [[[6, 4], [6, 5]], [[2, 6], [2, 7]]],
        'selected_channels': np.ones(60)
    }
    # 固定参数生成数据
    # 如果selected_channels=None，可以通过初始化是设置PSTH_time_len和threshold自动选择通道
    preprocess = Preprocessing(data_params, threshold=.5)
    # preprocess = Preprocessing(
    #     r"Z:\Organoid\codes\baseline\reconstructed_codes2.0_230222\data\Set7\01-14\12channels\4Hz\kern_sd_10\data.json",
    #     json_file_type=True)
    # # recordingData = preprocess.recording_data
    # # stimData = preprocess.stimulating_data
    # # print(recordingData.shape, stimData.shape)
    # preprocess.calPSTH(time_len_PSTH=40)
    # preprocess.drawPSTHs(time_len_PSTH=40, threshold=.4)


def testModelTrainer():
    special_params = {
        'L_factors': 0.05,
        'EI_ratio': 4,
        'l1_reg': False,
        'learning_rate': 0.001,
        'with_Tanh': False,
        'sparsity': 0,
        'e_clusters': 2,
        'i_clusters': 1,
        'inter_or_intra': 'Inter',
        'l2_norm': 0.001
    }

    extra_params = {
        'wrec_fix': None
    }

    simple_params = {
        'json_dir': r"/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set8/03-10/7channels/4Hz/kern_sd_10/data.json",
        'Tp': 10,
        'Sample_Size': 200000,
        'data_length': 3,  # Hyperparameter
        'HIDDEN_DIM': 100,
        'activation': 'ReLU',
        'final_activation': 'Softplus',
        'RNN_Type': 'EI-RNN',
        'l1_reg': False,
        'with_Tanh': False,
        'train_day': '03-10/',
        'init_ckpt_path': None,
        'test_sample_size': 100,
        'prediction_len': 500,
        'cross_validation': False,
        'num_folder': 5,
        'epochs': 50,
        'special_params': special_params,
        'extra_params': extra_params
    }

    testTrainer = ModelTrainer(simple_params=simple_params)
    testTrainer.training()


def generateXChannelsData():
    days = ['12-26', '12-27', '12-28', '12-29', '12-31', '01-05', '01-05-2', '01-08', '01-09', '01-09-02', '01-12',
            '01-12-02', '01-13', '01-12-02', '01-14', '01-14-02']
    rec_seq_index = [0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1]
    rec_seqs = {
        0: ['1-lv', '2-lv', '3-lv', '4-lv', '5-lv', '6-lv', '7-lv', '8-lv', '9-lv', '10-lv'],
        1: ['1-lv', '2-lv', '3-lv', '4-lv', '5-lv'],
        2: ['1-lv', '2-lv', '3-lv', '4-lv', '5-lv', '6-lv', '7-lv', '8-lv']
    }
    for i in range(len(days)):
        day = days[i]
        rec_seq = rec_seqs[rec_seq_index[i]]
        selected_channels = np.ones(60)
        data_params = {
            # 'dataPath': 'Z:/Organoid/科技部数据0710/课题四数据/第七批数据/',
            'dataPath': '/mnt/database6/Organoid/科技部数据0710/课题四数据/第七批数据/',
            'stimPath': ['random_stim1.mat', 'random_stim2.mat'],
            'set': 'Set7',
            'day': day + '/',
            'type': '系统辨识/',
            'rec_seq': rec_seq,
            'stim_seq': ['lv2-lan1', 'lv1-lan2'],
            'sample_fre': 4,
            'rec_channels': 60,
            'stim_channels': 2,
            'kern_sd': 10,
            'recording_time_len': 335,
            'ref': False,
            'stim_type': 3,
            'electrode_set': [[[2, 7], [3, 7]], [[4, 2], [5, 2]]],  # set 7
            'selected_channels': selected_channels_4c
        }
        a = Preprocessing(data_params, threshold=.4)


def dailyTraining():
    # days = ['01-05', '01-05-2', '01-08']
    # days = ['01-09', '01-09-02', '01-12']
    # days = ['01-12-02', '01-13', '01-14', '01-14-02']
    days = ['01-05', '01-05-2', '01-08', '01-09', '01-09-02', '01-12', '01-12-02', '01-13', '01-14', '01-14-02']
    for i in range(len(days)):
        # 每天PSTH 阈值=.4 选择电极
        jsonFilePath1 = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set7/%s/%s/4Hz/kern_sd_10/data.json" % (
            days[i], jsonDataDir[days[i]])
        # 1月5日 PSTH选取的4channels
        jsonFilePath2 = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set7/%s/4channels/4Hz/kern_sd_10/data.json" % (
            days[i])
        for jsonFilePath in [jsonFilePath1, jsonFilePath2]:
            for sparsity in [0, .5, .75, 1]:
                special_params = {
                    'L_factors': 0.05,
                    'EI_ratio': 4,
                    'l1_reg': False,
                    'learning_rate': 0.001,
                    'with_Tanh': False,
                    'sparsity': sparsity,
                    'e_clusters': 2,
                    'i_clusters': 1,
                    'inter_or_intra': 'Intra_with_E1_fixed_but_changed_E2',
                    'l2_norm': 0.001
                }
                extra_params = {
                    'wrec_fix': None
                }
                simple_params = {
                    'json_dir': jsonFilePath,
                    'Tp': 10,
                    'Sample_Size': 200000,
                    'data_length': 3,  # Hyperparameter
                    'HIDDEN_DIM': 100,
                    'activation': 'ReLU',
                    'final_activation': 'Softplus',
                    'RNN_Type': 'EI-RNN',
                    'l1_reg': False,
                    'with_Tanh': False,
                    'train_day': '03-03/',
                    'init_ckpt_path': None,
                    'test_sample_size': 100,
                    'prediction_len': 500,
                    'cross_validation': False,
                    'num_folder': 5,
                    'epochs': 200,
                    'special_params': special_params,
                    'extra_params': extra_params
                }
                testTrainer = ModelTrainer(simple_params=simple_params)
                testTrainer.training()


def generateFintuningData():
    # days = ['01-05', '01-05-2', '01-08', '01-09', '01-09-02', '01-12',
    #         '01-12-02', '01-13', '01-14', '01-14-02']
    days = ['01-09', '01-09-02']
    # rec_seq_index = [0, 1, 1, 1, 1, 1, 1, 1, 2, 1]
    rec_seq_index = [1, 1]
    rec_seqs = {
        0: ['1-lv', '2-lv', '3-lv', '4-lv', '5-lv', '6-lv', '7-lv', '8-lv', '9-lv', '10-lv'],
        1: ['1-lv', '2-lv', '3-lv', '4-lv', '5-lv'],
        2: ['1-lv', '2-lv', '3-lv', '4-lv', '5-lv', '6-lv', '7-lv', '8-lv']
    }
    for i in range(1, len(days)):
        day = days[i]
        rec_seq = rec_seqs[rec_seq_index[i]]
        # selected_channels = np.ones(60)
        init_day = days[i - 1]
        init_jsonFilePath = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set7/" + \
                            "%s/%s/4Hz/kern_sd_10/data.json" % (init_day, jsonDataDir[init_day])
        init_day_pro = Preprocessing(init_jsonFilePath, json_file_type=True)
        data_params = {
            # 'dataPath': 'Z:/Organoid/科技部数据0710/课题四数据/第七批数据/',
            'dataPath': '/mnt/database6/Organoid/科技部数据0710/课题四数据/第七批数据/',
            'stimPath': ['random_stim1.mat', 'random_stim2.mat'],
            'set': 'Set7',
            'day': day + '/',
            'type': '系统辨识/',
            'rec_seq': rec_seq,
            'stim_seq': ['lv2-lan1', 'lv1-lan2'],
            'sample_fre': 4,
            'rec_channels': 60,
            'stim_channels': 2,
            'kern_sd': 10,
            'recording_time_len': 335,
            'ref': False,
            'stim_type': 3,
            'electrode_set': [[[2, 7], [3, 7]], [[4, 2], [5, 2]]],
            'selected_channels': init_day_pro.selected_channels
        }
        # print(init_day_pro.selected_channels)
        a = Preprocessing(data_params)


def finetuning():
    # days = ['01-05', '01-05-2', '01-08', '01-09']
    # days = ['01-09', '01-09-02', '01-12', '01-12-02']
    # days = ['01-12-02', '01-13', '01-14', '01-14-02']
    days = ['01-05', '01-05-2', '01-08', '01-09', '01-09-02', '01-12', '01-12-02', '01-13', '01-14', '01-14-02']
    for i in range(1, len(days)):
        init_day = days[i - 1]
        day = days[i]
        # # 每天PSTH 阈值=.4 选择的电极
        # num_channels = jsonDataDir[init_day]
        # 1月5日 PSTH选取的4channels
        num_channels = '4channels'

        jsonFilePath = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set7/" + \
                       "%s/%s/4Hz/kern_sd_10/data.json" % (day, num_channels)
        for sparsity in [0, 0.5, 0.75, 1]:
            special_params = {
                'L_factors': 0.05,
                'EI_ratio': 4,
                'l1_reg': False,
                'learning_rate': 0.001,
                'with_Tanh': False,
                'sparsity': sparsity,
                'e_clusters': 2,
                'i_clusters': 1,
                'inter_or_intra': 'L1_reg',
                'l2_norm': 0.001
            }
            extra_params = {
                'wrec_fix': np.ones((50, 50))
            }
            init_ckpt_path = '/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/checkpoints/' + \
                             'random_test/EI-RNN/02-24/init_training/Set7_%s/' % init_day + \
                             '_%s_4Hz_kern_sd10/hidden_100_time_length_3_tp_10/' % num_channels + \
                             'L_factors_0.05_EI_ratio_4_l1_reg_False_learning_rate_0.001_with_Tanh_False_sparsity_%s' % sparsity + \
                             '_e_clusters_2_i_clusters_1_inter_or_intra_L1_reg_l2_norm_0.001.ckpt'
            simple_params = {
                'json_dir': jsonFilePath,
                'Tp': 10,
                'Sample_Size': 200000,
                'data_length': 3,  # Hyperparameter
                'HIDDEN_DIM': 100,
                'activation': 'ReLU',
                'final_activation': 'Softplus',
                'RNN_Type': 'EI-RNN',
                'l1_reg': False,
                'with_Tanh': False,
                'train_day': '03-01/',
                'init_ckpt_path': init_ckpt_path,
                'test_sample_size': 20,
                'prediction_len': 500,
                'cross_validation': False,
                'num_folder': 5,
                'epochs': 200,
                'special_params': special_params,
                'extra_params': extra_params
            }
            testTrainer = ModelTrainer(simple_params=simple_params)
            testTrainer.training()


def generateFixedChannelData():
    days = ['01-05', '01-05-2', '01-08', '01-09', '01-09-02', '01-12',
            '01-12-02', '01-13', '01-14', '01-14-02']
    # days = ['01-09', '01-09-02']
    rec_seq_index = [0, 1, 1, 1, 1, 1, 1, 1, 2, 1]
    # rec_seq_index = [1, 1]
    rec_seqs = {
        0: ['1-lv', '2-lv', '3-lv', '4-lv', '5-lv', '6-lv', '7-lv', '8-lv', '9-lv', '10-lv'],
        1: ['1-lv', '2-lv', '3-lv', '4-lv', '5-lv'],
        2: ['1-lv', '2-lv', '3-lv', '4-lv', '5-lv', '6-lv', '7-lv', '8-lv']
    }
    for i in range(len(days)):
        day = days[i]
        rec_seq = rec_seqs[rec_seq_index[i]]
        # selected_channels = np.ones(60)
        data_params = {
            # 'dataPath': 'Z:/Organoid/科技部数据0710/课题四数据/第七批数据/',
            'dataPath': '/mnt/database6/Organoid/科技部数据0710/课题四数据/第七批数据/',
            'stimPath': ['random_stim1.mat', 'random_stim2.mat'],
            'set': 'Set7',
            'day': day + '/',
            'type': '系统辨识/',
            'rec_seq': rec_seq,
            'stim_seq': ['lv2-lan1', 'lv1-lan2'],
            'sample_fre': 4,
            'rec_channels': 60,
            'stim_channels': 2,
            'kern_sd': 10,
            'recording_time_len': 335,
            'ref': False,
            'stim_type': 3,
            'electrode_set': [[[2, 7], [3, 7]], [[4, 2], [5, 2]]],
            'selected_channels': selected_channels_4c
        }
        # print(init_day_pro.selected_channels)
        a = Preprocessing(data_params)


def drawtrainingHeatmap(days, inters, sparsities, training_day, kern_sd, sample_fre, set_num, num_channels='5channels'):
    # days = ['01-05', '01-05-2', '01-08', '01-09', '01-09-02', '01-12', '01-12-02', '01-13', '01-14', '01-14-02']
    # training_day = '03-01'
    # fixed = True
    # sparsities = [0, 0.5, 0.75, 1]
    R2_init = np.zeros((len(days), len(sparsities)))
    EV_init = np.zeros((len(days), len(sparsities)))
    # R2_FT = np.zeros((len(days), 4))
    # EV_FT = np.zeros((len(days), 4))
    # R2_test = np.zeros((len(days), 4))
    # EV_test = np.zeros((len(days), 4))

    for inter in inters:
        i_cluster = 1 if inter != 'Inter_with_E1_I2_E2_I1_sparsity' else 2
        for i in range(len(days)):
            day = days[i]
            for j in range(len(sparsities)):
                sparsity = sparsities[j]
                # if fixed:
                #     num_channels = '4channels'
                # else:
                #     num_channels = jsonDataDir[day]

                filePath = r'./result_summary/random_test/EI-RNN/%s/init_training/Set%d_%s' % (
                    training_day, set_num, day) + \
                           '/_%s_%dHz_kern_sd%d/hidden_100_time_length_3_tp_10/L_factors_0.05_EI_ratio_4_l1_reg_False_' % (
                               num_channels, sample_fre, kern_sd) + \
                           'learning_rate_0.001_with_Tanh_False_sparsity_%s_e_clusters_2_i_clusters_%d_inter_or_intra_%s_l2_norm_0.001_.mat' % (
                               sparsity, i_cluster, inter)
                scores_ = scio.loadmat(filePath)
                R2_init[i, j] = scores_['R2']
                EV_init[i, j] = scores_['EV']

        # for i in range(1, len(days)):
        #     day = days[i]
        #     init_day = days[i-1]
        #     for j in range(4):
        #         sparsity = sparsities[j]
        #         # num_channels = jsonDataDir[init_day]
        #         num_channels = '4channels'
        #         filePath = r'./result_summary/random_test/EI-RNN/%s/fine_tuning/Set7_%s'% (training_day,day) + \
        #                    '/_%s_4Hz_kern_sd10/hidden_100_time_length_3_tp_10/L_factors_0.05_EI_ratio_4_l1_reg_False_' % num_channels + \
        #                    'learning_rate_0.001_with_Tanh_False_sparsity_%s_e_clusters_2_i_clusters_1_inter_or_intra_L1_reg_l2_norm_0.001_.mat' % sparsity
        #         scores_ = scio.loadmat(filePath)
        #         R2_FT[i, j] = scores_['R2']
        #         EV_FT[i, j] = scores_['EV']

        # for i in range(1, len(days)):
        #     day = days[i]
        #     init_day = days[i - 1]
        #     for j in range(4):
        #         sparsity = sparsities[j]
        #         if fixed:
        #             num_channels = '4channels'
        #         else:
        #             num_channels = jsonDataDir[init_day]
        #         filePath = r'./result_summary/random_test/EI-RNN/%s/init_training/Set7_%s' % (training_day, init_day) + \
        #                    '/_%s_4Hz_kern_sd10/hidden_100_time_length_3_tp_10/L_factors_0.05_EI_ratio_4_l1_reg_False_' % num_channels + \
        #                    'learning_rate_0.001_with_Tanh_False_sparsity_%s_e_clusters_2_i_clusters_1_inter_or_intra_L1_reg_l2_norm_0.001__cross_day_%s_.mat' % (
        #                        sparsity, day)
        #         scores_ = scio.loadmat(filePath)
        #         R2_test[i, j] = scores_['R2']
        #         EV_test[i, j] = scores_['EV']

        #     画热图
        cmap = 'Blues'
        center = None
        square = False
        vmin = 0.4
        vmax = .9
        annot = True
        x_ticks = sparsities
        y_ticks = days

        # if fixed:
        #     inWord = 'fixed4Channels'
        # else:
        #     inWord = 'PSTHSelectedChannels'

        print('R2 with init_training')
        pd_R2 = ndarray2Dataframe(R2_init, x_ticks, y_ticks)
        drawHeatmap(pd_R2,
                    name='./data_analysis/statistic_fig/training_result/%s_R2_0105-0114_EIRNN_%s_initTraining.png' % (
                        training_day, inWord),
                    title='R2 with init_training', x_lable='Sparsity', y_lable='Day',
                    cmap=cmap, square=square, vmin=vmin, vmax=vmax, center=center, annot=annot)

        print('EV with init_training')
        pd_EV = ndarray2Dataframe(EV_init, x_ticks, y_ticks)
        drawHeatmap(pd_EV,
                    name='./data_analysis/statistic_fig/training_result/%s_EV_0105-0114_EIRNN_%s_initTraining.png' % (
                        training_day, inWord),
                    title='EV with init_training', x_lable='Sparsity', y_lable='Day',
                    cmap=cmap, square=square, vmin=vmin, vmax=vmax, center=center, annot=annot)

        # print('R2 with fine-tuning')
        # pd_R2_FT = ndarray2Dataframe(R2_FT, x_ticks, y_ticks)
        # drawHeatmap(pd_R2_FT,
        #             name='./data_analysis/statistic_fig/training_result/R2_0105-0114_EIRNN_%s_fine-tuning.png' % inWord,
        #             title='R2 with fine-tuning', x_lable='Sparsity', y_lable='Day',
        #             cmap=cmap, square=square, vmin=vmin, vmax=vmax, center=center, annot=annot)

        # print('EV with fine-tuning')
        # pd_EV_FT = ndarray2Dataframe(EV_FT, x_ticks, y_ticks)
        # drawHeatmap(pd_EV_FT,
        #             name='./data_analysis/statistic_fig/training_result/EV_0105-0114_EIRNN_%s_fine-tuning.png' % inWord,
        #             title='EV with fine-tuning', x_lable='Sparsity', y_lable='Day',
        #             cmap=cmap, square=square, vmin=vmin, vmax=vmax, center=center, annot=annot)

        print('R2 with cross-day testing')
        pd_R2_test = ndarray2Dataframe(R2_test, x_ticks, y_ticks)
        drawHeatmap(pd_R2_test,
                    name='./data_analysis/statistic_fig/training_result/%s_R2_0105-0114_EIRNN_%s_cross-day-testing.png' % (
                        training_day, inWord),
                    title='R2 with cross-day testing', x_lable='Sparsity', y_lable='Day',
                    cmap=cmap, square=square, vmin=vmin, vmax=vmax, center=center, annot=annot)

        print('EV with cross-day testing')
        pd_EV_test = ndarray2Dataframe(EV_test, x_ticks, y_ticks)
        drawHeatmap(pd_EV_test,
                    name='./data_analysis/statistic_fig/training_result/%s_EV_0105-0114_EIRNN_%s_cross-day-testing.png' % (
                        training_day, inWord),
                    title='EV with cross-day testing', x_lable='Sparsity', y_lable='Day',
                    cmap=cmap, square=square, vmin=vmin, vmax=vmax, center=center, annot=annot)

        # print(pd_R2[0.0]) # 对应列的结果和日期
        # print(list(pd_R2))
        sparsities = [0.0, 0.5, 0.75, 1.0]
        colors = ['red', 'yellow', 'green', 'blue', 'black']
        colors = ['#E889BD', '#67C2A3', '#FC8A61', '#8EA0C9']

        # 条形图
        axes = pd_EV.plot.bar(rot=0,
                              color={sparsities[0]: colors[0], sparsities[1]: colors[1], sparsities[2]: colors[2],
                                     sparsities[3]: colors[3]})
        # axes[1].legend(loc=2)
        plt.title('Training Results')
        plt.xlabel('Days')
        plt.ylabel('Accuracy(EV)')
        plt.ylim((0, 1))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            './data_analysis/statistic_fig/training_result/%s_init-training-results_%s.png' % (training_day, inWord))
        plt.close()

        # axes = pd_EV_FT.plot.bar(rot=0, color={sparsities[0]: colors[0], sparsities[1]: colors[1], sparsities[2]: colors[2],
        #                                     sparsities[3]: colors[3]})
        # # axes[1].legend(loc=2)
        # plt.title('Training Results')
        # plt.xlabel('Days')
        # plt.ylabel('Accuracy(EV)')
        # plt.ylim((0, 1))
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig('./data_analysis/statistic_fig/training_result/$s_fine-tuning-results_%s.png' % (training_day,inWord))
        # plt.close()

        axes = pd_EV_test.plot.bar(rot=0,
                                   color={sparsities[0]: colors[0], sparsities[1]: colors[1], sparsities[2]: colors[2],
                                          sparsities[3]: colors[3]})
        # axes[1].legend(loc=2)
        plt.title('Training Results')
        plt.xlabel('Days')
        plt.ylabel('Accuracy(EV)')
        plt.ylim((0, 1))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            './data_analysis/statistic_fig/training_result/%s_cross-day-testing-results_%s.png' % (
                training_day, inWord))
        plt.close()

        # 折线图
        for i in range(4):
            sparsity = sparsities[i]
            color = colors[i]
            plt.title('Sparsity = %s' % sparsity)
            plt.xlabel('Days')
            plt.ylabel('Accuracy(EV)')
            plt.ylim((0, 1))
            plt.xticks(rotation=45)
            # label = 'sparsity=%s' % sparsity
            plt.plot(pd_EV[sparsity], color=color, label='init-training', marker='.')
            plt.plot(pd_EV_test[sparsity].drop(index='01-05'), color=color, label='fine-tuning', linestyle='--',
                     marker='.')
            # plt.plot(pd_EV_FT[sparsity].drop(index='01-05'), color=color, label='fine-tuning', linestyle='--', marker='v')

            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('./data_analysis/statistic_fig/training_result/%s_multi-contrast-results_%s_sparsity-%s.png' % (
                training_day, sparsity, inWord))
            plt.close()
        # init training 和 fine-tuning 比较

        # pd_R2[sparsities[0]].plot.bar(color=colors[0], label='init-training')
        # pd_R2_FT[sparsities[0]].plot.bar(color=colors[0], label='init-training')
        # plt.title('First_Drawing')
        # plt.xlabel('Days')
        # plt.ylabel('Accuracy(R2)')
        # plt.ylim((0, 1))
        # plt.xticks(rotation=45)
        # plt.tight_layout()
        # plt.savefig('./data_analysis/statistic_fig/training_result/test_contra.png')


def drawCrossDayTrainingHeatmap(days, inter_or_intras, sparsities, training_day, sample_fre, kern_sd, set_num=8,
                                num_channels='5channels'):
    # days = ['01-05', '01-05-2', '01-08', '01-09', '01-09-02', '01-12', '01-12-02', '01-13', '01-14', '01-14-02']
    #
    # inter_or_intras = ['Intra_with_E1_fixed_but_changed_E2', 'Intra_with_E2_fixed_but_changed_E1',
    #                    'Inter_with_E1_I2_E2_I1_sparsity', 'Inter']
    # training_day = '03-03'
    fixed = True
    print('----------------------------------------strat---------------------------------------------------')
    for inter_or_intra in inter_or_intras:
        # sparsities = ['0', '0.5', '0.75', '1']
        print(
            '----------------------------------------%s---------------------------------------------------' % inter_or_intra)
        R2_test = np.zeros((len(days), len(days), len(sparsities)))
        EV_test = np.zeros((len(days), len(days), len(sparsities)))
        for j in range(len(sparsities)):
            sparsity = sparsities[j]
            for i in range(len(days)):
                day = days[i]
                # if fixed:
                #     num_channels = '4channels'
                # else:
                #     num_channels = jsonDataDir[day]
                i_cluster = 1 if inter_or_intra != 'Inter_with_E1_I2_E2_I1_sparsity' else 2
                filePath = r'./result_summary/random_test/EI-RNN/%s/init_training/Set%d_%s' % (
                    training_day, set_num, day) + \
                           '/_%s_%dHz_kern_sd%d/hidden_100_time_length_3_tp_10/L_factors_0.05_EI_ratio_4_l1_reg_False_' % (
                               num_channels, sample_fre, kern_sd) + \
                           'learning_rate_0.001_with_Tanh_False_sparsity_%s_e_clusters_2_i_clusters_%d_inter_or_intra_%s_l2_norm_0.001_.mat' % (
                               sparsity, i_cluster, inter_or_intra)
                scores_ = scio.loadmat(filePath)
                R2_test[i, i, j] = scores_['R2']
                EV_test[i, i, j] = scores_['EV']
                for k in range(i + 1, len(days)):
                    filePath = r'./result_summary/random_test/EI-RNN/%s/init_training/Set%d_%s' % (
                        training_day, set_num, day) + \
                               '/_%s_%dHz_kern_sd%d/hidden_100_time_length_3_tp_10/L_factors_0.05_EI_ratio_4_l1_reg_False_' % (
                                   num_channels, sample_fre, kern_sd) + \
                               'learning_rate_0.001_with_Tanh_False_sparsity_%s_e_clusters_2_i_clusters_%d_inter_or_intra_%s_l2_norm_0.001__cross_day_%s_.mat' % (
                                   sparsity, i_cluster, inter_or_intra, days[k])
                    scores_ = scio.loadmat(filePath)
                    R2_test[i, k, j] = scores_['R2']
                    EV_test[i, k, j] = scores_['EV']

        #     画热图
        cmap = 'Blues'
        center = None
        square = False
        vmin = 0.0
        vmax = .9
        annot = True
        x_ticks = days
        y_ticks = days

        # if fixed:
        #     inWord = 'fixed4Channels'
        # else:
        #     inWord = 'PSTHSelectedChannels'

        pd_R2 = {}
        pd_EV = {}
        for i in range(len(sparsities)):
            sparsity = sparsities[i]
            pd_R2[i] = ndarray2Dataframe(R2_test[:, :, i], x_ticks, y_ticks)
            pd_EV[i] = ndarray2Dataframe(EV_test[:, :, i], x_ticks, y_ticks)
            save_dir = './data_analysis/statistic_fig/training_result/%s/Long_kern_sd_%d_%dHz/' % (
                training_day, kern_sd, sample_fre,)
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            drawHeatmap(pd_R2[i],
                        name=save_dir + 'R2_set%d_0310-0319_EIRNN_cross-day_testing_sparsity_%s_%s.png' % (
                            set_num, sparsity, inter_or_intra),
                        title='R2 of Cross-day Testing\n %s Sparsity:%s' % (inter_or_intra, sparsity),
                        x_lable='Testing Day',
                        y_lable='Training Day',
                        cmap=cmap, square=square, vmin=vmin, vmax=vmax, center=center, annot=annot)
            drawHeatmap(pd_EV[i],
                        name=save_dir + 'EV_set%d_0310-0319_EIRNN_cross-day_testing_sparsity_%s_%s.png' % (
                            set_num, sparsity, inter_or_intra),
                        title='EV of Cross-day Testing\n %s Sparsity:%s' % (inter_or_intra, sparsity),
                        x_lable='Testing Day',
                        y_lable='Training Day',
                        cmap=cmap, square=square, vmin=vmin, vmax=vmax, center=center, annot=annot)
            # print(pd_EV[i])

        # print(pd_R2[0.0]) # 对应列的结果和日期
        # print(list(pd_R2))
        # sparsities = [0.0, 0.5, 0.75, 1.0]
        colors = ['red', 'yellow', 'green', 'blue', 'black']
        colors = ['#E889BD', '#67C2A3', '#FC8A61', '#8EA0C9', '#E889BD']
        linestyles = []
        markers = []

        metrics = ['EV', 'R2']
        # metric = metrics[0]

        # 折线图
        for metric in metrics:
            for i in range(len(days) - 1):
                plt.title('Cross-day Testing\n Training Day: %s' % (days[i]))
                plt.xlabel('Days')
                plt.ylabel('Accuracy(%s)' % metric)
                plt.ylim((0, 1))
                plt.xticks(rotation=45)
                for j in range(len(sparsities)):
                    sparsity = sparsities[j]
                    color = colors[j]
                    label = 'sparsity=%s' % sparsity
                    pd_input = pd_EV[j] if metric == 'EV' else pd_R2[j]
                    # print(pd_input)
                    if i > 0:
                        for k in range(0, i):
                            pd_input = pd_input.drop(columns=days[k])
                    pd_input = pd_input.drop(columns=days[-1])
                    plt.plot(pd_input.loc[days[i]], color=color, label=label, marker='.')

                plt.tight_layout()
                plt.legend()
                plt.savefig(save_dir + 'crossDay_test_allSparsities_%s_trainingDay=%s_%s.png' % (
                    inter_or_intra, days[i], metric))
                plt.close()


def testCrossDayTesting():
    days = ['01-05', '01-05-2', '01-08', '01-09', '01-09-02', '01-12', '01-12-02', '01-13', '01-14', '01-14-02']
    inter_or_intras = ['Intra_with_E1_fixed_but_changed_E2', 'Intra_with_E2_fixed_but_changed_E1',
                       'Inter_with_E1_I2_E2_I1_sparsity', 'Inter']
    training_day = '03-03'
    for i in range(len(days) - 1):
        for inter_or_intra in inter_or_intras:
            # 1月5日 PSTH选取的4channels
            num_channels = '4channels'
            jsonFilePath = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set7/%s/%s/4Hz/kern_sd_10/data.json" % (
                days[i], num_channels)
            for j in range(i + 1, len(days)):
                test_json_dir = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set7/%s/%s/4Hz/kern_sd_10/data.json" % (
                    days[j], num_channels)
                for sparsity in [0, .5, .75, 1]:
                    special_params = {
                        'L_factors': 0.05,
                        'EI_ratio': 4,
                        'l1_reg': False,
                        'learning_rate': 0.001,
                        'with_Tanh': False,
                        'sparsity': sparsity,
                        'e_clusters': 2,
                        'i_clusters': 1,
                        'inter_or_intra': 'L1_reg',
                        'l2_norm': 0.001
                    }
                    extra_params = {
                        'wrec_fix': None
                    }
                    simple_params = {
                        'json_dir': jsonFilePath,
                        'test_json_dir': test_json_dir,
                        'Tp': 10,
                        'Sample_Size': 200000,
                        'data_length': 3,  # Hyperparameter
                        'HIDDEN_DIM': 100,
                        'activation': 'ReLU',
                        'final_activation': 'Softplus',
                        'RNN_Type': 'EI-RNN',
                        'l1_reg': False,
                        'with_Tanh': False,
                        'train_day': '03-03/',
                        'init_ckpt_path': None,
                        'test_sample_size': 100,
                        'prediction_len': 500,
                        'cross_validation': False,
                        'num_folder': 5,
                        'epochs': 200,
                        'special_params': special_params,
                        'extra_params': extra_params
                    }
                    json_path = './checkpoints/random_test/EI-RNN/%s/init_training/Set7_%s/_%s_4Hz_kern_sd10/hidden_100_time_length_3_tp_10/' % (
                        training_day, days[i], num_channels) + \
                                'L_factors_0.05_EI_ratio_4_l1_reg_False_learning_rate_0.001_with_Tanh_False_sparsity_%s_e_clusters_2_i_clusters_1_inter_or_intra_%s_l2_norm_0.001.json' % (
                                    sparsity, inter_or_intra)
                    testTrainer = ModelTrainer(simple_params=json_path, json_file_type=True)
                    testTrainer.set_testPreprocessing(test_json_dir)
                    testTrainer.cross_day_testing()

            # 每天PSTH 阈值=.4 选择电极
            num_channels = jsonDataDir[days[i]]
            jsonFilePath = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set7/%s/%s/4Hz/kern_sd_10/data.json" % (
                days[i], num_channels)
            test_json_dir = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set7/%s/%s/4Hz/kern_sd_10/data.json" % (
                days[i + 1], num_channels)
            for sparsity in [0, .5, .75, 1]:
                special_params = {
                    'L_factors': 0.05,
                    'EI_ratio': 4,
                    'l1_reg': False,
                    'learning_rate': 0.001,
                    'with_Tanh': False,
                    'sparsity': sparsity,
                    'e_clusters': 2,
                    'i_clusters': 1,
                    'inter_or_intra': 'L1_reg',
                    'l2_norm': 0.001
                }
                extra_params = {
                    'wrec_fix': None
                }
                simple_params = {
                    'json_dir': jsonFilePath,
                    'test_json_dir': test_json_dir,
                    'Tp': 10,
                    'Sample_Size': 200000,
                    'data_length': 3,  # Hyperparameter
                    'HIDDEN_DIM': 100,
                    'activation': 'ReLU',
                    'final_activation': 'Softplus',
                    'RNN_Type': 'EI-RNN',
                    'l1_reg': False,
                    'with_Tanh': False,
                    'train_day': '02-24/',
                    'init_ckpt_path': None,
                    'test_sample_size': 100,
                    'prediction_len': 500,
                    'cross_validation': False,
                    'num_folder': 5,
                    'epochs': 200,
                    'special_params': special_params,
                    'extra_params': extra_params
                }
                json_path = './checkpoints/random_test/EI-RNN/%s/init_training/Set7_%s/_%s_4Hz_kern_sd10/hidden_100_time_length_3_tp_10/' % (
                    training_day, days[i], num_channels) + \
                            'L_factors_0.05_EI_ratio_4_l1_reg_False_learning_rate_0.001_with_Tanh_False_sparsity_%s_e_clusters_2_i_clusters_1_inter_or_intra_%s_l2_norm_0.001.json' % (
                                sparsity, inter_or_intra)
                testTrainer = ModelTrainer(simple_params=json_path, json_file_type=True)
                testTrainer.set_testPreprocessing(test_json_dir)
                testTrainer.cross_day_testing()


def xiemenle():
    day = '03-10'
    channels = ['5channels', '7channels', '60channels']
    for i in range(len(channels)):
        # 每天PSTH 阈值=.4 选择电极
        jsonFilePath = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set8/%s/%s/4Hz/kern_sd_10/data.json" % (
            day, channels[i])
        for sparsity in [0, .5, .75, 1]:
            special_params = {
                'L_factors': 0.05,
                'EI_ratio': 4,
                'l1_reg': False,
                'learning_rate': 0.001,
                'with_Tanh': False,
                'sparsity': sparsity,
                'e_clusters': 2,
                'i_clusters': 1,
                'inter_or_intra': 'Inter',
                'l2_norm': 0.001
            }
            extra_params = {
                'wrec_fix': None
            }
            simple_params = {
                'json_dir': jsonFilePath,
                'Tp': 10,
                'Sample_Size': 200000,
                'data_length': 3,  # Hyperparameter
                'HIDDEN_DIM': 100,
                'activation': 'ReLU',
                'final_activation': 'Softplus',
                'RNN_Type': 'EI-RNN',
                'l1_reg': False,
                'with_Tanh': False,
                'train_day': '03-11/',
                'init_ckpt_path': None,
                'test_sample_size': 100,
                'prediction_len': 500,
                'cross_validation': False,
                'num_folder': 5,
                'epochs': 200,
                'special_params': special_params,
                'extra_params': extra_params
            }
            testTrainer = ModelTrainer(simple_params=simple_params)
            testTrainer.training()


def testOptimize(ref):
    ckpt_json = '/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/checkpoints/random_test/EI-RNN/03-11/fine_tuning/Set8_03-12/_5channels_4Hz_kern_sd10/hidden_100_time_length_3_tp_10/L_factors_0.05_EI_ratio_4_l1_reg_False_learning_rate_0.001_with_Tanh_False_sparsity_0.75_e_clusters_2_i_clusters_1_inter_or_intra_Inter_l2_norm_0.001.json'
    a = ModelTrainer(simple_params=ckpt_json, json_file_type=True)
    params = {
        'N_trials': 1,
        'Control_Horizon': 10,
        'Pred_Horizon': 10,
        'iters': 1000,
        'Y_ref': np.array(ref) / 3,
        'min_input': [0, 0.3, 0, 0.3],
        'max_input': [20, 0.9, 20, 0.9],
        'minimize_hz': 4
    }
    a.generate_optimized_stim_sequence(
        data_json="/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set8/03-12/5channels/4Hz/kern_sd_10/after-5min-spon2_data.json",
        optimize_params=params
    )


def testDrawTraingHeatmap0311():
    day = '03-11'
    training_day = '03-11'
    channels = ['5channels', '7channels']  # , '60channels']
    sparsities = [0, 0.5, 0.75, 1]
    R2_init = np.zeros((len(channels), 4))
    EV_init = np.zeros((len(channels), 4))
    # R2_FT = np.zeros((len(days), 4))
    # EV_FT = np.zeros((len(days), 4))
    for i in range(len(channels)):
        num_channels = channels[i]
        for j in range(4):
            sparsity = sparsities[j]

            filePath = r'./result_summary/random_test/EI-RNN/%s/fine_tuning/Set8_%s' % (training_day, day) + \
                       '/_%s_4Hz_kern_sd10/hidden_100_time_length_3_tp_10/L_factors_0.05_EI_ratio_4_l1_reg_False_' % num_channels + \
                       'learning_rate_0.001_with_Tanh_False_sparsity_%s_e_clusters_2_i_clusters_1_inter_or_intra_Inter_l2_norm_0.001_.mat' % sparsity
            scores_ = scio.loadmat(filePath)
            R2_init[i, j] = scores_['R2']
            EV_init[i, j] = scores_['EV']

    # for i in range(1, len(days)):
    #     day = days[i]
    #     init_day = days[i-1]
    #     for j in range(4):
    #         sparsity = sparsities[j]
    #         # num_channels = jsonDataDir[init_day]
    #         num_channels = '4channels'
    #         filePath = r'./result_summary/random_test/EI-RNN/%s/fine_tuning/Set7_%s'% (training_day,day) + \
    #                    '/_%s_4Hz_kern_sd10/hidden_100_time_length_3_tp_10/L_factors_0.05_EI_ratio_4_l1_reg_False_' % num_channels + \
    #                    'learning_rate_0.001_with_Tanh_False_sparsity_%s_e_clusters_2_i_clusters_1_inter_or_intra_L1_reg_l2_norm_0.001_.mat' % sparsity
    #         scores_ = scio.loadmat(filePath)
    #         R2_FT[i, j] = scores_['R2']
    #         EV_FT[i, j] = scores_['EV']

    #     画热图
    cmap = 'Blues'
    center = None
    square = False
    vmin = 0.4
    vmax = .9
    annot = True
    x_ticks = sparsities
    y_ticks = channels

    inWord = '0311_init_training'

    print('R2 with init_training')
    pd_R2 = ndarray2Dataframe(R2_init, x_ticks, y_ticks)
    drawHeatmap(pd_R2,
                name='./data_analysis/statistic_fig/training_result/%s_R2_0310_EIRNN_%s_initTraining.png' % (
                    training_day, inWord),
                title='R2 with init_training', x_lable='Sparsity', y_lable='Channel',
                cmap=cmap, square=square, vmin=vmin, vmax=vmax, center=center, annot=annot)

    print('EV with init_training')
    pd_EV = ndarray2Dataframe(EV_init, x_ticks, y_ticks)
    drawHeatmap(pd_EV,
                name='./data_analysis/statistic_fig/training_result/%s_EV_0310_EIRNN_%s_initTraining.png' % (
                    training_day, inWord),
                title='EV with init_training', x_lable='Sparsity', y_lable='Channel',
                cmap=cmap, square=square, vmin=vmin, vmax=vmax, center=center, annot=annot)


def readSponData(day, kern_sd, optimized_round):
    channels = ['5channels']  # , '7channels', '60channels']
    for channel in channels:
        # data_json = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set8/03-10/%s/4Hz/kern_sd_10/data.json" % channel
        # a = Preprocessing(data_json, json_file_type=True)
        # print(np.sum(a.selected_channels))
        data_params = {
            'dataPath': '/mnt/database6/Organoid/科技部数据0710/课题四数据/第八批数据/',
            # 'dataPath': '/mnt/database6/Organoid/科技部数据0710/课题四数据/第七批数据/',
            'stimPath': [],
            'set': 'Set8',
            'day': day + '/',
            'type': optimized_round + '/',
            'rec_seq': ['01', '02', '03', '04', '06', '07', '08', '09', '10'],
            # , '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],
            'filename': optimized_round,  # 'optimized-round1',
            'stim_seq': ['lv2-lan1', 'lv1-lan2'],
            'sample_fre': 4,
            'rec_channels': 60,
            'stim_channels': 2,
            'kern_sd': kern_sd,
            'recording_time_len': 8,
            'ref': 'Ref',
            'stim_type': 3,
            'electrode_set': [[[6, 4], [6, 5]], [[2, 6], [2, 7]]],  # set 8
            'selected_channels': selected_channels_2c
        }
        preprocess = Preprocessing(data_params)


def OptimizeU_0313(ref):
    ref_str = ''
    for i in ref:
        ref_str += '_' + str(i)


def testReadOptimizedStimData(filename):
    # channel = '5channels'
    # data_json = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set8/03-10/%s/4Hz/kern_sd_10/data.json" % channel
    # a = Preprocessing(data_json, json_file_type=True)
    # print(np.sum(a.selected_channels))
    data_params = {
        'dataPath': 'data/optimized_stim/03-15/',
        'stimPath': [filename],
        'set': 'Set8',
        'day': '03-15/',
        'type': '',
        'rec_seq': ['01', '02', '03', '04', '06', '07', '08', '09', '10'],
        # , '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],
        'filename': 'optimized-round1-stim',
        'stim_seq': ['lv2-lan1', 'lv1-lan2'],
        'sample_fre': 4,
        'rec_channels': 60,
        'stim_channels': 2,
        'kern_sd': 5,
        'recording_time_len': 8,
        'ref': 'stim',
        'stim_type': 3,
        'electrode_set': [[[6, 4], [6, 5]], [[2, 6], [2, 7]]],  # set 8
        'selected_channels': selected_channels_2c
    }
    preprocess = Preprocessing(data_params)


def generateSet8Data():
    days = ['03-10', '03-11', '03-12', '03-13', '03-15', '03-16', '03-17', '03-19']
    rec_seq_index = {'03-10': 0,
                     '03-11': 0,
                     '03-12': 0,
                     '03-13': 0,
                     '03-15': 1,
                     '03-16': 1,
                     '03-17': 1,
                     '03-19': 2
                     }
    rec_seqs = {
        0: ['1-lv', '2-lv', '3-lv', '4-lv', '5-lv'],
        1: ['1-lv', '2-lv', '3-lv', '4-lv'],
        2: ['1-lv', '2-lv', '3-lv', '4-lv', '5-lv', '6-lv', '7-lv', '8-lv']
    }

    data_json = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set8/03-10/5channels/4Hz/kern_sd_10/data.json"
    a = Preprocessing(data_json, json_file_type=True)
    selected_channels = a.selected_channels
    kern_sds = []  # [3, 5, 10]
    # kern_sds = [10]
    sample_fres = [4]  # [4, 10, 20]

    sparsities = [0, 0.25, 0.5, 0.75, 1]
    inter_types = ['Intra_with_E2_fixed_but_changed_E1', 'Inter_with_E1_I2_E2_I1_sparsity', 'Inter']

    training_day = '03-19-runpeng'
    for sample_fre in sample_fres:
        for kern_sd_ in kern_sds:

            for i in range(len(days)):
                # read data
                day = days[i]
                rec_seq = rec_seqs[rec_seq_index[day]]
                data_params = {
                    # 'dataPath': 'Z:/Organoid/科技部数据0710/课题四数据/第七批数据/',
                    'dataPath': '/mnt/database6/Organoid/科技部数据0710/课题四数据/第八批数据/',
                    'stimPath': ['random_stim1.mat', 'random_stim2.mat'],
                    'set': 'Set8',
                    'day': day + '/',
                    'type': '系统辨识/',
                    'rec_seq': rec_seq,
                    'stim_seq': ['lv2-lan1', 'lv1-lan2'],
                    'sample_fre': sample_fre,
                    'rec_channels': 60,
                    'stim_channels': 2,
                    'kern_sd': kern_sd_,
                    'recording_time_len': 335,
                    'ref': False,
                    'stim_type': 3,
                    'electrode_set': [[[6, 4], [6, 5]], [[2, 6], [2, 7]]],  # set 7
                    'selected_channels': selected_channels
                }
                a = Preprocessing(data_params)
                # training
                for sparsity in sparsities:
                    for inter_type in inter_types:
                        special_params = {
                            'L_factors': 0.05,
                            'EI_ratio': 4,
                            'l1_reg': False,
                            'learning_rate': 0.001,
                            'with_Tanh': False,
                            'sparsity': sparsity,
                            'e_clusters': 2,
                            'i_clusters': 1,
                            'inter_or_intra': inter_type,
                            'l2_norm': 0.001
                        }
                        if inter_type == 'Inter_with_E1_I2_E2_I1_sparsity':
                            special_params['i_clusters'] = 2
                        extra_params = {
                            'wrec_fix': None
                        }
                        jsonFilePath = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set8/%s/5channels/%dHz/kern_sd_%d/data.json" % (
                            day, sample_fre, kern_sd_)
                        simple_params = {
                            'json_dir': jsonFilePath,
                            'Tp': 10,
                            'Sample_Size': 200000,
                            'data_length': 3,  # Hyperparameter
                            'HIDDEN_DIM': 100,
                            'activation': 'ReLU',
                            'final_activation': 'Softplus',
                            'RNN_Type': 'EI-RNN',
                            'l1_reg': False,
                            'with_Tanh': False,
                            'train_day': training_day + '/',
                            'init_ckpt_path': None,
                            'test_sample_size': 100,
                            'prediction_len': 500,
                            'cross_validation': False,
                            'num_folder': 5,
                            'epochs': 200,
                            'special_params': special_params,
                            'extra_params': extra_params
                        }
                        testTrainer = ModelTrainer(simple_params=simple_params)
                        testTrainer.training()


def crossDayTesting(days, inters, sparsities, training_day, kern_sd, sample_fre, num_channels='5channels'):
    '''

    :param days: 测试天数
    :param inters: EI-RNN mask类型
    :param sparsities: mask稀疏程度
    :param training_day: 模型被训练的时间
    :param kern_sd: 平滑核方差
    :param sample_fre: 采样率
    :return:
    '''

    for inter_or_intra in inters:
        i_cluster = 1 if inter_or_intra != 'Inter_with_E1_I2_E2_I1_sparsity' else 2
        for i in range(len(days) - 1):
            # jsonFilePath = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set8/%s/%s/%dHz/kern_sd_%d/data.json" % (
            #     days[i], num_channels, sample_fre, kern_sd)
            for j in range(i + 1, len(days)):
                test_json_dir = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set8/%s/%s/%dHz/kern_sd_%d/data.json" % (
                    days[j], num_channels, sample_fre, kern_sd)
                for sparsity in sparsities:
                    json_path = './checkpoints/random_test/EI-RNN/%s/init_training/Set8_%s/_%s_%dHz_kern_sd%d/hidden_100_time_length_3_tp_10/' % (
                        training_day, days[i], num_channels, sample_fre, kern_sd) + \
                                'L_factors_0.05_EI_ratio_4_l1_reg_False_learning_rate_0.001_with_Tanh_False_sparsity_%s_e_clusters_2_i_clusters_%d_inter_or_intra_%s_l2_norm_0.001.json' % (
                                    sparsity, i_cluster, inter_or_intra)
                    testTrainer = ModelTrainer(simple_params=json_path, json_file_type=True)
                    testTrainer.set_testPreprocessing(test_json_dir)
                    testTrainer.cross_day_testing()


def avg_testing(days, inters, sparsities, training_day, kern_sd, sample_fre, set_num, num_channels='5channels'):
    for inter_or_intra in inters:
        i_cluster = 1 if inter_or_intra != 'Inter_with_E1_I2_E2_I1_sparsity' else 2
        for i in range(len(days)):
            test_json_dir = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set%d/%s/%s/%dHz/kern_sd_%d/data.json" % (
                set_num, days[i], num_channels, sample_fre, kern_sd)
            for sparsity in sparsities:
                json_path = './checkpoints/random_test/EI-RNN/%s/init_training/Set%d_%s/_%s_%dHz_kern_sd%d/hidden_100_time_length_3_tp_10/' % (
                    training_day, set_num, days[i], num_channels, sample_fre, kern_sd) + \
                            'L_factors_0.05_EI_ratio_4_l1_reg_False_learning_rate_0.001_with_Tanh_False_sparsity_%s_e_clusters_2_i_clusters_%d_inter_or_intra_%s_l2_norm_0.001.json' % (
                                sparsity, i_cluster, inter_or_intra)
                testTrainer = ModelTrainer(simple_params=json_path, json_file_type=True)
                testTrainer.set_testPreprocessing(test_json_dir)
                if set_num == 7:
                    testTrainer.avg_testing(20, 800, max_y=40)
                else:
                    testTrainer.avg_testing(20, 800)
                testTrainer.input_baseline_test(20, 800, run_times=100)
                testTrainer.output_baseline_test(20, 800, run_times=100)


def dynamic_analysis_testing(days, inters, sparsities, training_day, kern_sd, sample_fre, set_num,
                             num_channels='5channels', generate_length=5000, draw_start_index=1000):
    for inter_or_intra in inters:
        i_cluster = 1 if inter_or_intra != 'Inter_with_E1_I2_E2_I1_sparsity' or set_num == 7 else 2
        for i in range(len(days)):
            test_json_dir = "/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set%d/%s/%s/%dHz/kern_sd_%d/data.json" % (
                set_num, days[i], num_channels, sample_fre, kern_sd)
            for sparsity in sparsities:
                json_path = './checkpoints/random_test/EI-RNN/%s/init_training/Set%d_%s/_%s_%dHz_kern_sd%d/hidden_100_time_length_3_tp_10/' % (
                    training_day, set_num, days[i], num_channels, sample_fre, kern_sd) + \
                            'L_factors_0.05_EI_ratio_4_l1_reg_False_learning_rate_0.001_with_Tanh_False_sparsity_%s_e_clusters_2_i_clusters_%d_inter_or_intra_%s_l2_norm_0.001.json' % (
                                sparsity, i_cluster, inter_or_intra)
                testTrainer = ModelTrainer(simple_params=json_path, json_file_type=True)
                functional_dynamics, range_functional_dynamics, original_signal = testTrainer.dynamics_analysis(
                    generate_length=generate_length)
                testTrainer.draw_PCA_FA(functional_dynamics, range_functional_dynamics, original_signal)
    # return functional_dynamics, range_functional_dynamics, load_result

    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=10, svd_solver='arpack')
    # print(functional_dynamics.shape)
    # proj_zero_inputs_signal = pca.fit_transform(functional_dynamics.T)
    # proj_with_inputs = pca.transform(range_functional_dynamics.T)
    #
    # print(pca.explained_variance_ratio_)
    # print(proj_with_inputs.shape)
    #
    # print(proj_zero_inputs_signal.shape)
    # # sample_seqs = load_result['Origin_Signal']
    # # print(sample_seqs.shape)
    # proj = pca.transform(original_signal)
    #
    # # plt.figure(figsize=(8,5))
    # # for i in range(20):
    # #     plt.plot(functional_dynamics[i,:200]+i*5)
    #
    # plt.figure(figsize=(3, 3))
    # plt.plot(np.linspace(1, 10, 10), pca.explained_variance_ratio_)
    # plt.savefig('./data_analysis/dynamics_analysis/test_dynamic_analysis_EV-ratio.png')
    # plt.close()
    #
    # fig = plt.figure(figsize=(5, 5))
    # for i in range(20):
    #     plt.plot(proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 0],
    #              proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 1])
    #
    # # for i in range(10000):
    # plt.scatter(proj[:, 0],
    #             proj[:, 1])
    #
    # plt.xlabel('$PC 1$', fontsize=15)
    # plt.ylabel('$PC 2$', fontsize=15)
    # plt.tight_layout()
    # plt.savefig('./data_analysis/dynamics_analysis/test_dynamic_analysis_PC-trajectory-1.png')
    # plt.close()
    #
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(projection='3d')
    #
    # for i in range(20):
    #     plt.plot(proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 0],
    #              proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 1],
    #              proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 2])
    #     # plt.plot(proj_zero_inputs_signal[2000:,0],
    #     #             proj_zero_inputs_signal[2000:,1],
    #     #             proj_zero_inputs_signal[2000:,2])
    #
    # # for i in range(200):
    # ax.scatter(proj[:, 0],
    #            proj[:, 1],
    #            proj[:, 2], color='grey')
    #
    # ax.set_xlabel('$PC 1$', fontsize=15)
    # ax.set_ylabel('$PC 2$', fontsize=15)
    # ax.set_zlabel('$PC 3$', fontsize=15)
    # plt.tight_layout()
    # plt.savefig('./data_analysis/dynamics_analysis/test_dynamic_analysis_PC-trajectory-2.png')
    # plt.close()
    #
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(projection='3d')
    #
    # for i in range(20):
    #     plt.plot(proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 0],
    #              proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 1],
    #              proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 2])
    #
    # # ax.scatter(proj_with_inputs[:,0],
    # #             proj_with_inputs[:,1],
    # #             proj_with_inputs[:,2],color='green')
    #
    # ax.set_xlabel('$PC 1$', fontsize=15)
    # ax.set_ylabel('$PC 2$', fontsize=15)
    # ax.set_zlabel('$PC 3$', fontsize=15)
    # plt.tight_layout()
    # plt.savefig('./data_analysis/dynamics_analysis/test_dynamic_analysis_PC-trajectory-2.png')
    # plt.close()
    #
    # fig = plt.figure(figsize=(5, 5))
    # ax = fig.add_subplot(projection='3d')
    # for i in range(20):
    #     plt.plot(proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 0],
    #              proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 1],
    #              proj_zero_inputs_signal[i * generate_length + draw_start_index:(i + 1) * generate_length, 2])
    # ax.scatter(proj_with_inputs[:, 0],
    #            proj_with_inputs[:, 1],
    #            proj_with_inputs[:, 2], color='green')
    #
    # ax.set_xlabel('$FA 1$', fontsize=15)
    # ax.set_ylabel('$FA 2$', fontsize=15)
    # ax.set_zlabel('$FA 3$', fontsize=15)
    #
    # plt.title('Functional Projection with Factor Analysis')
    # plt.show()
    # plt.savefig('./data_analysis/dynamics_analysis/test_dynamic_analysis_FA-trajectory-1.png')
    # plt.close()


def draw_set_statistic(days, inters, sparsities, training_day, kern_sd, sample_fre, set_num,
                       num_channels='5channels'):
    R2 = np.zeros((len(inters)), len(days), len(sparsities))
    EV = np.zeros((len(inters)), len(days), len(sparsities))
    for j in range(len(inters)):
        inter_or_intra = inters[j]
        i_cluster = 1 if inter_or_intra != 'Inter_with_E1_I2_E2_I1_sparsity' or set_num == 7 else 2
        for i in range(len(days)):
            for sparsity in sparsities:
                path = 'result_summary/random_test/EI-RNN/%s/' % training_day \
                       + 'init_training/Set%d_%s/_%s_%dHz_kern_sd%d/' % (set_num, days[i], num_channels, sample_fre, kern_sd) \
                       + 'hidden_100_time_length_3_tp_10/' \
                       + 'L_factors_0.05_EI_ratio_4_l1_reg_False_learning_rate_0.001_with_Tanh_False_' \
                       + 'sparsity_%s_e_clusters_2_i_clusters_%d_inter_or_intra_%s_l2_norm_0.001_.mat' % (sparsity, i_cluster, inter_or_intra)



def mainSet8():
    days = {
        'set7': ['01-05', '01-05-2', '01-08', '01-09', '01-09-02', '01-12', '01-12-02', '01-13', '01-14', '01-14-02'],
        'set8': ['03-10', '03-11', '03-12', '03-13', '03-15', '03-16', '03-17']
    }  # , '03-19']
    kern_sds = [3, 5, 10]
    sample_fres = [4, 10, 20]
    sparsities = [0, 0.5, 0.75, 1]
    inter_types = ['Intra_with_E2_fixed_but_changed_E1', 'Inter_with_E1_I2_E2_I1_sparsity', 'Inter']

    # crossDayTesting(days, inter_types, sparsities, training_day='03-19-runpeng', kern_sd=5, sample_fre=4,
    #                 num_channels='5channels')
    # drawCrossDayTrainingHeatmap(days, inter_types, sparsities, training_day='03-19-runpeng', kern_sd=3, sample_fre=4,
    #                             set_num=8, num_channels='5channels')
    # avg_testing(days['set8'], inter_types, sparsities, training_day='03-19-runpeng', kern_sd=5, sample_fre=4, set_num=8,
    #             num_channels='5channels')
    # avg_testing(['01-05'], inter_types, sparsities, training_day='03-14-zhichao', kern_sd=5, sample_fre=4, set_num=7,
    #             num_channels='4channels')
    # avg_testing(['03-10'], ['Intra_with_E2_fixed_but_changed_E1'], [0.75], training_day='03-19-runpeng', kern_sd=5,
    #             sample_fre=4, set_num=8,
    #             num_channels='5channels')
    # avg_testing(days['set7'], inter_types, sparsities, training_day='03-14-zhichao', kern_sd=5, sample_fre=4, set_num=7,
    #             num_channels='4channels')

    dynamic_analysis_testing(days['set8'], inter_types, sparsities, training_day='03-19-runpeng',
                             kern_sd=5, sample_fre=4, set_num=8, num_channels='5channels')
    # dynamic_analysis_testing(days['set7'], inter_types, sparsities, training_day='03-14-zhichao',
    #                          kern_sd=10, sample_fre=4, set_num=7, num_channels='4channels')


# def input_baseline_test():
#
# def output_baseline_test():


if __name__ == '__main__':
    mainSet8()
    # filename = '03-15-17-27-22_Sampling_4Hz_Stim_Parameter_kern_sd_5_Ref_5_5_2_2_5.mat'
    # testReadOptimizedStimData(filename)

    # testPreprocessing()
    # xiemenle()
    # readSponData()

    # generateSet8Data()

    # day='03-17'
    # kern_sd=3
    # optimized_round='optimized-round1'
    # readSponData(day,kern_sd,optimized_round)
    # # testOptimize()
    # data_json = 'data/Set8/%s/2channels/4Hz/kern_sd_%d/' %(day,kern_sd)+optimized_round+'_data.json'
    # ref = [20,6]
    # preprocess = Preprocessing(data_json)
    # preprocess.drawTrailAvg(ref=ref, plot_line=True)
    #
    #
    # ckpt_json = '/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/checkpoints/random_test/EI-RNN/03-13/fine_tuning/Set8_03-13/_5channels_4Hz_kern_sd%d/hidden_100_time_length_3_tp_10/L_factors_0.05_EI_ratio_4_l1_reg_False_learning_rate_0.001_with_Tanh_False_sparsity_0.75_e_clusters_2_i_clusters_1_inter_or_intra_Inter_l2_norm_0.001.json' % kern_sd
    # a = ModelTrainer(simple_params=ckpt_json, json_file_type=True)
    # params = {
    #     'N_trials': 1,
    #     'Control_Horizon': 10,
    #     'Pred_Horizon': 10,
    #     'iters': 1000,
    #     'Y_ref': np.array(ref) / 3,
    #     'min_input': [0, 0.3, 0, 0.3],
    #     'max_input': [20, 0.9, 20, 0.9],
    #     'minimize_hz': 4
    # }
    # a.generate_optimized_stim_sequence(
    #     data_json="/mnt/database6/Organoid/codes/baseline/reconstructed_codes2.0_230222/data/Set8/03-12/5channels/4Hz/kern_sd_10/after-5min-spon2_data.json",
    #     params=params
    # )
