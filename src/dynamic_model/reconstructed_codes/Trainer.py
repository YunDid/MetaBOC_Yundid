from DataSample import Input_Data
# from test_common_data_for_visulation.DeepKoopman import HighOrderDeepKoopman
import torch
import os
import numpy as np
import scipy.io as scio
from datetime import datetime
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import mean_squared_error, r2_score, median_absolute_error, explained_variance_score
from Models import GRU, LSTM, RNN

# batchsize = 200
# gpu = 0
epochs = 50
# ar_type = [1, 2]
prediction_len = 1000


# tp_prediction_type = [10]

def load_gru_model(train_dataloader, model_type, gru_model_params, checkpoint, conLearning=False, epochs=50):
    model_gru = None
    if model_type == 'gru':
        model_gru = GRU(gru_model_params)
        # trainer_gru = Trainer(max_epochs=epochs, gpus=[0])
    elif model_type == 'rnn':
        model_gru = RNN(gru_model_params)
        # trainer_gru = Trainer(max_epochs=epochs, gpus=[0])
    elif model_type == 'lstm':
        model_gru = LSTM(gru_model_params)
        # trainer_gru = Trainer(max_epochs=epochs, gpus=[0])
    if os.path.exists(checkpoint):
        print("Loading checkpoint...")
        model_gru = model_gru.load_from_checkpoint(checkpoint_path=checkpoint, params=gru_model_params)
        if conLearning != False:
            trainer_gru = Trainer(max_epochs=epochs, gpus=[0])
            trainer_gru.fit(model_gru, train_dataloader)
            # trainer_gru.test(model_gru, validation_dataloader)
            trainer_gru.save_checkpoint(conLearning)
    else:
        print("Trainning...")
        trainer_gru = Trainer(max_epochs=epochs, gpus=[0])
        trainer_gru.fit(model_gru, train_dataloader)
        # trainer_gru.test(model_gru, validation_dataloader)
        trainer_gru.save_checkpoint(checkpoint)
    return gru_model_params, model_gru


def model_prediction(model, load_data_params, data, start_index, model_param, model_type='koopman'):
    # print(data.X[start_index-1-model.data_len:(start_index+prediction_len+load_data_params['Tp']),:].shape)
    # print('start from %d and end at %d'%(start_index-1-model.data_len,start_index+prediction_len+load_data_params['Tp']))
    # print('data.X.shape: %s' % str(data.X.shape))
    Data = torch.reshape(torch.from_numpy(
        data.X[start_index - 1 - model.data_len:(start_index + prediction_len + load_data_params['Tp']), :].astype(
            'float32')),
        (1, prediction_len + 1 + model.data_len + load_data_params['Tp'], load_data_params['Num_Cortical'])).type(
        torch.DoubleTensor)
    Inputs = torch.reshape(torch.from_numpy(
        data.ext_input[start_index - 1 - model.data_len:(start_index + prediction_len + load_data_params['Tp']),
        :].astype('float32')),
        (1, prediction_len + 1 + model.data_len + load_data_params['Tp'], model.ext_input_dim)).type(
        torch.DoubleTensor)

    PredictionState = np.zeros((1, prediction_len, model.input_dim), dtype='float32')
    for time in range(0, prediction_len, load_data_params['Tp']):
        # print('Start')
        pre_X = torch.reshape(Data[:, time:(model.data_len + time), :], (1, 1, model.data_len * model.input_dim))
        Ext_IN = torch.reshape(Inputs[:, time:(time + model.data_len - 1), :],
                               (1, 1, (model.data_len - 1) * model.ext_input_dim))  # Previous_Input
        previous_state = torch.cat((pre_X, Ext_IN), dim=2)
        hidden_init = torch.permute(model.hidden_initial_nn(previous_state.type(torch.float32).to('cuda:0')),
                                    (1, 0, 2))  # Batch X 1 X H_dim -> 1 X Batch X H_dim

        for i in range(load_data_params['Tp']):
            inputs = Inputs[:, (time + model.data_len - 1 + i):(time + model.data_len + i), :]
            if model_type == "gru":
                output, hidden_init = model.gru(inputs.type(torch.float32).to('cuda:0'),
                                                hidden_init.type(torch.float32).to('cuda:0'))
            elif model_type == "rnn":
                output, hidden_init = model.rnn(inputs.type(torch.float32).to('cuda:0'),
                                                hidden_init.type(torch.float32).to('cuda:0'))
            elif model_type == "lstm":
                output, (hidden_init, hidden_init) = model.lstm(inputs.type(torch.float32).to('cuda:0'), (
                    hidden_init.type(torch.float32).to('cuda:0'), hidden_init.type(torch.float32).to('cuda:0')))

            PredictionState[:, time + i:time + i + 1, :] = model.decode(
                torch.permute(hidden_init.type(torch.float32).to('cuda:0'), (1, 0, 2))).to('cpu').detach().numpy()

    return PredictionState


def data_partition(input_data, batchsize=200, training_ratio=0.8):
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
    train_indices, val_indices = indices[split:], indices[:split]
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_dataloader = torch.utils.data.DataLoader(input_data, batch_size=batchsize,
                                                   sampler=train_sampler, num_workers=4)
    validation_dataloader = torch.utils.data.DataLoader(input_data, batch_size=batchsize,
                                                        sampler=valid_sampler)

    X, ext_in, start_indexs = next(iter(validation_dataloader))
    return train_dataloader, start_indexs


def load_model(train_dataloader, model_type, dateNum, unique_name, time_length, hidden, latent, ar, sample_fre, smth,
               load_data_params,
               modelDirPath, initial_checkpoint):
    gru_model_params = {
        'Tp': load_data_params['Tp'],
        'IN_DIM': load_data_params['Num_Cortical'],
        'HIDDEN_DIM': hidden,
        'LATENT_DIM': latent,  # Hyperparameter for the hidden dimension of GRU
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
        'device': 'cuda:0'
    }

    checkpoint = "%s/%s/%s_%s_hidden_" % (modelDirPath, model_type, dateNum, unique_name) + str(
        gru_model_params['HIDDEN_DIM']) + "_latent_" + str(gru_model_params['LATENT_DIM']) + "_time_length_" + str(
        gru_model_params['data_length']) + "_Tp_" + str(gru_model_params['Tp']) + "_%dHz_%s" % (
                     sample_fre, smth) + ".ckpt"
    if initial_checkpoint != None:
        gru_model_params, model_gru = load_gru_model(train_dataloader, model_type, gru_model_params,
                                                     initial_checkpoint, checkpoint)
    else:
        gru_model_params, model_gru = load_gru_model(train_dataloader, model_type, gru_model_params,
                                                     checkpoint)
    return model_gru


def virtualize_test_result(GroundTruth, pred_gru, run_index, recording_channels, model_type, latent, ar, time_length,
                           resultDirPath, dataNum, load_data_params, ratio=10):
    fs = 100
    fs_legend = 12
    if run_index % ratio == 0:
        plt.figure(figsize=(500, 500))

        for i in range(recording_channels):
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
            plt.yticks([0, .2, .4, .6, .8, 1], fontsize=fs)
            plt.ylabel('Channel %d ' % i, fontname='Arial', loc='center', fontsize=fs)
            # plt.xlabel('Time Steps',fontname='Arial',loc='center',fontsize=15)
            ax.legend(labels=['GT', '%s_Pred' % model_type], loc='upper center', ncol=3,
                      fancybox=True,
                      shadow=True, fontsize=fs)

        save_fig = '%s/%s/%s/Latent_' % (resultDirPath, model_type, dataNum) + str(
            latent) + '_AR_' + str(ar) + '_Time_Length_' + str(
            time_length) + '_Tp_Length_' + str(load_data_params['Tp']) + '_Test_' + str(
            run_index + 1) + '.pdf'
        plt.savefig(save_fig, dpi=400)
        plt.close()


def processing(recording_data, stim_data, model_type, unique_name, hidden_type, sample_fre, smth, time_length_type,
               recording_channels, stim_channels,
               data_Sample_Size=90000, test_sample_size=100, tp_prediction_type=[10], resultDirPath='./result/',
               modelDirPath='./checkpoints/', saveTestFig=True, save_prediction=False, initial_checkpoint=None):
    dateNum = datetime.today().date()  # 跨天训练可能会保存到两个文件夹
    if (not os.path.isdir('%s/%s/%s' % (resultDirPath, model_type, dateNum))):
        os.makedirs('%s/%s/%s' % (resultDirPath, model_type, dateNum))
    if (not os.path.isdir('%s/%s' % (modelDirPath, model_type))):
        os.makedirs('%s/%s' % (modelDirPath, model_type))

    load_data_params = {
        'Num_Cortical': recording_channels,
        'Tp': 30,
        'Sample_Size': data_Sample_Size,
        'data_length': 100,  # Hyperparameter
        'with_input': True,
        'data_type': 'GroundTruth',  # 'GroundTruth'  '1dB'  '5dB'  '10dB'  '15dB'  '20dB'
        'input_index': 'Seizure',  # Seizure  NonSeizure  Both
        'ext_input_dim': stim_channels,  # input_index= Seizure  1; input_index= NonSeizure  1; input_index= Both  2;
        'AR_order': 3
    }
    input_data = Input_Data(load_data_params, recording_data, stim_data)
    train_dataloader, start_indexs = data_partition(input_data)

    ############################
    # set MSE....
    model_types = 1
    ar_types = 1
    hidden_types = len(hidden_type)
    length_types = len(time_length_type)

    R2_record = np.zeros((11, hidden_types, length_types, ar_types, model_types, test_sample_size))
    EV_record = np.zeros((11, hidden_types, length_types, ar_types, model_types, test_sample_size))
    MeAE_record = np.zeros((11, hidden_types, length_types, ar_types, model_types, test_sample_size))
    mse_record = np.zeros((11, hidden_types, length_types, ar_types, model_types, test_sample_size))

    for tp_index in range(len(tp_prediction_type)):
        load_data_params['Tp'] = tp_prediction_type[tp_index]
        for hidden_index in range(len(hidden_type)):
            hidden = hidden_type[hidden_index]
            latent = hidden
            ar_index = 0
            ar = 1
            for time_length_index in range(len(time_length_type)):
                time_length = time_length_type[time_length_index]
                result_save_path = "%s/%s/%s/Time_length_" % (resultDirPath, model_type, dateNum) + str(
                    time_length) + "_Latent_" + str(hidden) + "_Tp_" + str(load_data_params['Tp']) + ".mat"
                gru_model_params = {
                    'Tp': load_data_params['Tp'],
                    'IN_DIM': load_data_params['Num_Cortical'],
                    'HIDDEN_DIM': hidden,
                    'LATENT_DIM': latent,  # Hyperparameter for the hidden dimension of GRU
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
                    'device': 'cuda:0'
                }

                checkpoint = "%s/%s/%s_%s_hidden_" % (modelDirPath, model_type, dateNum, unique_name) + str(
                    gru_model_params['HIDDEN_DIM']) + "_latent_" + str(
                    gru_model_params['LATENT_DIM']) + "_time_length_" + str(
                    gru_model_params['data_length']) + "_Tp_" + str(gru_model_params['Tp']) + "_%dHz_%s" % (
                                 sample_fre, smth) + ".ckpt"
                if initial_checkpoint != None:
                    gru_model_params, model_gru = load_gru_model(train_dataloader, model_type, gru_model_params,
                                                                 initial_checkpoint, checkpoint)
                else:
                    gru_model_params, model_gru = load_gru_model(train_dataloader, model_type, gru_model_params,
                                                                 checkpoint)

                prediction_result = np.zeros([test_sample_size, prediction_len, recording_channels])
                for run_index in range(test_sample_size):
                    # print('run_index:',run_index)
                    start_index = start_indexs[run_index]

                    GroundTruth = input_data.X[start_index:(start_index + prediction_len), :]

                    pred_gru = model_prediction(model_gru.to('cuda:0'), load_data_params, input_data, start_index,
                                                model_param=None,
                                                model_type=model_type)
                    prediction_result[run_index, :, :] = pred_gru[0, :, :]
                    if saveTestFig:
                        virtualize_test_result(GroundTruth, pred_gru, run_index, recording_channels, model_type, latent,
                                               ar, time_length, resultDirPath, dateNum, load_data_params)
                        # hidden_types,length_types,ar_types,model_types,sample_size
                    R2_record[tp_index, hidden_index, time_length_index, ar_index, 0, run_index] = r2_score(
                        GroundTruth[:, :], pred_gru[0, :, :])
                    mse_record[
                        tp_index, hidden_index, time_length_index, ar_index, 0, run_index] = mean_squared_error(
                        GroundTruth[:, :], pred_gru[0, :, :])
                    MeAE_record[
                        tp_index, hidden_index, time_length_index, ar_index, 0, run_index] = median_absolute_error(
                        GroundTruth[:, :], pred_gru[0, :, :])
                    EV_record[
                        tp_index, hidden_index, time_length_index, ar_index, 0, run_index] = explained_variance_score(
                        GroundTruth[:, :], pred_gru[0, :, :])

                print("hidden_size:", hidden, "_Time_length:", time_length, "_AR:", ar, "Predict_length:",
                      load_data_params['Tp'])
                print("R2:",
                      np.mean(R2_record[tp_index, hidden_index, time_length_index, ar_index, 0, :]), "_std:",
                      np.std(R2_record[tp_index, hidden_index, time_length_index, ar_index, 0, :]), "\n")
                print("MSE:",
                      np.mean(mse_record[tp_index, hidden_index, time_length_index, ar_index, 0, :]), "_std:",
                      np.std(mse_record[tp_index, hidden_index, time_length_index, ar_index, 0, :]), "\n")  # ,
                print("MeAE:",
                      np.mean(MeAE_record[tp_index, hidden_index, time_length_index, ar_index, 0, :]), "_std:",
                      np.std(MeAE_record[tp_index, hidden_index, time_length_index, ar_index, 0, :]), "\n")  # ,
                print("EV:",
                      np.mean(EV_record[tp_index, hidden_index, time_length_index, ar_index, 0, :]), "_std:",
                      np.std(EV_record[tp_index, hidden_index, time_length_index, ar_index, 0, :]), "\n")

                if save_prediction:
                    scio.savemat(result_save_path,
                                 {"R2": R2_record[tp_index, hidden_index, time_length_index, ar_index, 0, :],
                                  "MSE": mse_record[tp_index, hidden_index, time_length_index, ar_index, 0, :],
                                  "MeAE": MeAE_record[tp_index, hidden_index, time_length_index, ar_index, 0,
                                          :],
                                  "EV": EV_record[tp_index, hidden_index, time_length_index, ar_index, 0, :],
                                  "Prediction": prediction_result})
                else:
                    scio.savemat(result_save_path,
                                 {"R2": R2_record[tp_index, hidden_index, time_length_index, ar_index, 0, :],
                                  "MSE": mse_record[tp_index, hidden_index, time_length_index, ar_index, 0, :],
                                  "MeAE": MeAE_record[tp_index, hidden_index, time_length_index, ar_index, 0,
                                          :],
                                  "EV": EV_record[tp_index, hidden_index, time_length_index, ar_index, 0, :]}
                                 )
