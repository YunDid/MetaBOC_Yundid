import os
from tkinter import Y
# from scipy.io import loadmat, savemat
import pandas as pd
import scipy.io as scio
import torch
import numpy as np
from torch.utils.data import Dataset


class Input_Data(Dataset):
    def __init__(self, params):
        super().__init__()
        self.Data_Path = params['file_path']
        self.Cortical = params['Num_Cortical']
        self.tp_step = params['Tp']
        self.sample_size = params['Sample_Size']
        self.data_length = params['data_length']  # The data length for parameter estimation

        self.with_input = params['with_input']
        self.input_index = params['input_index']
        self.ext_input_dim = params['ext_input_dim']
        self.ar_order = params['AR_order']
        print(self.Data_Path)
        Seizure = np.array(scio.loadmat(self.Data_Path)['Cortex'])[:self.Cortical, :]  # Cortical*T

        # if params['data_type']=='GroundTruth':
        self.data = np.transpose(Seizure)  # T*Channels
        print(self.data.shape)

        self.max_value = max(self.data_length, self.tp_step)

        self.datalen = self.data.shape[0] - self.max_value - self.tp_step - self.ar_order - 2000
        # self.indexs=np.random.choice(self.datalen-self.max_value-self.tp_step-self.ar_order-1000,self.sample_size,replace=False)
        self.indexs = np.random.randint(500, self.datalen, self.sample_size)  # ,replace=False)

        range_start = 0
        range_end = self.datalen + range_start + self.max_value + self.tp_step + self.ar_order + 1500

        self.X = self.data[range_start:range_end, :]
        self.ext_input = np.reshape(
            np.transpose(np.array(scio.loadmat(self.Data_Path)['Inputs']))[range_start + 1:range_end + 1, :],
            (range_end, self.ext_input_dim))

        print(self.X.shape, self.ext_input.shape)

    def __init__(self, params, recording_data, stim_data):
        # initialize with input data, Channels*T
        super().__init__()
        # self.Data_Path = params['file_path']
        self.Cortical = params['Num_Cortical']  # number of recording channels
        self.tp_step = params['Tp']
        self.sample_size = params['Sample_Size']
        self.data_length = params['data_length']  # The data length for parameter estimation

        self.with_input = params['with_input']
        self.input_index = params['input_index']
        self.ext_input_dim = params['ext_input_dim'] # number of stimulating channels
        self.ar_order = params['AR_order']
        # print(self.Data_Path)
        Seizure = np.array(recording_data)[:self.Cortical, :]  # Cortical*T

        # if params['data_type']=='GroundTruth':
        self.data = np.transpose(Seizure)  # T*Channels
        print(self.data.shape)

        self.max_value = max(self.data_length, self.tp_step)

        self.datalen = self.data.shape[0] - self.max_value - self.tp_step - self.ar_order - 2000
        # self.indexs=np.random.choice(self.datalen-self.max_value-self.tp_step-self.ar_order-1000,self.sample_size,replace=False)
        self.indexs = np.random.randint(500, self.datalen, self.sample_size)  # ,replace=False)

        range_start = 0
        range_end = self.datalen + range_start + self.max_value + self.tp_step + self.ar_order + 1500

        self.X = self.data[range_start:range_end, :]
        self.ext_input = np.reshape(
            np.transpose(np.array(stim_data))[range_start + 1:range_end + 1, :],
            (range_end, self.ext_input_dim))

        print(self.X.shape, self.ext_input.shape)

    def __len__(self):
        return self.sample_size

    def __getitem__(self, index):
        """
        :X     : Batch x (Length+Tp+AR) x Dim
        :Ext_In: Batch x (Length+Tp) x Ext_Dim
        :return:
        """
        X = torch.from_numpy(
            self.X[self.indexs[index]:(self.indexs[index] + self.data_length + self.tp_step + self.ar_order), :]).to(
            torch.float32)
        ext_in = torch.from_numpy(
            self.ext_input[self.indexs[index]:(self.indexs[index] + self.data_length + self.tp_step + self.ar_order),
            :]).to(torch.float32)

        return X, ext_in, self.indexs[index]

