from os import X_OK
from tkinter.tix import Y_REGION
from turtle import forward
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule


class GRU(LightningModule):
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
        self.L_factors = 0.01
        self.ar_order = params['AR_order']
        self.ext_input_dim = params['ext_input_dim']
        self.run_device = params['device']
        # Koopman Operator
        self.K = None
        self.K_inv = None
        self.B = None
        self.K_B = None

        self.iters = 0

        self.hidden_initial_nn = nn.Sequential(
            torch.nn.Linear(self.data_len * (self.input_dim) + (self.data_len - 1) * (self.ext_input_dim),
                            self.latent_dim * 2),
            nn.ReLU(),
            # nn.Tanh(),
            torch.nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(),
            # nn.Tanh(),
        )

        self.gru = torch.nn.GRU(input_size=self.ext_input_dim, hidden_size=self.latent_dim, num_layers=1,
                                batch_first=True)

        self.decode = nn.Sequential(
            torch.nn.Linear(self.latent_dim, int(self.latent_dim / 2)),
            nn.ReLU(),
            # nn.Tanh(),
            torch.nn.Linear(int(self.latent_dim / 2), self.input_dim),
            nn.ReLU(),
        )

    def training_step(self, input):
        X, ext_in, start_index = input  # Batch x (Length+Tp+ar) x origin_dim
        # print(X.shape,ext_in.shape)
        batch, length, ori_dim = X.size()

        # Data Init Position
        init_pos = length - self.Tp - self.data_len

        X_ = torch.reshape(X[:, init_pos:init_pos + self.data_len, :], (batch, 1, self.data_len * self.input_dim))
        Ext_IN = torch.reshape(ext_in[:, :(self.data_len - 1), :],
                               (batch, 1, (self.data_len - 1) * self.ext_input_dim))  # Previous_Input
        previous_state = torch.cat((X_, Ext_IN), dim=2)
        hidden_init = torch.permute(self.hidden_initial_nn(previous_state),
                                    (1, 0, 2))  # Batch X 1 X H_dim -> 1 X Batch X H_dim
        # print(hidden_init.shape)
        Tp_prediction = torch.zeros((batch, self.Tp, self.input_dim), device=self.run_device)
        for i in range(self.Tp):
            inputs = ext_in[:, (self.data_len - 1 + i):(self.data_len + i), :]
            output, hidden_init = self.gru(inputs, hidden_init)
            Tp_prediction[:, i:i + 1, :] = self.decode(torch.permute(hidden_init, (1, 0, 2)))
        loss_fun = torch.nn.MSELoss(reduction='mean')
        # loss_X_recon=loss_fun(X_,recon_X)
        loss = loss_fun(X[:, (init_pos + self.data_len):(init_pos + self.data_len + self.Tp), :], Tp_prediction)
        return loss

    def test_step(self, input, index):
        X, ext_in, start_index = input  # Batch x (Length+Tp+ar) x origin_dim
        # print(X.shape,ext_in.shape)
        batch, length, ori_dim = X.size()

        # Data Init Position
        # Just for alignment with the Start Predict Position with with Koopman
        init_pos = length - self.Tp - self.data_len

        X_ = torch.reshape(X[:, init_pos:init_pos + self.data_len, :], (batch, 1, self.data_len * self.input_dim))

        Ext_IN = torch.reshape(ext_in[:, init_pos:(init_pos + self.data_len - 1), :],
                               (batch, 1, (self.data_len - 1) * self.ext_input_dim))  # Previous_Input
        previous_state = torch.cat((X_, Ext_IN), dim=2)
        hidden_init = torch.permute(self.hidden_initial_nn(previous_state),
                                    (1, 0, 2))  # Batch X 1 X H_dim -> 1 X Batch X H_dim
        # print(hidden_init.shape)
        Tp_prediction = torch.zeros((batch, self.Tp, self.input_dim), device=self.run_device)
        for i in range(self.Tp):
            inputs = ext_in[:, (init_pos + self.data_len - 1 + i):(init_pos + self.data_len + i), :]
            output, hidden_init = self.gru(inputs, hidden_init)
            Tp_prediction[:, i:i + 1, :] = self.decode(torch.permute(hidden_init, (1, 0, 2)))
        loss_fun = torch.nn.MSELoss(reduction='mean')
        # loss_X_recon=loss_fun(X_,recon_X)
        loss = loss_fun(X[:, init_pos + self.data_len:(init_pos + self.data_len + self.Tp), :], Tp_prediction)
        # print('loss:',loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class RNN(LightningModule):
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
        self.L_factors = 0.01
        self.ar_order = params['AR_order']
        self.ext_input_dim = params['ext_input_dim']
        self.run_device = params['device']
        # Koopman Operator
        self.K = None
        self.K_inv = None
        self.B = None
        self.K_B = None

        self.iters = 0

        self.hidden_initial_nn = nn.Sequential(
            torch.nn.Linear(self.data_len * (self.input_dim) + (self.data_len - 1) * (self.ext_input_dim),
                            self.latent_dim * 2),
            nn.ReLU(),
            torch.nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(),
        )

        self.rnn = torch.nn.RNN(input_size=self.ext_input_dim, hidden_size=self.latent_dim, num_layers=1,
                                batch_first=True)

        self.decode = nn.Sequential(
            torch.nn.Linear(self.latent_dim, int(self.latent_dim / 2)),
            nn.ReLU(),
            torch.nn.Linear(int(self.latent_dim / 2), self.input_dim),
            nn.ReLU(),
        )

    def training_step(self, input):
        X, ext_in, start_index = input  # Batch x (Length+Tp+ar) x origin_dim
        # print(X.shape,ext_in.shape)
        batch, length, ori_dim = X.size()

        # Data Init Position
        init_pos = length - self.Tp - self.data_len

        X_ = torch.reshape(X[:, init_pos:init_pos + self.data_len, :], (batch, 1, self.data_len * self.input_dim))
        Ext_IN = torch.reshape(ext_in[:, :(self.data_len - 1), :],
                               (batch, 1, (self.data_len - 1) * self.ext_input_dim))  # Previous_Input
        previous_state = torch.cat((X_, Ext_IN), dim=2)
        hidden_init = torch.permute(self.hidden_initial_nn(previous_state),
                                    (1, 0, 2))  # Batch X 1 X H_dim -> 1 X Batch X H_dim
        # print(hidden_init.shape)
        Tp_prediction = torch.zeros((batch, self.Tp, self.input_dim), device=self.run_device)
        for i in range(self.Tp):
            inputs = ext_in[:, (self.data_len - 1 + i):(self.data_len + i), :]
            # print(hidden_init.shape,inputs.shape)
            output, hidden_init = self.rnn(inputs, hidden_init)
            Tp_prediction[:, i:i + 1, :] = self.decode(torch.permute(hidden_init, (1, 0, 2)))
        loss_fun = torch.nn.MSELoss(reduction='mean')
        # loss_X_recon=loss_fun(X_,recon_X)
        loss = loss_fun(X[:, (init_pos + self.data_len):(init_pos + self.data_len + self.Tp), :], Tp_prediction)
        return loss

    def test_step(self, input, index):
        X, ext_in, start_index = input  # Batch x (Length+Tp+ar) x origin_dim
        # print(X.shape,ext_in.shape)
        batch, length, ori_dim = X.size()

        # Data Init Position
        # Just for alignment with the Start Predict Position with with Koopman
        init_pos = length - self.Tp - self.data_len

        X_ = torch.reshape(X[:, init_pos:init_pos + self.data_len, :], (batch, 1, self.data_len * self.input_dim))

        Ext_IN = torch.reshape(ext_in[:, init_pos:(init_pos + self.data_len - 1), :],
                               (batch, 1, (self.data_len - 1) * self.ext_input_dim))  # Previous_Input
        previous_state = torch.cat((X_, Ext_IN), dim=2)
        hidden_init = torch.permute(self.hidden_initial_nn(previous_state),
                                    (1, 0, 2))  # Batch X 1 X H_dim -> 1 X Batch X H_dim
        # print(hidden_init.shape)
        Tp_prediction = torch.zeros((batch, self.Tp, self.input_dim), device=self.run_device)
        for i in range(self.Tp):
            inputs = ext_in[:, (init_pos + self.data_len - 1 + i):(init_pos + self.data_len + i), :]
            output, hidden_init = self.rnn(inputs, hidden_init)
            Tp_prediction[:, i:i + 1, :] = self.decode(torch.permute(hidden_init, (1, 0, 2)))
        loss_fun = torch.nn.MSELoss(reduction='mean')
        # loss_X_recon=loss_fun(X_,recon_X)
        loss = loss_fun(X[:, init_pos + self.data_len:(init_pos + self.data_len + self.Tp), :], Tp_prediction)
        # print('loss:',loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


class LSTM(LightningModule):
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
        self.L_factors = 0.01
        self.ar_order = params['AR_order']
        self.ext_input_dim = params['ext_input_dim']
        self.run_device = params['device']
        # Koopman Operator
        self.K = None
        self.K_inv = None
        self.B = None
        self.K_B = None

        self.iters = 0

        self.hidden_initial_nn = nn.Sequential(
            torch.nn.Linear(self.data_len * (self.input_dim) + (self.data_len - 1) * (self.ext_input_dim),
                            self.latent_dim * 2),
            nn.ReLU(),
            torch.nn.Linear(self.latent_dim * 2, self.latent_dim),
            nn.ReLU(),
        )

        self.lstm = torch.nn.LSTM(input_size=self.ext_input_dim, hidden_size=self.latent_dim, num_layers=1,
                                  batch_first=True)

        self.decode = nn.Sequential(
            torch.nn.Linear(self.latent_dim, int(self.latent_dim / 2)),
            nn.ReLU(),
            torch.nn.Linear(int(self.latent_dim / 2), self.input_dim),
            nn.ReLU(),
        )

    def training_step(self, input):
        X, ext_in, start_index = input  # Batch x (Length+Tp+ar) x origin_dim
        # print(X.shape,ext_in.shape)
        batch, length, ori_dim = X.size()

        # Data Init Position
        init_pos = length - self.Tp - self.data_len

        X_ = torch.reshape(X[:, init_pos:init_pos + self.data_len, :], (batch, 1, self.data_len * self.input_dim))
        Ext_IN = torch.reshape(ext_in[:, :(self.data_len - 1), :],
                               (batch, 1, (self.data_len - 1) * self.ext_input_dim))  # Previous_Input
        previous_state = torch.cat((X_, Ext_IN), dim=2)
        hidden_init = torch.permute(self.hidden_initial_nn(previous_state),
                                    (1, 0, 2))  # Batch X 1 X H_dim -> 1 X Batch X H_dim
        # print(hidden_init.shape)
        Tp_prediction = torch.zeros((batch, self.Tp, self.input_dim), device=self.run_device)
        for i in range(self.Tp):
            inputs = ext_in[:, (self.data_len - 1 + i):(self.data_len + i), :]
            # print(hidden_init.shape,inputs.shape)
            output, (hidden_init, hidden_init) = self.lstm(inputs, (hidden_init, hidden_init))
            Tp_prediction[:, i:i + 1, :] = self.decode(torch.permute(hidden_init, (1, 0, 2)))
        loss_fun = torch.nn.MSELoss(reduction='mean')
        # loss_X_recon=loss_fun(X_,recon_X)
        loss = loss_fun(X[:, (init_pos + self.data_len):(init_pos + self.data_len + self.Tp), :], Tp_prediction)
        return loss

    def test_step(self, input, index):
        X, ext_in, start_index = input  # Batch x (Length+Tp+ar) x origin_dim
        # print(X.shape,ext_in.shape)
        batch, length, ori_dim = X.size()

        # Data Init Position
        # Just for alignment with the Start Predict Position with with Koopman
        init_pos = length - self.Tp - self.data_len

        X_ = torch.reshape(X[:, init_pos:init_pos + self.data_len, :], (batch, 1, self.data_len * self.input_dim))

        Ext_IN = torch.reshape(ext_in[:, init_pos:(init_pos + self.data_len - 1), :],
                               (batch, 1, (self.data_len - 1) * self.ext_input_dim))  # Previous_Input
        previous_state = torch.cat((X_, Ext_IN), dim=2)
        hidden_init = torch.permute(self.hidden_initial_nn(previous_state),
                                    (1, 0, 2))  # Batch X 1 X H_dim -> 1 X Batch X H_dim
        # print(hidden_init.shape)
        Tp_prediction = torch.zeros((batch, self.Tp, self.input_dim), device=self.run_device)
        for i in range(self.Tp):
            inputs = ext_in[:, (init_pos + self.data_len - 1 + i):(init_pos + self.data_len + i), :]
            output, (hidden_init, hidden_init) = self.lstm(inputs, (hidden_init, hidden_init))
            Tp_prediction[:, i:i + 1, :] = self.decode(torch.permute(hidden_init, (1, 0, 2)))
        loss_fun = torch.nn.MSELoss(reduction='mean')
        # loss_X_recon=loss_fun(X_,recon_X)
        loss = loss_fun(X[:, init_pos + self.data_len:(init_pos + self.data_len + self.Tp), :], Tp_prediction)
        # print('loss:',loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
