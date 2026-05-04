from Trainer import *
from OptimizeU import generate_stim

def testTrainer_1():
    Data_Path = './data/firing_rate_data6-1107_50Hz_smth_PSTH_5channels_newStimET.mat'
    recording_data = scio.loadmat(Data_Path)['Cortex']
    stim_data = scio.loadmat(Data_Path)['Inputs']
    print(stim_data.shape)
    model_type = 'gru'
    hidden_type = [30]
    time_length_type = [10]
    recording_channels = 5
    stim_channels = 4
    unique_name='test-1'
    sample_fre = 50
    smth = 'smth_PSTH_5channels_newStimET'
    initial_checkpoint = "./checkpoints/gru/gru_1107data6-1105-smth_PSTH_5channels_newStimET-test_hidden_30_latent_30_time_length_10_Tp_10_50Hz_smth_PSTH_5channels_newStimET.ckpt"
    processing(recording_data,stim_data,model_type,unique_name,hidden_type,sample_fre,smth,time_length_type,recording_channels,stim_channels,initial_checkpoint=initial_checkpoint)

def testTrainer_2():
    Data_Path = './data/firing_rate_data6-1107_50Hz_smth_PSTH_5channels_newStimET.mat'
    recording_data = scio.loadmat(Data_Path)['Cortex']
    stim_data = scio.loadmat(Data_Path)['Inputs']
    print(stim_data.shape)
    # 自主设定
    unique_name = 'test-2'
    model_type = 'gru'
    hidden_type = [30]
    time_length_type = [10]
    # 与数据相匹配
    recording_channels = 5
    stim_channels = 4
    sample_fre = 50
    smth = 'smth_PSTH_5channels_newStimET'
    processing(recording_data, stim_data, model_type, unique_name, hidden_type, sample_fre, smth, time_length_type,
               recording_channels, stim_channels)

def testOptimizeU_1():
    load_data_params = {
        'file_path': r'./data/firing_rate_data6-1107_50Hz_smth_PSTH_5channels_newStimET.mat',
        'Num_Cortical': 5,
        'Tp': 10,
        'Sample_Size': 90000,
        'data_length': 100,  # Hyperparameter
        'with_input': True,
        'data_type': 'GroundTruth',  # 'GroundTruth'  '1dB'  '5dB'  '10dB'  '15dB'  '20dB'
        'input_index': 'Seizure',  # Seizure  NonSeizure  Both
        'ext_input_dim': 4,  # input_index= Seizure  1; input_index= NonSeizure  1; input_index= Both  2;
        'AR_order': 3
    }
    Data_Path = load_data_params['file_path']
    recording_data = scio.loadmat(Data_Path)['Cortex']
    stim_data = scio.loadmat(Data_Path)['Inputs']
    jr_data = Input_Data(load_data_params, recording_data, stim_data)

    gru_model_params = {
        'Tp': load_data_params['Tp'],
        'IN_DIM': load_data_params['Num_Cortical'],
        'HIDDEN_DIM': 30,
        'LATENT_DIM': 30,  # Hyperparameter for the hidden dimension of GRU
        # symbol for loss selection
        'X_recon': 100,
        'Y_recon': 1,
        'Tp_recon': 1,
        'Linear_Loss': 200,
        'data_length': 10,  # Hyperparameter Previous Steps for Initializing the Hidden State
        'with_input': load_data_params['with_input'],
        'ext_input_dim': load_data_params['ext_input_dim'],
        'AR_order': 1,
        'StateDependent': False,
        'Conv': False,
        'device': 'cuda:0'
    }
    # checkpoint="/mnt/database6/Organoid/codes/baseline/gru/gru_1103data6-1103-smth_PSTH_5channels_2stimET-testX10_hidden_30_latent_30_time_length_10_Tp_10_100Hz_smth_PSTH_5channels_2stimET.ckpt"
    # checkpoint = "/mnt/database6/Organoid/codes/baseline/gru/gru_1103data6-1103_AN-smth_PSTH_5channels_2stimET-test_hidden_30_latent_30_time_length_10_Tp_10_100Hz_smth_PSTH_5channels_2stimET.ckpt"
    # checkpoint = "/mnt/database6/Organoid/codes/baseline/gru/gru_1107data6-1107-smth_PSTH_5channels_newStimET-test_hidden_30_latent_30_time_length_10_Tp_10_50Hz_smth_PSTH_5channels_newStimET.ckpt"
    checkpoint = './checkpoints/gru/gru_1107data6-1107-smth_PSTH_5channels_newStimET-test_hidden_30_latent_30_time_length_10_Tp_10_50Hz_smth_PSTH_5channels_newStimET.ckpt'
    U = generate_stim(recording_data,gru_model_params,checkpoint,1,0.2,iters=10)
    print()

if __name__ == '__main__':
    testOptimizeU_1()
