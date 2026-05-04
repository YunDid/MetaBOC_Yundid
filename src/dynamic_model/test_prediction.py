# import matplotlib.pyplot as plt
# import numpy as np
#
# from Trainer import *
# from OptimizeU import generate_stim
import matplotlib.pyplot as plt

from DataPreprocessing import *

def generate_recording_data_fig():
    #

    # 第六批数据
    data_path = 'Z:/Organoid/科技部数据0710/课题四数据/第七批数据/'
    data_path_stim = ['', '']
    stims = ['random_stim1.mat', 'random_stim2.mat']
    days = ['12-26/','12-27/','12-28/','12-29/','12-31/','01-05/','01-05-2/','01-08/','01-09/','01-09-02/','01-12/','01-12-02/','01-13/','01-13-02/','01-14/','01-14-02/']
    types = ['图像/', '系统辨识/']
    rec_seq_1 = ['1-lv', '2-lv', '3-lv', '4-lv', '5-lv']#, '6-lv', '7-lv', '8-lv', '9-lv', '10-lv']
    # rec_seq_1=['1-trail']#,'2-trail','3-trail','4-trail','5-trail','6-trail','7-trail','8-trail','9-trail','10-trail']#系统辨识序列
    # rec_seq_0=['trail1-','trail2-','trail3-','trail4-','trail5-','trail6-','trail7-','trail8-','trail9-','trail10-','trail11-','trail12-','trail13-','trail14-','trail15-','trail16-','trail17-','trail18-','trail19-','trail20-']#图像序列
    stim_seq_1 = ['lan1-lv2', 'lan2-lv1']  # 系统辨识刺激类型（用于10-2及之后）
    # stim_seq_0=['1X','1L','2X','2L']
    seq_1 = [[[2, 6], [2, 7]], [[7, 2], [7, 3]]]  # sitimulus electrodes sets * nodes num * 2

    # 设置需要的seq，type,时常和采样率
    seq = rec_seq_1
    type = types[1]
    time_len = 355  # 记录总时间 s
    sample_freq = 50  # 采样频率 Hz
    # for day in days:
    stim_spike_count = np.zeros(
        [len(days), len(seq), 60, time_len * sample_freq])  # 80 seconds with 100Hz sampling frequency
    for i in range(len(days)):

        day = days[i]
        #     print(day)

        for j in range(len(seq)):
            stim_ind = seq[j]

            # 找到包含stim_ind的记录文件
            for filewalks in os.walk(data_path + day + type):
                #             print(filewalks)
                for files in filewalks[2]:
                    if stim_ind in files and stim_ind + "0" not in files:
                        file = os.path.join(filewalks[0], files)
            #                     print(stim_ind,' is in',file)

            # 读取单个记录文件
            print(file)
            filetype = 'h5'
            try:
                spikes_sequence=h5py.File(file)
            except:
                spikes_sequence = scio.loadmat(file)
                filetype = 'mat'
            electrode = -1
            for row in range(1, 9):
                for col in range(1, 9):
                    str_read_key = 'AnSt_Label_E_00159_' + str(row) + str(col) + '_ID_'
                    key_name = [k for k, v in spikes_sequence.items() if str_read_key in k]
                    #                 print(key_name)
                    #                 for k,v in spikes_sequence.items():
                    #                     print(k)

                    # 当对应电极存在神经响应记录时，滑动窗计算spike count
                    if len(key_name) > 0:
                        #                     print(spikes_sequence[key_name[0]].shape)
                        if filetype=='h5':
                            spike_data_=np.array(spikes_sequence[key_name[0]][0,:])#h5 file
                        elif filetype=='mat':
                            spike_data_ = np.array(spikes_sequence[key_name[0]][:, 0])  # loadmat type
                        else:
                            print('No Data!')
                        # print(spike_data_)
                        electrode += 1

                        for stim_time_index in range(time_len * sample_freq):
                            stim_time_cur = stim_time_index / sample_freq
                            intervel = 1 / sample_freq
                            stim_time_lower = stim_time_cur - intervel / 2
                            stim_time_upper = stim_time_cur + intervel / 2
                            stim_spike_count[i, j, electrode, stim_time_index] = np.sum(
                                np.all([spike_data_ > stim_time_lower, spike_data_ < stim_time_upper],
                                       axis=0) == True)
                    #                         data_pos+=1

                    # 当没有记录时，排除四角电极（11，18，81，88），其余填空
                    else:
                        if (row == 1 or row == 8) and (col == 1 or col == 8):
                            continue
                        else:
                            electrode += 1
        plt.figure(figsize=(50, 120), dpi=80)
        for j in range(60):
            plt.subplot(60, 1, j + 1)
            plt.ylim((0, 20))
            plt.plot(np.linspace(0.0, time_len, time_len * sample_freq), stim_spike_count[0, 0, j],label='channel%d'%(j+1))
            #     plt.plot(range(800),firing_rate[j,0],label='seq_'+str(1))
            #     plt.plot(range(800),firing_rate[j,3],label='seq_'+str(4))
            plt.legend(loc=2)

        if (not os.path.isdir('fig/%s'%day)):
            os.makedirs('fig/%s'%day)
        plt.savefig('fig/%s/rec_data.png'%day)
        plt.show()

        # resize data
        # firing_rate_per_channel = stim_spike_count * sample_freq
        firing_rate_per_channel = stim_spike_count * sample_freq
        print(firing_rate_per_channel.shape)
        resized_stim_spike_count = resizeData(firing_rate_per_channel)
        print(resized_stim_spike_count.shape)

        import scipy.signal as signal
        kern_sd_ms = 10
        kern_sd = int(round(kern_sd_ms / 10))
        window = signal.gaussian(kern_sd * 6, kern_sd, sym=True)
        window /= np.sum(window)
        filt = lambda x: np.convolve(x, window, 'same')
        # resized_stim_inputs_smth = np.apply_along_axis(filt, 1, resized_stim_inputs)
        resized_stim_spike_count_smth = np.apply_along_axis(filt, 1, resized_stim_spike_count)

        # 作图
        plt.figure(figsize=(50, 50), dpi=80)
        index = 0
        for j in range(60):
            if np.mean(resized_stim_spike_count_smth[j]) > 1 and np.max(resized_stim_spike_count_smth[j]) < 50:
                plt.subplot(60, 1, index + 1)
                plt.ylim((0, 50))
                plt.axis('off')
                plt.plot(np.linspace(0.0, time_len / 2, time_len * sample_freq // 2),
                         resized_stim_spike_count_smth[j, :8875], linewidth=10.0, color='k',label='channel%d'%(j+1))
                index += 1
            #     plt.plot(range(800),firing_rate[j,0],label='seq_'+str(1))
            #     plt.plot(range(800),firing_rate[j,3],label='seq_'+str(4))
            plt.legend(loc=2)
        plt.savefig('fig/%s/smth_recData.png'%day)
        plt.show()

if __name__ == '__main__':
    generate_recording_data_fig()