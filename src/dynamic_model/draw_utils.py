import matplotlib.pyplot as plt
import numpy as np



def draw_wrec_eig_dist(wrec,savepath):
    
    w, v = np.linalg.eig(wrec)

    w_real = np.real(w)
    w_imag = np.imag(w)
    
    plt.figure(figsize=(5, 3))
    
    x_ = []
    y_ = []
    for i in range(1001):
        x_.append(np.sin(i / 1000 * 2 * np.pi))
        y_.append(np.cos(i / 1000 * 2 * np.pi))
    plt.scatter(w_real, w_imag)
    plt.plot(x_, y_)
    plt.savefig(savepath, dpi=200)

def draw_low_dim_traj(proj_zeros_inputs_signal,generate_length,draw_start_index,trials,save_path,method_type='PC'):
    # import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(projection='3d')


    for i in range(trials):
        plt.plot(proj_zeros_inputs_signal[i*generate_length+draw_start_index:(i+1)*generate_length,0],
                proj_zeros_inputs_signal[i*generate_length+draw_start_index:(i+1)*generate_length,1],
                proj_zeros_inputs_signal[i*generate_length+draw_start_index:(i+1)*generate_length,2])

    # for i in range(200):
    #     ax.scatter(proj[i*model_params['Tp']:i*model_params['Tp']+1,0],
    #                 proj[i*model_params['Tp']:i*model_params['Tp']+1,1],
    #                 proj[i*model_params['Tp']:i*model_params['Tp']+1,2])
    # ax.set_xlim(-10,10)
    # ax.set_ylim(-10,10)
    # ax.set_zlim(-10,10)
    ax.set_xlabel('$%s 1$'%(method_type), fontsize=15)
    ax.set_ylabel('$%s 2$'%(method_type), fontsize=15)
    ax.set_zlabel('$%s 3$'%(method_type), fontsize=15)
    plt.tight_layout()
    plt.savefig(save_path,dpi=1000)
    
    
def draw_null_plots():
    
    return None

def draw_null_plots():
    
    return None