import h5py
import scipy.io as sio
import os

def convert_h5_to_mat(h5_path, mat_path):
    # 指定文件路径
    h5_path = h5_path
    mat_path = mat_path

    # 读取H5文件并转换为MAT
    with h5py.File(h5_path, 'r') as f:
        # 创建字典存储所有数据
        mat_data = {}
        
        for key in f.keys():
            mat_data[key] = f[key][...]  # 读取数据
            # print(f"转换: {key}, shape: {f[key].shape}")
        
        # 方法1：尝试用hdf5storage
        try:
            import hdf5storage
            if os.path.exists(mat_path):
                os.remove(mat_path)  # 删除已存在的文件
            hdf5storage.savemat(mat_path, mat_data, format='7.3')
            print(f"已保存到: {mat_path} (hdf5storage)")
        except ImportError:
            print("hdf5storage未安装，使用scipy...")
            # 方法2：用scipy的v7.3格式
            try:
                sio.savemat(mat_path, mat_data, format='7.3', 
                            =True)
                print(f"已保存到: {mat_path} (scipy v7.3)")
            except:
                print("文件太大，分块保存...")
                # 方法3：分块保存
                for key, value in mat_data.items():
                    chunk_path = mat_path.replace('.mat', f'_{key}.mat')
                    sio.savemat(chunk_path, {key: value}, do_compression=True)
                    print(f"已保存: {chunk_path}")

# 批处理逻辑
base_dir = r"K:\AI-train-summary\Sa"
for date_folder in os.listdir(base_dir):
    date_path = os.path.join(base_dir, date_folder)
    if os.path.isdir(date_path):
        raw_path = os.path.join(date_path, "raw")
        if os.path.isdir(raw_path):
            for h5_file in os.listdir(raw_path):
                if h5_file.endswith(".h5"):
                    h5_full_path = os.path.join(raw_path, h5_file)
                    mat_file = os.path.splitext(h5_file)[0] + ".mat"
                    mat_full_path = os.path.join(raw_path, mat_file)
                    convert_h5_to_mat(h5_full_path, mat_full_path)