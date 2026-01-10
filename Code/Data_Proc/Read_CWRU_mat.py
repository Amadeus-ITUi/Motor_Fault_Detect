import os
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import numpy as np
from scipy.io import loadmat

def get_cwru_path(fault_type, rpm, diameter=None, component=None, base_dir='/home/angela/Motor_Fault_Detect/Rawdata/CWRU'):
    """
    自动合成 CWRU 数据集文件地址
    fault_type: 'DE' (驱动端), 'FE' (风扇端), 'Normal' (正常)
    rpm: '1730', '1750', '1772', '1797'
    diameter: '0.007', '0.014', '0.021', '0.028'
    component: 'Ball', 'InnerRace', 'OuterRace3', 'OuterRace6', 'OuterRace12'
    """
    if fault_type == 'Normal':
        return os.path.join(base_dir, "NormalBaseline", str(rpm), "Normal.mat")
    
    folder_map = {'DE': '12DriveEndFault', 'FE': '12FanEndFault'}
    folder = folder_map.get(fault_type)
    
    file_name = f"{diameter}-{component}.mat"
    
    return os.path.join(base_dir, folder, str(rpm), file_name)

def load_and_preprocess(path):
    """读取文件并提取驱动端振动数据"""
    if not os.path.exists(path):
        print(f"路径不存在: {path}")
        return None
    
    mat_data = loadmat(path)
    de_key = [k for k in mat_data.keys() if '_DE_time' in k][0]
    return mat_data[de_key].flatten()


if __name__ == "__main__":
    # 使用示例 1：获取 1730 转速下的 0.007英寸 滚珠故障数据
    path_ball = get_cwru_path('DE', 1730, '0.007', 'Ball')
    print(f"生成的故障路径: {path_ball}")
    
    # 使用示例 2：获取 1797 转速下的 正常数据
    path_normal = get_cwru_path('Normal', 1797)
    print(f"生成的正常路径: {path_normal}")

    # 实际读取测试
    data = load_and_preprocess(path_ball)
    if data is not None:
        print(f"成功读取数据，长度: {len(data)}")
        df = pd.DataFrame({'Vibration': data})
        print(df.head())