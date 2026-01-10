import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_se_csv(path,skiprows=16):
    """读取SE数据集的CSV文件"""
    df = pd.read_csv(path, skiprows=skiprows, header=None, sep='\t')
    df.columns = [
        'motor', #电机振动
        'pt_x', 'pt_y', 'pt_z', #行星齿轮箱
        'torque', #扭矩
        'pa_x', 'pa_y', 'pa_z', #并行减速齿轮箱
        '_'
    ]

    return df

def quick_analysis(df, channel='motor'):
    sample_data = df[channel].values[:4096]

    plt.figure(figsize=(10, 4))
    plt.plot(sample_data, color='blue', linewidth=0.7)
    plt.title(f'Sample Vibration Data - {channel}')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    file_path = "/home/angela/Motor_Fault_Detect/Rawdata/Southeast/bearingset/outer_30_2.csv"

    df = load_se_csv(file_path)
    print(df.head())
    quick_analysis(df, channel='motor')