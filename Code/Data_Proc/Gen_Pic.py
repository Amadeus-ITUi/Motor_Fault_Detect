import os
import pandas as pd
import matplotlib.pyplot as plt
import pywt
import numpy as np
from scipy.io import loadmat
from Read_CWRU_mat import get_cwru_path, load_and_preprocess

def fast_plot(df, title='Vibration Signal', sample_num=2000):
    plt.figure(figsize=(12, 4))
    plt.plot(df['Vibration'][:sample_num], color='blue', linewidth=0.8) 
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amp')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    path_ball = get_cwru_path('DE', 1730, diameter=0.007, component='OuterRace3')
    print(f"生成的故障路径: {path_ball}")
    data = load_and_preprocess(path_ball)
    if data is not None:
        print(f"成功读取数据，长度: {len(data)}")
        df = pd.DataFrame({'Vibration': data})
        fast_plot(df, title='DE-1730-0.007-Ball Fault Signal', sample_num=512)