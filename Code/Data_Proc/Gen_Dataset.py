import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from Gen_10Pic import create_10class_table

def split_data_with_overlap(df, window_size=512, overlap_ratio=0.5):
    """
    按照固定参数切分数据
    window_size: 样本长度 (512)
    overlap_ratio: 重叠率 (0.5)
    """
    stride = int(window_size * (1 - overlap_ratio))
    X, y = [], []
    
    # 对 10 个分类分别进行切分
    for label_idx, col in enumerate(df.columns):
        signal = df[col].values
        # 滑动窗口切分
        for i in range(0, len(signal) - window_size + 1, stride):
            X.append(signal[i : i + window_size])
            y.append(label_idx)
            
    return np.array(X), np.array(y)

def make_datasets(X, y, split_rate=[0.7, 0.2, 0.1]):
    """
    按照 7:2:1 划分训练集、验证集、测试集
    """
    total_samples = len(X)
    # 打乱顺序
    indices = np.random.permutation(total_samples)
    X, y = X[indices], y[indices]
    
    # 计算切分点
    train_end = int(total_samples * split_rate[0])
    val_end = train_end + int(total_samples * split_rate[1])
    
    train_set = (X[:train_end], y[:train_end])
    val_set = (X[train_end:val_end], y[train_end:val_end])
    test_set = (X[val_end:], y[val_end:])
    
    return train_set, val_set, test_set

if __name__ == "__main__":
    # 1. 制作 10 分类原始表
    df_raw = create_10class_table(rpm=1730)
    
    # 2. 切分样本 (步长 512，重叠 0.5)
    print("正在进行滑动窗口切分...")
    X_all, y_all = split_data_with_overlap(df_raw, window_size=512, overlap_ratio=0.5)
    print(f"切分完成，总样本数: {len(X_all)}")
    
    # 3. 划分数据集 (7:2:1)
    train, val, test = make_datasets(X_all, y_all)
    
    print("-" * 30)
    print(f"训练集规模: {train[0].shape}") # (样本数, 512)
    print(f"验证集规模: {val[0].shape}")
    print(f"测试集规模: {test[0].shape}")
    
    # 保存结果 (你可以取消注释来保存)
    # np.save('train_x.npy', train[0]); np.save('train_y.npy', train[1])