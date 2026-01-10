import os
import pandas as pd
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from Read_CWRU_mat import get_cwru_path, load_and_preprocess
from matplotlib.gridspec import GridSpec

def create_10class_table(rpm=1730):
    """
    制作 PDF 中的 10 列数据存储表
    rpm: 转速
    """
    df_10c = pd.DataFrame()
    min_len = 119808 # PDF 中统一使用的裁剪长度
    
    # 定义 10 个分类的组合逻辑
    # 结构：(列名, 故障类型, 直径, 部件)
    configs = [
        ('de_normal',   'Normal', None,    None),
        ('de_7_inner',  'DE',     '0.007', 'InnerRace'),
        ('de_7_ball',   'DE',     '0.007', 'Ball'),
        ('de_7_outer',  'DE',     '0.007', 'OuterRace6'),
        ('de_14_inner', 'DE',     '0.014', 'InnerRace'),
        ('de_14_ball',  'DE',     '0.014', 'Ball'),
        ('de_14_outer', 'DE',     '0.014', 'OuterRace6'),
        ('de_21_inner', 'DE',     '0.021', 'InnerRace'),
        ('de_21_ball',  'DE',     '0.021', 'Ball'),
        ('de_21_outer', 'DE',     '0.021', 'OuterRace6')
    ]

    print(f"--- 正在构建 {rpm} RPM 的 10 分类数据表 ---")
    for col_name, f_type, dia, comp in configs:
        path = get_cwru_path(f_type, rpm, dia, comp)
        if os.path.exists(path):
            mat = loadmat(path)
            # 找到带 _DE_time 的键
            de_key = [k for k in mat.keys() if '_DE_time' in k][0]
            # 提取并裁剪到统一长度
            data = mat[de_key].flatten()[:min_len]
            df_10c[col_name] = data
            print(f"已装载: {col_name}")
        else:
            print(f"跳过 (文件不存在): {path}")
            
    return df_10c

def plot_10_classes(df, points=1000):
    """
    自定义布局绘图：
    第一行：正常信号 (跨三列)
    第二行：0.007" (内圈, 滚珠, 外圈)
    第三行：0.014" (内圈, 滚珠, 外圈)
    第四行：0.021" (内圈, 滚珠, 外圈)
    """
    fig = plt.figure(figsize=(15, 12))
    # 创建 4行 3列 的网格
    gs = GridSpec(4, 3, figure=fig)
    
    # 1. 第一行：Normal (占据第0行的所有列)
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(df.iloc[:points, 0], color='green', linewidth=0.6)
    ax0.set_title("Normal Baseline", fontsize=12, fontweight='bold')
    ax0.set_ylim(-3, 3)
    ax0.grid(True, alpha=0.3)

    # 2. 剩下的 9 个故障信号
    # 我们从 df 的第 1 列开始取 (第 0 列是 normal)
    fault_cols = df.columns[1:]
    
    for i, col in enumerate(fault_cols):
        # 计算网格位置：从第 1 行开始，i // 3 决定行，i % 3 决定列
        row = (i // 3) + 1
        col_idx = i % 3
        
        ax = fig.add_subplot(gs[row, col_idx])
        ax.plot(df[col][:points], color='#1f77b4', linewidth=0.6)
        ax.set_title(col, fontsize=10)
        ax.set_ylim(-3, 3) # 统一坐标轴方便对比
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle("CWRU Bearing Fault Matrix Comparison", fontsize=16, y=1.02)
    plt.show()


if __name__ == "__main__":
    # 1. 制作表格
    df_final = create_10class_table(rpm=1730)
    
    # 2. 查看表格信息
    print("\n表格预览 (前5行):")
    print(df_final.head())
    print(f"表格形状: {df_final.shape}")
    
    # 3. 绘制全景对比图
    if not df_final.empty:
        plot_10_classes(df_final, points=1024)