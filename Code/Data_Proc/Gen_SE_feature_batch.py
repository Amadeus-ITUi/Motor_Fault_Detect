import os
import pandas as pd
import numpy as np
import Gen_feature  # 引用你已有的 23 种特征提取函数
import Read_SE_csv  # 引用你上传的东南大学 CSV 读取工具

# --- 配置参数 ---
CONFIG = {
    'input_dir': '/home/angela/Motor_Fault_Detect/Rawdata/Southeast/bearingset', # 输入 CSV 文件夹
    'output_dir': '/home/angela/Motor_Fault_Detect/Dataset/SE_Features',        # 结果保存文件夹
    'window_size': 1024,
    'overlap_rate': 0.5,
    'fs': 5120,  # 东南大学数据集采样率通常为 5.12kHz
    'feature_list': ['max', 'min', 'mean', 'peak', 'arv', 'var', 'std',
                    'kurtosis', 'skewness', 'rms', 'waveformF', 'peakF', 
                    'impulseF', 'clearanceF', 'FC', 'MSF', 'RMSF', 'VF', 
                    'RVF', 'SKMean', 'SKStd', 'SKSkewness', 'SKKurtosis']
}

# 需要分析的振动通道列表
VIB_CHANNELS = ['motor', 'pt_x', 'pt_y', 'pt_z', 'pa_x', 'pa_y', 'pa_z']

def get_label_from_filename(filename):
    """根据 SE 数据集文件名关键词映射标签"""
    fn = filename.lower()
    if 'health' in fn:
        return 'Normal'
    elif 'inner' in fn:
        return 'InnerRace'
    elif 'outer' in fn:
        return 'OuterRace'
    elif 'ball' in fn:
        return 'Ball'
    elif 'comb' in fn:
        return 'Combined'
    else:
        return 'Unknown'

def run_batch_extraction():
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
    
    csv_files = [f for f in os.listdir(CONFIG['input_dir']) if f.endswith('.csv')]
    if not csv_files:
        print("错误：未找到 CSV 文件。")
        return

    step_size = int(CONFIG['window_size'] * (1 - CONFIG['overlap_rate']))
    
    for file_name in csv_files:
        file_path = os.path.join(CONFIG['input_dir'], file_name)
        base_name = os.path.splitext(file_name)[0]
        
        # --- 核心修复：根据文件名确定标签 ---
        current_label = get_label_from_filename(base_name)
        
        print(f"\n>>> 处理文件: {file_name} | 识别标签: {current_label}")
        
        try:
            df_raw = Read_SE_csv.load_se_csv(file_path)
            for channel in VIB_CHANNELS:
                if channel not in df_raw.columns:
                    continue
                
                signal = df_raw[channel].values
                sliced_data = Gen_feature.sliding_window(signal, CONFIG['window_size'], step_size)
                
                df_features = Gen_feature.genFeatureTF(sliced_data, CONFIG['fs'], CONFIG['feature_list'])
                
                # 保存追溯信息
                df_features['source_file'] = base_name
                df_features['channel'] = channel
                # --- 核心修复：添加 label 列，让 Train_MLP.py 能够识别 ---
                df_features['label'] = current_label
                
                save_name = f"{base_name}_{channel}.csv"
                save_path = os.path.join(CONFIG['output_dir'], save_name)
                df_features.to_csv(save_path, index=False)
                
        except Exception as e:
            print(f"处理 {file_name} 出错: {e}")

if __name__ == "__main__":
    run_batch_extraction()
    print("\n[完成] 特征已提取完毕，且已添加 'label' 列。")