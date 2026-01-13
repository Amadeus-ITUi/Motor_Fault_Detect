import os
import pandas as pd
import numpy as np
import Gen_feature # 引用你已有的特征提取函数
import Read_CWRU_mat # 你的数据读取工具

# --- 配置参数 ---
CONFIG = {
    'window_size': 1024,
    'overlap_rate': 0.5,
    'fs': 12000,
    'output_dir': '/home/angela/Motor_Fault_Detect/Dataset/CWRU_Features',
    'feature_list': ['max', 'min', 'mean', 'peak', 'arv', 'var', 'std',
                    'kurtosis', 'skewness', 'rms', 'waveformF', 'peakF', 
                    'impulseF', 'clearanceF', 'FC', 'MSF', 'RMSF', 'VF', 
                    'RVF', 'SKMean', 'SKStd', 'SKSkewness', 'SKKurtosis']
}

# 数据集定义：这里的 key 必须和 Read_CWRU_mat.get_cwru_path 的参数名一致
# 注意：CWRU 的外圈故障通常有位置区分 (3, 6, 12)，这里默认为 6 (OuterRace6)
DATASET_MAP = {
    'Normal':    {'fault_type': 'Normal', 'rpm': 1730},
    'InnerRace': {'fault_type': 'DE',     'rpm': 1730, 'diameter': '0.007', 'component': 'InnerRace'},
    'OuterRace': {'fault_type': 'DE',     'rpm': 1730, 'diameter': '0.007', 'component': 'OuterRace6'},
    'Ball':      {'fault_type': 'DE',     'rpm': 1730, 'diameter': '0.007', 'component': 'Ball'},
}

def run_extraction():
    # 自动创建输出文件夹
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
    
    # 计算步长（重叠 50% 即步长为窗口的一半）
    step_size = int(CONFIG['window_size'] * (1 - CONFIG['overlap_rate']))
    
    for label_name, params in DATASET_MAP.items():
        print(f"\n>>> 正在处理类别: {label_name}")
        
        # 1. 动态生成路径 (使用 **params 自动匹配参数)
        try:
            path = Read_CWRU_mat.get_cwru_path(**params)
            print(f"尝试加载文件: {path}")
            
            # 2. 加载原始振动信号
            raw_data = Read_CWRU_mat.load_and_preprocess(path)
            
            if raw_data is None:
                print(f"跳过 {label_name}: 文件加载失败或路径错误。")
                continue
            
            # 3. 滑动窗口切分 (一维 -> 二维)
            sliced_data = Gen_feature.sliding_window(raw_data, CONFIG['window_size'], step_size)
            print(f"数据切分完成，样本数: {len(sliced_data)}")
            
            # 4. 提取特征 (23 种因子)
            df_features = Gen_feature.genFeatureTF(sliced_data, CONFIG['fs'], CONFIG['feature_list'])
            
            # 5. 打上标签并保存
            df_features['label'] = label_name
            save_path = os.path.join(CONFIG['output_dir'], f"{label_name}_features.csv")
            df_features.to_csv(save_path, index=False)
            print(f"特征提取成功！保存至: {save_path}")
            
        except Exception as e:
            print(f"处理 {label_name} 时发生错误: {e}")

if __name__ == "__main__":
    run_extraction()
    print("\n所有类别处理完毕！")