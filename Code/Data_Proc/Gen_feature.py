import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import stft
import Read_CWRU_mat

def sliding_window(data, window_size, step_size):
    """
    辅助函数：将长的一维信号切分为多个重叠的窗口
    
    参数:
    data: 原始一维信号
    window_size: 每个窗口的长度 (比如 2048)
    step_size: 步长 (比如 1024，表示重叠 50%)
    
    返回:
    windowed_data: 切分后的二维矩阵 (样本数 x 窗口长度)
    """
    # 确保是 numpy 数组
    data = np.array(data)
    
    # 计算能切多少段
    n_samples = data.shape[0]
    n_windows = (n_samples - window_size) // step_size + 1
    
    if n_windows <= 0:
        raise ValueError("数据太短，不够切分一个窗口")
        
    # 利用 numpy 的 stride tricks 高效切分 (不占用额外内存)
    # 或者用简单的列表推导式 (为了代码易读性，这里用简单方式)
    windows = []
    for i in range(n_windows):
        start = i * step_size
        end = start + window_size
        windows.append(data[start:end])
        
    return np.array(windows)

def genFeatureTF(data, fs, feature_names):
    """
    时域、频域相关算法的信号特征提取函数 (Python版)
    
    参数:
    data: (numpy.ndarray) 维度为 m*n，m为样本数，n为信号长度。
    fs: (int/float) 采样频率
    feature_names: (list) 包含特征名称字符串的列表
    
    返回:
    features_df: (pandas.DataFrame) 提取的特征，行对应样本，列对应特征
    """
    
    # 0. 数据格式规整：确保是 2D 数组 (Samples x TimeSteps)
    data = np.array(data)
    if data.ndim == 1:
        # 如果用户传了一维数组，我们把它变成 1行 x N列
        data = data.reshape(1, -1)
    
    m, n = data.shape
    results = {name: [] for name in feature_names}
    
    # --- 预计算：时域基础统计量 (利用 Numpy 向量化加速) ---
    # axis=1 表示沿着时间轴计算 (对每一行单独计算)
    
    # 绝对值数据
    abs_data = np.abs(data)
    # 均方根 (RMS)
    rms_val = np.sqrt(np.mean(data**2, axis=1))
    # 方差 (Variance)
    var_val = np.var(data, axis=1)
    # 峰值 (Max Abs) - 用于计算脉冲因子等
    max_abs = np.max(abs_data, axis=1)
    
    
    # --- 预计算：频域数据 (仅当需要频域特征时计算) ---
    freq_features_needed = any(f in feature_names for f in ['FC', 'MSF', 'RMSF', 'VF', 'RVF'])
    if freq_features_needed:
        # 计算 FFT 频谱 (取单边谱)
        fft_res = np.fft.rfft(data, axis=1)
        mag_spec = np.abs(fft_res) / n # 幅值谱 S(f)
        # 频率轴
        freq_axis = np.fft.rfftfreq(n, d=1/fs)
        
        # 频谱能量总和 (分母)
        sum_mag = np.sum(mag_spec, axis=1)
        # 防止除零
        sum_mag[sum_mag == 0] = 1e-9

    # --- 预计算：谱峭度 SK (仅当需要谱峭度特征时计算) ---
    sk_features_needed = any('SK' in f for f in feature_names)
    if sk_features_needed:
        # 使用 STFT 计算时频图: f(频率), t(时间), Zxx(复数结果)
        # nperseg=256 是个经验值，根据信号长度可调
        f_stft, t_stft, Zxx = stft(data, fs, nperseg=256)
        
        # 计算谱峭度 SK(f): 对于每个频率 f，计算其在时间轴 t 上的峭度
        # axis=-1 是时间轴 (Zxx 形状通常是 Samples x Freqs x Time)
        # abs(Zxx)**2 是能量谱
        spec_mag_sq = np.abs(Zxx)**2
        
        # 峭度计算：(E[x^4] / E[x^2]^2) - 2 (或者 -3，取决于定义，这里保持原始定义匹配 Matlab 习惯)
        # 这里手动计算每一行的峭度以避免维度混淆
        # SK 向量长度 = 频率点数
        
        # 简易版：直接对每个频率带的时间序列算 Kurtosis
        # 结果维度: Samples x Freqs
        sk_vectors = kurtosis(spec_mag_sq, axis=-1, fisher=False) 

    # --- 特征提取循环 ---
    
    # 1. max : 最大值
    if 'max' in feature_names:
        results['max'] = np.max(data, axis=1)
        
    # 2. min : 最小值
    if 'min' in feature_names:
        results['min'] = np.min(data, axis=1)
        
    # 3. mean : 平均值
    if 'mean' in feature_names:
        results['mean'] = np.mean(data, axis=1)
        
    # 4. peak : 峰峰值 (Max - Min)
    if 'peak' in feature_names:
        results['peak'] = np.max(data, axis=1) - np.min(data, axis=1)
        
    # 5. arv : 整流平均值 (Mean of Abs)
    if 'arv' in feature_names:
        results['arv'] = np.mean(abs_data, axis=1)
        
    # 6. var : 方差
    if 'var' in feature_names:
        results['var'] = var_val
        
    # 7. std : 标准差
    if 'std' in feature_names:
        results['std'] = np.std(data, axis=1)
        
    # 8. kurtosis : 峭度 (Fisher=False 让正态分布为3，对齐 Matlab)
    if 'kurtosis' in feature_names:
        results['kurtosis'] = kurtosis(data, axis=1, fisher=False)
        
    # 9. skewness : 偏度
    if 'skewness' in feature_names:
        results['skewness'] = skew(data, axis=1)
        
    # 10. rms : 均方根
    if 'rms' in feature_names:
        results['rms'] = rms_val
        
    # --- 复杂时域指标 ---
    
    # 11. waveformF : 波形因子 (RMS / ARV)
    if 'waveformF' in feature_names:
        denom = np.mean(abs_data, axis=1)
        denom[denom==0] = 1e-9
        results['waveformF'] = rms_val / denom
        
    # 12. peakF : 峰值因子 (MaxAbs / RMS)
    if 'peakF' in feature_names:
        denom = rms_val.copy()
        denom[denom==0] = 1e-9
        results['peakF'] = max_abs / denom
        
    # 13. impulseF : 脉冲因子 (MaxAbs / ARV)
    if 'impulseF' in feature_names:
        denom = np.mean(abs_data, axis=1)
        denom[denom==0] = 1e-9
        results['impulseF'] = max_abs / denom
        
    # 14. clearanceF : 裕度因子 (MaxAbs / (Mean(Sqrt(Abs)))^2)
    if 'clearanceF' in feature_names:
        denom = (np.mean(np.sqrt(abs_data), axis=1))**2
        denom[denom==0] = 1e-9
        results['clearanceF'] = max_abs / denom

    # --- 频域指标 ---
    
    if freq_features_needed:
        # 重心频率 FC: sum(f * S(f)) / sum(S(f))
        fc = np.sum(freq_axis * mag_spec, axis=1) / sum_mag
        
        if 'FC' in feature_names:
            results['FC'] = fc
            
        # 均方频率 MSF: sum(f^2 * S(f)) / sum(S(f))
        msf = np.sum((freq_axis**2) * mag_spec, axis=1) / sum_mag
        
        if 'MSF' in feature_names:
            results['MSF'] = msf
            
        # 均方根频率 RMSF: sqrt(MSF)
        if 'RMSF' in feature_names:
            results['RMSF'] = np.sqrt(msf)
            
        # 频率方差 VF: sum((f - FC)^2 * S(f)) / sum(S(f))
        # 需要利用广播机制计算 (f - FC)^2
        # freq_axis: (F,) -> (1, F)
        # fc: (m,) -> (m, 1)
        vf = np.sum(((freq_axis.reshape(1, -1) - fc.reshape(-1, 1))**2) * mag_spec, axis=1) / sum_mag
        
        if 'VF' in feature_names:
            results['VF'] = vf
            
        # 频率标准差 RVF: sqrt(VF)
        if 'RVF' in feature_names:
            results['RVF'] = np.sqrt(vf)
            
    # --- 谱峭度指标 ---
    
    if sk_features_needed:
        # sk_vectors 维度: Samples x FreqBins
        
        if 'SKMean' in feature_names:
            results['SKMean'] = np.mean(sk_vectors, axis=1)
            
        if 'SKStd' in feature_names:
            results['SKStd'] = np.std(sk_vectors, axis=1)
            
        if 'SKSkewness' in feature_names:
            results['SKSkewness'] = skew(sk_vectors, axis=1)
            
        if 'SKKurtosis' in feature_names:
            results['SKKurtosis'] = kurtosis(sk_vectors, axis=1, fisher=False)

    # 转换为 DataFrame 以便查看和后续处理
    return pd.DataFrame(results)

# ==========================================
# 测试 Demo：演示如何处理长数据
# ==========================================
if __name__ == "__main__":
    
    # 1. 模拟读取 CWRU 数据 (这里会读入约 12万个点的一维数据)
    # 你需要把这里换成真实的路径或者调用 Read_CWRU_mat
    raw_data = Read_CWRU_mat.load_and_preprocess(
        Read_CWRU_mat.get_cwru_path('DE', 1730, '0.007', 'Ball')
    )
    
    if raw_data is None:
        print("未找到数据，生成模拟数据代替...")
        raw_data = np.random.randn(120000) # 模拟 12万个点
        
    print(f"原始数据长度: {len(raw_data)}")
    
    # 2. 关键步骤：滑窗切片 (Sliding Window)
    # 目的：将一维长数据 -> 二维矩阵 [样本数, 窗口长度]
    # 窗口长度: 2048 (常用), 步长: 1024 (重叠 50%)
    window_len = 2048
    step = 1024
    
    sliced_data = sliding_window(raw_data, window_len, step)
    print(f"切片后数据形状: {sliced_data.shape} (行=样本数, 列=特征计算用的长度)")
    
    fs = 12000
    
    # 3. 批量特征提取
    feature_list = ['max', 'min', 'mean', 'peak', 'arv', 'var', 'std',
                    'kurtosis', 'skewness', 'rms',
                    'waveformF', 'peakF', 'impulseF', 'clearanceF',
                    'FC', 'MSF', 'RMSF', 'VF', 'RVF',
                    'SKMean', 'SKStd', 'SKSkewness', 'SKKurtosis']
    print("开始批量特征提取...")
    
    # 这里的 sliced_data 是多行的，函数会自动对每一行算出一个特征值
    df = genFeatureTF(sliced_data, fs, feature_list)
    
    print("\n特征提取结果 (前5行):")
    print(df.head())
    
    print(f"\n总共获得了 {len(df)} 组特征数据。")
    print("你可以画出 'impulseF' 的曲线来看趋势了。")