import os
import numpy as np
import pandas as pd
import pywt
import matplotlib
# 必须在导入 pyplot 之前指定非交互式后端 Agg，防止 tkinter 报错
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
from tqdm import tqdm


class CWRUDatasetGenerator:
    """CWRU轴承故障数据集生成器"""
    
    def __init__(self, 
                 base_dir=r"/home/angela/Motor_Fault_Detect/Rawdata/CWRU",
                 save_dir=r"/home/angela/Motor_Fault_Detect/Dataset/CWT_Dataset(sample=all)",
                 rpm=1730,
                 window_size=512,
                 overlap_ratio=0.5,
                 wavename='cmor1-1',
                 totalscale=128,
                 sampling_rate=12000):
        """
        初始化CWRU数据集生成器
        
        Args:
            base_dir: CWRU原始数据目录
            save_dir: 生成数据集保存目录
            rpm: 电机转速
            window_size: 窗口大小
            overlap_ratio: 重叠率
            wavename: 小波基名称
            totalscale: 小波尺度数
            sampling_rate: 采样率
        """
        self.base_dir = base_dir
        self.save_dir = save_dir
        self.rpm = rpm
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.wavename = wavename
        self.totalscale = totalscale
        self.sampling_rate = sampling_rate
        
        # 故障类型配置: (类别名, 故障类型, 故障直径, 故障部件)
        self.configs = [
            ('normal',   'Normal', None,    None),
            ('7_inner',  'DE',     '0.007', 'InnerRace'),
            ('7_ball',   'DE',     '0.007', 'Ball'),
            ('7_outer',  'DE',     '0.007', 'OuterRace6'),
            ('14_inner', 'DE',     '0.014', 'InnerRace'),
            ('14_ball',  'DE',     '0.014', 'Ball'),
            ('14_outer', 'DE',     '0.014', 'OuterRace6'),
            ('21_inner', 'DE',     '0.021', 'InnerRace'),
            ('21_ball',  'DE',     '0.021', 'Ball'),
            ('21_outer', 'DE',     '0.021', 'OuterRace6')
        ]
    
    def get_mat_path(self, f_type, dia=None, comp=None):
        """获取CWRU数据文件路径"""
        if f_type == 'Normal':
            return os.path.join(self.base_dir, "NormalBaseline", str(self.rpm), "Normal.mat")
        folder = '12DriveEndFault' if f_type == 'DE' else '12FanEndFault'
        return os.path.join(self.base_dir, folder, str(self.rpm), f"{dia}-{comp}.mat")
    
    def save_cwt_image(self, signal, save_path):
        """将1D信号转为CWT图像并保存"""
        sampling_period = 1.0 / self.sampling_rate
        fc = pywt.central_frequency(self.wavename)
        cparam = 2 * fc * self.totalscale
        scales = cparam / np.arange(self.totalscale, 0, -1)
        
        coef, freqs = pywt.cwt(signal, scales, self.wavename, sampling_period)
        amp = np.abs(coef)
        
        # 绘图并保存（Agg 后端渲染，不会弹出窗口，不会报错）
        plt.figure(figsize=(2.56, 2.56), dpi=100) 
        plt.axes([0, 0, 1, 1]) 
        plt.axis('off')
        plt.contourf(amp, cmap='jet')
        plt.savefig(save_path)
        plt.close('all')  # 显式关闭所有画布，释放内存
    
    def build_dataset(self, max_samples_per_class=None):
        """
        生成CWRU数据集
        
        Args:
            max_samples_per_class: 每类最大样本数，None表示使用所有样本
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        label_records = []
        
        for label, (name, f_type, dia, comp) in enumerate(self.configs):
            print(f"正在处理类别: {name}...")
            path = self.get_mat_path(f_type, dia, comp)
            
            if not os.path.exists(path):
                print(f"找不到文件: {path}")
                continue
            
            mat = loadmat(path)
            key = [k for k in mat.keys() if '_DE_time' in k][0]
            full_signal = mat[key].flatten()
            
            stride = int(self.window_size * (1 - self.overlap_ratio))
            samples = [full_signal[i:i+self.window_size] 
                      for i in range(0, len(full_signal)-self.window_size, stride)]
            
            # 如果指定了最大样本数，则截取
            if max_samples_per_class is not None:
                samples = samples[:max_samples_per_class]
            
            class_path = os.path.join(self.save_dir, name)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
            
            for i, s in enumerate(tqdm(samples, desc=name)):
                img_name = f"{name}_{i}.png"
                img_path = os.path.join(class_path, img_name)
                
                self.save_cwt_image(s, img_path)
                label_records.append({'path': img_path, 'label': label, 'class': name})
        
        df_labels = pd.DataFrame(label_records)
        df_labels.to_csv(os.path.join(self.save_dir, "labels.csv"), index=False)
        print(f"数据集制作完成！标签已保存至 {self.save_dir}/labels.csv")

#--------------------------------------------------------------------------------------------------------------------------

class SEDatasetGenerator:
    """Southeast轴承故障数据集生成器"""

    def __init__(self, 
                 base_dir=r"/home/angela/Motor_Fault_Detect/Rawdata/Southeast/bearingset",
                 save_dir=r"/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset",
                 window_size=512,
                 overlap_ratio=0.5,
                 wavename='cmor1-1',
                 totalscale=128,
                 sampling_rate=25600,
                 skiprows=16):
        """
        初始化SE数据集生成器
        
        Args:
            base_dir: SE原始CSV数据目录
            save_dir: 生成数据集保存目录
            window_size: 窗口大小
            overlap_ratio: 重叠率
            wavename: 小波基名称
            totalscale: 小波尺度数
            sampling_rate: 采样率（SE数据集为25.6kHz）
            skiprows: CSV文件跳过的行数
        """
        self.base_dir = base_dir
        self.save_dir = save_dir
        self.window_size = window_size
        self.overlap_ratio = overlap_ratio
        self.wavename = wavename
        self.totalscale = totalscale
        self.sampling_rate = sampling_rate
        self.skiprows = skiprows
        
        # SE数据集的8个通道
        self.channels = [
            'motor',      # 电机振动
            'pt_x', 'pt_y', 'pt_z',  # 行星齿轮箱 x,y,z
            'torque',     # 扭矩
            'pa_x', 'pa_y', 'pa_z'   # 并行减速齿轮箱 x,y,z
        ]
    
    def load_se_csv(self, csv_path):
        """读取SE数据集的CSV文件，自动检测分隔符"""
        # 先尝试用制表符读取
        try:
            df = pd.read_csv(csv_path, skiprows=self.skiprows, header=None, sep='\t')
            # 检查列数是否正确（应该是9列：8个数据列+1个空列）
            if len(df.columns) >= 8:
                df.columns = self.channels + ['_']  # 最后一列是空列
                return df
        except Exception as e:
            pass
        
        # 如果失败，尝试用逗号读取
        try:
            df = pd.read_csv(csv_path, skiprows=self.skiprows, header=None, sep=',')
            # 去掉可能的空列
            df = df.dropna(axis=1, how='all')
            if len(df.columns) >= 8:
                df.columns = self.channels[:len(df.columns)]
                return df
        except Exception as e:
            raise ValueError(f"无法读取文件 {csv_path}: {e}")
        
        raise ValueError(f"文件格式不符合预期: {csv_path}")
    
    def save_cwt_image(self, signal, save_path):
        """将1D信号转为CWT图像并保存"""
        sampling_period = 1.0 / self.sampling_rate
        fc = pywt.central_frequency(self.wavename)
        cparam = 2 * fc * self.totalscale
        scales = cparam / np.arange(self.totalscale, 0, -1)
        
        coef, freqs = pywt.cwt(signal, scales, self.wavename, sampling_period)
        amp = np.abs(coef)
        
        # 绘图并保存（Agg 后端渲染，不会弹出窗口，不会报错）
        plt.figure(figsize=(2.56, 2.56), dpi=100) 
        plt.axes([0, 0, 1, 1]) 
        plt.axis('off')
        plt.contourf(amp, cmap='jet')
        plt.savefig(save_path)
        plt.close('all')  # 显式关闭所有画布，释放内存
    
    def build_dataset(self, max_samples_per_class=None):
        """
        生成SE数据集
        
        Args:
            max_samples_per_class: 每类最大样本数，None表示使用所有样本
        """
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        label_records = []
        global_label = 0
        
        # 获取base_dir下所有的CSV文件
        csv_files = [f for f in os.listdir(self.base_dir) if f.endswith('.csv')]
        
        if not csv_files:
            print(f"在 {self.base_dir} 中未找到CSV文件！")
            return
        
        print(f"找到 {len(csv_files)} 个CSV文件")
        
        # 遍历每个CSV文件
        for csv_file in sorted(csv_files):
            csv_path = os.path.join(self.base_dir, csv_file)
            
            # 从文件名提取类别名（去除.csv后缀）
            fault_type = csv_file.replace('.csv', '')
            print(f"\n正在处理文件: {csv_file} (故障类型: {fault_type})")
            
            # 读取CSV文件
            try:
                df = self.load_se_csv(csv_path)
            except Exception as e:
                print(f"读取文件 {csv_file} 失败: {e}")
                continue
            
            # 遍历每个通道
            for channel in self.channels:
                print(f"  处理通道: {channel}...")
                
                full_signal = df[channel].values
                
                # 滑窗切分
                stride = int(self.window_size * (1 - self.overlap_ratio))
                samples = [full_signal[i:i+self.window_size] 
                          for i in range(0, len(full_signal)-self.window_size, stride)]
                
                # 如果指定了最大样本数，则截取
                if max_samples_per_class is not None:
                    samples = samples[:max_samples_per_class]
                
                # 创建子母文件夹结构: fault_type/channel/
                class_path = os.path.join(self.save_dir, fault_type, channel)
                if not os.path.exists(class_path):
                    os.makedirs(class_path)
                
                # 生成CWT图像
                for i, s in enumerate(tqdm(samples, desc=f"{fault_type}/{channel}")):
                    img_name = f"{fault_type}_{channel}_{i}.png"
                    img_path = os.path.join(class_path, img_name)
                    
                    self.save_cwt_image(s, img_path)
                    
                    # 记录标签：fault_type+channel作为组合类别
                    class_name = f"{fault_type}_{channel}"
                    label_records.append({
                        'path': img_path, 
                        'label': global_label, 
                        'class': class_name,
                        'fault_type': fault_type,
                        'channel': channel
                    })
                
                global_label += 1
        
        # 保存标签文件
        df_labels = pd.DataFrame(label_records)
        df_labels.to_csv(os.path.join(self.save_dir, "labels.csv"), index=False)
        print(f"\n数据集制作完成！标签已保存至 {self.save_dir}/labels.csv")
        print(f"共生成 {len(df_labels)} 个样本，{global_label} 个类别")

Select_TYPE = 'SE'
if __name__ == "__main__":
    if Select_TYPE == 'CWRU':
        # 使用CWRU数据集生成器
        generator = CWRUDatasetGenerator(
            base_dir=r"/home/angela/Motor_Fault_Detect/Rawdata/CWRU",
            save_dir=r"/home/angela/Motor_Fault_Detect/Dataset/CWT_Dataset(sample=all)",
            rpm=1730,
            window_size=512,
            overlap_ratio=0.5,
            wavename='cmor1-1',
            totalscale=128,
            sampling_rate=12000
        )
        # 生成完整数据集，如果只想测试，可以设置 max_samples_per_class=100
        generator.build_dataset(max_samples_per_class=None)
        
    elif Select_TYPE == 'SE':
        # 使用SE数据集生成器
        generator = SEDatasetGenerator(
            base_dir=r"/home/angela/Motor_Fault_Detect/Rawdata/Southeast/bearingset/ball_2",
            save_dir=r"/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset",
            window_size=512,
            overlap_ratio=0.5,
            wavename='cmor1-1',
            totalscale=128,
            sampling_rate=25600,
            skiprows=16
        )
        # 生成完整数据集，如果只想测试，可以设置 max_samples_per_class=100
        generator.build_dataset(max_samples_per_class=None)