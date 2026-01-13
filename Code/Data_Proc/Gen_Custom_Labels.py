import os
import pandas as pd
from pathlib import Path

# ==================== 宏定义 / 配置区 ====================
# 在这里方便地修改你的标签地址组
# 格式: (标签数字, 文件夹路径, 自定义类名)
# 如果多个文件夹需要打上同一个标签，只需重复该标签数字即可
LABEL_CONFIG = [
    # 标签 0 对应的三个文件夹地址，自定义类名可以相同也可以不同
    (0, "/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset/ball_20_0/motor", "ball_20_0"),
    (1, "/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset/comb_20_0/motor", "comb_20_0"),
    (2, "/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset/outer_20_0/motor", "outer_20_0"),
    (3, "/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset/inner_20_0/motor", "inner_20_0"),
    (4, "/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset/health_20_0/motor", "health_20_0"),
    (5, "/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset/ball_30_2/motor", "ball_30_2"),
    (6, "/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset/comb_30_2/motor", "comb_30_2"),
    (7, "/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset/outer_30_2/motor", "outer_30_2"),
    (8, "/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset/inner_30_2/motor", "inner_30_2"),
    (9, "/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset/health_30_2/motor", "health_30_2"),
]

# 输出的文件名
OUTPUT_CSV = "/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset/labels_bearingset_motor.csv"
# ========================================================

class CustomLabelGenerator:
    def __init__(self, output_path):
        self.output_path = output_path

    def run(self, config):
        """
        根据配置组生成标签文件
        config: [(label, path, class_name), ...]
        """
        label_records = []

        for label, folder_path, custom_class_name in config:
            root = Path(folder_path)
            if not root.exists():
                print(f"跳过不存在的目录: {folder_path}")
                continue

            # rglob 会递归扫描所有子文件夹下的 .png 文件
            image_files = list(root.rglob('*.png'))
            
            print(f"处理类别 [{label}] - {custom_class_name}: 扫描到 {len(image_files)} 张图片")

            for img_path in image_files:
                # 按照要求的顺序排列：path, label, class
                label_records.append({
                    'path': str(img_path.absolute()),
                    'label': int(label),
                    'class': custom_class_name
                })

        if not label_records:
            print("未找到任何图片，操作取消。")
            return

        # 生成 DataFrame
        df = pd.DataFrame(label_records)
        
        # 按照 path, label, class 保存
        # index=False 不保存行索引，确保格式纯净
        df.to_csv(self.output_path, index=False)
        
        print("-" * 30)
        print(f"CSV 已生成: {self.output_path}")
        if not df.empty:
            print(f"首行示例: {df.iloc[0]['path']},{df.iloc[0]['label']},{df.iloc[0]['class']}")

if __name__ == "__main__":
    generator = CustomLabelGenerator(OUTPUT_CSV)
    generator.run(LABEL_CONFIG)