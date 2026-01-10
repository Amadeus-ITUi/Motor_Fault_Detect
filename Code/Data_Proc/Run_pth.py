import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import pandas as pd

# 1. 定义模型结构 (必须与 Train_CNN.py 完全一致)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 128 -> 64
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 64 -> 32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # 32 -> 16
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128), 
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def get_class_info(csv_path):
    """
    自动从 CSV 标签文件中提取类名列表和类别总数
    """
    if not os.path.exists(csv_path):
        print(f"警告: 找不到标签文件 {csv_path}，将无法获取类名！")
        return [], 0
    
    df = pd.read_csv(csv_path)
    # 按照 label 排序并去重，确保索引与 label 数字完美对应
    mapping = df.sort_values('label').drop_duplicates('label')
    class_names = mapping['class'].tolist()
    num_classes = len(class_names)
    
    return class_names, num_classes

def predict(model_path, image_path, csv_path):
    # 检测设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # A. 自动获取类名和类别总数
    class_names, num_classes = get_class_info(csv_path)
    if num_classes == 0: return

    # B. 实例化模型并加载权重
    # ---------------------------------------------------------
    # 解决 RuntimeError 的关键点：
    # 你的 .pth 文件是在 num_classes=10 时训练的。
    # 即使 CSV 只有 5 类，加载旧模型也必须先按 10 类初始化，否则 Shape 不匹配。
    # ---------------------------------------------------------
    try:
        # 尝试按 CSV 实际类别数加载
        model = SimpleCNN(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        print("\n[错误提示] 类别数量不匹配！")
        print(f"当前 CSV 记录为 {num_classes} 类，但模型文件是按 10 类训练的。")
        print("正在尝试强制以 10 类模式加载以解决报错...")
        
        # 强制使用 10 类初始化（兼容你那个已经训练好的 pth 文件）
        model = SimpleCNN(num_classes=10).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.eval()

    # C. 图像预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # D. 读取图片并处理
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device) 
    except Exception as e:
        print(f"读取图片失败: {e}")
        return

    # E. 开始预测
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted_idx = torch.max(output, 1)
        result_idx = predicted_idx.item()

    # F. 输出结果
    print("-" * 30)
    print(f"分析图片: {os.path.basename(image_path)}")
    
    # 增加一层保护，防止索引超出 class_names 范围
    if result_idx < len(class_names):
        print(f"鉴定结果: {class_names[result_idx]} (Label ID: {result_idx})")
    else:
        print(f"鉴定结果: 标签 ID {result_idx} (但在 CSV 中未找到对应类名，请重新训练)")
    print("-" * 30)

if __name__ == "__main__":
    # ==================== 配置区 ====================
    # 模型文件路径
    MODEL_FILE = "/home/angela/Motor_Fault_Detect/Model/SE/bearing_motor_20_0/bearing_fault_cnn.pth"
    # 测试图片路径
    TEST_IMAGE = "/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset/outer_20_0/motor/outer_20_0_motor_3.png"
    # 标签 CSV 路径 (用于自动获取类名)
    CSV_FILE = "/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset/labels_motor_20_0.csv"
    # ===============================================
    
    if os.path.exists(MODEL_FILE):
        predict(MODEL_FILE, TEST_IMAGE, CSV_FILE)
    else:
        print(f"错误: 找不到模型文件 {MODEL_FILE}")