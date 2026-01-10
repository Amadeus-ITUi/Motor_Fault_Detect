import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 1. 基础配置
CSV_PATH = "/home/angela/Motor_Fault_Detect/Dataset/SE_CWT_Dataset(sample=all)/bearingset/labels_motor_20_0.csv"
MODEL_PATH = "/home/angela/Motor_Fault_Detect/Model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001

# 2. 自定义数据集类
class CWTRDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['path']
        label = self.df.iloc[idx]['label']
        # 读取图片并转为 RGB
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# 3. 定义简单的 CNN 模型
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

# 4. 主训练流程
def train_model():
    # A. 加载标签
    df = pd.read_csv(CSV_PATH)
    
    # --- 核心修改：动态识别类别数量 ---
    # 通过 nunique() 统计 label 列有多少个互不相同的数字
    num_classes = df['label'].nunique()
    print(f"检测到数据集包含 {num_classes} 个类别。")
    # --------------------------------
    
    df = df.sample(frac=1).reset_index(drop=True) # 打乱数据
    
    n = len(df)
    train_df = df.iloc[:int(n*0.7)]
    val_df = df.iloc[int(n*0.7):int(n*0.9)]
    test_df = df.iloc[int(n*0.9):]

    # B. 图像预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)), 
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # C. 准备 DataLoader
    train_loader = DataLoader(CWTRDataset(train_df, transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(CWTRDataset(val_df, transform), batch_size=BATCH_SIZE)
    test_loader = DataLoader(CWTRDataset(test_df, transform), batch_size=BATCH_SIZE)

    # D. 初始化模型、损失函数和优化器 (传入动态获取的 num_classes)
    model = SimpleCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {'train_loss': [], 'val_acc': []}

    # E. 训练循环
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        avg_loss = train_loss / len(train_loader)
        acc = 100 * correct / len(val_df)
        
        history['train_loss'].append(avg_loss)
        history['val_acc'].append(acc)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Val Acc: {acc:.2f}%")

    # F. 保存模型   
    file_dir = os.path.join(MODEL_PATH, "SE/bearing_motor_20_0")
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    save_path = os.path.join(file_dir, "bearing_fault_cnn.pth")
    torch.save(model.state_dict(), save_path)
    print(f"模型已保存为 {save_path}")

    # 绘图逻辑
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCHS + 1), history['train_loss'], label='Train Loss', color='red', marker='o')
    plt.title('Training Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCHS + 1), history['val_acc'], label='Val Accuracy', color='blue', marker='s')
    plt.title('Validation Accuracy')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_log.png')
    print("训练曲线图已保存为 training_log.png")
    plt.show()

if __name__ == "__main__":
    train_model()