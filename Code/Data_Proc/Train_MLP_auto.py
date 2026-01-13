import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- 配置与模型定义 ---
DATA_DIR = '/home/angela/Motor_Fault_Detect/Dataset/SE_Features'
RESULT_SAVE_PATH = 'feature_combination_results.csv'
EPOCHS = 200
LR = 0.005

ALL_FEATURES = ['max', 'min', 'mean', 'peak', 'arv', 'var', 'std',
                'kurtosis', 'skewness', 'rms', 'waveformF', 'peakF', 
                'impulseF', 'clearanceF', 'FC', 'MSF', 'RMSF', 'VF', 
                'RVF', 'SKMean', 'SKStd', 'SKSkewness', 'SKKurtosis']

class SignalMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SignalMLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )
    def forward(self, x):
        return self.layers(x)

def evaluate_combination(features, full_df, le):
    """针对给定的特征组合训练并评估模型"""
    X = full_df[list(features)].values
    y = le.transform(full_df['label'].values)
    
    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 转换为 Tensor
    X_train, y_train = torch.FloatTensor(X_train), torch.LongTensor(y_train)
    X_test, y_test = torch.FloatTensor(X_test), torch.LongTensor(y_test)
    
    model = SignalMLP(input_size=len(features), num_classes=len(le.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # 训练循环
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
    
    # 评估
    model.eval()
    with torch.no_grad():
        preds = torch.argmax(model(X_test), dim=1)
        acc = (preds == y_test).sum().item() / y_test.size(0)
    
    return acc

def run_auto_test():
    # 1. 加载数据
    print("正在加载数据集...")
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    full_df = pd.concat([pd.read_csv(f) for f in all_files], ignore_index=True)
    
    le = LabelEncoder()
    le.fit(full_df['label'].values)
    
    results = []
    
    # --- 策略 1: 测试所有单个因子 ---
    print("\n>>> 开始单因子敏感度测试...")
    for f in ALL_FEATURES:
        acc = evaluate_combination([f], full_df, le)
        print(f"Feature: {f:12} | Accuracy: {acc*100:6.2f}%")
        results.append({'combination': f, 'count': 1, 'accuracy': acc})
    
    # --- 策略 2: 测试全因子组合 ---
    print("\n>>> 开始全因子综合测试...")
    acc_all = evaluate_combination(ALL_FEATURES, full_df, le)
    print(f"All Features Combined | Accuracy: {acc_all*100:6.2f}%")
    results.append({'combination': 'ALL_23_FEATURES', 'count': 23, 'accuracy': acc_all})

    # --- 策略 3: 测试特定的 5 因子组合 ---
    print("\n>>> 开始特定组合测试...")
    combo = ['std', 'FC', 'RMSF', 'max', 'var']
    acc_specific = evaluate_combination(combo, full_df, le)
    print(f"Specific Combo ({'+'.join(combo)}) | Accuracy: {acc_specific*100:6.2f}%")
    results.append({'combination': '+'.join(combo), 'count': len(combo), 'accuracy': acc_specific})

    # 2. 保存结果
    res_df = pd.DataFrame(results)
    # 按准确率从高到低排序
    res_df = res_df.sort_values(by='accuracy', ascending=False)
    res_df.to_csv(RESULT_SAVE_PATH, index=False)
    print(f"\n[测试完成] 结果已保存至: {RESULT_SAVE_PATH}")

if __name__ == "__main__":
    run_auto_test()