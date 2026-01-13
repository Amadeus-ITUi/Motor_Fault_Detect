import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. 定义与之前一致的模型结构
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

# --- 分析配置：在这里修改你想测试的因子 ---
# 你可以只选 ['rms', 'kurtosis'] 来测试敏感度
SELECTED_FEATURES = ['rms', 'kurtosis', 'peakF', 'SKMean', 'FC', 'var', 'std'] 
DATA_DIR = '/home/angela/Motor_Fault_Detect/Dataset/SE_Features'

def train_and_eval():
    # 1. 加载所有 CSV 数据并合并
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    df_list = [pd.read_csv(f) for f in all_files]
    full_df = pd.concat(df_list, ignore_index=True)
    
    print(f"总数据集大小: {full_df.shape}")

    # 2. 准备特征 (X) 和 标签 (y)
    X = full_df[SELECTED_FEATURES].values
    y_raw = full_df['label'].values
    
    # 编码标签 (Normal -> 0, Ball -> 1 等)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # 数据标准化 (MLP 对量纲很敏感，这步必做)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 3. 切分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 转换为 Tensor
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.LongTensor(y_test)

    # 4. 初始化模型
    model = SignalMLP(input_size=len(SELECTED_FEATURES), num_classes=len(le.classes_))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # 5. 训练循环
    print(f"开始使用 {len(SELECTED_FEATURES)} 个特征进行训练...")
    for epoch in range(200):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/200], Loss: {loss.item():.4f}")

    # 6. 测试集评估
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        accuracy = (predicted == y_test).sum().item() / y_test.size(0)
        
    print("-" * 30)
    print(f"特征组合: {SELECTED_FEATURES}")
    print(f"测试集准确率: {accuracy * 100:.2f}%")
    print(f"类别对应关系: {dict(enumerate(le.classes_))}")

if __name__ == "__main__":
    train_and_eval()