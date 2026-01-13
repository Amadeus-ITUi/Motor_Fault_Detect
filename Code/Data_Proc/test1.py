import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
import time

# ==========================================
# 1. 模拟数据生成 (模仿你的场景)
# ==========================================
# n_features=6: 你的5-6个因子
# n_classes=7: 你的7个分类结果
# n_informative=5: 假设其中5个因子是有用的，1个是干扰项
print("正在生成模拟数据...")
X, y = make_classification(n_samples=2000, 
                           n_features=6, 
                           n_informative=5, 
                           n_classes=7, 
                           n_clusters_per_class=1,
                           random_state=42)

# 划分训练集和测试集 (80% 训练, 20% 验证)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"数据准备完毕: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
print("-" * 30)

# ==========================================
# 2. 定义模型
# ==========================================
models = {
    "决策树 (Decision Tree)": DecisionTreeClassifier(max_depth=10, random_state=42),
    
    "随机森林 (Random Forest)": RandomForestClassifier(n_estimators=100, random_state=42),
    
    # MLP即多层感知机，一种基础的神经网络
    # hidden_layer_sizes=(100, 50): 两个隐藏层，分别有100和50个神经元
    "神经网络 (Neural Network)": MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

# ==========================================
# 3. 训练与评估
# ==========================================
results = {}

for name, model in models.items():
    start_time = time.time()
    
    # 训练
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    acc = accuracy_score(y_test, y_pred)
    elapsed = time.time() - start_time
    results[name] = acc
    
    print(f"\n模型: 【{name}】")
    print(f"耗时: {elapsed:.4f} 秒")
    print(f"准确率: {acc:.2%}")
    #如果你需要看每一类的详细准确率，取消下面这行的注释
    #print(classification_report(y_test, y_pred))

# ==========================================
# 4. 可视化：谁才是最重要的因子？
# ==========================================
# 通常随机森林的特征重要性最具参考价值
rf_model = models["随机森林 (Random Forest)"]
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("各因子重要性排名 (基于随机森林)")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [f"因子 {i}" for i in indices]) # 这里可以换成你真实的因子名字
plt.xlim([-1, X.shape[1]])
plt.ylabel("重要性权重")
plt.tight_layout()

print("\n正在显示特征重要性图表...")
plt.show()

# 对于强化学习背景的提示：
# 如果你想把这些应用到 RL (强化学习) 中，
# 神经网络 (MLP) 通常对应 RL 中的 Policy Network (策略网络) 或 Q-Network。
# 它是 Deep Q-Learning (DQN) 的核心组件。