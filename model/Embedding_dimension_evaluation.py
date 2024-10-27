import torch
import torch.optim as optim
from complex_model import ComplEx
from dataset_loader import KnowledgeGraphDataset
from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  # 用于划分训练集和验证集

# 加载数据集
dataset = KnowledgeGraphDataset(folder_path='../dataset/All_animal')
train_heads, train_relations, train_tails = dataset.get_train_data()

# 划分训练集和验证集（80% 训练，20% 验证）
train_heads, val_heads, train_relations, val_relations, train_tails, val_tails = train_test_split(
    train_heads, train_relations, train_tails, test_size=0.2, random_state=42)

# 超参数设置
embedding_dims = list(range(500, 50001, 1000))  # 嵌入维度从500到10000，步长为500
epochs = 50
lr = 0.001
weight_decay = 1e-5  # L2正则化参数
early_stopping_patience = 5  # Early Stopping耐心参数
best_dim = 0
best_performance = float('inf')
performances = {}
no_improvement_count = 0  # 用于Early Stopping计数


# 定义训练和验证函数
def train_and_evaluate(model, train_heads, train_relations, train_tails, val_heads, val_relations, val_tails):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)  # 引入L2正则化
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        scores = model(train_heads, train_relations, train_tails)
        labels = torch.ones_like(scores)
        loss = loss_fn(scores, labels)
        loss.backward()
        optimizer.step()

        # 验证模型
        model.eval()
        with torch.no_grad():
            val_scores = model(val_heads, val_relations, val_tails)
            val_labels = torch.ones_like(val_scores)
            val_loss = loss_fn(val_scores, val_labels).item()

        # Early stopping 检查
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_count = 0  # 重置early stopping计数
        else:
            no_improvement_count += 1
            if no_improvement_count >= early_stopping_patience:
                print(f"早停触发，停止训练，最佳验证损失: {best_val_loss}")
                break

    return best_val_loss


# 遍历不同维度进行模型训练与评估
for dim in embedding_dims:
    print(f"正在训练嵌入维度为 {dim} 的模型...")
    entity_count = len(dataset.entity2id)
    relation_count = len(dataset.relation2id)
    model = ComplEx(entity_count, relation_count, embedding_dim=dim)

    # 训练并评估模型
    val_performance = train_and_evaluate(model, train_heads, train_relations, train_tails, val_heads, val_relations,
                                         val_tails)
    performances[dim] = val_performance

    print(f"嵌入维度为 {dim} 的模型验证损失为: {val_performance}")

    if val_performance < best_performance:
        best_performance = val_performance
        best_dim = dim

# 输出最优维度
print(f"最优的嵌入维度为: {best_dim}，对应的验证损失为: {best_performance}")

# 可视化不同维度下的模型性能
dims = list(performances.keys())
losses = list(performances.values())

plt.figure(figsize=(8, 6))
plt.plot(dims, losses, marker='o', linestyle='-', color='b')
plt.xlabel('Embedding Dimension')
plt.ylabel('Validation Loss')
plt.title('Model Performance Across Different Embedding Dimensions')
plt.grid(True)
plt.show()
