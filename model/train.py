import torch
import torch.optim as optim
from complex_model import ComplEx
from dataset_loader import KnowledgeGraphDataset

# 加载数据集
dataset = KnowledgeGraphDataset(folder_path='../dataset/All_animal')
train_heads, train_relations, train_tails = dataset.get_train_data()

# 定义模型
entity_count = len(dataset.entity2id)
relation_count = len(dataset.relation2id)
embedding_dim = 10
model = ComplEx(entity_count, relation_count, embedding_dim)

# 优化器
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCEWithLogitsLoss()

# 训练
epochs = 200
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    scores = model(train_heads, train_relations, train_tails)
    labels = torch.ones_like(scores)
    loss = loss_fn(scores, labels)
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# 保存模型
torch.save(model.state_dict(), 'complex_model.pth')
