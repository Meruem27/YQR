import torch
import os
from complex_model import ComplEx
from dataset_loader import KnowledgeGraphDataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import numpy as np
from scipy.ndimage import gaussian_filter1d  # 用于平滑曲线
from collections import defaultdict  # 用于统计常见三元组组合

# 加载数据集和模型
dataset = KnowledgeGraphDataset(folder_path='../dataset/All_animal')
entity2id, relation2id = dataset.get_entity_relation_maps()
test_files = dataset.get_test_files()

# 使用训练时相同的实体和关系数量
entity_count = len(entity2id)
relation_count = len(relation2id)
embedding_dim = 10
model = ComplEx(entity_count, relation_count, embedding_dim)

# 加载训练好的模型，加入异常捕获机制
try:
    model.load_state_dict(torch.load('complex_model.pth'))
    print("模型成功加载！")
except RuntimeError as e:
    print(f"加载模型时发生错误: {e}")
    exit(1)

# 新的保存路径
new_folder = '../dataset/All_animal_predictions'
if not os.path.exists(new_folder):
    os.makedirs(new_folder)

# 知识图谱矫正和预测
model.eval()

# 初始化列表存储真实标签和预测的分数
true_labels = []
predicted_scores = []

# 为头节点-尾节点对建立关系统计
head_tail_relation_stats = defaultdict(set)

# 统计函数：记录常见头尾实体对的关系
def update_head_tail_relation_stats(head_id, tail_id, relation_id):
    head_tail_relation_stats[(head_id, tail_id)].add(relation_id)

# 函数：矫正现有三元组，并返回预测分数
def predict_and_correct(head, relation, tail):
    head_id = entity2id.get(head)
    relation_id = relation2id.get(relation)
    tail_id = entity2id.get(tail)

    if head_id is not None and relation_id is not None and tail_id is not None:
        score = model(torch.tensor([head_id]), torch.tensor([relation_id]), torch.tensor([tail_id]))
        predicted_score = score.item()  # 获取模型输出的分数

        # 记录常见头尾组合和关系
        update_head_tail_relation_stats(head_id, tail_id, relation_id)

        # 返回矫正后的尾实体和预测分数
        if predicted_score < 0.5:  # 如果得分较低，进行矫正
            for triple in dataset.triples:
                if triple[0] == head_id and triple[1] == relation_id:
                    return list(entity2id.keys())[list(entity2id.values()).index(triple[2])], predicted_score
        return tail, predicted_score
    return tail, 0.0  # 如果没有找到相应的实体，则返回分数为0

# 函数：根据统计关系，补充常见的头尾实体组合
def complete_frequent_relations():
    completed_triples = []
    for (head_id, tail_id), relations in head_tail_relation_stats.items():
        for relation_id in range(relation_count):  # 尝试所有关系
            if relation_id not in relations:  # 补全不存在的关系
                head = list(entity2id.keys())[list(entity2id.values()).index(head_id)]
                relation = list(relation2id.keys())[list(relation2id.values()).index(relation_id)]
                tail = list(entity2id.keys())[list(entity2id.values()).index(tail_id)]
                completed_triples.append((head, relation, tail))
    return completed_triples

# 遍历测试集并保存矫正和补全后的结果
with torch.no_grad():
    for test_file in test_files:
        try:
            corrected_triples = []
            file_path = os.path.join('../dataset/All_animal', test_file)
            modified_flag = False  # 用于标记是否文件有任何修改

            with open(file_path, 'r') as f:
                print(f"正在处理文件: {test_file}")
                for line in f:
                    # 确保每行能正确分割成三元组
                    elements = line.strip().split()
                    if len(elements) == 3:
                        head, relation, tail = elements
                        corrected_tail, score = predict_and_correct(head, relation, tail)

                        # 记录真实值和预测分数，准备计算精确度-召回率曲线
                        true_labels.append(1 if corrected_tail == tail else 0)  # 如果矫正后与原始值相同，则为1，否则为0
                        predicted_scores.append(score)  # 模型预测的分数

                        # 检查是否三元组有变化
                        if corrected_tail != tail:
                            modified_flag = True

                        corrected_triples.append((head, relation, corrected_tail))
                    else:
                        print(f"Skipping invalid line: {line.strip()}")

            # 保存矫正后的三元组，并确保使用单个空格分隔
            output_path = os.path.join(new_folder, test_file)
            with open(output_path, 'w') as out_file:
                for h, r, t in corrected_triples:
                    out_file.write(f'{h} {r} {t}\n')  # 使用单个空格分隔实体、关系和实体

            # 如果文件有任何修改，打印出来
            if modified_flag:
                print(f"文件 {test_file} 已修改")
            else:
                print(f"文件 {test_file} 无修改")

            print(f"文件 {test_file} 的预测结果已保存为 {output_path}。")

        except Exception as e:
            print(f"处理文件 {test_file} 时发生错误: {e}")
            continue

# 补充常见的头尾实体组合
completed_triples = complete_frequent_relations()

# 保存补全后的三元组
completed_output_path = os.path.join(new_folder, 'completed_triples.txt')
with open(completed_output_path, 'w') as completed_file:
    for h, r, t in completed_triples:
        completed_file.write(f'{h} {r} {t}\n')
print(f"补全的三元组已保存为 {completed_output_path}。")

# 将真实标签和预测的分数转换为numpy数组
true_labels = np.array(true_labels)
predicted_scores = np.array(predicted_scores)

# 计算精确度和召回率
precision, recall, _ = precision_recall_curve(true_labels, predicted_scores)

# 使用高斯滤波器来平滑曲线
precision_smooth = gaussian_filter1d(precision, sigma=2)
recall_smooth = gaussian_filter1d(recall, sigma=2)

# 绘制平滑后的精确度-召回率曲线
plt.figure(figsize=(8, 6))
plt.plot(recall_smooth, precision_smooth, marker='.', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.show()
