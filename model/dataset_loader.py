import os
import random
import torch

class KnowledgeGraphDataset:
    def __init__(self, folder_path, split_ratio=0.8):
        self.folder_path = folder_path
        self.txt_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.txt')])

        # 随机打乱文件列表
        random.shuffle(self.txt_files)

        # 按照比例划分训练集和测试集
        split_point = int(len(self.txt_files) * split_ratio)
        self.train_files = self.txt_files[:split_point]
        self.test_files = self.txt_files[split_point:]

        self.entity2id = {}
        self.relation2id = {}
        self.triples = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        entity_count = 0
        relation_count = 0
        for file in self.train_files:
            with open(os.path.join(self.folder_path, file), 'r') as f:
                for line in f:
                    # 确保每行只有三个元素
                    elements = line.strip().split()
                    if len(elements) == 3:
                        head, relation, tail = elements
                        if head not in self.entity2id:
                            self.entity2id[head] = entity_count
                            entity_count += 1
                        if tail not in self.entity2id:
                            self.entity2id[tail] = entity_count
                            entity_count += 1
                        if relation not in self.relation2id:
                            self.relation2id[relation] = relation_count
                            relation_count += 1
                        self.triples.append((self.entity2id[head], self.relation2id[relation], self.entity2id[tail]))
                    else:
                        print(f"Skipping invalid line: {line.strip()}")

    def get_train_data(self):
        heads, relations, tails = [], [], []
        for h, r, t in self.triples:
            heads.append(h)
            relations.append(r)
            tails.append(t)
        return torch.tensor(heads), torch.tensor(relations), torch.tensor(tails)

    def get_test_files(self):
        return self.test_files

    def get_entity_relation_maps(self):
        return self.entity2id, self.relation2id
