import torch
import torch.nn as nn

class ComplEx(nn.Module):
    def __init__(self, entity_count, relation_count, embedding_dim):
        super(ComplEx, self).__init__()
        self.embedding_dim = embedding_dim
        self.entity_real = nn.Embedding(entity_count, embedding_dim)
        self.entity_imag = nn.Embedding(entity_count, embedding_dim)
        self.relation_real = nn.Embedding(relation_count, embedding_dim)
        self.relation_imag = nn.Embedding(relation_count, embedding_dim)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.entity_real.weight.data)
        nn.init.xavier_uniform_(self.entity_imag.weight.data)
        nn.init.xavier_uniform_(self.relation_real.weight.data)
        nn.init.xavier_uniform_(self.relation_imag.weight.data)

    def forward(self, head, relation, tail):
        head_real = self.entity_real(head)
        head_imag = self.entity_imag(head)
        tail_real = self.entity_real(tail)
        tail_imag = self.entity_imag(tail)
        relation_real = self.relation_real(relation)
        relation_imag = self.relation_imag(relation)

        score_real = torch.sum(head_real * relation_real * tail_real +
                               head_imag * relation_imag * tail_real +
                               head_real * relation_imag * tail_imag -
                               head_imag * relation_real * tail_imag, -1)
        return score_real
