# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

"""
建立网络模型结构
"""


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    # 输入为问题字符编码
    def forward(self, x):
        x = self.embedding(x)
        # 使用lstm
        # x, _ = self.lstm(x)
        # 使用线性层
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config, margin=1.0):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        self.loss = TripletLoss(margin)

    # 计算余弦距离  1-cos(a,b)
    # cos=1时两个向量相同，余弦距离为0；cos=0时，两个向量正交，余弦距离为1
    def cosine_distance(self, tensor1, tensor2):
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), axis=-1)
        return 1 - cosine

    def cosine_triplet_loss(self, a, p, n, margin=None):
        ap = self.cosine_distance(a, p)
        an = self.cosine_distance(a, n)
        if margin is None:
            diff = ap - an + 0.1
        else:
            diff = ap - an + margin.squeeze()
        return torch.mean(diff[diff.gt(0)])  # greater than

    # sentence : (batch_size, max_length)
    def forward(self, anchor, positive=None, negative=None):
        # 同时传入两个句子
        # 当同时传入三个句子时，认为是在计算Triplet Loss
        if positive is not None and negative is not None:
            vector_anchor = self.sentence_encoder(anchor)  # vec:(batch_size, hidden_size)
            vector_positive = self.sentence_encoder(positive)
            vector_negative = self.sentence_encoder(negative)
            loss = self.loss(vector_anchor, vector_positive, vector_negative)
            return loss
        # 单独传入一个或两个句子时，保持原有行为
        else:
            return self.sentence_encoder(anchor)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config

    Config["vocab_size"] = 10
    Config["max_length"] = 4
    Config["hidden_size"] = 64  # 假设隐藏层大小为64
    model = SiameseNetwork(Config, margin=1.0)

    # 模拟数据
    a = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])  # 锚点样本
    p = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])  # 正样本
    n = torch.LongTensor([[0, 0, 0, 0], [1, 0, 0, 0]])  # 负样本

    # 计算Triplet Loss
    y = model(a, p, n)
    print(y)
