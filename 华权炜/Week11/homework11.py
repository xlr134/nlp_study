#coding:utf8

import json
import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader



class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(path, return_dict=False, attn_implementation='eager')
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask=None, y=None):
        if y is not None:
            #训练时，构建一个下三角的mask矩阵，让上下文之间没有交互
            mask = torch.tril(torch.ones((x.shape[0], x.shape[1], x.shape[1])))
            if torch.cuda.is_available():
                mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            #预测时，可以不使用mask
            x, _ = self.bert(x)
            y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
            return torch.softmax(y_pred, dim=-1)

#加载语料
def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line)
            corpus.append([line["title"], line["content"]])
    return corpus



def build_mask(a, b):
    mask = torch.ones(a + b + 3, a + b +3)
    for i in range(a + 2):
        mask[i, a + 2 :] = 0  
    for i in range(b + 1):
        mask[a + 2 + i, a + 3 + i:] = 0
    return mask

def pad_mask(tensor, target_shape):
    height, width = tensor.shape
    target_height, target_width = target_shape
    result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
    h_start = 0
    w_start = 0
    h_end = min(height, target_height)
    w_end = min(width, target_width)
    result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
    return result


def build_dataset(tokenizer, corpus, max_length, batch_size):
    dataset = []
    for i, (prompt, answer) in enumerate(corpus):
        prompt_encode = tokenizer.encode(prompt, add_special_tokens=False)
        answer_encode = tokenizer.encode(answer, add_special_tokens=False)
        x = [tokenizer.cls_token_id] + prompt_encode + [tokenizer.sep_token_id] + answer_encode + [tokenizer.sep_token_id]
        y = len(prompt_encode) * [-1] + [-1] + answer_encode + [tokenizer.sep_token_id] + [-1]
        mask = build_mask(len(prompt_encode), len(answer_encode))
        x = x[:max_length] + [0] * (max_length - len(x))
        y = y[:max_length] + [0] * (max_length - len(y))
        x = torch.LongTensor(x)
        y = torch.LongTensor(y)
        mask = pad_mask(mask, (max_length, max_length))
        dataset.append([x, mask, y])
        
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    

#建立模型
def build_model(vocab, char_dim, pretrain_model_path):
    model = LanguageModel(768, 21128, path)
    return model

#文本生成测试代码
def generate_sentence(openings, model, tokenizer):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        #生成文本超过30字则终止迭代
        while len(openings) <= 50:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)



def main(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 32       #每次训练样本个数
    char_dim = 768        #每个字的维度
    max_length = 50       #样本文本长度
    vocab_size = 21128      #字表大小
    learning_rate = 0.001  #学习率
    

    pretrain_model_path = r'C:\Users\hqw19\Desktop\ai学习\第六周 语言模型\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    corpus = load_corpus(corpus_path)     #加载语料
    train_data = build_dataset(tokenizer, corpus, max_length, batch_size)  #建立数据集
    model = build_model(vocab_size, char_dim, pretrain_model_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for x, mask, y in train_data: #构建一组训练样本
            if torch.cuda.is_available():
                x, mask, y = x.cuda(), mask.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, mask, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return



if __name__ == "__main__":
    main("sample_data.json", False)
