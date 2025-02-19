#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertModel
from transformers import BertTokenizer
"""
基于pytorch的LSTM语言模型
"""


class LanguageModel(nn.Module):
    def __init__(self, bert_path):
        super(LanguageModel, self).__init__()
        self.layer = BertModel.from_pretrained(bert_path, return_dict=False)
        self.classify = nn.Linear(768, 21128)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, mask,y=None):
        x,_= self.layer(x,attention_mask=mask)        #output shape:(batch_size, sen_len, 768)
        y_pred = self.classify(x)   #output shape:(batch_size, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)


#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

#随机生成一个样本
#从文本中截取随机窗口，前n个字作为输入，最后一个字作为输出
def build_sample( window_size, corpus,tokenizer):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  #输入输出错开一位
    x = [tokenizer.vocab.get(word, tokenizer.vocab.get("[UNK]")) for word in window]  # 将字转换成序号
    y = [tokenizer.vocab.get(word, tokenizer.vocab.get("[UNK]")) for word in target]
    return x, y

#建立数据集
def build_dataset(sample_length,window_size, corpus,tokenizer):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(window_size, corpus,tokenizer)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model( bert_path):
    model = LanguageModel(bert_path)
    return model

#文本生成测试代码
def generate_sentence(openings, model,  window_size,tokenizer):
    model.eval()
    with torch.no_grad():
        pred_char = ""

        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "\n" and len(openings) <= 30:
            openings += pred_char

            x = [tokenizer.vocab.get(word, tokenizer.vocab.get("[UNK]")) for word in openings[-window_size:]]  # 将字转换成序号
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x,None)[0][-1]
            index = sampling_strategy(y)
            pred_char =  tokenizer.convert_ids_to_tokens(index)
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.2:
        strategy = "greedy"
    else:
        strategy = "sampling"
    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return int(np.random.choice(list(range(len(prob_distribution))), p=prob_distribution))


#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 20        #训练轮数
    batch_size = 256       #每次训练样本个数
    train_sample = 10000   #每轮训练总共训练的样本总数
    window_size = 10       #样本文本长度
    bert_path = r"/Volumes/komorebi/model/bert-base-chinese"
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model(bert_path)    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)   #建立优化器
    mask = generate_causal_mask(window_size).unsqueeze(0).expand(batch_size, -1, -1)
    tokenizer = BertTokenizer.from_pretrained(bert_path)
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, window_size, corpus,tokenizer) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x,mask, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model,  window_size,tokenizer))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model,  window_size,tokenizer))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

def generate_causal_mask(seq_len):
    mask = torch.triu(torch.ones(seq_len, seq_len))
    return mask

if __name__ == "__main__":
    train("corpus.txt", False)
