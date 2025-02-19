import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertModel, BertTokenizer


class BertLanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, bert_model_path):
        super(BertLanguageModel, self).__init__()
        # 加载预训练的BERT模型
        self.bert = BertModel.from_pretrained(bert_model_path, return_dict=False)
        self.classify = nn.Linear(self.bert.config.hidden_size, vocab_size)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, attention_mask=None, y=None):
        # 获取BERT的输出
        output = self.bert(input_ids=x, attention_mask=attention_mask)
        last_hidden_state = output[0]  # [batch_size, seq_len, hidden_dim]

        # 分类层
        logits = self.classify(last_hidden_state)  # [batch_size, seq_len, vocab_size]

        if y is not None:
            return self.loss(logits.view(-1, logits.shape[-1]), y.view(-1))  # 计算交叉熵损失
        else:
            return torch.softmax(logits, dim=-1)  # 计算生成的softmax概率分布


# 加载字表
def build_vocab(vocab_path):
    vocab = {"<pad>": 0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]  # 去掉结尾换行符
            vocab[char] = index + 1  # 留出0位给pad token
    return vocab


# 加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


# 随机生成一个样本
def build_sample(vocab, window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]  # 输入输出错开一位
    x = [vocab.get(word, vocab["<UNK>"]) for word in window]  # 将字转换成序号
    y = [vocab.get(word, vocab["<UNK>"]) for word in target]
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, hidden_dim, bert_model_path):
    model = BertLanguageModel(len(vocab), hidden_dim, bert_model_path)
    return model


# 文本生成测试代码
def generate_sentence(openings, model, vocab, window_size, tokenizer, max_len=30):
    reverse_vocab = {v: k for k, v in vocab.items()}
    model.eval()
    with torch.no_grad():
        pred_char = ""
        while pred_char != "\n" and len(openings) < max_len:
            openings += pred_char
            x = [vocab.get(char, vocab["<UNK>"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            attention_mask = generate_attention_mask(x)
            # print('attention_mask', attention_mask)
            if torch.cuda.is_available():
                x = x.cuda()
                attention_mask = attention_mask.cuda()
            y = model(x, attention_mask=attention_mask)  # 预测下一个字符
            prob = y[0, -1]  # 取最后一个token的概率分布
            index = sampling_strategy(prob)
            pred_char = reverse_vocab[index]
    return openings



def generate_attention_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    print("subsequent_mask:", subsequent_mask)
    return subsequent_mask



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


# 计算文本ppl
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
    return 2 ** (prob * (-1 / len(sentence)))


# 训练函数
def train(corpus_path, vocab_path, bert_model_path, save_weight=True):
    epoch_num = 20  # 训练轮数
    batch_size = 64  # 每次训练样本个数
    train_sample = 50000  # 每轮训练总共训练的样本总数
    hidden_dim = 256  # 每个字的维度
    window_size = 10  # 样本文本长度
    vocab = build_vocab(vocab_path)  # 建立字表
    corpus = load_corpus(corpus_path)  # 加载语料
    model = build_model(vocab, hidden_dim, bert_model_path)  # 建立模型
    tokenizer = BertTokenizer.from_pretrained(bert_model_path)  # 初始化tokenizer

    if torch.cuda.is_available():
        model = model.cuda()

    optim = torch.optim.Adam(model.parameters(), lr=1e-5)  # 建立优化器
    print("文本词表模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, window_size, corpus)  # 构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            attention_mask = (x != 0).float()  # attention mask
            optim.zero_grad()  # 梯度归零
            loss = model(x, attention_mask=attention_mask, y=y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, vocab, window_size, tokenizer))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, vocab, window_size, tokenizer))

    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return


if __name__ == "__main__":
    train("./corpus.txt", "./vocab.txt", r"D:\AI\homeWork\pretrain_models\bert-base-chinese", save_weight=False)
