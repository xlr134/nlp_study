# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.data = None
        self.config = config
        self.path = data_path
        self.use_bert = config["use_bert"]
        self.vocab = load_vocab(config)
        self.config["vocab_size"] = len(self.vocab)
        self.tokenizer = load_vocab(config) if self.use_bert else None
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                # labels = []
                labels = [] if not self.use_bert else [101]  # [101]表示[CLS] [102]表示[SEP]
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])

                self.sentences.append("".join(sentenece))
                input_ids = self.encode_sentence(sentenece)

                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    def encode_sentence(self, text, padding=True):
        if self.use_bert:
            if isinstance(text, list):
                text = " ".join(text)
            return self.tokenizer.encode(
                text, padding="max_length", max_length=self.config["max_length"], truncation=True
            )
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


# 加载字表或词表
def load_vocab(config):
    if config['use_bert']:
        return BertTokenizer.from_pretrained(config["bert_path"])
    vocab_path = config["vocab_path"]
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    # tokenizer = BertTokenizer.from_pretrained(Config["bert_path"])
    # texts = ['上午好', '你好', '早上好']
    # for text in texts:
    #     print(tokenizer.encode(text))
    # encoded = tokenizer.encode('你好')
    # print(encoded)
    dg = DataGenerator("../ner_data/train.txt", Config)
