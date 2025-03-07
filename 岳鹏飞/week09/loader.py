# -*- coding: utf-8 -*-

import json
import re
import os
from distutils.command.config import config

import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                # labels = [] # 配合add_special_tokens=False参数，可以去掉cls token
                labels = [-1] # 对齐cls token.
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])
                sentence = "".join(sentenece)
                self.sentences.append(sentence)
                # input_ids = self.tokenizer.encode(sentence,
                #                                   max_length=self.config["max_length"],
                #                                   pad_to_max_length=True,
                #                                   add_special_tokens=False # 去掉cls sep token
                #                                   )
                input_ids = self.tokenizer.encode(sentence,
                                                  max_length=self.config["max_length"],
                                                  pad_to_max_length=True)
                labels = self.padding(labels, -1)
                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    #补齐或截断输入的序列，使其可以在一个batch内运算
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

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("ner_data/train", Config)
    dl = DataLoader(dg, batch_size=32)  
    for x,y in dl:
        print(x, y)