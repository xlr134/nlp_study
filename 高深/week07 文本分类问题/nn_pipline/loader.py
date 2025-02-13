# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: 0, 1: 1, }
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                tag = line["label"]
                label = self.label_to_index[tag]
                title = line["review"]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(title)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config

    # check if Config["train_data_path"] and Config["valid_data_path"]  exists
    if not os.path.exists(Config["train_data_path"]) or not os.path.exists(Config["valid_data_path"]):
        print("train valid split not exist, creating...")
        
        import pandas as pd
        df = pd.read_csv("../data/文本分类练习.csv")
        df = df.sample(frac=1).reset_index(drop=True) # shuffle rows
        train_percent = 0.8 # train = 80% of sample
        train_size, valid_size = int(len(df) * train_percent), int(len(df) * (1 - train_percent))
        df_train = df.iloc[:train_size].to_json(Config["train_data_path"], orient="records", lines=True, force_ascii=False) # pandas defaults conversion to ascii. we want utf-8 to preserve chinese characters
        df_valid = df.iloc[:valid_size].to_json(Config["valid_data_path"], orient="records", lines=True, force_ascii=False)
    
    # dg = DataGenerator("../data/文本分类练习.csv", Config)
    # print(dg[1])
