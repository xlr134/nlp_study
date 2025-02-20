import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from transformers import BertModel,BertTokenizer

class DataGenerator:
    def __init__(self,path,config):
        self.config = config
        self.max_seq_len = config["max_seq_len"]
        self.tokenizer = BertTokenizer.from_pretrained(r"C:\Users\58353\Desktop\Depp leraing\第十一周 大模型相关内容第一讲\week11 大语言模型相关第一讲\model")
        self.path = path
        self.load()
    def load(self):
        self.data = []
        with open(self.path,encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                title = line["title"]
                encode_t = self.tokenizer(title,add_special_tokens=False)["input_ids"]
                content = line["content"]
                encode_c = self.tokenizer(content,max_length=self.max_seq_len-len(encode_t)-1
                                ,padding='max_length'
                                ,add_special_tokens=False
                                ,truncation=True
                                )["input_ids"]
                encode_c.insert(0,102)
                encode_t_c = encode_t+encode_c
                label = encode_t_c[1:]
                label.append(0)
                for index in range(len(label)):#eos
                    if label[index]==0:
                        label[index] = 10434#eos
                        break
                label[:len(encode_t)-1] = [0]*(len(encode_t)-1)
                if len(encode_t_c)!=300 or len(label)!=300:
                    continue
                self.data.append([ torch.LongTensor(encode_t_c)
                                  ,torch.LongTensor(label)])
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        return self.data[index]

def load_data(data_path,logger,config,shuffle=True):
    data = DataGenerator(data_path,config)
    data = DataLoader(data,batch_size=config["batch_size"],shuffle=shuffle)
    return data
