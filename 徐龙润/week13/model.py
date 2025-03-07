# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from peft import LoraConfig,TaskType,get_peft_model
from transformers import AutoModelForTokenClassification,BertTokenizer
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        self.bert = AutoModelForTokenClassification.from_pretrained(config["bert_path"],num_labels=class_num)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-1)
        self.lora_config = LoraConfig(r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"])
        self.layer = get_peft_model(self.bert,self.lora_config)
        for param in self.layer.get_submodule("model").get_submodule("classifier").parameters():
            param.requires_grad = True
    def forward(self, x, target=None):
        x = self.layer(x)["logits"]
        if target != None:
            loss = self.loss_func(x.view(-1,x.shape[-1]),target.view(-1))
            return loss
        return x

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)