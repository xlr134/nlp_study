import torch
import torch.nn as nn
import torch.optim.optimizer
import numpy as np
import os
import time
import logging
from config import Config
from load import load_data
from transformers import BertModel,BertTokenizer

logging.basicConfig(level=logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class Model(nn.Module):
    def __init__(self, config,*args, **kwargs):
        super(Model,self).__init__(*args, **kwargs)
        self.config = config
        self.layer = BertModel.from_pretrained(r"C:\Users\58353\Desktop\Depp leraing\第十一周 大模型相关内容第一讲\week11 大语言模型相关第一讲\model")
        self.tokenizer = BertTokenizer.from_pretrained(r"C:\Users\58353\Desktop\Depp leraing\第十一周 大模型相关内容第一讲\week11 大语言模型相关第一讲\model")
        self.ids_to_tokens = self.tokenizer.ids_to_tokens
        self.linear = nn.Linear(768,21128)
    def forward(self,x):
        x = self.layer(x)["last_hidden_state"]
        x = self.linear(x)
        return x
    def decoder(self,index):
        return self.ids_to_tokens[index]
    def encoder(self,question):
        id = self.tokenizer(question,add_special_tokens=False)["input_ids"]
        return id
    def Generate(self,question):
        id = self.encoder(question)
        id.append(102)#sep
        for i in range(300):
            Tensor_id = torch.LongTensor(id).unsqueeze(0) 
            x = self.forward(Tensor_id)
            x = (x[0,-1,:]).reshape(1,-1)
            index = x.argmax(-1)
            index = index.item()
            if index==10434:#eos
                break
            c = self.decoder(index)
            id.append(index)
            question+=c
        print(question)


def choose_optimizer(config,model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer=="adam":
        return torch.optim.Adam(model.parameters(),lr=learning_rate)
    elif optimizer=="sgd":
        return torch.optim.SGD(model.parameters(),lr=learning_rate)

def main(config):
    if not os.path.isdir(config["check_point_path"]):
        os.mkdir(config["check_point_path"])
    
    model = Model(config)
    optimizer = choose_optimizer(config,model)
    train_data = load_data(config["data_path"],logger,config)
    loss_func = nn.CrossEntropyLoss(ignore_index=0)
    for epoch in range(config["epoch"]):
        model.train()
        train_loss = []
        logger.info("epoch %d begin" % epoch)
        for d in train_data:
            input_data,target_seq = d
            target_seq = target_seq.reshape(-1)
            pred = model(input_data)
            pred = pred.reshape(pred.shape[0]*pred.shape[1],-1)
            loss = loss_func(pred,target_seq)
            train_loss.append(float(loss))
            loss.backward() 
            optimizer.step()
            optimizer.zero_grad()
            break
        model.Generate("少女峰脚下 还藏着这样一处风景")
        logger.info("epoch average loss: %f" % np.mean(train_loss))
    torch.save(model,config["check_point_path"])
    
        

if __name__=="__main__":
    main(Config)
