'''
Author: Zhao
Date: 2025-01-08 18:43:51
LastEditTime: 2025-01-09 15:24:50
FilePath: model.py
Description: 

'''
'''
Author: Zhao
Date: 2025-01-08 15:43:51
LastEditors: Please set LastEditors
LastEditTime: 2025-01-09 09:23:10
FilePath: model.py
Description: 建立网络模型结构

'''
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel


class TorchModel(nn.Module):
    def __init__(self, config):
        """ 
        初始化模型，根据配置选择不同的编码器 
        :param config: 配置字典，包含各种超参数和路径 
        """
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.pooling_style = config["pooling_style"]
        self.useBert = False

        # 嵌入层
        self.embedding = nn.Embedding(num_embeddings=vocab_size,embedding_dim=hidden_size,padding_idx=0)

        # 根据模型类型选择编码器
        if model_type == "fast_text":
            self.encoder = lambda x:x
        elif model_type == "lstm":
            self.encoder = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        elif model_type == "cnn2":
            self.encoder = CNN2(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "stack_gated_cnn":
            self.encoder = StackGatedCNN(config)
        elif model_type == "rcnn":
            self.encoder = RCNN(config)
        elif model_type == "bert":
            self.useBert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict = False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_lstm":
            self.useBert = True
            self.encoder = BertLSTM(config)
        elif model_type == "bert_cnn":
            self.useBert = True
            self.encoder = BertCNN(config)
        elif model_type == "bert_mid_layer":
            self.useBert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size

        # 分类层
        self.classify = nn.Linear(in_features=hidden_size, out_features=class_num)

        # 损失函数
        self.loss = nn.functional.cross_entropy #loss采用交叉熵损失
        
    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target = None):
        if self.useBert:
        ## bert返回的结果是 (sequence_output, pooler_output)
            x = self.encoder(x)
        else:
            x = self.embedding(x)
            x = self.encoder(x)
        
        if isinstance(x, tuple): #RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]

        #可以采用pooling的方式得到句向量
        if self.pooling_style == "max":
            self.pooling_style = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_style = nn.AvgPool1d(x.shape[1])
        x = self.pooling_style(x.transpose(1, 2)).squeeze() #input shape:(batch_size, sen_len, input_dim)

        #也可以直接使用序列最后一个位置的向量
        # x = x[:, -1, :]

        predict = self.classify(x) #input shape:(batch_size, input_dim)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict

# 定义CNN模型       
class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        in_channels = config["in_channels"]
        out_channels = config["out_channels"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1)/2)
        self.cnn = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, padding=pad)

    def forward(self, x): #x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)

# 定义CNN2模型       
class CNN2(nn.Module):
    def __init__(self, config):
        super(CNN2, self).__init__()
        in_channels = config["in_channels"]
        out_channels = config["out_channels"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) // 2)
        self.cnn = nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(kernel_size,kernel_size),bias=False, padding=(pad,pad))

    def forward(self, x): # x : (batch_size, max_len, embedding_size)
        return self.cnn(x)

# 定义GatedCNN模型
class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN,self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)
    
    def forward(self, x): 
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)

# 定义StackGatedCNN模型
class StackGatedCNN(nn.Module):
    def __init__(self, config):
        super(StackGatedCNN,self).__init__()
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        #ModuleList类内可以放置多个模型，取用时类似于一个列表
        self.gcnn_layers = nn.ModuleList(
            GatedCNN(config) for i in range(self.num_layers)
        )
        self.ff_liner_layers1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.ff_liner_layers2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )
    
    def forward(self, x): 
        for i in range(self.num_layers):
            gcnn_x = self.gcnn_layers[i](x)
            x = gcnn_x + x                      #通过gcnn+残差
            x = self.bn_after_gcnn[i](x)        #之后bn
            # 仿照feed-forward层，使用两个线性层
            l1 = self.ff_liner_layers1[i](x)    #一层线性
            l1 = torch.relu(l1)                 #在bert中这里是gelu
            l2 = self.ff_liner_layers2[i](l1)   #二层线性
            x = self.bn_after_ff[i](l2 + x)     #残差后过bn2
        return x

# 定义RCNN模型
class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN,self).__init__()
        hidden_size = config["hidden_size"]
        self.rnn = nn.RNN(input_size=hidden_size,hidden_size=hidden_size)
        self.gate = GatedCNN(config)
    
    def forward(self, x): 
        x,_ = self.rnn(x)
        x = self.gate(x)
        return x
    
# 定义BertLSTM模型
class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM,self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.rnn = nn.LSTM(input_size=self.bert.config.hidden_size,hidden_size=self.bert.config.hidden_size,batch_first=True)
    
    def forward(self, x): 
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x

# 定义BertCNN模型
class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN,self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        #config["hidden_size"] = self.bert.config.hidden_size
        self.cnn = CNN(config)
    
    def forward(self, x): 
        x = self.bert(x)[0]
        x = self.cnn(x)
        return x

# 定义BertMidLayer模型    
class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer,self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.bert.config.output_hidden_states = True
    
    def forward(self, x): 
        layer_states = self.bert(x)[2]
        #使用 torch.add 方法，将倒数第二层和最后一层的隐藏状态逐元素相加，结果存储在 layer_states 中
        layer_states = torch.add(layer_states[-2], layer_states[-1]) 
        return layer_states

#优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    # 定义adam和sgd
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    
# if __name__ == "__main__":
#     from config import Config
#     import numpy as np
#     #Config["class_num"] = 3
#     Config["hidden_size"] = 3
#     Config["kernel_size"] = 3
#     #Config["model_type"] = "bert"
#     Config["vocab_size"] = 1000
#     Config["num_layers"] = 3
#     Config["in_channels"] = 768
#     Config["out_channels"] = 3
#     Config["model_type"] = "bert_mid_layer"

#     # 创建 TorchModel 实例 
#     torch_model = TorchModel(Config) 

#     torch_w = torch_model.state_dict()
#     #print(torch_w)
#     # 生成一些测试数据
#     batch_size = 3
#     max_len = 3
#     embedding_size = Config["hidden_size"]
#     # 生成一些随机的浮点数数据，形状为 (batch_size, max_len, embedding_size) 
#     test_data = torch.randint(0, Config["vocab_size"], (batch_size, max_len), dtype=torch.long)
#     print("输入数据形状:", test_data.shape)
#     print("输入数据:", test_data)
#     output = torch_model(test_data)
#     print("输出数据:", output)