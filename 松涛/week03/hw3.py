import torch
import torch.nn as nn
import numpy as np
import random
import json

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.RNN = nn.RNN(vector_dim, vector_dim, bias=False, batch_first=True)
        self.pool = nn.AvgPool1d(sentence_length)
        self.classify = nn.Linear(vector_dim, sentence_length)
        self.loss = nn.functional.cross_entropy
        
    def forward(self, x, y=None):
        x = self.embedding(x)               #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = self.RNN(x)[0]                  #(batch_size, sen_len, vector_dim) -> (batch_size, sen_len, vector_dim)
        x = x.transpose(1, 2)               #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        x = self.pool(x)                    #(batch_size, vector_dim, sen_len) -> (batch_size, vector_dim, 1)
        x = x.squeeze()                     #(batch_size, vector_dim, 1) -> (batch_size, vector_dim)
        x = self.classify(x)                #(batch_size, vector_dim) -> (batch_size, sen_len)
        y_pred = x
        if y is not None:
            return self.loss(y_pred, y.squeeze().long())
        else:
            return y_pred
        
def build_vocab():
    '''
    特定字符设为感叹号：!
    '''
    chars = "你我他defghijklmnopqrstuvwxyz!"
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length):
    '''
    特定字符设为感叹号：!
    '''
    x = []
    
    while len(x) < sentence_length - 1:
        to_be_appended = random.choice(list(vocab.keys()))
        if to_be_appended != '!':
            x.append(to_be_appended)
            
    index = random.randint(0, sentence_length-1)
    x.insert(index, '!') # insert '!' at random position     
    y = index

    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            if torch.argmax(y_p, dim=0) == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    #保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)      # 建立模型
    model.load_state_dict(torch.load(model_path))              # 加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab.get(char, vocab['unk']) for char in input_string])  # 将输入序列化
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        logits = model.forward(torch.LongTensor(x))  # 模型预测
        probs = torch.softmax(logits, dim=1)         # 转换为概率值
        preds = torch.argmax(probs, dim=1)           # 获取预测类别
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%s" % (input_string, preds[i].item(), probs[i].tolist()))  # 打印结果



if __name__ == "__main__":
    main()
    test_strings = ["fnvf!e", "!z你dfg", "rqwde!", "n我!www"]
    predict("model.pth", "vocab.json", test_strings)
