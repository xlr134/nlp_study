# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
import time
import csv
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from analysis import data_analysis

# [DEBUG, INFO, WARNING, ERROR, CRITICAL]
# logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

seed = Config["seed"]
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def main(config):
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config)
    # 加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    # 训练
    cost_time_record = []
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data  # 输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:
                logger.info("batch loss %f" % loss)
        logger.info("epoch average loss: %f" % np.mean(train_loss))
        start_time = time.time()
        acc = evaluator.eval(epoch)
        cost_time = time.time() - start_time
        cost_time_record.append(cost_time)

    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    avg_cost = np.mean(cost_time_record)
    return acc, avg_cost


def save_output(output_list):
    with open(Config["output_path"], "w", newline='', encoding="utf8") as f:  # 保存结果
        writer = csv.writer(f)
        writer.writerows(output_list)


if __name__ == "__main__":
    # 数据分析
    data_analysis()
    # 对比所有模型
    # 中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    output_list = [["Model", "Learning_Rate", "Hidden_Size", "pooling_style", "acc", "avg_cost"]]
    for model in ["gated_cnn", 'bert', 'lstm']:
        Config["model_type"] = model
        for lr in [1e-3, 1e-5]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg", 'max']:
                        Config["pooling_style"] = pooling_style
                        acc, avg_cost = main(Config)
                        print("最后一轮准确率：", acc, "预测耗时", avg_cost, "当前配置：", Config)
                        output_list.append([model, lr, hidden_size, pooling_style, acc, avg_cost])
    # 保存输出结果
    save_output(output_list)
