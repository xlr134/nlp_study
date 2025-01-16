'''
Author: Zhao
Date: 2025-01-14 19:54:13
LastEditTime: 2025-01-15 18:23:25
FilePath: config.py
Description: 

'''
Config = {
    "model_path": "week8/model_output",
    "schema_path": "week8/data/schema.json",
    "train_data_path": "week8/data/train.json",
    "valid_data_path": "week8/data/valid.json",
    "vocab_path":"week8/data/chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
}