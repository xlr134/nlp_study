# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "./data/train.json",
    "test_data_path": "./data/test.json",
    "valid_data_path": "./data/valid.json",
    "vocab_path":"./data/chars.txt",
    "model_type":"bert",
    "max_length": 400,#通过评论长度分布图决定最大长度取400
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 200,
    "pooling_style":"max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path":"bert-base-chinese",
    "seed": 987
}

