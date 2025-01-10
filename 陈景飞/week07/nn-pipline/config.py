# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "../data/train_tag_news.json",
    "valid_data_path": "../data/valid_tag_news.json",
    "vocab_path": "chars.txt",
    "model_type": "rnn",
    "max_length": 30,
    "hidden_size": 256,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 2,
    "batch_size": 128,
    "pooling_style": "max",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "pretrain_model_path": r"D:\Chen\Code\python\demo\BaDou\teacherCode\第6周\下午\bert-base-chinese\bert-base-chinese",
    "seed": 987
}
