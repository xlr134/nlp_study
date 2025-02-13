# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 3,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "pooling_style":"max",
    "learning_rate": 1e-5,
    "use_crf": False,
    "class_num": 10,
    "pretrain_model_path":r"D:\learning\AI\第六周 语言模型\bert-base-chinese",
}

