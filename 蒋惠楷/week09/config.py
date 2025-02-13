# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_type": ["Bert-CRF", "LSTM-CRF", "Bert-LSTM-CRF"], # 
    "model_path": "model_output",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train.txt",
    "valid_data_path": "ner_data/test.txt",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": True,
    "class_num": 9,
    "dropout": 0.3,
    "bidirectional": True,
    "attention_mask": True,
    "bert_path": "E:\\AIGC\\NLP算法\\【9】序列标注\课件\\ner_bert\\ner_bert\\bert-base-chinese"
}
