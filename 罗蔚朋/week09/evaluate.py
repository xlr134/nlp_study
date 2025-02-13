# -*- coding: utf-8 -*-
import re
from collections import defaultdict

import numpy as np
import torch.cuda

from loader import load_data

"""
模型效果测试
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.stats_dict = None
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config['valid_data_path'], config, shuffle=False)

    def eval(self, epoch):
        self.logger.info(f"开始测试第{epoch}轮模型效果")
        self.stats_dict = {"LOCATION": defaultdict(int),
                           "TIME": defaultdict(int),
                           "PERSON": defaultdict(int),
                           "ORGANIZATION": defaultdict(int)}
        self.model.eval()
        batch_size = self.config['batch_size']
        for index, batch_data in enumerate(self.valid_data):
            # 原句子
            sentences = self.valid_data.dataset.sentences[index * batch_size:(index + 1) * batch_size]
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data
            with torch.no_grad():
                pred_results = self.model(input_id)
            self.write_stats(labels, pred_results, sentences)
        self.show_stats()
        return

    def write_stats(self, labels, pred_results, sentences):
        assert len(labels) == len(pred_results) == len(sentences)
        if not self.config['use_crf']:
            pred_results = torch.argmax(pred_results, dim=-1)
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config['use_crf']:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            true_entities = self.decode(sentence, true_label)
            pred_entities = self.decode(sentence, pred_label)
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]['正确识别'] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]])
                self.stats_dict[key]['样本实体数'] += len(true_entities[key])
                self.stats_dict[key]['识别出实体数'] += len(pred_entities[key])
        return

    def show_stats(self):
        f1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            f1 = (2 * precision * recall) / (precision + recall + 1e-5)
            f1_scores.append(f1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, f1: %f" % (key, precision, recall, f1))
        self.logger.info("Macro-f1: %f" % np.mean(f1_scores))
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum(
            [self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        self.logger.info("Micro-f1 %f" % micro_f1)
        self.logger.info("--------------------")
        return

    def decode(self, sentence, labels):
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        return results
