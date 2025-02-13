#!/usr/bin/env python3  
# coding: utf-8

# 基于训练好的词向量模型进行聚类
# 聚类采用KMeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


# 计算两点之间的欧氏距离
def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def main():
    model = load_word2vec_model(r"F:\BaiduNetdiskDownload\week5 词向量及文本向量\model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  # 取出句子和标签
        sentence_label_dict[label].append(sentence)  # 同标签的放到一起

    # 对每个类内部的句子按距离排序
    for label, sentences_in_cluster in sentence_label_dict.items():
        print(f"Cluster {label} (Total {len(sentences_in_cluster)} sentences):")

        # 获取该类的中心（质心）
        cluster_center = kmeans.cluster_centers_[label]

        # 计算每个句子到质心的欧氏距离，并存储
        sentence_distance = []
        for sentence in sentences_in_cluster:
            sentence_vector = sentences_to_vectors([sentence], model)[0]
            distance = euclidean_distance(sentence_vector, cluster_center)
            sentence_distance.append((sentence, distance))

        # 按照距离从小到大排序
        sorted_sentences = sorted(sentence_distance, key=lambda x: x[1])

        # 打印排序后的前10个句子（距离最小的）
        for i in range(min(10, len(sorted_sentences))):
            print(f"Distance: {sorted_sentences[i][1]:.4f} - {sorted_sentences[i][0].replace(' ', '')}")

        print("---------")


if __name__ == "__main__":
    main()
