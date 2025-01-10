#!/usr/bin/env python3
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
from operator import index

import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
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

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)

# 计算两个向量的欧式距离
def calculate_euclidean_distance(A, B):
    return np.linalg.norm(A - B)

def main():
    model = load_word2vec_model(r"..\model.w2v") #加载词向量模型
    sentences = load_sentence(r"..\titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    label_dict_list = sentence_label_dict.copy()

    center_list = kmeans.cluster_centers_
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_index = list(sentences).index(sentence)
        sentence_vector = vectors[sentence_index]
        label_center = center_list[label]

        label_euclidean = calculate_euclidean_distance(sentence_vector,label_center)
        # 类别,该类别的所有欧式距离
        label_dict_list[label].append(label_euclidean)
        # 类别,该类别的所有文本
        sentence_label_dict[label].append(sentence)         #同标签的放到一起

    mean_group_label ={}
    for label in label_dict_list:
        group_label_list = label_dict_list[label]
        # 该组的平均欧式距离
        mean = np.mean(group_label_list)
        mean_group_label[label] = mean
    # 平均欧式距离的组按从小到大排序
    mean_group_label = sorted(mean_group_label.items(), key=lambda k: k[1], reverse=False)
    for mean_label in mean_group_label:
        label = mean_label[0]
        mean_euclidean = mean_label[1]
        sentences = sentence_label_dict[label]
        print("mean %s : mean_euclidean=%s" % (label,mean_euclidean))
        for i in range(min(10, len(sentences))):  # 随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))

    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")

if __name__ == "__main__":
    main()

