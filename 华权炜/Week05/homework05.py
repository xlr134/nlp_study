#!/usr/bin/env python3  
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
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
    
#计算欧式距离
def Euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum(np.square(vector1 - vector2)))


def main():
    model = load_word2vec_model(r"C:\Users\hqw19\Desktop\ai学习\第五周 词向量\week5 词向量及文本向量\model.w2v") #加载词向量模型
    sentences = load_sentence(r"C:\Users\hqw19\Desktop\ai学习\第五周 词向量\week5 词向量及文本向量\titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算


    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        sentence_label_dict[label].append(sentence)         #同标签的放到一起


    

    #计算类内距离
    intra_distance_dict = defaultdict(list)
    for index, label in enumerate(kmeans.labels_):
        vector = vectors[index]   #某句话的向量
        center = kmeans.cluster_centers_[label]  #对应的类别中心向量
        distance = Euclidean_distance(vector, center)  #计算距离
        intra_distance_dict[label].append(distance)    #保存下来
    for label, distance_list in intra_distance_dict.items():
        intra_distance_dict[label] = np.mean(distance_list)   #对于每一类，将类内所有文本到中心的距离取平均
    distance_order = sorted(intra_distance_dict.items(),key=lambda x:x[1], reverse=True)       
    print(distance_order)

    #按照距离顺序输出
    for label, distance_avg in distance_order:
        print("cluster %s , avg distance %f: " % (label, distance_avg))
        sentences = sentence_label_dict[label]
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            print(sentences[i].replace(" ", ""))
        print("---------")




if __name__ == "__main__":
    main()
