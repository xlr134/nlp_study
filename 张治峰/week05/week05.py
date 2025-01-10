# 【第五周作业】

# 实现基于kmeans结果类内距离的排序

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


def main():
    model = load_word2vec_model(r"C:\Users\komorebi\Desktop\ai\week05\model.w2v") #加载词向量模型
    sentences = load_sentence(r"C:\Users\komorebi\Desktop\ai\week05\titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    sentence_label_score = defaultdict(float) # 每种标签的分数 ， 计分方式为：同标签内的所有点到中心点的距离 和 / 总数

    for sentence, label,vector in zip(sentences, kmeans.labels_,vectors):  #取出句子和标签 并计算累加标签分数
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
        center_vector =  kmeans.cluster_centers_[label]  # 根据标签获取中心点
        distance = math.sqrt(np.sum((vector-center_vector)**2)) # 计算向量到中心点的距离
        sentence_label_score[label]+=distance #累加相同标签类内元素到中心点的距离
    
    for label,scope in sentence_label_score.items():
        sentence_label_score[label] = scope/len(sentence_label_dict[label]) # 计算 标签类内元素到中心点的距离 平均值

    sentence_label_score = sorted(sentence_label_score.items(), key=lambda item: item[1], reverse=False) # 按 分数排序

    # 打印排名前十的分类
    for index in range(10):
        item = sentence_label_score[index]
        label = item[0]
        print("cluster %s,scope:%s :" % (label,item[1]))
        sentences = sentence_label_dict[label] #回去当前类别的文章 打印前十条
        for i in range(min(10,len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

