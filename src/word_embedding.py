import torch
import re
import numpy as np
import random
import torch.nn as nn

np.set_printoptions(threshold=np.inf)

def clean_stopwords(sentence):
    new_sentence = sentence
    stopwords = set()
    with open('.././data/stopwords.txt', 'r') as f:
        for stopword in f.readlines():
            stopword=stopword.rstrip("\n")
            stopwords.add(stopword)
    for word in sentence.split():
        if word in stopwords:
            new_sentence=new_sentence.replace(word, '')
            new_sentence = re.sub(r"\s+", " ", new_sentence)
        if word.isdigit():
            new_sentence=new_sentence.replace(word, '')
            new_sentence = re.sub(r"\s+", " ", new_sentence)
    return new_sentence


def get_voc(filepath):
    vocabulary = []
    word_statistic = {}
    with open(filepath, 'r') as f:
        for line in f.readlines():
            void_stopwords = clean_stopwords(line.split(" ", 1)[1])
            for word in void_stopwords.split():
                if word in word_statistic:
                    word_statistic[word] += 1
                else:
                    word_statistic[word] = 1
    for word in word_statistic:
        if word_statistic[word] > 2:
            vocabulary.append(word)
    return vocabulary


def get_vector(filepath, vocabulary):
    word_list=[]
    vector = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            void_stopwords = line.split(" ", 1)[1]
            pre_vec=np.zeros(len(vocabulary))
            for word in void_stopwords.split():
                if word in vocabulary:
                    pre_vec[vocabulary.index(word)]=pre_vec[vocabulary.index(word)]+1
            for word in void_stopwords.split():
                vec_temp=pre_vec
                if word in vocabulary:
                    vec_temp[vocabulary.index(word)]=0
                if word not in word_list:
                    word_list.append(word)
                    vector.append(vec_temp)
                else:
                    vector[word_list.index(word)]=vector[word_list.index(word)]+vec_temp
    return word_list,vector


def randomly_embedding(filepath, dimension):
    vocabulary = get_voc(filepath)
    word_list,vector = get_vector(filepath, vocabulary)
    torch.manual_seed(1)
    embedding = nn.Embedding(len(vocabulary), dimension)
    label = torch.LongTensor(vector)
    initialize = embedding(label)
    return initialize


if __name__ == '__main__':
    print("Randomly initialised word embeddings")
    initial_embedding = randomly_embedding('.././data/train.txt', 1000)
    print(initial_embedding[0])