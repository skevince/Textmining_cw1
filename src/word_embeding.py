import torch
import re
import random
import torch.nn as nn


def clean_stopwords(sentence):
    new_sentence = sentence
    stopwords = set()
    with open('/Users/kai/Desktop/Textmining/tm_cw1/data/stopwords.txt', 'r') as f:
        for stopword in f.readlines():
            stopword=stopword.rstrip("\n")
            stopwords.add(stopword)
    for word in sentence.split():
        if word in stopwords:
            new_sentence = re.sub(word, "", new_sentence)
        if word.isdigit():
            new_sentence = re.sub(word, "", new_sentence)
    new_sentence = re.sub(r"\s+", " ", new_sentence)
    return new_sentence


def get_voc(filepath):
    vocabulary = {}
    voc_set = set()
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
                if word_statistic[word] > 3:
                    voc_set.add(word)
    for i in range(len(voc_set)):
        vocabulary[voc_set[i]] = i
    return vocabulary


def get_vector(filepath, vocabulary):
    vector = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            sentence = []
            void_stopwords = clean_stopwords(line.split(" ", 1)[1])
            for word in void_stopwords.split():
                sentence.append(vocabulary[word])
            vector.append(sentence)
    return vector


def randomly_embedding(filepath, dimension):
    vocabulary = get_voc(filepath)
    vector = get_vector(filepath, vocabulary)
    torch.manual_seed(1)
    embedding = nn.Embedding(len(vocabulary), dimension)
    label = torch.LongTensor(vector)
    initialize = embedding(label)
    return initialize


if __name__ == '__main__':
    print("Randomly initialised word embeddings")
    initial_embedding = randomly_embedding('.././data/train.txt', 1000)
