import re
import numpy as np
import random

np.set_printoptions(threshold=np.inf)


def clean_stopwords(sentence):
    new_sentence = sentence
    stopwords = set()
    with open('D://Users\AW\Documents\GitHub\Textmining_cw1/data/stopwords.txt', 'r') as f:
        for stopword in f.readlines():
            stopword = stopword.rstrip("\n")
            stopwords.add(stopword)
    for word in sentence.split():
        if word in stopwords:
            new_sentence = new_sentence.replace(word, '')
            new_sentence = re.sub(r"\s+", " ", new_sentence)
        if word.isdigit():
            new_sentence = new_sentence.replace(word, '')
            new_sentence = re.sub(r"\s+", " ", new_sentence)
    return new_sentence


def randomly_embedding(filepath):
    wordlist = []
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
        wordlist.append(word)
        if word_statistic[word] > 10:
            vocabulary.append(word)
    vocabulary.append('PAD')
    return wordlist, vocabulary
