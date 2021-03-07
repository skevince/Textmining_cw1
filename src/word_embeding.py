import torch
import re
import random
import torch.nn as nn

torch.manual_seed(1)


def clean_stopwords(sentence):
    new_sentence = ''
    stopwords = set()
    with open ('.././data/stopwords.txt', 'r') as f:
        for stopword in f.readlines():
            stopwords.add(stopword)
    for word in sentence.split():
        if word in stopwords:
            new_sentence = re.sub(word, '', sentence)
    return new_sentence


def get_voc(filepath):
    vocabulary = {}
    voc_set = set()
    with open (filepath, 'r') as f:
        for line in f.readlines():
            void_stopwords = clean_stopwords(line.split(" ", 1)[1])
            for word in void_stopwords.split():
