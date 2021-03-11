import torch
import codecs
import torch.nn as nn

import numpy as np


def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf-8') as f:
        words = []
        word_to_vec_map = {}
        glove_vector = []

        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            curr_vector = np.array(line[1:], dtype=np.float64)
            glove_vector.append(curr_vector)
            words.append(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    lowerwords = [x.lower() for x in words]
    return lowerwords, torch.FloatTensor(glove_vector)


def build_matrix(word_index, path,freeze = True):
    w, embedding_index,glove_vec= read_glove_vecs(path)
    #wordlist = list(w)
    vocabulary_size = len(w)
    vocabulary_dim = glove_vec[0].shape[0]
    print(vocabulary_dim)
    vocabulary_dim = embedding_index['a'].shape[0]
    print(vocabulary_dim)
    embedding = nn.Embedding(vocabulary_size, vocabulary_dim,padding_idx= len(w)-1)
    weight = torch.FloatTensor(glove_vec)
    embedding = embedding.from_pretrained(weight,freeze= freeze)
    tensorlist = []
    for i in range (len(word_index)):
        if word_index[i] in w:
            tensorlist.append(w.index(word_index[i]))
        #if word_index[i] not in w:
            #tensorlist.append(len(w)-1)

    input = torch.LongTensor(tensorlist)
    embedding_matrix = embedding(input)
    return embedding_matrix

