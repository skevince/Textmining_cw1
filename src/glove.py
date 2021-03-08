import numpy as np
import torch
import codecs

file1 = open(".././data/train.txt")
textdata = file1.read()

file2 = open(".././data/glove.small.txt", 'r', encoding='utf-8')
glovedata = file2.read()


def read_glove_vecs(glove_file):
    with open(glove_file, 'r', encoding='utf-8') as f:
        words = set()
        word_to_vec_map = {}

        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

    return words, word_to_vec_map


def build_matrix(word_index, path):
    w, embedding_index = read_glove_vecs(path)
    embedding_matrix = np.zeros((len(word_index), 50))
    for i in range(len(word_index)):
        try:
            embedding_matrix[i] = embedding_index[word_index[i]]
        except KeyError:
            pass
    return embedding_matrix
