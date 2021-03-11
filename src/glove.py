import torch
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
