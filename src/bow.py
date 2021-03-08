import glove
import numpy


def bow(embeddingmatrix):
    n = len(embeddingmatrix)
    bowvector = 0
    for i in range(len(embeddingmatrix)):
        bowvector += embeddingmatrix[i]

    bowvector = bowvector /n
    # print(bowvector)
    return bowvector
