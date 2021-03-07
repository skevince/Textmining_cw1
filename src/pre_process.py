import re
from spacy.lang.en import English
from spacy.pipeline import Sentencizer
from spacy.tokenizer import Tokenizer
import collections
'''
tokenisation the data

'''


def preprocessdata (filename):
    train = []
    with open(filename, 'r', encoding='utf-8') as f:
        lines = [re.sub(' [^a-z] +', ' ', line.strip().lower()) for line in f]

    train = [sentence.split(' ') for sentence in lines]
    for i in range(len(train)):
        train[i] = train[i][0:len(train[i]) - 1]
    return train


'''
    if token == 'word':
        return [sentence.split(' ') for sentence in lines]
    #elif token == 'char':
        #return [list(sentence) for sentence in lines]
    else:
        print('ERROR: unkown token type ' + token)
'''

# nlp = English()
# # Create a tokeniser with just the default vocabulary
# tokenizer = Tokenizer(nlp.vocab)






if __name__ == '__main__':
    trains = preprocessdata('.././data/train_5500.txt')
    print(trains[0])
