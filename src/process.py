from src.question_classifier import get_config

import argparse

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--config', help='The path of the configuration file',type=str, default='../data/Bag_of_words.config')
parser.add_argument("--train", help="To train the model", action="store_true")
parser.add_argument("--test", help="To test the model", action="store_true")

path_stopwords = get_config('PATH','path_stopwords')
print(path_stopwords)


def processfile (filename,outfilename):
    stopwords = set()
    with open(path_stopwords, 'r') as f:
        for stopword in f.readlines():
            stopword = stopword.rstrip("\n")
            stopwords.add(stopword)

    with open(filename, 'r', encoding='utf-8') as f:
        outstr = ""
        for line in f:
            line = line.strip().split()
            for i in range (len(line)):
                if line[i] not in stopwords:
                    outstr += line[i] + " "
            outstr += "\n"
    outputs = open(outfilename,'w', encoding="utf8")
    outputs.write(outstr)





