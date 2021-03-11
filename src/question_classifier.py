import configparser
from src.bilstm import BiLSTMTagger
from src.word_embedding import initial_embedding
from src.glove import build_matrix
import configparser


config = configparser.ConfigParser()
config.sections()
config.read( "config.ini" )
print (config.keys())
path_train = config[ "PATH" ][ "path_train" ]
path_dev = config[ "PATH" ][ "path_dev" ]
path_test = config[ "PATH" ][ "path_test" ]


def input_vector(embedding):
    if embedding == 'randomly':
        return initial_embedding()
    if embedding == 'pre_train':
        return build_matrix()