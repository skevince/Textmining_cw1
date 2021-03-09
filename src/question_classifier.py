import configparser
from src.bilstm import BiLSTMTagger
import configparser


config = configparser.ConfigParser()
config.sections()
config.read( "config.ini" )
print (config.keys())
path_train = config[ "PATH" ][ "path_train" ]
path_dev = config[ "PATH" ][ "path_dev" ]
path_test = config[ "PATH" ][ "path_test" ]