import torch
import random
from configparser import ConfigParser
import argparse

torch.manual_seed(1)
random.seed(1)

config = ConfigParser()
config.sections()
config.read( ".././data/Bag_of_words.config" )
print (config.keys())
path_train = config[ "PATH" ][ "path_train" ]
path_dev = config[ "PATH" ][ "path_dev" ]
path_test = config[ "PATH" ][ "path_test" ]

print(path_train)
print(path_dev)

# parser = argparse.ArgumentParser(description='manual to this script')
# parser.add_argument('--config', help='The path of the configuration file',type=str, default='.././data/Bag_of_words.config')
# parser.add_argument("--train", help="To train the model", action="store_true")
# parser.add_argument("--test", help="To test the model", action="store_true")
#
# args = parser.parse_args()



# if(args.train):
#     # do the train function
#     print('train')
#
#
# if(args.test):
#     # do the test function
#     print('test')
#
# if __name__ == '__main__':
#     # print(conf.get('param', 'model'))
#     pass