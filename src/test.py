import torch
import random
from configparser import ConfigParser
import argparse

torch.manual_seed(1)
random.seed(1)

config = ConfigParser()
config.sections()
# config.read( ".././data/Bag_of_words.config" )


parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--config', help='The path of the configuration file',type=str, default='../data/Bag_of_words.config')
parser.add_argument("--train", help="To train the model", action="store_true")
parser.add_argument("--test", help="To test the model", action="store_true")

args = parser.parse_args()

config = ConfigParser()
config.read(args.config)



if(args.train):
    # do the train function
    print('train')


if(args.test):
    # do the test function
    print('test')

if __name__ == '__main__':
    pass