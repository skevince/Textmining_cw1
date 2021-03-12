import torch
import random
from configparser import ConfigParser
import argparse
# from src.main import train, test


torch.manual_seed(1)
random.seed(1)


#
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--config', help='The path of the configuration file',type=str, default='../data/Bag_of_words.config')
parser.add_argument("--train", help="To train the model", action="store_true")
parser.add_argument("--test", help="To test the model", action="store_true")


def get_config(section, para):
    args = parser.parse_args()
    config = ConfigParser()
    config.read(args.config)
    config_get = config.get(section, para)
    return config_get





#
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
#     pass