from configparser import ConfigParser
import argparse

#
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--config', help='The path of the configuration file', type=str, default='../data/model.config')
parser.add_argument("--train", help="To train the model", action="store_true")
parser.add_argument("--test", help="To test the model", action="store_true")

parser.add_argument("--bow", help="To use bow model", action="store_true")
parser.add_argument("--bilstm", help="To use bilstm model", action="store_true")
parser.add_argument("--bowandlstm", help="To use bow+bilstm model", action="store_true")

parser.add_argument("--random", help="To use random initialization", action="store_true")
parser.add_argument("--glove", help="To use glove", action="store_true")

parser.add_argument("--freeze", help="To freeze the embedding layer", action="store_true")
parser.add_argument("--finetune", help="To finetune the embedding layer", action="store_true")

parser.add_argument("--confu_matrix", help="To finetune the embedding layer", action="store_true")

parser.add_argument("--ensemble", help="To test ensemble model", action="store_true")

args = parser.parse_args()
config = ConfigParser()
config.read(args.config)


def get_config(section, para):
    config_get = config.get(section, para)
    return config_get


if __name__ == '__main__':

    from utl import train, test, ensemble, bowandlstm_train

    if (args.train):
        # do the train function
        print('train')
        if (args.random):
            if_glove = False
        if (args.glove):
            if_glove = True
        if (args.bow):
            if_biLSTM = False
        if (args.bilstm):
            if_biLSTM = True
        if (args.freeze):
            if_freeze = True
        if (args.finetune):
            if_freeze = False
        if (args.bowandlstm):
            bowandlstm_train(glove=if_glove, freeze=if_freeze)
        else:
            train(glove=if_glove, biLSTM=if_biLSTM, freeze=if_freeze)

    if (args.test):
        # do the test function
        print('test')
        confu_matrix = False
        if (args.confu_matrix):
            confu_matrix = True
        if (args.random):
            if_glove = False
        if (args.glove):
            if_glove = True
        test(if_glove=if_glove, confu_matrix=confu_matrix)
    if (args.ensemble):
        # do the test function
        print('ensemble')
        confu_matrix = False
        if (args.confu_matrix):
            confu_matrix = True
        ensemble(confu_matrix=confu_matrix)
    pass
