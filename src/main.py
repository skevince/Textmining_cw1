import torch
from src.run import model_run,build_model_and_dataset
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
    embedding = 'glove'
    ngpu = 1
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))
    batch_size = 8
    if_GPU = True
    print('if_GPU:', if_GPU)
    if if_GPU:
        device = device
    else:
        device = None
    model, traindataloader, devdataloader, testdataloader = build_model_and_dataset(if_glove=False, if_biLSTM=False,
                                                                                    if_freeze=False,
                                                                                    batch_size=batch_size,
                                                                                    device=device)
    if if_GPU:
        model = model.to(device)

    train_epoch = 10
    dev_epoch = 1
    model, train_loss, train_acc = model_run(model, run_type='Train', epoch_range=train_epoch, batch_size=batch_size,
                                             device=device, dataloader=traindataloader)
    print(train_loss)
    print(train_acc)
    model, dev_loss, dev_acc = model_run(model, run_type='Develop', epoch_range=dev_epoch, batch_size=batch_size,
                                             device=device, dataloader=devdataloader)
    print(dev_loss)
    print(dev_acc)
    prediction, test_loss, test_acc = model_run(model, run_type='Test', batch_size=1,
                                         device=device, dataloader=testdataloader)
    print(len(prediction))
    print(test_loss)
    print('test_accuracy：',test_acc)