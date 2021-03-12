import torch
from src.run import model_run,build_model_and_dataset
import torch
import random
from src.question_classifier import get_config


torch.manual_seed(1)
random.seed(1)


def train(glove=False, biLSTM=False, freeze=False):
    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))

    if_GPU = False
    print('if_GPU:', if_GPU)
    if if_GPU:
        device = device
    else:
        device = None

    batch_size = get_config('Model', 'batch_size')
    train_epoch = get_config('Model', 'epoch')
    dev_epoch = get_config('Model', 'dev_epoch')

    model, traindataloader, devdataloader, testdataloader = build_model_and_dataset(if_glove=glove, if_biLSTM=biLSTM,
                                                                                    if_freeze=freeze,
                                                                                    batch_size=batch_size,
                                                                                    device=device)
    if if_GPU:
        model = model.to(device)

    model, train_loss, train_acc = model_run(model, run_type='Train', epoch_range=train_epoch, batch_size=batch_size,
                                             device=device, dataloader=traindataloader)
    print(train_loss)
    print(train_acc)

    model, dev_loss, dev_acc = model_run(model, run_type='Develop', epoch_range=dev_epoch, batch_size=batch_size,
                                         device=device, dataloader=devdataloader)
    print(dev_loss)
    print(dev_acc)
    return


def test(glove=False, biLSTM=False, freeze=False):
    ngpu = 1
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))

    if_GPU = False
    print('if_GPU:', if_GPU)
    if if_GPU:
        device = device
    else:
        device = None

    batch_size = get_config('Model', 'batch_size')

    model, traindataloader, devdataloader, testdataloader = build_model_and_dataset(if_glove=glove, if_biLSTM=biLSTM,
                                                                                    if_freeze=freeze,
                                                                                    batch_size=batch_size,
                                                                                    device=device)
    if if_GPU:
        model = model.to(device)
    prediction, test_loss, test_acc = model_run(model, run_type='Test', batch_size=1,
                                                device=device, dataloader=testdataloader)
    print(len(prediction))
    print(test_loss)
    print('test_accuracy：', test_acc)
    return


# if __name__ == '__main__':
#
#     # Decide which device we want to run on
#     device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
#     print(device)
#     print(torch.cuda.get_device_name(0))
#
#     if_GPU = False
#     print('if_GPU:', if_GPU)
#     if if_GPU:
#         device = device
#     else:
#         device = None
#
#     if (get_config('Model', 'model') == "bow"):
#         model, traindataloader, devdataloader, testdataloader = build_model_and_dataset(if_glove=False, if_biLSTM=False,
#                                                                                         if_freeze=False,
#                                                                                         batch_size=batch_size,
#                                                                                         device=device)
#         if if_GPU:
#             model = model.to(device)
#
#
#     if (get_config('Model', 'model') == "bilstm"):
#         model, traindataloader, devdataloader, testdataloader = build_model_and_dataset(if_glove=False, if_biLSTM=True,
#                                                                                         if_freeze=False,
#                                                                                         batch_size=batch_size,
#                                                                                         device=device)
#         if if_GPU:
#             model = model.to(device)