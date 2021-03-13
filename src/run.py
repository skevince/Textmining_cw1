from word_embeding import randomly_embedding
from glove import read_glove_vecs
from model import Model
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
from process import processfile
import question_classifier


# get pad size
def get_pad_size(filepath):
    seq_len = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            void_stopwords = line.split(" ", 1)[1]
            word_seg = void_stopwords.split()
            seq_len_temp = len(word_seg)
            seq_len.append(seq_len_temp)
    pad_size = max(seq_len)
    return pad_size


# get label list
def get_label_list(filepath):
    label_list = []
    with open(filepath, 'r') as f:
        for line in f.readlines():
            label_temp = line.split(" ", 1)[0]
            if label_temp not in label_list:
                label_list.append(label_temp)
    return label_list


# read data and return as tensor
def read_data(filepath, pad_size, word_list, label_list=None):
    data = []
    seq_len = []
    label = []
    # get label list
    if label_list == None:
        label_list = []
    else:
        label_list = label_list
    with open(filepath, 'r') as f:
        for line in f.readlines():
            data_pre = []
            void_stopwords = line.split(" ", 1)[1]
            label_temp = line.split(" ", 1)[0]
            if label_temp in label_list:
                label.append(label_list.index(label_temp))
            else:
                label_list.append(label_temp)
                label.append(label_list.index(label_temp))
            word_seg = void_stopwords.split()
            seq_len_temp = len(word_seg)
            for i in range(seq_len_temp):
                # if not in word_list, use 'PAD' (the last word in word list) instead
                if word_seg[i] in word_list:
                    data_pre.append(word_list.index(word_seg[i]))
                else:
                    data_pre.append(len(word_list) - 1)
            # use'PAD' (the last word in word list) to PAD sentence
            if seq_len_temp < pad_size:
                for i in range(pad_size - seq_len_temp):
                    data_pre.append(len(word_list) - 1)
            if seq_len_temp > pad_size:
                data_pre = data_pre[:pad_size]
            seq_len.append(seq_len_temp)
            data.append(data_pre)
    data = torch.LongTensor(data)
    seq_len = torch.LongTensor(seq_len)
    label = torch.LongTensor(label)
    return data, seq_len, label, label_list


# build model and datasetloader
def build_model_and_dataset(if_glove, if_biLSTM, if_freeze, batch_size, device, bowandlstm=False):
    torch.manual_seed(1)
    random.seed(1)
    path_glove = question_classifier.get_config('PATH', 'path_pre_emb')

    path_train = question_classifier.get_config('PATH', 'path_train')
    path_dev = question_classifier.get_config('PATH', 'path_dev')
    path_test = question_classifier.get_config('PATH', 'path_test')

    processfile(path_train, '././data/train_precess.txt')
    processfile(path_dev, '././data/dev_precess.txt')
    processfile(path_test, '././data/test_precess.txt')
    path_train = '././data/train_precess.txt'
    path_dev = '././data/dev_precess.txt'
    path_test = '././data/test_precess.txt'

    if if_glove:
        word_list, vector = read_glove_vecs(path_glove)
        embedding_dim = vector[0].shape[0]
    else:
        word_list, vector = randomly_embedding(path_train)
        embedding_dim = len(vector)

    pad_size = get_pad_size(path_train)
    label_list = get_label_list('././data/train.txt')
    # read data
    data_train, seq_len_train, label_train, label_list = read_data(path_train, pad_size=pad_size, word_list=word_list,
                                                                   label_list=label_list)
    # prepare model
    class_number = len(label_list)

    model = Model(vocabulary_size=len(word_list), embedding_dim=embedding_dim, hidden_dim=500,
                  num_classes=class_number, pretrain_char_embedding=vector,
                  if_glove=if_glove, if_biLSTM=if_biLSTM, freeze=if_freeze, device=device, bowandlstm=bowandlstm)

    # build dataset loader (train develop test)
    trainset = TensorDataset(data_train, seq_len_train, label_train)
    traindataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0,
                                 pin_memory=True,
                                 drop_last=True)

    data_dev, seq_len_dev, label_dev, class_number_dev = read_data(path_dev, pad_size=pad_size, label_list=label_list,
                                                                   word_list=word_list)
    devset = TensorDataset(data_dev, seq_len_dev, label_dev)
    devdataloader = DataLoader(devset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0,
                               pin_memory=True,
                               drop_last=True)

    data_test, seq_len_test, label_test, class_number_test = read_data(path_test, pad_size=pad_size,
                                                                       label_list=label_list, word_list=word_list)
    testset = TensorDataset(data_test, seq_len_test, label_test)
    testdataloader = DataLoader(testset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0,
                                pin_memory=True,
                                drop_last=True)
    return model, traindataloader, devdataloader, testdataloader, label_list

# run model as train, develop or test
def model_run(model, run_type=None, epoch_range=0, batch_size=1, device=None, dataloader=None):
    if run_type == 'Train' or run_type == 'Develop':
        model.train()
        lr = float(question_classifier.get_config('Model', 'lr_param'))
        momentum = float(question_classifier.get_config('Model', 'momentum'))
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        train_loss = []
        train_acc = []
        for epoch in range(epoch_range):
            epoch_acc = []
            epoch_loss = []
            running_loss = 0.0
            correct = 0.0
            for i, data in enumerate(dataloader, 0):
                # get the input
                data, seq_len, labels = data
                if device != None:
                    data = data.to(device)
                    seq_len = seq_len.to(device)
                    labels = labels.to(device)
                # zeros the paramster gradients
                optimizer.zero_grad()  #
                # forward + backward + optimize
                outputs, idx_sort = model(data, seq_len)
                labels = labels[idx_sort]
                loss_function = nn.NLLLoss()
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                _, pred = torch.max(outputs, 1)
                correct += (pred == labels).sum().item()
                epoch_acc.append(correct / batch_size)
                running_loss += loss.item()
                epoch_loss.append(running_loss / batch_size)
                running_loss = 0.0
                correct = 0.0
            print('epoch: ', epoch, ' train loss: ', np.average(epoch_loss), ' train accuracy: ', np.average(epoch_acc))
            train_loss.append(np.average(epoch_loss))
            train_acc.append(np.average(epoch_acc))
        return model, train_loss, train_acc
    if run_type == 'Test':
        model.eval()
        prediction_result = np.array([])
        true_label = np.array([])
        test_acc = []
        correct = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the input
            data, seq_len, labels = data
            if device != None:
                data = data.to(device)
                seq_len = seq_len.to(device)
                labels = labels.to(device)
            # forward + backward + optimize
            outputs, idx_sort = model(data, seq_len)
            labels = labels[idx_sort]
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            pred = pred.cpu()
            prediction = pred.numpy()
            prediction_result = np.append(prediction_result, prediction)
            labels = labels.cpu()
            labels = labels.numpy()
            true_label = np.append(true_label, labels)
            test_acc.append(correct / batch_size)
            correct = 0.0
        return prediction_result, true_label, np.average(test_acc)

# run the ensemble model
def model_ensemble_run(ensemble_size, emsemble_path_model, batch_size=500, device=None, dataloader=None):
    prediction_result = np.array([])
    true_label = np.array([])
    test_acc = []
    correct = 0.0
    for i, data in enumerate(dataloader, 0):
        # get the input
        data, seq_len, labels = data
        if device != None:
            data = data.to(device)
            seq_len = seq_len.to(device)
            labels = labels.to(device)
        # forward + backward + optimize
        outputs_all = []
        for i in range(ensemble_size):
            model = torch.load(emsemble_path_model + '.' + str(i), map_location='cpu')
            if device != None:
                model = model.to(device)
            model.eval()
            outputs, idx_sort = model(data, seq_len)
            outputs_all.append(outputs)
        outputs = outputs_all[len(outputs_all) - 1]
        for i in range(len(outputs_all) - 1):
            outputs = outputs + outputs_all[i]
        outputs = outputs / ensemble_size
        labels = labels[idx_sort]
        _, pred = torch.max(outputs, 1)
        correct += (pred == labels).sum().item()
        pred = pred.cpu()
        prediction = pred.numpy()
        prediction_result = np.append(prediction_result, prediction)
        labels = labels.cpu()
        labels = labels.numpy()
        true_label = np.append(true_label, labels)
        test_acc.append(correct / batch_size)
        correct = 0.0
    return prediction_result, true_label, np.average(test_acc)
