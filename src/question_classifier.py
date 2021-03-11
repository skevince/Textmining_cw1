from src.word_embeding import randomly_embedding
from src.glove import read_glove_vecs
from src.model import Model
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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

def read_data(filepath, pad_size,word_list,label_list=None):
    data = []
    seq_len = []
    label = []
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
                # 不在字典里则用UKN(word_list最后一个词)代替
                if word_seg[i] in word_list:
                    data_pre.append(word_list.index(word_seg[i]))
                else:
                    data_pre.append(len(word_list) - 1)
            # 同样使用UKN(word_list最后一个词) pad所有句子，若超出则删掉掉超出部分
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

def build_model_and_dataset(if_glove, if_biLSTM, if_freeze,batch_size):
    print('if_glove:', if_glove, ' if_biLSTM: ', if_biLSTM, ' if_freeze: ', if_freeze)
    path_glove = '.././data/glove.small.txt'
    path_train = '.././data/train.txt'
    path_dev = '.././data/dev.txt'
    path_test = '.././data/test.txt'

    if if_glove:
        word_list, vector = read_glove_vecs(path_glove)
        embedding_dim = vector[0].shape[0]
    else:

        word_list, vector = randomly_embedding(path_train)
        embedding_dim = len(vector)

    pad_size = get_pad_size(path_train)
    data_train, seq_len_train, label_train, label_list = read_data(path_train, pad_size=pad_size,word_list=word_list)
    class_number = len(label_list)

    model = Model(vocabulary_size=len(word_list), embedding_dim=embedding_dim, hidden_dim=1000,
                  num_classes=class_number, pretrain_char_embedding=vector,
                  if_glove=if_glove, if_biLSTM=if_biLSTM, freeze=if_freeze, device=device)

    batch_size = batch_size

    trainset = TensorDataset(data_train, seq_len_train, label_train)
    traindataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0,
                                 pin_memory=True,
                                 drop_last=True)

    data_dev, seq_len_dev, label_dev, class_number_dev = read_data(path_dev, pad_size=pad_size, label_list=label_list,word_list=word_list)
    devset = TensorDataset(data_dev, seq_len_dev, label_dev)
    devdataloader = DataLoader(devset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0,
                               pin_memory=True,
                               drop_last=True)

    data_test, seq_len_test, label_test, class_number_test = read_data(path_test, pad_size=pad_size, label_list=label_list,word_list=word_list)
    testset = TensorDataset(data_test, seq_len_test, label_test)
    testdataloader = DataLoader(testset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0,
                                pin_memory=True,
                                drop_last=True)
    return model, traindataloader, devdataloader, testdataloader


if __name__ == '__main__':
    path_train = '.././data/train.txt'
    embedding = 'glove'
    ngpu = 1
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))
    batch_size=32
    model, traindataloader, devdataloader, testdataloader = build_model_and_dataset(if_glove=False, if_biLSTM=False,
                                                                                    if_freeze=True,batch_size=batch_size)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)

    loss_function = nn.NLLLoss()

    for epoch in range(10):
        model.train()
        print(epoch+1)
        acc = []
        train_loss = []
        running_loss = 0.0
        correct = 0.0
        for i, data in enumerate(traindataloader, 0):
            # get the input
            data, seq_len, labels = data
            data = data.to(device)
            seq_len = seq_len.to(device)
            labels = labels.to(device)
            # zeros the paramster gradients
            optimizer.zero_grad()  #

            # forward + backward + optimize
            outputs, idx_sort = model(data, seq_len)
            labels = labels[idx_sort]
            loss = loss_function(outputs, labels)  # 计算loss
            loss.backward()  # loss 求导
            optimizer.step()  # 更新参数

            # print accuracy
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            acc.append(correct / batch_size)
            # print statistics
            running_loss += loss.item()  # tensor.item()  获取tensor的数值
            train_loss.append(running_loss / batch_size)
            running_loss = 0.0
            correct = 0.0
        print('train-- loss: ', np.average(train_loss), ' acc: ', np.average(acc))

        # test
        model.eval()

        acc = []
        train_loss = []
        running_loss = 0.0
        correct = 0.0
        for i, data in enumerate(testdataloader, 0):
            # get the input
            data, seq_len, labels = data
            data = data.to(device)
            seq_len = seq_len.to(device)
            labels = labels.to(device)
            # zeros the paramster gradients
            optimizer.zero_grad()  #

            # forward + backward + optimize
            outputs, idx_sort = model(data, seq_len)
            labels = labels[idx_sort]
            # print(outputs)
            loss = F.cross_entropy(outputs, labels)  # 计算loss
            # print accuracy
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            acc.append(correct / batch_size)
            # print statistics
            running_loss += loss.item()  # tensor.item()  获取tensor的数值
            train_loss.append(running_loss / batch_size)
            running_loss = 0.0
            correct = 0.0
        print('test-- loss: ', np.average(train_loss), ' acc: ', np.average(acc))
