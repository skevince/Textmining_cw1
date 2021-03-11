from src.word_embeding import randomly_embedding
from src.glove import read_glove_vecs
from src.bilstm import BiLSTMTagger
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


def read_data(filepath, pad_size):
    data = []
    seq_len = []
    label = []
    label_list = []
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
    class_number = len(label_list)
    return data, seq_len, label, class_number


if __name__ == '__main__':
    path_glove = '.././data/glove.small.txt'
    path_train = '.././data/train.txt'
    embedding = 'glove'
    ngpu = 1
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))

    if embedding == 'random':
        # 获取word_list, vector
        word_list, vector = read_glove_vecs(path_glove)
        embedding_dim = vector[0].shape[0]
        filepath = '.././data/train.txt'
        pad_size = get_pad_size(filepath)
        data, seq_len, label, class_number = read_data(filepath, pad_size=pad_size)
        BiLstm = BiLSTMTagger(vocabulary_size=len(word_list), embedding_dim=embedding_dim, hidden_dim=150,
                              num_classes=class_number,
                              pretrain_char_embedding=vector, pre=True, freeze=False)
    else:
        # 获取word_list, embedding_dim  这里的vector是出现次数>n次的词，取其长度为embedding维度
        word_list, vector = randomly_embedding(path_train)
        embedding_dim = len(vector)
        filepath = '.././data/train.txt'
        pad_size = get_pad_size(filepath)
        data, seq_len, label, class_number = read_data(filepath, pad_size=pad_size)
        BiLstm = BiLSTMTagger(vocabulary_size=len(word_list), embedding_dim=embedding_dim, hidden_dim=1000,
                              num_classes=class_number,
                              pre=False, freeze=False)

    BiLstm = BiLstm.to(device)
    batch_size=32
    trainset = TensorDataset(data, seq_len, label)
    traindataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, sampler=None, num_workers=0, pin_memory=True,
                                 drop_last=True)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    #optimizer = optim.SGD(BiLstm.parameters(), lr=0.01, momentum=0.9)
    optimizer = torch.optim.Adam(BiLstm.parameters(), lr=0.01, weight_decay=1e-5)
    for epoch in range(20):
        print(epoch)
        acc=[]
        train_loss=[]
        running_loss = 0.0
        correct = 0.0
        total = 0.0
        correctPlus = 0.0
        running_lossPlus = 0.0
        for i, data in enumerate(traindataloader, 0):

            # get the input
            data, seq_len, labels = data
            data = data.to(device)
            seq_len = seq_len.to(device)
            labels = labels.to(device)
            # zeros the paramster gradients
            optimizer.zero_grad()  #

            # forward + backward + optimize
            outputs = BiLstm(data, seq_len)

            loss = F.cross_entropy(outputs, labels)  # 计算loss
            loss.backward()  # loss 求导
            optimizer.step()  # 更新参数

            # print accuracy
            _, pred = torch.max(outputs, 1)
            correct += (pred == labels).sum().item()
            acc.append(correct/batch_size)
            # print statistics
            running_loss += loss.item()  # tensor.item()  获取tensor的数值
            train_loss.append(running_loss/batch_size)
            running_loss = 0.0
            correct = 0.0
        print('loss: ',np.average(train_loss), ' acc: ',np.average(acc))

