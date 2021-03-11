from src.word_embeding import randomly_embedding

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

np.set_printoptions(threshold=np.inf)


class BiLSTMTagger(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, hidden_dim, num_classes, pretrain_char_embedding=None, pre=False,
                 freeze=False, dropout=0.1, ifBow=False,device=None):
        super(BiLSTMTagger, self).__init__()
        self.device=device
        self.hidden_dim = hidden_dim
        if pre == True:
            self.word_embeddings = nn.Embedding.from_pretrained(pretrain_char_embedding, freeze=freeze)
        else:
            self.word_embeddings = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=vocabulary_size - 1)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim*2, 4000)
        self.relu_1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(4000, 2000)
        self.relu_2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(2000,num_classes)

    def forward(self, x, seq_len):

        out = self.word_embeddings(x)
        # 排序，并去掉padding词
        _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
        _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
        out = torch.index_select(out, 0, idx_sort)
        seq_len = list(seq_len[idx_sort])
        out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
        out, (hn, _) = self.lstm(out)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        out = torch.sum(out, dim=1)
        seq_len = torch.FloatTensor(seq_len)
        seq_len = seq_len.reshape([32,1])
        seq_len = seq_len.expand([32, self.hidden_dim*2])
        seq_len = seq_len.to(self.device)
        out = torch.div(out, seq_len)
        out = self.fc1(out)
        out = self.relu_1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu_2(out)
        out = self.dropout1(out)
        out = self.fc3(out)

        return out,idx_sort
