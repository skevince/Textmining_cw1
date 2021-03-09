from src.word_embeding import randomly_embedding
import os
import torch
import torch.nn as nn
import torch.nn.functional as F


if (os.cpu_count() > 8):
    USE_CUDA = True
else:
    USE_CUDA = False


class BiLSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, dropout=0.1):
        super(BiLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dropout = nn.Dropout(dropout)
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # The BiLSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def get_state(self, input):
        """Get cell states and hidden states."""
        batch_size = input.size(1)
        c0_encoder = torch.zeros(2, batch_size, self.hidden_dim // 2)
        h0_encoder = torch.zeros(2, batch_size, self.hidden_dim // 2)  ### * self.num_directions = 2 if bi
        if USE_CUDA:
            h0_encoder = h0_encoder.cuda()
            c0_encoder = c0_encoder.cuda()
        return (h0_encoder, c0_encoder)


    def forward(self, sentence, sentence_lengths):
        embeds = self.word_embeddings(sentence)
        embeds = self.embedding_dropout(embeds)
        embeds = nn.utils.rnn.pack_padded_sequence(embeds, sentence_lengths, batch_first=False)

        hidden = self.get_state(sentence)  # [m,b,e]
        lstm_out, self.hidden = self.lstm(embeds, hidden)
        return lstm_out
