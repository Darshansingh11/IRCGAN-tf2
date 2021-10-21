import torch
from torch import nn


class BiLSTMTextEncoder(nn.Module):
    def __init__(self, src_vocab_size, input_dims, units: int = 16):
        super(BiLSTMTextEncoder, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.input_dims = input_dims
        self.units = units

        self.embedding = nn.Embedding(
            num_embeddings=src_vocab_size, embedding_dim=input_dims
        )

        # bidirectional LSTM with units//2 size.
        self.lstm = nn.LSTM(
            input_size=input_dims,
            hidden_size=units // 2,
            bidirectional=True,
            batch_first=True,
        )

    def forward(self, src_seq, src_mask):
        # src_seq: [batch_size, seq_len]
        # src_mask: [batch_size, seq_len]
        # embed: [batch_size, seq_len, input_dims]
        embed = self.embedding(src_seq)
        # output: [batch_size, seq_len, units]
        # hidden: [num_layers * num_directions, batch_size, units]
        # cell: [num_layers * num_directions, batch_size, units]
        output, (hidden, cell) = self.lstm(embed, src_mask)

        return output
