import torch
import torch.nn as nn
import torch.nn.functional as F
from src.EncoderBlockLayer import EncoderBlockLayer
from src.PositionalEncodingLayer import PositionalEncodingLayer

class EncoderLayer(nn.Module):

    def __init__(self, vocab_size, max_len, d_model, n_heads, hidden_size, kernel_size, dropout, n_layers, scale):
        super(EncoderLayer, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.scale = scale
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncodingLayer(d_model=d_model, max_len=max_len)
        self.encoder_block_layers = nn.ModuleList(
            [EncoderBlockLayer(d_model=d_model, n_heads=n_heads, hidden_size=hidden_size,
                               dropout=dropout) for _ in range(n_layers)])
        self.fc_embedding_hidden = nn.Linear(d_model, hidden_size)
        self.fc_hidden_embedding = nn.Linear(hidden_size, d_model)
        self.conv1ds = nn.ModuleList([nn.Conv1d(hidden_size, hidden_size * 2, kernel_size=kernel_size,
                                                padding=(kernel_size - 1) // 2) for _ in range(n_layers)])

    def forward(self, src_sequences, src_mask):
        """
        :param Tensor[batch_size, src_len] src_sequences
        :param Tensor[batch_size, src_len] src_mask
        :return Tensor[batch_size, src_len, d_model] outputs
        """
        token_embedded = self.token_embedding(src_sequences)  # [batch_size, src_len, d_model]
        position_encoded = self.position_encoding(src_sequences)  # [batch_size, src_len, d_model]
        outputs = self.dropout(token_embedded) + position_encoded  # [batch_size, src_len, d_model]

        embedded = outputs
        for layer in self.encoder_block_layers:
            outputs = layer(src_inputs=outputs, src_mask=src_mask)  # [batch_size, src_len, d_model]

        conv_output = self.fc_embedding_hidden(embedded)  # [batch_size, src_len, hidden_size]
        conv_output = conv_output.permute(0, 2, 1)  # [batch_size, hidden_size, src_len]
        for conv1d in self.conv1ds:
            conv_output = self.dropout(conv_output)
            conved = conv1d(conv_output)  # [batch_size, hidden_size * 2, src_len]
            conved = F.glu(conved, dim=1)  # [batch_size, hidden_size, src_len]
            conv_output = (conved + conv_output) * self.scale  # [batch_size, hidden_size, src_len] Residual connection
        conv_output = conv_output.permute(0, 2, 1)  # [batch_size, src_len, hidden_size]
        conv_output = self.fc_hidden_embedding(conv_output)  # [batch_size, src_len, d_model]

        outputs = outputs + conv_output
        return outputs