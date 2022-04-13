import torch
import torch.nn as nn
import torch.nn.functional as F
from src.DecoderBlockLayer import DecoderBlockLayer
from src.PositionalEncodingLayer import PositionalEncodingLayer

class DecoderLayer(nn.Module):

    def __init__(self, vocab_size, max_len, d_model, n_heads, hidden_size, dropout, n_layers, seq_thresh):
        super(DecoderLayer, self).__init__()
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(p=dropout)
        self.n_layers = n_layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_encoding = PositionalEncodingLayer(d_model=d_model, max_len=max_len)
        self.decoder_block_layers = nn.ModuleList(
            [DecoderBlockLayer(d_model=d_model, n_heads=n_heads, hidden_size=hidden_size, dropout=dropout, seq_thresh=seq_thresh) for _ in
             range(n_layers)])
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, dest_sequences, src_encoded, dest_mask, src_mask):
        """
        :param Tensor[batch_size, dest_len] dest_sequences
        :param Tensor[batch_size, src_len, d_model] src_encoded
        :param Tensor[batch_size, dest_len, d_model] dest_mask
        :param Tensor[batch_size, src_len, d_model] src_mask
        :return Tensor[batch_size, dest_len, vocab_size] logits
        :return Tensor[batch_size, n_heads, dest_len, src_len] attention_weights
        """
        token_embedded = self.token_embedding(dest_sequences)  # [batch_size, dest_len, d_model]
        position_encoded = self.position_encoding(dest_sequences)  # [batch_size, dest_len, d_model]
        outputs = self.dropout(token_embedded) + position_encoded  # [batch_size, dest_len, d_model]
        for layer in self.decoder_block_layers:
            outputs, attention_weights = layer(dest_inputs=outputs, src_encoded=src_encoded, dest_mask=dest_mask,
                                               src_mask=src_mask)
        logits = self.fc(outputs)
        return logits, attention_weights