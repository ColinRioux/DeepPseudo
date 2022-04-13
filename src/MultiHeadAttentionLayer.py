import torch
import torch.nn as nn
import torch.nn.functional as F
from src.ScaleUp import ScaleUp

def scaling_factor(sequence_threshold):
    return np.log2((sequence_threshold ** 2) - sequence_threshold)

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, d_model, n_heads, sequence_threshold):
        super(MultiHeadAttentionLayer, self).__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_size = d_model // n_heads
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.scaleup = ScaleUp(scaling_factor(sequence_threshold))

    def forward(self, query, key, value, mask):
        """
        :param Tensor[batch_size, q_len, d_model] query
        :param Tensor[batch_size, k_len, d_model] key
        :param Tensor[batch_size, v_len, d_model] value
        :param Tensor[batch_size, ..., k_len] mask
        :return Tensor[batch_size, q_len, d_model] context
        :return Tensor[batch_size, n_heads, q_len, k_len] attention_weights
        """
        Q = self.fc_q(query)  # [batch_size, q_len, d_model]
        K = self.fc_k(key)  # [batch_size, k_len, d_model]
        V = self.fc_v(value)  # [batch_size, v_len, d_model]

        Q = Q.view(Q.size(0), -1, self.n_heads, self.head_size).permute(0, 2, 1,
                                                                        3)  # [batch_size, n_heads, q_len, head_size]
        K = K.view(K.size(0), -1, self.n_heads, self.head_size).permute(0, 2, 1,
                                                                        3)  # [batch_size, n_heads, k_len, head_size]
        V = V.view(V.size(0), -1, self.n_heads, self.head_size).permute(0, 2, 1,
                                                                        3)  # [batch_size, n_heads, v_len, head_size]

        #         scores = torch.matmul(Q, K.transpose(-1, -2)) # [batch_size, n_heads, q_len, k_len]
        #         scores = scores / torch.sqrt(torch.FloatTensor([self.head_size]).to(Q.device))
        Q = F.normalize(Q, p=2, dim=-1)
        K = F.normalize(K, p=2, dim=-1)
        scaleup = self.scaleup
        scores = scaleup(torch.matmul(Q, K.transpose(-2, -1)))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e18)
        attention_weights = F.softmax(scores, dim=-1)  # [batch_size, n_heads, q_len, k_len]

        context = torch.matmul(attention_weights, V)  # [batch_size, n_heads, q_len, v_len]
        context = context.permute(0, 2, 1, 3).contiguous()  # [batch_size, q_len, n_heads, v_len]
        context = context.view(context.size(0), -1, self.d_model)
        context = self.fc_o(context)  # [batch_size, q_len, d_model]

        return context, attention_weights