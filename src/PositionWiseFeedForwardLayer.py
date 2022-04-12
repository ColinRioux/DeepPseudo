import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionWiseFeedForwardLayer(nn.Module):

    def __init__(self, d_model, hidden_size):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.fc_in = nn.Linear(d_model, hidden_size)
        self.fc_ou = nn.Linear(hidden_size, d_model)

    def forward(self, inputs):
        """
        :param Tensor[batch_size, seq_len, d_model] inputs
        :return Tensor[batch_size, seq_len, d_model] outputs
        """
        outputs = F.relu(self.fc_in(inputs))  # [batch_size, seq_len, hidden_size]
        return self.fc_ou(outputs)  # [batch_size, seq_len, d_model]