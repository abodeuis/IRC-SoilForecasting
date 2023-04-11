import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, data_shape, target_shape=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=data_shape, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, target_shape)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    
class RNN_Model(nn.Module):
    def __init__(self, data_shape, target_shape=1):
        super(RNN_Model, self).__init__()
        self.rnn = nn.RNN(input_size=data_shape, hidden_size=50, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, target_shape)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x
    
class GRU_Model(nn.Module):
    def __init__(self, data_shape, target_shape=1):
        super(GRU_Model, self).__init__()
        self.gru = nn.GRU(input_size=data_shape, hidden_size=50)
        self.linear = nn.Linear(50, target_shape)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.linear(x)
        return x
