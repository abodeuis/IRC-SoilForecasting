import torch
import torch.nn as nn
import torch.nn.functional as F

def get_model(model, model_params):
    models = {
        'rnn': RNN_Model,
        'lstm': LSTM_Model,
        'gru': GRU_Model
    }
    return models.get(model.lower())(**model_params)

class LSTM_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=50, output_size=1, layers=1):
        super().__init__()
        self.model_name = 'LSTM'
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_size = output_size
        self.layers = layers
        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)
    def forward(self, x):
        # Initializing hidden state for first input with zeros
        h0 = torch.zeros(self.layers, x.size(0), self.hidden_dim).requires_grad_()
        # Initializing cell state for first input with zeros
        c0 = torch.zeros(self.layers, x.size(0), self.hidden_dim).requires_grad_()
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        # Forward propagation by passing in the input, hidden state, and cell state into the model
        x, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        x = x[:, -(self.output_size+1):-1, :]
        x = self.linear(x)
        return x
    
class RNN_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=50, layers=1):
        super(RNN_Model, self).__init__()
        self.model_name = 'RNN'
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = layers

        self.rnn = nn.RNN(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.layers, batch_first=True)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x
    
class GRU_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=50, layers=1):
        super(GRU_Model, self).__init__()
        self.model_name = 'GRU'
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.layers = layers

        self.gru = nn.GRU(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.layers)
        self.linear = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.linear(x)
        return x

class Dense_Model(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=50, hidden_layers=1):
        super(Dense_Model, self).__init__()
        self.model_name = 'DenseNet'
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        for i in range(0,hidden_layers):
            self.layers.append(nn.Linear(self.hidden_dim,self.hidden_dim))
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x