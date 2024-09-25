import torch
import torch.nn as nn
from model_config import config_iris
    
class IrisNN(nn.Module):
    def __init__(self):
        super(IrisNN, self).__init__()
        input_size = config_iris['input_size']
        hidden_size = config_iris['hidden_size']
        output_size = config_iris['output_size']
        layer_names = config_iris['layer_names']

        setattr(self, layer_names['input'], nn.Linear(input_size, hidden_size))
        setattr(self, layer_names['hidden'], nn.Linear(hidden_size, hidden_size))
        setattr(self, layer_names['output'], nn.Linear(hidden_size, output_size))

        self.layer_names = layer_names

    def forward(self, x):
        x = torch.relu(getattr(self, self.layer_names['input'])(x))
        x = torch.relu(getattr(self, self.layer_names['hidden'])(x))
        x = torch.softmax(getattr(self, self.layer_names['output'])(x), dim=1)
        return x