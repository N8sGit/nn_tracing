import torch
import torch.nn as nn

# Define the model 
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        self.neuron_ids = {}

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

    def initialize_neuron_ids(self):
        layer_idx = 0
        # Assign IDs for the first layer (input layer has no activations)
        for neuron_idx in range(self.fc1.in_features):
            self.neuron_ids[(layer_idx, neuron_idx)] = (layer_idx, neuron_idx)
        
        layer_idx += 1
        # Assign IDs for the first hidden layer
        for neuron_idx in range(self.fc1.out_features):
            self.neuron_ids[(layer_idx, neuron_idx)] = (layer_idx, neuron_idx)
        
        layer_idx += 1
        # Assign IDs for the output layer
        for neuron_idx in range(self.fc2.out_features):
            self.neuron_ids[(layer_idx, neuron_idx)] = (layer_idx, neuron_idx)

model = SimpleNN()
# Dry run to initialize the model with neural ids
model.initialize_neuron_ids()
print(model.neuron_ids)