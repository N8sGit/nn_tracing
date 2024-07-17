import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(20, 10)
        self.fc2 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()
        self.neuron_ids = {}

    def initialize_neuron_ids(self):
        # TO DO: generalize for arbitary depth 
        self.neuron_ids = {
            "input": [(0, i) for i in range(self.fc1.in_features)],
            "layer1": [(1, i) for i in range(self.fc1.out_features)],
            "layer2": [(2, i) for i in range(self.fc2.out_features)]
        }

class TraceObject:
    def __init__(self):
        self.paths = {0: [], 1: []}  # Separate lists for classification results 0 and 1

    def record_path(self, layer, neuron_id, source_neurons, activation, classification_result):
        path_data = {
            "layer": layer,
            "neuron_id": neuron_id,
            "source_neurons": source_neurons,
            "activation": activation
        }
        self.paths[classification_result].append(path_data)
        print(f"Recorded path: {path_data} for classification result: {classification_result}")

class TracingNN(SimpleNN):
    def __init__(self):
        super(TracingNN, self).__init__()
        self.trace = TraceObject()
        self.initialize_neuron_ids()

    def forward(self, x):
        # TO DO: Generalize for arbitrary layers 
        x1 = torch.relu(self.fc1(x))
        self.record_activations(1, x1, self.neuron_ids["layer1"], self.neuron_ids["input"])
        x2 = self.sigmoid(self.fc2(x1))
        self.record_activations(2, x2, self.neuron_ids["layer2"], self.neuron_ids["layer1"])
        return x2

    def record_activations(self, layer, x, neuron_ids, previous_layer_neurons):
        predictions = (x > 0.5).float()
        for i in range(x.size(0)):  # Iterate over batch size
            for neuron_id in range(x.size(1)):  # Iterate over neurons
                if x[i, neuron_id] > 0:
                    classification_result = int(predictions[i, 0].item())
                    source_neurons = [previous_layer_neurons[j] for j in range(x.size(1)) if x[i, neuron_id] > 0]
                    self.trace.record_path(layer, neuron_ids[neuron_id], source_neurons, x[i, neuron_id].item(), classification_result)
                    print(f"Layer: {layer}, Neuron: {neuron_ids[neuron_id]}, Activation: {x[i, neuron_id].item()}, Classification: {classification_result}, Source Neurons: {source_neurons}")

    def predict(self, x):
        outputs = self.forward(x)
        predictions = (outputs > 0.5).float()
        return predictions