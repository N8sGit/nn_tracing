import torch
import torch.nn as nn
from typing import Union, List, Optional

class TraceObject:
    def __init__(self, epoch, layer, neuron_id):
        self.coordinates = {"E_": epoch, "L_": layer, "n_": neuron_id}
        self.history = []  # List to store (activation, input_neurons, output_neurons) tuples
        self.canonical_activation = None
        self.canonical_input_neurons = None
        self.canonical_output_neurons = None

    def record_state(self, activation, input_neurons, output_neurons):
        self.history.append((activation, input_neurons, output_neurons))

    def compute_canonical_values(self):
        # Placeholder for statistical analysis to compute canonical values
        if self.history:
            activations = [h[0] for h in self.history]
            input_neurons_list = [h[1] for h in self.history]
            output_neurons_list = [h[2] for h in self.history]
            
            # Example: Set canonical activation to the mean activation
            self.canonical_activation = sum(activations) / len(activations)
            # Example: Set canonical input/output neurons to the most common sets
            self.canonical_input_neurons = max(set(input_neurons_list), key=input_neurons_list.count)
            self.canonical_output_neurons = max(set(output_neurons_list), key=output_neurons_list.count)

    def to_dict(self):
        return {
            # "canonical_activation": self.canonical_activation,
            # "canonical_input_neurons": self.canonical_input_neurons,
            # "canonical_output_neurons": self.canonical_output_neurons,
            "history": self.history
        }

    def get_lookup_signature(self):
        return ''.join(f"{k}{v}" for k, v in self.coordinates.items())

class NetworkTrace:
    def __init__(self, model, num_epochs, epoch_interval: Optional[Union[int, float, List[Union[int, float]]]] = None):
        self.num_epochs = num_epochs
        self.epoch_interval = epoch_interval
        self.trace = self.initialize_trace(model)

    def should_execute(self, epoch):
        if isinstance(self.epoch_interval, list):
            return epoch in self.epoch_interval
        elif self.epoch_interval is not None:
            return epoch % self.epoch_interval == 0
        return True

    def initialize_trace(self, model):
        trace = {}
        for epoch in range(1, self.num_epochs + 1):
            self.neuron_counter = 1  # Reset neuron_counter for each epoch
            epoch_key = f"E_{epoch}"
            trace[epoch_key] = {}
            for layer_index, (layer_name, neurons) in enumerate(model.neuron_ids.items(), 1):
                layer_key = f"L_{layer_index}"
                trace[epoch_key][layer_key] = {}
                for i in range(len(neurons)):
                    neuron_key = f"n_{self.neuron_counter}"
                    trace[epoch_key][layer_key][neuron_key] = TraceObject(epoch, layer_index, self.neuron_counter)
                    self.neuron_counter += 1
        return trace

    def record_neuron_state(self, trace_obj, activation, input_neurons, output_neurons):
        trace_obj.record_state(activation, input_neurons, output_neurons)

    def get_trace(self, epoch=None, layer=None, neuron_id=None):
        if epoch is not None and layer is not None and neuron_id is not None:
            epoch_key = f"E_{epoch + 1}"
            layer_key = f"L_{layer + 1}"
            neuron_key = f"n_{neuron_id + 1}"
            return self.trace[epoch_key][layer_key].get(neuron_key, None)
        elif epoch is not None and layer is not None:
            epoch_key = f"E_{epoch + 1}"
            layer_key = f"L_{layer + 1}"
            return self.trace[epoch_key][layer_key]
        elif epoch is not None:
            epoch_key = f"E_{epoch + 1}"
            return self.trace[epoch_key]
        return self.trace

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_epochs):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.neuron_ids = {}
        self.neural_index = {}
        self.initialize_neuron_ids()
        self.network_trace = NetworkTrace(self, num_epochs)

    def initialize_neuron_ids(self):
        self.neuron_ids = {
            0: [i for i in range(self.fc1.in_features)],  # Input layer
            1: [i for i in range(self.fc1.out_features)], # Hidden layer 1
            2: [i for i in range(self.fc2.out_features)]  # Output layer
        }
        
        current_index = 1
        for layer in self.neuron_ids:
            self.neural_index[layer] = (current_index, current_index + len(self.neuron_ids[layer]) - 1)
            current_index += len(self.neuron_ids[layer])

    def forward(self, x, epoch):
        x1 = torch.relu(self.fc1(x))
        self.record_activations(epoch, 1, x1, self.neuron_ids[0])
        x2 = self.sigmoid(self.fc2(x1))
        self.record_activations(epoch, 2, x2, self.neuron_ids[1])
        return x2

    def record_activations(self, epoch, layer, activations, input_neurons):
        batch_size = activations.size(0)
        num_neurons = activations.size(1)
        start_index, end_index = self.neural_index[layer]

        for i in range(batch_size):
            for j in range(num_neurons):
                neuron_id = start_index + j  # Adjusted to start from 1-based indexing
                activation_value = activations[i, j].item()
                output_neurons = []  # Define logic to get output neurons if needed
                classification_result = None  # Define logic to get classification result if needed
                epoch_key = f"E_{epoch + 1}"
                layer_key = f"L_{layer + 1}"
                neuron_key = f"n_{neuron_id}"

                try:
                    trace_obj = self.network_trace.trace[epoch_key][layer_key][neuron_key]
                    print(f"Recording activation: epoch={epoch}, layer={layer}, neuron_id={neuron_id}, activation_value={activation_value}")
                    self.network_trace.record_neuron_state(trace_obj, activation_value, input_neurons, output_neurons)
                except KeyError as e:
                    print(f"Error: Neuron {neuron_key} not found in trace at {epoch_key}, {layer_key}. Continuing to next neuron.")
                    continue  # Skip to the next neuron if the current one is not found

    def predict(self, x, epoch):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, epoch)
            predictions = (outputs > 0.5).float()
        return predictions