import torch
import torch.nn as nn
import torch.optim as optim
from typing import Union, List, Optional
import numpy as np
from inspection import print_json_like_dump

class TraceObject:
    def __init__(self, epoch, layer, neuron_id):
        self.signature = f"E_{epoch}-{layer}-n_{neuron_id}"
        self.input_neurons = None  # To store the input neuron configuration
        self.mean_activation = None
        self.median_activation = None
        self.std_activation = None
        self.max_activation = None
        self.min_activation = None

    def to_dict(self):
        return {
            "input_neurons": self.input_neurons,
            "mean_activation": self.mean_activation,
            "median_activation": self.median_activation,
            "std_activation": self.std_activation,
            "max_activation": self.max_activation,
            "min_activation": self.min_activation,
            "signature": self.signature
        }

def parse_signature(signature):
    parts = signature.split('-')
    epoch = parts[0]
    layer = parts[1]
    neuron = parts[2]
    return epoch, layer, neuron

class NetworkTrace:
    def __init__(self, model, num_epochs, epoch_interval: Optional[Union[int, float, List[Union[int, float]]]] = None, drop_batches = False):
        self.num_epochs = num_epochs
        self.epoch_interval = epoch_interval
        self.drop_batches = drop_batches
        self.history = {}
        self.final_classification_results = {}  # Store final classification results at the epoch level
        self.trace = self.initialize_trace(model)
        print(f"Initialized trace: {self.trace}")

    def should_execute(self, epoch):
        if isinstance(self.epoch_interval, list):
            return epoch in self.epoch_interval
        elif self.epoch_interval is not None:
            return epoch % self.epoch_interval == 0
        return True

    def initialize_trace(self, model):
        trace = {}
        for epoch in range(0, self.num_epochs):
            self.neuron_counter = 0  # Reset neuron_counter for each epoch
            epoch_key = f"E_{epoch}"
            trace[epoch_key] = {}
            self.history[epoch_key] = {}  # Initialize history for each epoch
            self.final_classification_results[epoch_key] = None  # Initialize final classification result for each epoch
            print(f"Initialized history for {epoch_key}: {self.history[epoch_key]}")  # Debug statement
            for layer_name, neurons in model.neuron_ids.items():
                trace[epoch_key][layer_name] = {}
                for i in range(len(neurons)):
                    neuron_key = f"n_{self.neuron_counter}"
                    trace[epoch_key][layer_name][neuron_key] = TraceObject(epoch, layer_name, self.neuron_counter)
                    self.neuron_counter += 1
        return trace

    def record_neuron_state(self, signature, activation, input_neurons, batch_id):
        print(f"record_neuron_state called with signature: {signature}")
        epoch, layer, neuron_id = parse_signature(signature)
        epoch_key = f"E_{epoch}"
        layer_key = layer
        neuron_key = f"n_{neuron_id}"

        if epoch_key not in self.history:
            self.history[epoch_key] = {}

        if signature not in self.history[epoch_key]:
            self.history[epoch_key][signature] = {
                "activations": {},
                "input_neurons": input_neurons,
                "signature": signature
            }

        history_obj = self.history[epoch_key][signature]

        # Convert tensors to raw values
        activation = activation if not isinstance(activation, torch.Tensor) else activation.item()

        # Check if the input neuron configuration already exists
        if history_obj["input_neurons"] == input_neurons:
            if batch_id in history_obj["activations"]:
                history_obj["activations"][batch_id].append(activation)
            else:
                history_obj["activations"][batch_id] = [activation]
        else:
            history_obj["input_neurons"] = input_neurons
            history_obj["activations"][batch_id] = [activation]

        print(f"Updated history for {epoch_key} {layer_key} {neuron_key}: {history_obj}")  # Debug statement

    def set_final_classification_result(self, epoch, result):
        epoch_key = f"E_{epoch}"
        self.final_classification_results[epoch_key] = result
        print(f"Set final classification result for {epoch_key}: {result}")  # Debug statement
    
    # Handy accessor function to help access the network tree 
    def get_trace(self, signature, depth):
        epoch, layer, neuron = parse_signature(signature)
        if depth == 'neuron':
            # Get neuron associated with signature
            return self.trace[epoch][layer].get(neuron, None)
        elif depth == 'layer':
            # Get layer associated with signature
            return self.trace[epoch][layer]
        elif depth == 'epoch':
            # Get entire epoch
            return self.trace[epoch]
        return self.trace

    def print_history(self):
        for epoch, epoch_data in self.history.items():
            print(f"Epoch: {epoch}")
            for signature, trace_obj in epoch_data.items():
                print(f"Signature: {signature}, Trace Object: {trace_obj.to_dict()}")

    def calculate_statistics(self, activations):
        activations = np.array(activations)
        mean_activation = np.mean(activations)
        median_activation = np.median(activations)
        std_activation = np.std(activations)
        max_activation = np.max(activations)
        min_activation = np.min(activations)
        return mean_activation, median_activation, std_activation, max_activation, min_activation

    def update_trace_object_with_statistics(self):
        for epoch_key, epoch_data in self.history.items():
            for signature, history_obj in epoch_data.items():
                epoch, layer, neuron = parse_signature(history_obj['signature'])
                activations = [activation for batch_activations in history_obj["activations"].values() for activation in batch_activations]
                mean_activation, median_activation, std_activation, max_activation, min_activation = self.calculate_statistics(activations)
                trace_obj = self.trace[epoch][layer][neuron]
                trace_obj.input_neurons = history_obj['input_neurons']
                trace_obj.mean_activation = mean_activation
                trace_obj.median_activation = median_activation
                trace_obj.std_activation = std_activation
                trace_obj.max_activation = max_activation
                trace_obj.min_activation = min_activation
                print_json_like_dump(trace_obj)
                if self.drop_batches:
                    # If specified, drop the batch data after calculation to conserve space 
                    history_obj['activations'].clear()


    def describe_tree(self, epoch):
        # Get roots of tree
        # TO DO : Generalize over specified intervals 
        # epochs_to_search = [ 'E_' + interval for interval in self.epoch_interval]
        print('TRACE IN TREE',self.trace)
        output_layer = self.trace['E_'+ epoch]['L_output']
        for neuron in output_layer:
            print('NEURON IN OUTPUT', neuron)

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_epochs):
        super(SimpleNN, self).__init__()
        self.L_input = nn.Linear(input_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.L_output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.neuron_ids = {}
        self.neural_index = {}
        self.initialize_neuron_ids()
        self.network_trace = NetworkTrace(self, num_epochs)

    def initialize_neuron_ids(self):
        self.neuron_ids = {
            'L_input': [i for i in range(self.L_input.in_features)],  # Input layer
            'L_hidden_1': [i for i in range(self.fc1.out_features)],  # Hidden layer 1
            'L_output': [i for i in range(self.L_output.out_features)]  # Output layer
        }

        current_index = 0  # Start with zero-based indexing
        for layer in self.neuron_ids:
            self.neural_index[layer] = (current_index, current_index + len(self.neuron_ids[layer]) - 1)
            current_index += len(self.neuron_ids[layer])

    def forward(self, x, epoch):
        # Input layer
        input_neurons_signatures = [f"E_{epoch}-L_input-n_{i}" for i in range(self.L_input.in_features)]
        x1 = torch.relu(self.L_input(x))
        self.record_activations(epoch, 'L_input', x1, input_neurons_signatures)

        # Hidden layer
        input_neurons_signatures = [f"E_{epoch}-L_input-n_{i}" for i in range(self.L_input.out_features)]
        x2 = torch.relu(self.fc1(x1))
        self.record_activations(epoch, 'L_hidden_1', x2, input_neurons_signatures)

        # Output layer
        input_neurons_signatures = [f"E_{epoch}-L_hidden_1-n_{i}" for i in range(self.fc1.out_features)]
        x3 = self.sigmoid(self.L_output(x2))
        final_classification_result = (x3 >= 0.5).float()  # Get final classification result
        self.record_activations(epoch, 'L_output', x3, input_neurons_signatures)

        # Set final classification result for the epoch
        self.network_trace.set_final_classification_result(epoch, final_classification_result)

        return x3

    def record_activations(self, epoch, layer, activations, input_neurons):
        batch_size, num_neurons = activations.size()
        start_index, _ = self.neural_index[layer]

        for i in range(batch_size):
            input_neurons_signatures = input_neurons  # Directly pass the input neuron signatures
            batch_id = i
            for j in range(num_neurons):
                neuron_id = start_index + j  # Adjusted to start from 0-based indexing
                activation_value = activations[i, j].item()
                signature = f"E_{epoch}-{layer}-n_{neuron_id}"

                try:
                    trace_obj = self.network_trace.trace[f"E_{epoch}"][layer][f"n_{neuron_id}"]
                    print(f"Recording activation: epoch={epoch}, layer={layer}, neuron_id={neuron_id}, activation_value={activation_value}")
                    self.network_trace.record_neuron_state(
                        signature, activation_value, input_neurons_signatures, batch_id
                    )
                except KeyError:
                    print(f"Error: Neuron {neuron_id} not found in trace at E_{epoch}, {layer}. Continuing to next neuron.")
                    continue  # Skip to the next neuron if the current one is not found

    def predict(self, x, epoch):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, epoch)
            predictions = (outputs > 0.5).float()
        return predictions