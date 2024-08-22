import torch
import torch.nn as nn
from trace_nn import NetworkTrace
from typing import Union, List, Optional


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_epochs, epoch_interval: Optional[Union[int, float, List[Union[int, float]]]] = None,):
        super(SimpleNN, self).__init__()
        self.L_input = nn.Linear(input_size, hidden_size)
        self.L_hidden_1 = nn.Linear(hidden_size, hidden_size)
        self.L_output = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.epoch_interval = epoch_interval
        self.num_epochs = num_epochs
        self.neuron_ids = {}
        self.neural_index = {}
        self.initialize_neuron_ids()
        self.network_trace = NetworkTrace(self, num_epochs)

    def initialize_neuron_ids(self):
        self.neuron_ids = {
            'L_input': [i for i in range(self.L_input.in_features)],  # Input layer
            'L_hidden_1': [i for i in range(self.L_hidden_1.out_features)],  # Hidden layer 1
            'L_output': [i for i in range(self.L_output.out_features)]  # Output layer
        }

        current_index = 0  # Start with zero-based indexing
        for layer in self.neuron_ids:
            self.neural_index[layer] = (current_index, current_index + len(self.neuron_ids[layer]) - 1)
            current_index += len(self.neuron_ids[layer])

    def forward(self, x, epoch):
        # Input layer
        x1 = torch.relu(self.L_input(x))
        if self.should_execute(epoch):
            self.record_neural_states(epoch, 'L_input', x1)
            self.record_layer_state(epoch, 'L_input')
        # Hidden layer
        x2 = torch.relu(self.L_hidden_1(x1))
        if self.should_execute(epoch):
            self.record_neural_states(epoch, 'L_hidden_1', x2)
            self.record_layer_state(epoch, 'L_hidden_1')
        # Output layer
        x3 = self.sigmoid(self.L_output(x2))
        if self.should_execute(epoch):
            final_classification_result = (x3 >= 0.5).float()  # Get final classification result
            self.record_neural_states(epoch, 'L_output', x3)
            self.record_layer_state(epoch, 'L_output')
        # Set final classification result for the epoch
            self.network_trace.set_final_classification_result(epoch, final_classification_result)

        return x3
    
    def should_execute(self, epoch):
        if self.epoch_interval == -1:
            return epoch == self.num_epochs
        elif isinstance(self.epoch_interval, list):
            return epoch in self.epoch_interval
        elif self.epoch_interval is not None:
            return epoch % self.epoch_interval == 0
        return True

    def record_layer_state(self, epoch, layer):
        weights = self.get_layer_weights(layer)
        biases = self.get_layer_bias(layer)
        print(f'Weights in record_layer_state for {layer}:', weights)
        print(f'Biases in record_layer_state for {layer}:', biases)
        # Record weights and biases in the network trace
        self.network_trace.record_layer_trace(epoch, layer, weights, biases)

    def record_neural_states(self, epoch, layer, activations):
        batch_size, num_neurons = activations.size()
        start_index, _ = self.neural_index[layer]

        for i in range(batch_size):
            batch_id = i
            for j in range(num_neurons):
                neuron_id = start_index + j 
                activation_value = activations[i, j].item()
                signature = f"E_{epoch}-{layer}-n_{neuron_id}"
                try:
                    trace_obj = self.network_trace.trace[f"E_{epoch}"][layer][f"n_{neuron_id}"]
                    print(f"Recording activation: epoch={epoch}, layer={layer}, neuron_id={neuron_id}, activation_value={activation_value}")
                    self.network_trace.record_neuron_trace(
                        signature, activation_value, batch_id
                    )
                except KeyError:
                    print(f"Error: Neuron {neuron_id} not found in trace at E_{epoch}, {layer}. Continuing to next neuron.")
                    continue  # Skip to the next neuron if the current one is not found

    def get_layer_bias(self, layer_name):
        state_dict = self.state_dict()
        biases = state_dict[f'{layer_name}.bias']
        return biases

    def get_layer_weights(self, layer_name):
        state_dict = self.state_dict()
        weights = state_dict[f'{layer_name}.weight']
        return weights

    def predict(self, x, epoch):
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x, epoch)
            predictions = (outputs > 0.5).float()
        return predictions