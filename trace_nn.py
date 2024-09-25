import torch
import pandas as pd

class NeuronTrace:
    def __init__(self, global_neuron_id, epoch, layer_name, weight=None, bias=None):
        self.global_neuron_id = global_neuron_id  # Unique and universal ID
        self.epoch = epoch
        self.layer_name = layer_name
        self.weight = weight  # Tensor of weights from previous layer neurons
        self.bias = bias
        self.activations = []  # List to store activations per sample
        self.activation_sum = 0.0
        self.activation_squared_sum = 0.0
        self.activation_max = float('-inf')
        self.activation_min = float('inf')
        self.count = 0

    def update_activation(self, activation_values):
        for activation_value in activation_values:
            self.activation_sum += activation_value
            self.activation_squared_sum += activation_value ** 2
            self.count += 1
            self.activation_max = max(self.activation_max, activation_value)
            self.activation_min = min(self.activation_min, activation_value)
            self.activations.extend(activation_values)
    
    def get_activation_statistics(self):
        if self.count == 0:
            return {
                'mean': 0,
                'std_dev': 0,
                'max': None,
                'min': None,
                'count': 0
            }
        mean = self.activation_sum / self.count
        variance = (self.activation_squared_sum / self.count) - (mean ** 2)
        std_dev = variance ** 0.5
        return {
            'mean': mean,
            'std_dev': std_dev,
            'max': self.activation_max,
            'min': self.activation_min,
            'count': self.count
        }

import torch
import pandas as pd

class NetworkTrace:
    def __init__(self):
        self.trace = {}
        self.layer_order = []
        self.global_neuron_id_map = {}
        self.global_neuron_counter = 0
        self.predictions = {}

    def initialize_inference_trace(self):
        """Initialize trace for inference if not present."""
        if 'inference' not in self.trace:
            self.trace['inference'] = {}

    def initialize_training_trace(self, epoch=0):
        """Initialize trace for training if not present."""
        if epoch not in self.trace:
            self.trace[epoch] = {}

    def update_layer_activations(self, epoch, layer_name, activations):
        # activations: numpy array of shape (batch_size, num_neurons)
        if epoch not in self.trace:
            if epoch == 'inference':
                self.trace[epoch] = {}
            else:
                raise ValueError(f"No trace found for epoch {epoch}.")

        num_neurons = activations.shape[1]
        for neuron_index in range(num_neurons):
            global_neuron_id = self.global_neuron_id_map.get((layer_name, neuron_index))
            if global_neuron_id is None:
                continue

            # Get or create the NeuronTrace for this neuron
            neuron_trace = self.trace[epoch].get(global_neuron_id)
            if neuron_trace is None:
                neuron_trace = NeuronTrace(global_neuron_id, epoch, layer_name)
                self.trace[epoch][global_neuron_id] = neuron_trace

            # Get activations for this neuron across the batch
            neuron_activations = activations[:, neuron_index]
            neuron_trace.update_activation(neuron_activations.tolist())

    def store_predictions(self, epoch, predictions):
        self.predictions[epoch] = predictions.cpu().numpy() 
    
    def assign_global_neuron_ids(self, model):
        for layer_name, layer in model.named_modules():
            if hasattr(layer, 'weight') and layer.weight is not None:
                num_neurons = layer.weight.size(0)
                if layer_name not in self.layer_order:
                    self.layer_order.append(layer_name)
                for neuron_index in range(num_neurons):
                    # Assign a unique global neuron ID if not already assigned
                    if (layer_name, neuron_index) not in self.global_neuron_id_map:
                        self.global_neuron_id_map[(layer_name, neuron_index)] = self.global_neuron_counter
                        self.global_neuron_counter += 1

    def update_trace(self, epoch, model):
        if epoch == 0:
            # Only assign global neuron IDs during the first epoch
            self.assign_global_neuron_ids(model)

        for layer_name, layer in model.named_modules():
            if hasattr(layer, 'weight') and layer.weight is not None:
                weight_matrix = layer.weight.data.clone().detach().cpu()
                bias_vector = layer.bias.data.clone().detach().cpu() if hasattr(layer, 'bias') and layer.bias is not None else None

                num_neurons = weight_matrix.size(0)

                # Initialize dictionaries if not present
                self.trace.setdefault(epoch, {})

                for neuron_index in range(num_neurons):
                    neuron_weight = weight_matrix[neuron_index]
                    neuron_bias = bias_vector[neuron_index] if bias_vector is not None else None

                    # Get the unique global neuron ID
                    global_neuron_id = self.global_neuron_id_map[(layer_name, neuron_index)]

                    neuron_trace = NeuronTrace(global_neuron_id, epoch, layer_name, neuron_weight, neuron_bias)
                    self.trace[epoch][global_neuron_id] = neuron_trace
    
    def neurons_to_dataframe(self):
        data = []
        for epoch_key, neurons in self.trace.items():
            for global_neuron_id, neuron_trace in neurons.items():
                stats = neuron_trace.get_activation_statistics()
                data.append({
                    'global_neuron_id': global_neuron_id,
                    'epoch': epoch_key,
                    'layer': neuron_trace.layer_name,
                    'mean_activation': stats['mean'],
                    'std_dev_activation': stats['std_dev'],
                    'max_activation': stats['max'],
                    'min_activation': stats['min'],
                    'count': stats['count'],
                    'bias': neuron_trace.bias.item() if neuron_trace.bias is not None else None
                })
        return pd.DataFrame(data)

    def connections_to_dataframe(self):
        data = []
        for epoch_key, neurons in self.trace.items():
            # Build a mapping from layer_name to neurons in that layer
            layer_neurons = {}
            for global_neuron_id, neuron_trace in neurons.items():
                layer_neurons.setdefault(neuron_trace.layer_name, {})[global_neuron_id] = neuron_trace

            for layer_idx, layer_name in enumerate(self.layer_order):
                if layer_idx == 0:
                    continue  # Skip the first layer as it has no incoming connections
                prev_layer_name = self.layer_order[layer_idx - 1]

                curr_neurons = layer_neurons.get(layer_name, {})
                prev_neurons = layer_neurons.get(prev_layer_name, {})
                num_prev_neurons = len(prev_neurons)
                if num_prev_neurons == 0:
                    continue

                # Build mapping from neuron indices to global neuron IDs in previous layer
                prev_neuron_idx_to_id = {}
                for global_neuron_id, neuron_trace in prev_neurons.items():
                    neuron_index = self.get_neuron_index(global_neuron_id)
                    prev_neuron_idx_to_id[neuron_index] = global_neuron_id

                for global_neuron_id, neuron_trace in curr_neurons.items():
                    weight_vector = neuron_trace.weight.numpy()
                    for source_neuron_idx in range(len(weight_vector)):
                        weight = weight_vector[source_neuron_idx]
                        source_neuron_id = prev_neuron_idx_to_id.get(source_neuron_idx)
                        if source_neuron_id is None:
                            continue
                        data.append({
                            'epoch': epoch_key,
                            'source_layer': prev_layer_name,
                            'source_neuron': source_neuron_id,
                            'target_layer': neuron_trace.layer_name,
                            'target_neuron': global_neuron_id,
                            'weight': weight
                        })
        return pd.DataFrame(data)
    
    def get_neuron_index(self, global_neuron_id):
        for (layer_name, neuron_index), gid in self.global_neuron_id_map.items():
            if gid == global_neuron_id:
                return neuron_index
        return None