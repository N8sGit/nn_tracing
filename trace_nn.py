import torch
from typing import Union, List, Optional
import numpy as np
from inspection import print_json_like_dump
from helpers import set_model_level_label, parse_signature

class TraceObject:
    def __init__(self, epoch, layer, neuron_id):
        self.signature = f"E_{epoch}-{layer}-n_{neuron_id}"
        self.input_neurons = None  # To store the input neuron configuration
        self.activation_metrics = {
            'mean': None,
            'median': None,
            'std': None,
            'max': None,
            'min': None
        }
        self.bias_term_metrics = {
            'mean': None,
            'median': None,
            'std': None,
            'max': None,
            'min': None
        }
        self.weight_metrics = {
            'mean': None,
            'median': None,
            'std': None,
            'max': None,
            'min': None
        }
        self.breadcrumbs = []  # Store breadcrumbs as a list of input neuron signatures

    def to_dict(self):
        return {
            "input_neurons": self.input_neurons,
            "activation_metrics": self.activation_metrics,
            "bias_term_metrics": self.bias_term_metrics,
            "weight_metrics": self.weight_metrics,
            "breadcrumbs": self.breadcrumbs,
            "signature": self.signature
        }

class NetworkTrace:
    def __init__(self, model, num_epochs, epoch_interval: Optional[Union[int, float, List[Union[int, float]]]] = None, drop_batches=False):
        self.num_epochs = num_epochs
        self.epoch_interval = epoch_interval
        self.drop_batches = drop_batches
        self.history = {}
        self.final_classification_results = {}  # Store final classification results at the epoch level
        self.trace = self.initialize_trace(model)
        self.weights = self.initialize_weights(model)
        self.biases = self.initialize_biases(model)
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

    def initialize_weights(self, model):
        weights = {}
        for epoch in range(0, self.num_epochs):
            epoch_key = f"E_{epoch}"
            weights[epoch_key] = {}
            for layer_name in model.neuron_ids.keys():
                weights[epoch_key][layer_name] = None
        return weights

    def initialize_biases(self, model):
        biases = {}
        for epoch in range(0, self.num_epochs):
            epoch_key = f"E_{epoch}"
            biases[epoch_key] = {}
            for layer_name in model.neuron_ids.keys():
                biases[epoch_key][layer_name] = None
        return biases

    def record_neuron_trace(self, signature, activation, batch_id):
        print(f"record_neuron_trace called with signature: {signature}")
        epoch, layer, neuron_id = parse_signature(signature)
        epoch_key = f"E_{epoch}"
        layer_key = layer
        neuron_key = f"n_{neuron_id}"

        if epoch_key not in self.history:
            self.history[epoch_key] = {}

        if signature not in self.history[epoch_key]:
            self.history[epoch_key][signature] = {
                "activations": {},
                "signature": signature
            }

        history_obj = self.history[epoch_key][signature]

        # Convert tensors to raw values
        activation = activation if not isinstance(activation, torch.Tensor) else activation.item()
        
        history_obj["activations"][batch_id] = activation

        print(f"Updated history for {epoch_key} {layer_key} {neuron_key}: {history_obj}") 

    def record_layer_trace(self, epoch, layer_name, weights, biases):
        epoch_key = f"E_{epoch}"
        self.weights[epoch_key][layer_name] = weights
        self.biases[epoch_key][layer_name] = biases
        print(f"Recorded weights and biases for {layer_name} in {epoch_key}")  
    
    def set_final_classification_result(self, epoch, result):
        # Note: In future, we may want to associate results with batch numbers and not flatten them like such for simplicity
        epoch_key = f"E_{epoch}"
        # Flatten the result tensor to a 1D list
        flattened_result = result.view(-1).tolist()
        self.final_classification_results[epoch_key] = flattened_result
        print(f"Set final classification result for {epoch_key}: {flattened_result}")  
        
    # Handy accessor function to help access the network tree 
    def get_trace(self, epoch=None, layer=None, neuron_id=None):
        if epoch is not None and layer is not None and neuron_id is not None:
            epoch_key = f"{epoch}"
            layer_key = layer
            neuron_key = f"{neuron_id}"
            return self.trace[epoch_key][layer_key].get(neuron_key, None)
        elif epoch is not None and layer is not None:
            epoch_key = f"{epoch}"
            layer_key = layer
            return self.trace[epoch_key][layer_key]
        elif epoch is not None:
            epoch_key = set_model_level_label('epoch', 0)
            return self.trace[epoch_key]
        return self.trace

    def print_history(self):
        for epoch, epoch_data in self.history.items():
            print(f"Epoch: {epoch}")
            for signature, trace_obj in epoch_data.items():
                print(f"Signature: {signature}, Trace Object: {trace_obj}")

    def calculate_statistics(self, metrics):
        metrics = [metric for metric in metrics if metric is not None]  # Filter out None values
        if not metrics:
            return {'mean': None, 'median': None, 'std': None, 'max': None, 'min': None}
        metrics = np.array(metrics)
        mean = np.mean(metrics)
        median = np.median(metrics)
        std = np.std(metrics)
        max = np.max(metrics)
        min = np.min(metrics)
        return {'mean': mean, 'median': median, 'std': std, 'max': max, 'min': min}

    def update_trace_object_with_statistics(self):
        for epoch_key, epoch_data in self.history.items():
            for signature, history_obj in epoch_data.items():
                epoch, layer, neuron = parse_signature(history_obj['signature'])
                activations = [activation for activation in history_obj["activations"].values()]
                trace_obj = self.trace[epoch][layer][neuron]
                trace_obj.activation_metrics = self.calculate_statistics(activations)
                print_json_like_dump(trace_obj)
                if self.drop_batches:
                    # If specified, drop the historical batch data after calculation to conserve space 
                    history_obj['activations'].clear()
