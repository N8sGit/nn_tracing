# traceable_pipeline.py

import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from traceable_model import TraceableModel
from trace_nn import NetworkTrace
from data import DataHandler
from model_config import ModelConfigurator

class TraceablePipeline:
    def __init__(self, model, config, model_type='multi_class_classification'):
        # Extract configurations
        self.num_epochs = config.num_epochs
        self.layer_names = config.layer_names
        self.batch_size = config.batch_size or 32

        # Initialize data handler
        self.data_handler = DataHandler(
            dataset_type='iris',
            n_samples=config.num_samples,
            n_features=config.input_size,
            n_classes=config.output_size,
            test_size=0.2,
            random_state=42
        )
        self.data_handler.generate_data()
        self.train_loader, self.test_loader = self.data_handler.get_data_loaders(batch_size=self.batch_size)

        # Initialize the NetworkTrace
        self.network_trace = NetworkTrace()

        # Use the layer names for tracing
        layers_to_trace = list(self.layer_names.values())

        # Wrap the base model with TraceableModel
        self.model = TraceableModel(model, self.network_trace, layers_to_trace=layers_to_trace)

        # Use ModelConfigurator
        self.configurator = ModelConfigurator(model_type)
        self.loss_function = self.configurator.get_loss_function()
        self.output_activation = self.configurator.get_output_activation()

        # Define optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        for epoch in range(self.num_epochs):
            self.model.train()
            # Use the context manager to handle tracing for the epoch
            with self.model.trace(epoch, mode='training'):
                for inputs, labels in self.train_loader:
                    self.optimizer.zero_grad()
                    labels = labels.view(-1).long()
                    outputs = self.model(inputs)
                    if self.output_activation is not None:
                        outputs = self.output_activation(outputs)
                    loss = self.loss_function(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Loss: {loss.item():.4f}')

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        # Optionally trace during evaluation
        with self.model.trace(epoch='evaluation', mode='inference'):
            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    labels = labels.view(-1).long()
                    outputs = self.model(inputs)
                    if self.output_activation is not None:
                        outputs = self.output_activation(outputs)
                    loss = self.loss_function(outputs, labels)
                    total_loss += loss.item() * inputs.size(0)

                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += labels.size(0)
                    correct_predictions += (predicted == labels).sum().item()

        average_loss = total_loss / total_samples
        accuracy = correct_predictions / total_samples * 100

        print(f'Evaluation Results - Loss: {average_loss:.4f}, Accuracy: {accuracy:.2f}%')

    def save_results(self, model_path='outputs/trained_model_full.pth', trace_path='outputs/network_trace.pkl'):
        # Save the model's state
        torch.save(self.model.state_dict(), model_path)
        # Export the network_trace
        with open(trace_path, 'wb') as f:
            pickle.dump(self.network_trace, f)