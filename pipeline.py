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
        self.num_time_steps = config.num_time_steps
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
        for time_step in range(self.num_time_steps):
            self.model.train()
            total_loss = 0.0
            total_samples = 0

            with self.model.trace(time_step, mode='training'):
                for inputs, labels in self.train_loader:
                    self.optimizer.zero_grad()

                    labels = labels.view(-1)
                    outputs = self.model(inputs)

                    if self.output_activation is not None:
                        outputs = self.output_activation(outputs)

                    # Adjust labels for different loss functions
                    if isinstance(self.loss_function, nn.CrossEntropyLoss):
                        labels = labels.long()
                    elif isinstance(self.loss_function, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                        labels = labels.float()
                    elif isinstance(self.loss_function, nn.MSELoss):
                        labels = labels.float().unsqueeze(1)
                        outputs = outputs.float()
                    else:
                        # Custom handling or raise an error
                        raise ValueError('Unsupported loss function type.')

                    loss = self.loss_function(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss.item() * inputs.size(0)
                    total_samples += inputs.size(0)
                    average_loss = total_loss / total_samples
                    print(f'Time Step [{time_step+1}/{self.num_time_steps}], Loss: {average_loss:.4f}')
    
    def evaluate(self, evaluation_function=None):
        """
        Evaluates the model on the test dataset.

        Args:
            evaluation_function (callable, optional): A custom function to evaluate the model's performance.
                If not provided, a default evaluation based on the model type will be used.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_outputs = []
        all_labels = []

        # Initialize time_step for batch-level tracing
        time_step = 0

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                labels = labels.view(-1)
                # Use batch-level tracing if needed
                with self.model.trace(time_step, mode='inference'):
                    outputs = self.model(inputs)

                    if self.output_activation is not None:
                        outputs = self.output_activation(outputs)

                    # Adjust labels and outputs for different loss functions
                    if isinstance(self.loss_function, nn.CrossEntropyLoss):
                        labels = labels.long()
                    elif isinstance(self.loss_function, (nn.BCELoss, nn.BCEWithLogitsLoss)):
                        labels = labels.float()

                    loss = self.loss_function(outputs, labels)
                    total_loss += loss.item() * inputs.size(0)
                    total_samples += inputs.size(0)

                    all_outputs.append(outputs)
                    all_labels.append(labels)

                # Increment the time step for each batch
                time_step += 1

        average_loss = total_loss / total_samples

        # Concatenate all outputs and labels
        all_outputs = torch.cat(all_outputs)
        all_labels = torch.cat(all_labels)

        # Store predictions in the network trace
        self.network_trace.store_predictions('evaluation', all_outputs)

        # Use custom evaluation function if provided
        if evaluation_function is not None:
            metrics = evaluation_function(all_outputs, all_labels)
        else:
            metrics = self._default_evaluation(all_outputs, all_labels)

        print(f'Evaluation Results - Loss: {average_loss:.4f}, {metrics}')
        print(f'Total Predictions: {all_outputs.size(0)}')

    def _default_evaluation(self, outputs, labels):
        """
        Default evaluation method based on the model's loss function.

        Args:
            outputs (torch.Tensor): Model predictions.
            labels (torch.Tensor): True labels.

        Returns:
            str: Formatted string of evaluation metrics.
        """
        if isinstance(self.loss_function, nn.CrossEntropyLoss):
            # Multi-class classification
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0) * 100
            return f'Accuracy: {accuracy:.2f}%'
        elif isinstance(self.loss_function, (nn.BCELoss, nn.BCEWithLogitsLoss)):
            # Binary classification
            predicted = (outputs >= 0.5).float()
            correct = (predicted == labels).sum().item()
            accuracy = correct / labels.size(0) * 100
            return f'Accuracy: {accuracy:.2f}%'
        elif isinstance(self.loss_function, nn.MSELoss):
            # Regression
            mse = nn.MSELoss()(outputs, labels)
            return f'Mean Squared Error: {mse.item():.4f}'
        else:
            return 'No default evaluation available for this loss function.'

    def save_results(self, model_path='outputs/trained_model_full.pth', trace_path='outputs/network_trace.pkl'):
        # Save the model's state
        torch.save(self.model.state_dict(), model_path)
        # Export the network_trace
        with open(trace_path, 'wb') as f:
            pickle.dump(self.network_trace, f)
