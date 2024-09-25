import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from model_config import config_iris
from traceable_model import TraceableModel
from trace_nn import NetworkTrace
from sample_models import IrisNN
from model_config import ModelConfigurator

def main():
    # Extract configurations specific to the Iris dataset
    num_epochs = config_iris['num_epochs']
    layer_names = config_iris['layer_names']

    # Retrieve DataLoaders from the config
    train_loader = config_iris['data']['train_loader']
    test_loader = config_iris['data']['test_loader']

    # Initialize the base model for the Iris dataset
    base_model = IrisNN()

    # Initialize the NetworkTrace
    network_trace = NetworkTrace()

    # Use the layer names from config_iris for tracing
    layers_to_trace = list(layer_names.values())

    # Wrap the base model with TraceableModel
    model = TraceableModel(base_model, network_trace, layers_to_trace=layers_to_trace)

    # Use ModelConfigurator for multi-class classification
    configurator = ModelConfigurator('multi_class_classification')
    criterion = configurator.get_loss_function()
    output_activation = configurator.get_output_activation()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start training trace
    model.start_training_trace()

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()

            # Ensure labels are in the correct shape (1D tensor of class indices)
            labels = labels.view(-1).long()  # Flatten labels to be 1D and convert to long (int64)

            outputs = model(inputs)
            if output_activation is not None:
                outputs = output_activation(outputs)  # Apply activation if specified
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
        # End of epoch: record weights and biases
        model.end_training_epoch()

    # Export the network_trace
    with open('outputs/network_trace.pkl', 'wb') as f:
        pickle.dump(network_trace, f)

    # Save the model's state
    torch.save(model.state_dict(), 'outputs/trained_model_full.pth')

if __name__ == "__main__":
    main()