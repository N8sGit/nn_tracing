import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from data import X_test
from model import TracableNN, NetworkTrace
from model_config import config
from inspection import print_network_trace


def main():
    # Initialize the model
    input_size = config['input_size']
    hidden_size = config['hidden_size']
    output_size = config['output_size']
    num_epochs = config['num_epochs']
    epoch_interval = config['epoch_interval']  # Initialize trace at specific epochs
    model = TracableNN(input_size, hidden_size, output_size, num_epochs)
    network_trace = NetworkTrace(model, num_epochs)

    # Initialize the optimizer and loss function
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(torch.randn(5, input_size), epoch)
        targets = torch.rand(5, output_size)  # Ensure target values are between 0 and 1
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        model.network_trace.update_trace_object_with_statistics()
        
    # Record predictions on test data
        model.eval()
    with torch.no_grad():
        predictions = model.predict(X_test, num_epochs - 1)

    # Print the network trace
    print_network_trace(model.network_trace)
    # Export the network_trace
    with open('outputs/network_trace.pkl', 'wb') as f:
        pickle.dump(model.network_trace, f)
    # Save the model's state   
    torch.save(model.state_dict(), 'outputs/trained_model_full.pth')

if __name__ == "__main__":
    main()