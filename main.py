import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from data import X_train, y_train, X_test
from model import SimpleNN, NetworkTrace
# from plot import visualize_diff, visualize_filtered_activations, calculate_overlap_and_divergence, plot_model_skeleton
from inspection import print_network_trace, print_all_weights

def main():
    # Initialize the model
    input_size = 20
    hidden_size = 10
    output_size = 1
    num_epochs = 100
    epoch_interval = [10, 20, 30]  # Initialize trace at specific epochs
    model = SimpleNN(input_size, hidden_size, output_size, num_epochs)
    network_trace = NetworkTrace(model, num_epochs, epoch_interval=None)

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
    
    with open('outputs/network_trace.pkl', 'wb') as f:
        pickle.dump(model.network_trace, f)


    print_all_weights(model.network_trace)

    # Here we convert the network trace schema to a dataframe to prepare it for analysis

    # df = convert_nn_schema_to_df(model.network_trace.trace)
    # print('dataframe', df)
if __name__ == "__main__":
    main()