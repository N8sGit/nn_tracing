import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data import X_train, y_train, X_test
from model import SimpleNN

class TraceObject:
    def __init__(self):
        self.paths = {0: [], 1: []}  # Separate lists for classification results 0 and 1

    def record_path(self, layer, neuron_id, input_ids, activation, classification_result):
        path_data = {
            "layer": layer,
            "neuron_id": neuron_id,
            "input_ids": input_ids,
            "activation": activation
        }
        self.paths[classification_result].append(path_data)

class TracingNN(SimpleNN):
    def __init__(self):
        super(TracingNN, self).__init__()
        self.trace = TraceObject()

    def forward(self, x):
        input_ids = torch.arange(x.size(0)).tolist()  # Using batch size as input IDs
        x = torch.relu(self.fc1(x))
        self.record_activations(1, x, input_ids)
        input_ids = torch.arange(x.size(0)).tolist()  # Update input IDs for next layer
        x = self.sigmoid(self.fc2(x))
        self.record_activations(2, x, input_ids)
        return x

    def record_activations(self, layer, x, input_ids):
        predictions = (x > 0.5).float()
        for i in range(x.size(0)):  # Iterate over batch size
            for neuron_id in range(x.size(1)):  # Iterate over neurons
                if x[i, neuron_id] > 0:
                    # Using int(predictions[i, 0].item()) to ensure single scalar value
                    self.trace.record_path(layer, neuron_id, input_ids, x[i, neuron_id].item(), int(predictions[i, 0].item()))

    def predict(self, x):
        outputs = self.forward(x)
        predictions = (outputs > 0.5).float()
        return predictions

# Initialize the model, loss function, and optimizer
model = TracingNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training complete.")

# Run the model on test data to record activations and classification results
model.eval()
with torch.no_grad():
    predictions = model.predict(X_test)

def stain(trace):
    # Colors for different classifications
    colors = {0: 'red', 1: 'blue'}

    colored_paths = []
    for classification_result, paths in trace.paths.items():
        color = colors.get(classification_result, 'black')  # Default to 'black' if result is not in colors
        for path in paths:
            colored_paths.append((path, color))
    
    return colored_paths

def filter_paths_by_classification(trace, classification_result):
    return trace.paths.get(classification_result, [])

def calculate_overlap_and_divergence(trace):
    activations_0 = trace.paths[0]
    activations_1 = trace.paths[1]

    # Create sets of neuron IDs for each classification result
    neurons_0 = {(path["layer"], path["neuron_id"]) for path in activations_0}
    neurons_1 = {(path["layer"], path["neuron_id"]) for path in activations_1}

    # Calculate overlap and divergence
    overlap = neurons_0 & neurons_1
    divergence_0 = neurons_0 - neurons_1
    divergence_1 = neurons_1 - neurons_0

    return overlap, divergence_0, divergence_1

def visualize_diff(overlap, divergence_0, divergence_1):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot overlap in green
    for neuron in overlap:
        ax.scatter(neuron[0], neuron[1], 0, color='green')

    # Plot divergence for classification result 0 in red
    for neuron in divergence_0:
        ax.scatter(neuron[0], neuron[1], 1, color='red')

    # Plot divergence for classification result 1 in blue
    for neuron in divergence_1:
        ax.scatter(neuron[0], neuron[1], 1, color='blue')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Neuron')
    ax.set_zlabel('Activation')
    plt.title('Overlap and Divergence of Neuron Activations')
    plt.show()

# Example usage
overlap, divergence_0, divergence_1 = calculate_overlap_and_divergence(model.trace)
visualize_diff(overlap, divergence_0, divergence_1)

# def visualize_filtered_activations(trace, classification_result, save_result=False):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')

#     # Filter paths by classification result
#     filtered_paths = filter_paths_by_classification(trace, classification_result)

#     # Get colored paths
#     colored_paths = stain(trace)

#     # Plot activations
#     for path, color in colored_paths:
#         if path in filtered_paths:
#             layer = path["layer"]
#             neuron_id = path["neuron_id"]
#             input_ids = path["input_ids"]
#             activation = path["activation"]
        
#             for input_id in input_ids:
#                 ax.plot([layer-1, layer], [input_id, neuron_id], [0, 1], color=color)

#     ax.set_xlabel('Layer')
#     ax.set_ylabel('Neuron')
#     ax.set_zlabel('Activation')
#     plt.title(f'Activations for Classification Result: {classification_result}')
#     ## Optionally save and export plot result
#     if save_result:
#         file_path = os.path.join('results', f'activations_class_{classification_result}.png')
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         plt.savefig(file_path)
#         print(f"Plot saved to {file_path}")
    
#     plt.show()

# Example usage
# classification_result_to_visualize = 0  # Choose the classification result to visualize (0 or 1)
# visualize_filtered_activations(model.trace, classification_result_to_visualize, save_result=True)