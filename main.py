import torch
import torch.nn as nn
import torch.optim as optim
from data import X_train, y_train, X_test
from model import SimpleNN
from plot import visualize_diff, visualize_filtered_activations, calculate_overlap_and_divergence, plot_all_paths

class TraceObject:
    def __init__(self):
        self.paths = {0: [], 1: []}  # Separate lists for classification results 0 and 1

    def record_path(self, layer, neuron_id, source_neurons, activation, classification_result):
        path_data = {
            "layer": layer,
            "neuron_id": neuron_id,
            "source_neurons": source_neurons,
            "activation": activation
        }
        self.paths[classification_result].append(path_data)
        print(f"Recorded path: {path_data} for classification result: {classification_result}")

class TracingNN(SimpleNN):
    def __init__(self):
        super(TracingNN, self).__init__()
        self.trace = TraceObject()
        self.initialize_neuron_ids()

    def forward(self, x):
        x1 = torch.relu(self.fc1(x))
        self.record_activations(1, x1, self.neuron_ids["layer1"], self.neuron_ids["input"])
        x2 = self.sigmoid(self.fc2(x1))
        self.record_activations(2, x2, self.neuron_ids["layer2"], self.neuron_ids["layer1"])
        return x2

    def record_activations(self, layer, x, neuron_ids, previous_layer_neurons):
        predictions = (x > 0.5).float()
        for i in range(x.size(0)):  # Iterate over batch size
            for neuron_id in range(x.size(1)):  # Iterate over neurons
                if x[i, neuron_id] > 0:
                    classification_result = int(predictions[i, 0].item())
                    source_neurons = [previous_layer_neurons[j] for j in range(x.size(1)) if x[i, neuron_id] > 0]
                    self.trace.record_path(layer, neuron_ids[neuron_id], source_neurons, x[i, neuron_id].item(), classification_result)
                    print(f"Layer: {layer}, Neuron: {neuron_ids[neuron_id]}, Activation: {x[i, neuron_id].item()}, Classification: {classification_result}, Source Neurons: {source_neurons}")

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


# Calculate overlap and divergence
# overlap, divergence_0, divergence_1 = calculate_overlap_and_divergence(model.trace)
# visualize_diff(overlap, divergence_0, divergence_1)

# Example usage
# Example usage
plot_all_paths(model.trace)
# classification_result_to_visualize = 0  # Choose the classification result to visualize (0 or 1)
# visualize_filtered_activations(model.trace, classification_result_to_visualize, save_result=True)