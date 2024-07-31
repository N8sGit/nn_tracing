import os
import matplotlib.pyplot as plt

def stain(trace):
    # Colors for different classifications
    colors = {0: 'red', 1: 'blue'}

    colored_paths = []
    for classification_result, paths in trace.paths.items():
        color = colors.get(classification_result, 'black')  # Default to 'black' if result is not in colors
        for path in paths:
            colored_paths.append((path, color))
    
    return colored_paths

def plot_model_skeleton(neuron_ids):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot neurons for each layer
    for layer, neurons in neuron_ids.items():
        for neuron in neurons:
            ax.scatter(neuron[0], neuron[1], 0, color='black')

    ax.set_xlabel('Layer')
    ax.set_ylabel('Neuron')
    ax.set_zlabel('Activation')
    plt.title('Model Skeleton')
    plt.show()

