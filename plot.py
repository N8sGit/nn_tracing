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



def visualize_filtered_activations(trace, classification_result, save_result=False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Filter paths by classification result
    filtered_paths = filter_paths_by_classification(trace, classification_result)

    # Get colored paths
    colored_paths = stain(trace)

    # Plot activations
    for path, color in colored_paths:
        if path in filtered_paths:
            layer = path["layer"]
            neuron_id = path["neuron_id"]
            source_neurons = path["source_neruons"]
            activation = path["activation"]
        
            for neuron in source_neurons:
                ax.plot([layer-1, layer], [source_neurons, neuron_id], [0, 1], color=color)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Neuron')
    ax.set_zlabel('Activation')
    plt.title(f'Activations for Classification Result: {classification_result}')
    ## Optionally save and export plot result
    if save_result:
        file_path = os.path.join('results', f'activations_class_{classification_result}.png')
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.savefig(file_path)
        print(f"Plot saved to {file_path}")
    
    plt.show()

def plot_all_paths(trace):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot all paths
    for classification_result, paths in trace.paths.items():
        for path in paths:
            layer = path["layer"]
            neuron_id = path["neuron_id"]
            activation = path["activation"]
            color = 'blue' if classification_result == 1 else 'red'
            
            # Plot the neuron activation
            ax.scatter(layer, neuron_id[1], activation, color=color)

    ax.set_xlabel('Layer')
    ax.set_ylabel('Neuron')
    ax.set_zlabel('Activation')
    plt.title('Neuron Activations in All Layers')
    plt.show()