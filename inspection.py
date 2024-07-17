# Helper functions to inspect internal model states, outputs, etc

def print_model_parameters(neuron_ids, clusters, layer_mapping, trace):
    def print_dict(dictionary, name):
        print(f"\n{name}:")
        for key, value in dictionary.items():
            print(f"  {key}: {value}")

    print_dict(neuron_ids, "Neuron IDs")
    print_dict(layer_mapping, "Layer Mapping")
    
    print("\nClusters:")
    for neuron_id, cluster_id in clusters.items():
        print(f"  Neuron ID: {neuron_id}, Cluster ID: {cluster_id}")

    print("\nTrace Paths:")
    for path in trace.get_all_paths():
        print(f"  Layer: {path['layer']}, Neuron ID: {path['neuron_id']}, "
            f"Source Neurons: {path['source_neurons']}, Activation: {path['activation']}, "
            f"Classification Result: {path['classification_result']}")

    print("\nCo-Activations:")
    co_activations = trace.get_co_activations()
    for neuron_id, co_activated_neurons in co_activations.items():
        print(f"  Neuron ID: {neuron_id}, Co-Activated Neurons: {list(co_activated_neurons)}")