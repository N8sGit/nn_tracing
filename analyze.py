## Mathematical tools to analyze results

def get_shape_info(neuron_ids, clusters, layer_mapping, trace):
    def get_dict_info(dictionary, name):
        info = {}
        info['length'] = len(dictionary)
        info['data_types'] = {key: type(value) for key, value in dictionary.items()}
        return {name: info}

    def get_trace_info(trace):
        paths = trace.get_all_paths()
        num_paths = len(paths)
        layers = set(path['layer'] for path in paths)
        unique_neurons = set(path['neuron_id'] for path in paths)
        activations = [path['activation'] for path in paths]

        activation_stats = {
            'mean': sum(activations) / len(activations) if activations else 0,
            'min': min(activations) if activations else 0,
            'max': max(activations) if activations else 0,
            'count': len(activations)
        }

        co_activations = trace.get_co_activations()
        co_activations_info = {
            'unique_neurons': len(co_activations),
            'max_co_activations': max(len(neurons) for neurons in co_activations.values()) if co_activations else 0
        }

        return {
            'num_paths': num_paths,
            'layers': layers,
            'unique_neurons': unique_neurons,
            'activation_stats': activation_stats,
            'co_activations_info': co_activations_info
        }

    neuron_ids_info = get_dict_info(neuron_ids, "Neuron IDs")
    layer_mapping_info = get_dict_info(layer_mapping, "Layer Mapping")

    cluster_counts = {}
    for neuron_id, cluster_id in clusters.items():
        if cluster_id not in cluster_counts:
            cluster_counts[cluster_id] = 0
        cluster_counts[cluster_id] += 1

    clusters_info = {
        'length': len(clusters),
        'data_type': {neuron_id: type(cluster_id) for neuron_id, cluster_id in clusters.items()},
        'unique_clusters': len(set(clusters.values())),
        'cluster_counts': cluster_counts
    }

    trace_info = get_trace_info(trace)

    return {
        'neuron_ids_info': neuron_ids_info,
        'layer_mapping_info': layer_mapping_info,
        'clusters_info': clusters_info,
        'trace_info': trace_info
    }


def print_shape_info(shape_info):
    def print_dict(info, name):
        print(f"\n{name}:")
        for key, value in info.items():
            print(f"  {key}: {value}")

    print_dict(shape_info['neuron_ids_info'], "Neuron IDs Info")
    print_dict(shape_info['layer_mapping_info'], "Layer Mapping Info")
    print_dict(shape_info['clusters_info'], "Clusters Info")
    print_dict(shape_info['trace_info'], "Trace Info")
