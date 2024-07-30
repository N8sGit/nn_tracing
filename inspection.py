# Helper functions to inspect and format internal model states, outputs, etc
import json


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        return json.JSONEncoder.default(self, obj)

def print_json_like_dump(dict):
    print(json.dumps(dict, cls=CustomEncoder, indent=4))

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

def print_network_trace(network_trace):
    for epoch, layers in network_trace.items():
        print(f"Epoch {epoch}:")
        for layer, neurons in layers.items():
            print(f"  Layer {layer}:")
            for neuron_id, data in neurons.items():
                print(f"    Neuron ID {neuron_id}:")
                for activation, state in data["activations"].items():
                    print(f"      Activation: {activation}")
                    print(f"        Input Neurons: {state['input_state']}")
                    print(f"        Output Neurons: {state['output_state']}")
                    print(f"        Classification Result: {state['classification_result']}")
        print("\n")


def network_trace_to_dict(network_trace):
    trace_dict = {}
    for epoch, layers in network_trace.get_trace().items():
        trace_dict[epoch] = {}
        for layer, neurons in layers.items():
            trace_dict[epoch][layer] = {}
            for neuron_id, trace_obj in neurons.items():
                trace_dict[epoch][layer][neuron_id] = trace_obj.to_dict()
    return trace_dict

def print_network_trace(network_trace):
    trace_dict = network_trace_to_dict(network_trace)
    print(json.dumps(trace_dict, indent=2))
