## helper functions and accessors to make lookups and assignments more consistent 

def set_model_level_label(model_level, value):
    str_val = str(value)
    match model_level:
        case 'epoch':
            return f'E_{str_val}'
        case 'layer':
            return f'L_{str_val}'
        case 'neuron':
            return f'N_{str_val}'
        case _:
            return None

def get_model_level_integer(level_label: str) -> int:
    # Split the string by the delimiter '_'
    parts = level_label.split('_')
    # The numeric part is always after the delimiter
    numeric_part = parts[1]
    # Convert the numeric part to an integer and return
    return int(numeric_part)

def parse_signature(signature):
    parts = signature.split('-')
    epoch = parts[0]
    layer = parts[1]
    neuron = parts[2]
    return epoch, layer, neuron

def fetch_neuron_by_signature(signature, network_trace):
    epoch, layer, neuron = parse_signature(signature)
    return network_trace[epoch][layer][neuron]