from data import  data_handler_iris, train_loader_iris, test_loader_iris
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int
    output_size: int
    num_samples: int
    num_time_steps: int
    time_step_interval: List[int]
    layer_names: Dict[str, str]
    batch_size: Optional[int] = 16
    inference_batch_size: Optional[int] = 30
    data: Optional[Dict] = field(default_factory=dict)

# Configuration for a general model
config = ModelConfig(
    input_size=20,
    hidden_size=10,
    output_size=1,
    num_samples=100,
    num_time_steps=30,  
    time_step_interval=[10, 20, 30],
    layer_names={
        'input': 'fc1',
        'hidden': 'fc2',
        'output': 'fc3'
    }
)

# Configuration for the Iris dataset model
config_iris = ModelConfig(
    input_size=4,
    hidden_size=10,
    output_size=3,
    num_samples=150,
    num_time_steps=20,
    batch_size=16,
    inference_batch_size=30,
    time_step_interval=[10, 20], 
    layer_names={
        'input': 'L_input',
        'hidden': 'L_hidden_1',
        'output': 'L_output'
    },
    data={
        'train_loader': train_loader_iris,  # Ensure these variables are defined
        'test_loader': test_loader_iris,
        'X_test': data_handler_iris.X_test,
        'y_test': data_handler_iris.y_test
    }
)

import torch.nn as nn

class ModelConfigurator:
    """
    The ModelConfigurator class groups common model configurations together based on their type.
    It provides a stateful API for configuring a custom model for testing.
    Currently supported configurations:
    - binary_classification
    - multi_class_classification
    - regression
    - custom

    If the custom flag is specified, you must include
    the loss function, activation function, label format, etc., as an additional dictionary parameter.
    """
    def __init__(self, model_type, custom_config=None):
        self.model_type = model_type
        self.config = self._get_config(model_type, custom_config)

    def _get_config(self, model_type, custom_config=None):
        if model_type == 'binary_classification':
            return {
                'loss_function': nn.BCELoss(),
                'output_activation': nn.Sigmoid(),
                'label_format': 'single_column',
            }
        elif model_type == 'multi_class_classification':
            return {
                'loss_function': nn.CrossEntropyLoss(),
                'output_activation': None,
                'label_format': 'integer_labels',
            }
        elif model_type == 'regression':
            return {
                'loss_function': nn.MSELoss(),
                'output_activation': None,
                'label_format': 'continuous_values',
            }
        elif model_type == 'custom': 
            if custom_config is None:
                raise ValueError("For custom model type, custom_config must be provided.")
            return {
                'loss_function': custom_config.get('loss_function'),
                'output_activation': custom_config.get('output_activation'),
                'label_format': custom_config.get('label_format')
            }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    # Helper getter functions
    def get_loss_function(self):
        return self.config['loss_function']

    def get_output_activation(self):
        return self.config['output_activation']

    def get_label_format(self):
        return self.config['label_format']