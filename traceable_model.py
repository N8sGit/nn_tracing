import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from contextlib import contextmanager
class TraceableModel(nn.Module):
    """
    A wrapper class that adds tracing functionality to a PyTorch model.

    This class supports tracing activations and layer order during training and
    inference, and allows recording predictions during inference for later analysis.

    Key features:
    - Traces activations and layer order dynamically
    - Modes: 'training' and 'inference'
    - Supports easy switching between modes with hooks for layer activations
    """
    VALID_MODES = {'training', 'inference'}

    def __init__(self, model, network_trace, layers_to_trace=None):
        super(TraceableModel, self).__init__()
        self.model = model
        self.network_trace = network_trace
        self.layers_to_trace = layers_to_trace
        self.time_step = 0
        self.handles = []
        self._mode = 'training'
        self.layer_order_recorded = False
        self._register_hooks()

    @contextmanager
    def trace(self, time_step, mode='training'):
        """Context manager for tracing, handles both training and inference."""
        self.set_mode(mode)
        self.start_trace(time_step)
        try:
            yield
        finally:
            self.end_trace()

    def start_trace(self, time_step):
        """Method to start tracing based on the current mode."""
        self.time_step = time_step
        
        # Initialize the trace for the given time step and mode
        self.network_trace.initialize_trace(self.time_step, self.mode)
        
        # If switching to inference mode, ensure hooks are registered
        if self.mode == 'inference' and not self.handles:
            self._register_hooks()

    def end_trace(self):
        """Ends tracing and handles mode-specific operations."""
        self.network_trace.update_trace(self.time_step, self.model)
        self.time_step += 1  # Increment the time step
        if self.mode == 'inference':
            self.remove_hooks()  # Remove hooks after inference if desired

    def _get_activation_hook(self, layer_name):
        """
    Creates a forward hook for tracing activations and layer order during the forward pass.

    This method returns a hook function that:
    1. Records the order of layers during the first forward pass. This is only done once 
    per session to ensure that the layer order is recorded for analysis.
    2. During inference, it records the activations of the specified layer, detaching 
    the output from the computational graph to store it in a trace-friendly format.

    Parameters:
        layer_name (str): The name of the layer for which the hook is being created. 
        Used to identify the layer in the network trace.

    Returns:
        hook (function): A forward hook function to be registered with the specified layer. 
        This function captures the output (activations) of the layer 
        during the forward pass, depending on the model's current mode.

    Hook Function Behavior:
        - If this is the first forward pass and the layer order hasn't been recorded, it appends
        the layer's name to `self.network_trace.layer_order`.
        - If the model is in 'inference' mode, it records the activations of the layer, detaching 
        them from the computational graph, reshaping them to 2D, and updating the `network_trace`
        object with the activations for the current time step.
    """
        def hook(module, input, output):
            # Record layer order during the first forward pass
            if not self.layer_order_recorded:
                if layer_name not in self.network_trace.layer_order:
                    self.network_trace.layer_order.append(layer_name)
            # Record activations during inference
            if self.mode == 'inference':
                activations = output.detach().view(output.size(0), -1)
                self.network_trace.update_layer_activations(self.time_step, layer_name, activations)
        return hook

    def _register_hooks(self):
        """Register forward hooks to capture activations and record layer order."""
        for name, module in self.model.named_modules():
            if self.layers_to_trace is None or name in self.layers_to_trace:
                handle = module.register_forward_hook(self._get_activation_hook(name))
                self.handles.append(handle)

    def remove_hooks(self):
        """Remove all forward hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def forward(self, x):
        return self.model(x)

    def predict(self, x, batch_size=None, return_probabilities=False, device=None):
        """Perform inference with activation recording."""
        self.eval_mode()
        if device is not None:
            self.to(device)
            x = x.to(device)
        outputs = []
        
        with self.trace(time_step=self.time_step, mode='inference'):
            with torch.no_grad():
                if batch_size:
                    dataset = TensorDataset(x)
                    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
                    for batch in dataloader:
                        batch_x = batch[0]
                        output = self.forward(batch_x)
                        outputs.append(output.cpu())
                    outputs = torch.cat(outputs, dim=0)
                else:
                    output = self.forward(x)
                    outputs.append(output.cpu())
                    outputs = outputs[0]
                if return_probabilities:
                    if outputs.shape[1] == 1:
                        outputs = torch.sigmoid(outputs)
                    else:
                        outputs = torch.softmax(outputs, dim=1)
                
            # Store predictions using the current time step
            self.network_trace.store_predictions(self.time_step, outputs)
            
            # Increment the time step to ensure unique recording for each prediction
            self.time_step += 1
        
        return outputs

    @property
    def mode(self):
        return self._mode

    def set_mode(self, mode):
        """
        Sets the mode of the model to either 'training' or 'inference'.
        Raises a ValueError if the mode is invalid.
        """
        if not isinstance(mode, str):
            raise TypeError(f"Mode must be a string, got {type(mode).__name__}")
        mode = mode.lower()
        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Valid modes are 'training' and 'inference'.")
        self._mode = mode
        self.network_trace.set_mode(mode)  # Synchronize mode with NetworkTrace

    def train_mode(self):
        """Convenience method to set the model to training mode."""
        self.set_mode('training')
        self.train()

    def eval_mode(self):
        """Convenience method to set the model to evaluation mode."""
        self.set_mode('inference')
        self.eval()