import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from contextlib import contextmanager

class TraceableModel(nn.Module):
    def __init__(self, model, network_trace, layers_to_trace=None):
        super(TraceableModel, self).__init__()
        self.model = model
        self.network_trace = network_trace
        self.layers_to_trace = layers_to_trace
        self.time_step = 0
        self.handles = []
        self.is_training_mode = True
        self.layer_order_recorded = False
        self._register_hooks()

    @contextmanager
    def trace(self, time_step, mode='training'):
        """Context manager for tracing, handles both training and inference."""
        self.start_trace(time_step, mode)
        try:
            yield
        finally:
            self.end_trace()

    def start_trace(self, time_step, mode='training'):
        """Method to start tracing in either training or inference mode."""
        self.mode = mode  # Use the mode directly instead of is_training_mode
        self.time_step = time_step
        
        # Initialize the trace for the given time step and mode
        self.network_trace.initialize_trace(self.time_step, self.mode)

    def end_trace(self):
        """Call this method to end tracing and unmount recording hooks."""
        self.network_trace.update_trace(self.time_step, self.model, self.mode)
        self.time_step += 1  # Increment the time step
        if self.mode == 'inference':
            self.remove_hooks()  # Only needed for inference mode, if desired

    def _get_activation_hook(self, layer_name):
        def hook(module, input, output):
            # Record layer order during the first forward pass
            if not self.layer_order_recorded:
                if layer_name not in self.network_trace.layer_order:
                    self.network_trace.layer_order.append(layer_name)
            # Record activations during inference
            if not self.is_training_mode:
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
    self.eval()
    if device is not None:
        self.to(device)
        x = x.to(device)
    outputs = []
    
    # Replace 'epoch' with 'time_step' and manage the time step properly
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