import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class TraceableModel(nn.Module):
    def __init__(self, model, network_trace, layers_to_trace=None):
        super(TraceableModel, self).__init__()
        self.model = model
        self.network_trace = network_trace
        self.layers_to_trace = layers_to_trace
        self.current_epoch = 0
        self.handles = []
        self.is_training_mode = True
        self.layer_order_recorded = False
        self._register_hooks()

    def start_training_trace(self):
        """Call this method at the beginning of training to initialize tracing."""
        self.is_training_mode = True
        # Initialize the trace at the beginning of training
        self.network_trace.initialize_training_trace()

    def start_inference_trace(self, epoch='inference'):
        """Call this method before inference to set up hooks for activation recording."""
        self.is_training_mode = False
        self.current_epoch = epoch
        self.network_trace.initialize_inference_trace()

    def _get_activation_hook(self, layer_name):
        def hook(module, input, output):
            # Record layer order during the first forward pass
            if not self.layer_order_recorded:
                if layer_name not in self.network_trace.layer_order:
                    self.network_trace.layer_order.append(layer_name)
            # Record activations during inference
            if not self.is_training_mode:
                activations = output.detach().view(output.size(0), -1)
                self.network_trace.update_layer_activations(self.current_epoch, layer_name, activations)
        return hook

    def end_training_epoch(self):
        """Call this method at the end of each training epoch to record weights and biases."""
        # Record weights and biases at the end of the epoch
        self.network_trace.update_trace(self.current_epoch, self.model)
        self.current_epoch += 1

    def end_inference_trace(self):
        """Call this method after inference to remove hooks."""
        self.remove_hooks()

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
            self.network_trace.store_predictions(self.current_epoch, outputs)
        return outputs