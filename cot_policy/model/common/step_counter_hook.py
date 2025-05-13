import torch.nn as nn


class ForwardCounter:
    def __init__(self, model: nn.Module):
        self.counter = 0
        self.hook = model.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inputs, outputs):
        self.counter += 1

    def reset(self):
        """Reset the counter to zero."""
        self.counter = 0

    def get_count(self):
        """Get the current count."""
        return self.counter

    def remove(self):
        """Remove the hook to stop counting."""
        self.hook.remove()
