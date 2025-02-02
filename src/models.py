import torch.nn as nn

class LinearProbe(nn.Module):
    """Improved linear probe with better initialization and normalization"""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
        # Improved initialization using Kaiming initialization
        nn.init.kaiming_normal_(self.linear.weight, mode='fan_out')
        nn.init.constant_(self.linear.bias, 0)
        
    def forward(self, x):
        return self.linear(x)
