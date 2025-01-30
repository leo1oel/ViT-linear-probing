import torch.nn as nn
import matplotlib.pyplot as plt

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

class GradientTracker:
    """Utility class to track gradient statistics during training"""
    def __init__(self):
        self.grad_norms = []
        self.param_norms = []
        
    def update(self, model: nn.Module):
        total_grad_norm = 0
        total_param_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm = p.grad.data.norm(2).item()
                param_norm = p.data.norm(2).item()
                total_grad_norm += grad_norm
                total_param_norm += param_norm
        
        self.grad_norms.append(total_grad_norm)
        self.param_norms.append(total_param_norm)
        
    def plot_statistics(self, save_path: str):
        """Plot gradient and parameter norm trends"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.grad_norms)
        plt.title('Gradient Norm History')
        plt.xlabel('Iteration')
        plt.ylabel('Norm')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.param_norms)
        plt.title('Parameter Norm History')
        plt.xlabel('Iteration')
        plt.ylabel('Norm')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
