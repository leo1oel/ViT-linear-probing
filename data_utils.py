import torch
from torch.utils.data import Dataset
import numpy as np
from rich.console import Console

console = Console()

class FeatureDataset(Dataset):
    """Enhanced dataset class with feature normalization and statistics tracking"""
    def __init__(self, features: np.ndarray, labels: np.ndarray, normalize: bool = True, is_train: bool = True):
        # Store original statistics for debugging
        
        if normalize:
            # Normalize features using robust statistics
            mean = features.mean(0, keepdims=True)
            std = features.std(0, keepdims=True)
            features = (features - mean) / (std + 1e-5)
            
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        
        # Verify label distribution
        self.label_distribution = torch.bincount(self.labels)
        self.is_train = is_train
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def print_statistics(self):
        """Print comprehensive dataset statistics"""
        prefix = "Training" if self.is_train else "Validation"
        console.print(f"\n[cyan]{prefix} Dataset:[/cyan]")
        console.print(f"Total samples: {len(self)}")
        console.print(f"Number of features: {self.features.shape[1]}")
        console.print(f"Label counts: {self.label_distribution.tolist()}")
        console.print(f"\nNumber of classes: {len(self.label_distribution)}")
        console.print(f"Samples per class min: {self.label_distribution.min().item()}")
        console.print(f"Samples per class max: {self.label_distribution.max().item()}")
        console.print(f"Samples per class mean: {self.label_distribution.float().mean().item():.2f}")
