import torch
from torch.utils.data import Dataset
import numpy as np
from rich.console import Console

console = Console()

class FeatureDataset(Dataset):
    """Enhanced dataset class with feature normalization and statistics tracking"""
    def __init__(self, features: np.ndarray, labels: np.ndarray, normalize: bool = True):
        # Store original statistics for debugging
        self.original_stats = {
            "mean": features.mean(),
            "std": features.std(),
            "min": features.min(),
            "max": features.max()
        }
        
        if normalize:
            # Normalize features using robust statistics
            mean = features.mean(0, keepdims=True)
            std = features.std(0, keepdims=True)
            features = (features - mean) / (std + 1e-5)
            
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).long()
        
        # Store normalized statistics
        self.normalized_stats = {
            "mean": self.features.mean().item(),
            "std": self.features.std().item(),
            "min": self.features.min().item(),
            "max": self.features.max().item()
        }
        
        # Verify label distribution
        self.label_distribution = torch.bincount(self.labels)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
    
    def print_statistics(self):
        """Print comprehensive dataset statistics"""
        console.print("\n[bold cyan]Dataset Statistics:[/bold cyan]")
        console.print("\nOriginal Features:")
        for k, v in self.original_stats.items():
            console.print(f"{k}: {v:.4f}")
        
        console.print("\nNormalized Features:")
        for k, v in self.normalized_stats.items():
            console.print(f"{k}: {v:.4f}")
        
        console.print(f"\nNumber of classes: {len(self.label_distribution)}")
        console.print(f"Samples per class min: {self.label_distribution.min().item()}")
        console.print(f"Samples per class max: {self.label_distribution.max().item()}")
        console.print(f"Samples per class mean: {self.label_distribution.float().mean().item():.2f}")
