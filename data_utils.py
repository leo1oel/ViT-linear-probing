import os
import torch
from torch.utils.data import Dataset
import numpy as np
from rich.console import Console
from torchvision import datasets, transforms
from typing import Optional, Tuple, Dict

console = Console()

class ImageDataset:
    """Base class for handling different image datasets"""
    def __init__(self, root: str, transform: Optional[transforms.Compose] = None):
        self.root = root
        self.transform = transform or self._get_default_transform()
        self.train_data = None
        self.val_data = None
        self._setup()
    
    def _get_default_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def _setup(self):
        raise NotImplementedError
    
    def get_data(self) -> Tuple[datasets.VisionDataset, datasets.VisionDataset]:
        return self.train_data, self.val_data

class ImageNetDataset(ImageDataset):
    """ImageNet dataset handler"""
    def _setup(self):
        self.train_data = datasets.ImageFolder(
            os.path.join(self.root, 'train'),
            transform=self.transform
        )
        self.val_data = datasets.ImageFolder(
            os.path.join(self.root, 'val'),
            transform=self.transform
        )

class CIFAR10Dataset(ImageDataset):
    """CIFAR-10 dataset handler"""
    def _setup(self):
        self.train_data = datasets.CIFAR10(
            self.root, train=True, transform=self.transform, download=True
        )
        self.val_data = datasets.CIFAR10(
            self.root, train=False, transform=self.transform, download=True
        )

class CIFAR100Dataset(ImageDataset):
    """CIFAR-100 dataset handler"""
    def _setup(self):
        self.train_data = datasets.CIFAR100(
            self.root, train=True, transform=self.transform, download=True
        )
        self.val_data = datasets.CIFAR100(
            self.root, train=False, transform=self.transform, download=True
        )

# 数据集注册表
DATASET_REGISTRY: Dict[str, type] = {
    'imagenet': ImageNetDataset,
    'cifar10': CIFAR10Dataset,
    'cifar100': CIFAR100Dataset,
}

def get_dataset(name: str, root: str, transform: Optional[transforms.Compose] = None) -> ImageDataset:
    """获取指定的数据集"""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(DATASET_REGISTRY.keys())}")
    
    dataset_cls = DATASET_REGISTRY[name]
    return dataset_cls(root, transform)

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
