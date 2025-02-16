import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, datasets
import h5py
import numpy as np
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from utils.progress_utils import create_progress_bar

console = Console()

@dataclass
class DatasetConfig:
    """Dataset configuration"""
    dataset_name: str
    data_path: str
    batch_size: int = 256
    num_workers: int = 8
    image_size: int = 224
    max_samples: Optional[int] = None  # Maximum samples per class, None means use all samples

class ImageFolderWithPaths(datasets.ImageFolder):
    """Extended ImageFolder that returns image paths"""
    def __init__(self, root, transform):
        super().__init__(root, transform)
        self.imgs = self.samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        path, target = self.imgs[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

class DatasetLoader:
    """Dataset loader class"""
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize(config.image_size),
            transforms.CenterCrop(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

    def load_dataset(self, split: str) -> Tuple[DataLoader, Dict[int, str]]:
        """Load dataset and return dataloader with class mapping
        
        Args:
            split: Dataset split ('train' or 'val')
            
        Returns:
            Tuple containing:
                - DataLoader: The dataset loader
                - Dict[int, str]: Mapping from class indices to class names
        """
        # Create dataset
        dataset = get_dataset(self.config.dataset_name, self.config.data_path, self.transform)
        if split == 'train':
            full_dataset = dataset.train_data
        else:
            full_dataset = dataset.val_data

        # Sample from each class if max_samples is specified
        if self.config.max_samples is not None:
            # Organize indices by class
            indices_by_class = {}
            for idx in range(len(full_dataset)):
                _, label, _ = full_dataset[idx]
                if label not in indices_by_class:
                    indices_by_class[label] = []
                indices_by_class[label].append(idx)
            
            # Sample from each class
            sampled_indices = []
            for indices in indices_by_class.values():
                if len(indices) > self.config.max_samples:
                    indices = np.random.choice(indices, self.config.max_samples, replace=False)
                sampled_indices.extend(indices)
            
            # Create subset
            full_dataset = Subset(full_dataset, sampled_indices)

        # Create data loader
        data_loader = DataLoader(
            full_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        if self.config.dataset_name == 'datacomp12m' or self.config.dataset_name == 'datacomp1.2m':
            return data_loader, {cls: idx for idx, cls in enumerate(['unknown'])}
        # Get class mapping
        class_to_idx = getattr(full_dataset, 'class_to_idx', None)
        if class_to_idx is None and hasattr(full_dataset, 'dataset'):
            # Handle Subset case
            class_to_idx = full_dataset.dataset.class_to_idx

        return data_loader, class_to_idx

class FeatureExtractor:
    """Feature extractor class"""
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        console.print(Panel(f"Feature extractor initialized on {device}", 
                          style="bold blue"))
    
    def _get_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features based on model type
        
        Args:
            images: Input tensor of images
            
        Returns:
            torch.Tensor: Extracted features
        """
        outputs = self.model(images, output_hidden_states=True)
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state[:, 0]  # Take CLS token
        else:
            # If no last_hidden_state, use the last layer hidden state
            features = outputs.hidden_states[-1][:, 0]
        
        return features

    def extract_features(
        self,
        data_loader: DataLoader,
        output_path: str,
        class_to_idx: Dict[str, int]
    ) -> None:
        """Extract features and save to file
        
        Args:
            data_loader: DataLoader containing the dataset
            output_path: Path to save the extracted features
            class_to_idx: Mapping from class names to indices
        """
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        all_features = []
        all_labels = []
        all_paths = []

        # Create progress bar
        progress = create_progress_bar()
        with progress:
            extract_task = progress.add_task(
                "[cyan]Extracting features...", 
                total=len(data_loader)
            )

            with torch.no_grad():
                for images, labels, paths in data_loader:
                    images = images.to(self.device)
                    
                    try:
                        features = self._get_features(images)
                        all_features.append(features.cpu().numpy())
                        all_labels.append(labels.numpy())
                        all_paths.extend(paths)
                    except Exception as e:
                        console.print(f"[red]Error processing batch: {str(e)}[/red]")
                        raise e
                    
                    progress.update(extract_task, advance=1)

        # Concatenate all features
        features = np.concatenate(all_features)
        labels = np.concatenate(all_labels)

        # Save features
        console.print(Panel("Saving features to file...", style="bold green"))
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('last_hidden_cls', data=features)
            f.create_dataset('targets', data=labels)
            # Save image paths and class mapping
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('paths', data=np.array(all_paths, dtype=object), dtype=dt)
            for class_name, idx in class_to_idx.items():
                f.attrs[f'class_{idx}'] = class_name

        # Print feature statistics
        stats_table = Table(title="Feature Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right", style="green")
        stats_table.add_row("Feature Shape", str(features.shape))
        stats_table.add_row("Number of Samples", str(len(labels)))
        stats_table.add_row("Feature Mean", f"{features.mean():.4f}")
        stats_table.add_row("Feature Std", f"{features.std():.4f}")
        console.print(stats_table)

def get_dataset(dataset_name, data_path, transform):
    """Get dataset instance
    
    Args:
        dataset_name: Name of the dataset
        data_path: Path to dataset
        transform: Transforms to apply to images
        
    Returns:
        Dataset instance
    """
    from data_utils import get_dataset as get_dataset_instance
    return get_dataset_instance(dataset_name, data_path, transform)
