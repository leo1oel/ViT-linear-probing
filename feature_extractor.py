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

console = Console()

@dataclass
class DatasetConfig:
    """数据集配置"""
    data_path: str
    batch_size: int = 256
    num_workers: int = 8
    image_size: int = 224
    max_samples: Optional[int] = None  # 每个类别的最大样本数，None表示使用所有样本

class ImageFolderWithPaths(datasets.ImageFolder):
    """扩展的 ImageFolder，返回图片路径"""
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int, str]:
        img, label = super().__getitem__(index)
        path = self.imgs[index][0]
        return img, label, path

def create_progress_bar() -> Progress:
    """创建统一的进度条样式"""
    return Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(complete_style="green", finished_style="green"),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    )

class DatasetLoader:
    """数据集加载器"""
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
        """加载数据集并返回数据加载器和类别映射"""
        # 构建数据集路径
        dataset_path = os.path.join(self.config.data_path, split)
        console.print(Panel(f"Loading {split} dataset from {dataset_path}", 
                          style="bold cyan"))

        # 创建完整数据集
        full_dataset = ImageFolderWithPaths(
            dataset_path,
            transform=self.transform
        )

        # 如果指定了最大样本数，对每个类别进行采样
        if self.config.max_samples is not None:
            # 按类别组织索引
            indices_by_class = {}
            for idx, (_, label) in enumerate(full_dataset.samples):
                if label not in indices_by_class:
                    indices_by_class[label] = []
                indices_by_class[label].append(idx)
            
            # 对每个类别进行采样
            selected_indices = []
            for label_indices in indices_by_class.values():
                if len(label_indices) > self.config.max_samples:
                    # 随机采样指定数量的样本
                    selected_indices.extend(
                        np.random.choice(
                            label_indices, 
                            self.config.max_samples, 
                            replace=False
                        ).tolist()
                    )
                else:
                    selected_indices.extend(label_indices)
            
            # 创建子集
            dataset = Subset(full_dataset, selected_indices)
            console.print(f"[yellow]Sampled {len(selected_indices)} images from original {len(full_dataset)} images[/yellow]")
        else:
            dataset = full_dataset

        # 创建数据加载器
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=True
        )

        # 打印数据集统计信息
        stats_table = Table(title=f"{split.capitalize()} Dataset Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right", style="green")
        stats_table.add_row("Total Images", str(len(dataset)))
        stats_table.add_row("Number of Classes", str(len(full_dataset.classes)))
        if self.config.max_samples is not None:
            stats_table.add_row("Max Samples per Class", str(self.config.max_samples))
        stats_table.add_row("Batch Size", str(self.config.batch_size))
        console.print(stats_table)

        return loader, full_dataset.class_to_idx

class FeatureExtractor:
    """特征提取器"""
    def __init__(self, model: nn.Module, device: str = "cuda"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        console.print(Panel(f"Feature extractor initialized on {device}", 
                          style="bold blue"))
    
    def _get_features(self, images: torch.Tensor) -> torch.Tensor:
        """根据模型类型获取特征"""
        outputs = self.model(images, output_hidden_states=True)
        if hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state[:, 0]  # 取 CLS token
        else:
            # 如果没有 last_hidden_state，使用最后一层隐藏状态
            features = outputs.hidden_states[-1][:, 0]
        
        return features

    def extract_features(
        self,
        data_loader: DataLoader,
        output_path: str,
        class_to_idx: Dict[str, int]
    ) -> None:
        """提取特征并保存到文件"""
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        all_features = []
        all_labels = []
        all_paths = []

        # 创建进度条
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

        # 合并所有特征
        features = np.concatenate(all_features)
        labels = np.concatenate(all_labels)

        # 保存特征
        console.print(Panel("Saving features to file...", style="bold green"))
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('last_hidden_cls', data=features)
            f.create_dataset('targets', data=labels)
            # 保存图片路径和类别映射
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('paths', data=np.array(all_paths, dtype=object), dtype=dt)
            for class_name, idx in class_to_idx.items():
                f.attrs[f'class_{idx}'] = class_name

        # 打印特征统计信息
        stats_table = Table(title="Feature Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", justify="right", style="green")
        stats_table.add_row("Feature Shape", str(features.shape))
        stats_table.add_row("Number of Samples", str(len(labels)))
        stats_table.add_row("Feature Mean", f"{features.mean():.4f}")
        stats_table.add_row("Feature Std", f"{features.std():.4f}")
        console.print(stats_table)
