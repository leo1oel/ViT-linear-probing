import os
import torch
from torch.utils.data import Dataset
import numpy as np
from rich.console import Console
from torchvision import datasets, transforms
from typing import Optional, Tuple, Dict
import json
import scipy.io as sio
from PIL import Image

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

class StanfordCarsDataset(ImageDataset):
    """Stanford Cars dataset handler"""
    def _setup(self):
        # 加载测试集标注
        annos = sio.loadmat(os.path.join(self.root, 'cars_test_annos_withlabels.mat'))
        test_annotations = annos['annotations'][0]
        
        class StanfordCarsSubset(datasets.VisionDataset):
            def __init__(self, root, annotations, transform=None):
                super().__init__(root, transform=transform)
                self.annotations = annotations
                
            def __getitem__(self, idx):
                anno = self.annotations[idx]
                img_name = anno[-1][0]  # 文件名在最后一个位置
                label = anno[-2][0][0] - 1  # 标签从1开始，需要减1
                
                img_path = os.path.join(self.root, img_name)
                img = Image.open(img_path).convert('RGB')
                
                if self.transform:
                    img = self.transform(img)
                
                return img, label
                
            def __len__(self):
                return len(self.annotations)
        
        self.train_data = StanfordCarsSubset(
            os.path.join(self.root, 'cars_train'),
            test_annotations,  # 需要替换为训练集标注
            transform=self.transform
        )
        
        self.val_data = StanfordCarsSubset(
            os.path.join(self.root, 'cars_test'),
            test_annotations,
            transform=self.transform
        )

class Flowers102Dataset(ImageDataset):
    """Oxford 102 Flowers dataset handler"""
    def _setup(self):
        # 读取jsonl文件
        train_data = []
        val_data = []
        
        with open(os.path.join(self.root, 'flowers102.jsonl'), 'r') as f:
            for line in f:
                item = json.loads(line)
                if item['split'] == 'train':
                    train_data.append(item)
                elif item['split'] in ['test', 'val']:
                    val_data.append(item)
        
        class Flowers102Subset(datasets.VisionDataset):
            def __init__(self, data_items, transform=None):
                super().__init__(root='', transform=transform)
                self.data_items = data_items
                # 创建标签到索引的映射
                self.label_to_idx = {item['label']: idx for idx, item in enumerate(sorted(set(x['label'] for x in data_items)))}
                
            def __getitem__(self, idx):
                item = self.data_items[idx]
                img_path = item['path']
                label = self.label_to_idx[item['label']]
                
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                
                return img, label
                
            def __len__(self):
                return len(self.data_items)
        
        self.train_data = Flowers102Subset(train_data, transform=self.transform)
        self.val_data = Flowers102Subset(val_data, transform=self.transform)

class CUBDataset(ImageDataset):
    """CUB-200-2011 dataset handler"""
    def _setup(self):
        dataset_path = os.path.join(self.root, 'CUB_200_2011')
        
        # 读取必要的文件
        def read_list_file(filename):
            with open(os.path.join(dataset_path, filename)) as f:
                return [line.strip().split() for line in f]
        
        # 读取图像列表和标签
        image_list = read_list_file('images.txt')
        image_labels = read_list_file('image_class_labels.txt')
        train_test_split = read_list_file('train_test_split.txt')
        
        # 创建映射
        filename_to_label = {img[1]: int(label[1]) - 1 for img, label in zip(image_list, image_labels)}
        filename_to_split = {img[1]: int(split[1]) for img, split in zip(image_list, train_test_split)}
        
        class CUBSubset(datasets.VisionDataset):
            def __init__(self, root, filenames, filename_to_label, transform=None):
                super().__init__(root, transform=transform)
                self.filenames = filenames
                self.filename_to_label = filename_to_label
                
            def __getitem__(self, idx):
                img_name = self.filenames[idx]
                img_path = os.path.join(self.root, 'images', img_name)
                label = self.filename_to_label[img_name]
                
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                
                return img, label
                
            def __len__(self):
                return len(self.filenames)
        
        # 分割训练集和测试集
        train_files = [f for f, is_train in filename_to_split.items() if is_train]
        test_files = [f for f, is_train in filename_to_split.items() if not is_train]
        
        self.train_data = CUBSubset(
            dataset_path,
            train_files,
            filename_to_label,
            transform=self.transform
        )
        
        self.val_data = CUBSubset(
            dataset_path,
            test_files,
            filename_to_label,
            transform=self.transform
        )

# 数据集注册表
DATASET_REGISTRY: Dict[str, type] = {
    'imagenet': ImageNetDataset,
    'cifar10': CIFAR10Dataset,
    'cifar100': CIFAR100Dataset,
    'cub': CUBDataset,
    'flowers102': Flowers102Dataset,
    'stanford_cars': StanfordCarsDataset,
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
