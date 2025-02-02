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
from feature_extractor import ImageFolderWithPaths

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
        self.train_data = ImageFolderWithPaths(
            os.path.join(self.root, 'train'),
            transform=self.transform
        )
        self.val_data = ImageFolderWithPaths(
            os.path.join(self.root, 'val'),
            transform=self.transform
        )

class CIFAR10WithPaths(datasets.CIFAR10):
    """扩展的 CIFAR10，返回图片的索引作为路径"""
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__(root, train=train, transform=transform, download=download)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        # 由于 CIFAR10 是内置数据集，没有实际的文件路径，我们使用索引作为标识
        path = f"cifar10_{'train' if self.train else 'test'}_{index}"
        return img, target, path

class CIFAR10Dataset(ImageDataset):
    """CIFAR-10 dataset handler"""
    def _setup(self):
        self.train_data = CIFAR10WithPaths(
            self.root, train=True, transform=self.transform, download=True
        )
        self.val_data = CIFAR10WithPaths(
            self.root, train=False, transform=self.transform, download=True
        )

class StanfordCarsDataset(ImageDataset):
    """Stanford Cars dataset handler"""
    def _setup(self):
        # 加载测试集标注
        test_annos = sio.loadmat(os.path.join(self.root, 'cars_test_annos_withlabels.mat'))
        train_annos = sio.loadmat(os.path.join(self.root, 'devkit/cars_train_annos.mat'))
        
        class StanfordCarsSubset(datasets.VisionDataset):
            def __init__(self, root, annotations, split, transform=None):
                super().__init__(root, transform=transform)
                self.annotations = annotations['annotations'][0]  # 获取标注数组
                self.split = split
                self.classes = [str(i) for i in range(196)]  # 196个车型类别
                self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
                
            def __getitem__(self, idx):
                anno = self.annotations[idx]
                img_name = anno[-1][0]  # 文件名在最后一个位置
                label = anno[-2][0][0] - 1  # 标签从1开始，需要减1
                
                # 根据split构建正确的图片路径
                img_path = os.path.join(self.root, f'cars_{self.split}', img_name)
                img = Image.open(img_path).convert('RGB')
                
                if self.transform:
                    img = self.transform(img)
                
                return img, label, img_path
                
            def __len__(self):
                return len(self.annotations)
        
        self.train_data = StanfordCarsSubset(
            self.root,
            train_annos,
            'train',
            transform=self.transform
        )
        
        self.val_data = StanfordCarsSubset(
            self.root,
            test_annos,
            'test',
            transform=self.transform
        )

class Flowers102Dataset(ImageDataset):
    """Oxford 102 Flowers dataset handler"""
    def _setup(self):
        # 读取jsonl文件
        train_data = []
        val_data = []
        
        with open(os.path.join(self.root, 'flowers.jsonl'), 'r') as f:
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
                unique_labels = sorted(set(x['label'] for x in data_items))
                self.classes = unique_labels
                self.class_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                
            def __getitem__(self, idx):
                item = self.data_items[idx]
                img_path = item['image']
                img_path = img_path.replace("./data", "/pasteur/u/yuhuiz/data")
                label = self.class_to_idx[item['label']]
                
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                
                return img, label, img_path
                
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
        classes_txt = read_list_file('classes.txt')
        
        # 创建映射
        self.classes = [c[1].split('.')[1] for c in classes_txt]  # 移除序号并获取类名
        filename_to_label = {img[1]: int(label[1]) - 1 for img, label in zip(image_list, image_labels)}
        filename_to_split = {img[1]: int(split[1]) for img, split in zip(image_list, train_test_split)}
        
        class CUBSubset(datasets.VisionDataset):
            def __init__(self, root, filenames, filename_to_label, classes, transform=None):
                super().__init__(root, transform=transform)
                self.filenames = filenames
                self.filename_to_label = filename_to_label
                self.classes = classes
                self.class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
                
            def __getitem__(self, idx):
                img_name = self.filenames[idx]
                img_path = os.path.join(self.root, 'images', img_name)
                label = self.filename_to_label[img_name]
                
                img = Image.open(img_path).convert('RGB')
                if self.transform:
                    img = self.transform(img)
                
                return img, label, img_path
                
            def __len__(self):
                return len(self.filenames)
        
        # 分割训练集和测试集
        train_files = [f for f, is_train in filename_to_split.items() if is_train]
        test_files = [f for f, is_train in filename_to_split.items() if not is_train]
        
        self.train_data = CUBSubset(
            dataset_path,
            train_files,
            filename_to_label,
            self.classes,
            transform=self.transform
        )
        
        self.val_data = CUBSubset(
            dataset_path,
            test_files,
            filename_to_label,
            self.classes,
            transform=self.transform
        )

class ImageNetV2Dataset(ImageDataset):
    """ImageNetV2 dataset handler - validation only dataset"""
    def _setup(self):
        self.train_data = None
        self.val_data = ImageFolderWithPaths(
            self.root,
            transform=self.transform
        )

class CIFAR101Dataset(ImageDataset):
    """CIFAR10.1 dataset handler - validation only dataset"""
    def _setup(self):
        self.train_data = None  # CIFAR10.1 is validation only
        
        # Load data and labels from NPY files
        data = np.load(os.path.join(self.root, 'cifar10.1_v6_data.npy'))
        labels = np.load(os.path.join(self.root, 'cifar10.1_v6_labels.npy'))
        
        # CIFAR10 class names
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
        
        class CIFAR101Subset(Dataset):
            def __init__(self, data, labels, transform=None):
                self.data = data
                self.labels = labels
                self.transform = transform
                self.classes = classes  # Use outer classes
                self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
            
            def __getitem__(self, index):
                img = self.data[index]
                target = int(self.labels[index])
                img = Image.fromarray(img)
                
                if self.transform:
                    img = self.transform(img)
                    
                path = f"cifar10.1_val_{index}"
                return img, target, path
            
            def __len__(self):
                return len(self.data)
        
        self.val_data = CIFAR101Subset(data, labels, transform=self.transform)

# 数据集注册表
DATASET_REGISTRY: Dict[str, type] = {
    'imagenet': ImageNetDataset,
    'imagenetv2': ImageNetV2Dataset,
    'cifar10': CIFAR10Dataset,
    'stanford_cars': StanfordCarsDataset,
    'flowers102': Flowers102Dataset,
    'cub': CUBDataset,
    'cifar10.1': CIFAR101Dataset,
}

def get_dataset(name: str, root: str, transform: Optional[transforms.Compose] = None) -> ImageDataset:
    """获取指定的数据集"""
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {name}. Available datasets: {list(DATASET_REGISTRY.keys())}")
    
    dataset_cls = DATASET_REGISTRY[name]
    return dataset_cls(root, transform)

class FeatureDataset(Dataset):
    """Enhanced dataset class with feature normalization and statistics tracking"""
    def __init__(self, features: np.ndarray, labels: np.ndarray, normalize: bool = True, train_stats: Dict = None, is_train: bool = True):
        # Store original statistics for debugging
        if normalize:
            if is_train:
                mean = features.mean(0, keepdims=True)
                std = features.std(0, keepdims=True)
                self.train_stats = {'mean': mean, 'std': std}
            else:
                assert train_stats is not None, "Need training stats for validation/test set"
                mean = train_stats['mean']
                std = train_stats['std']

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
