import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from transformers import AutoModel, CLIPVisionModel
from tqdm import tqdm
import os
import h5py
import numpy as np
from typing import Tuple, List

class FeatureExtractor:
    def __init__(self, model_name: str, model: nn.Module, device: str = 'cuda', batch_size: int = 256):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个 GPU 进行特征提取")
            self.model = nn.DataParallel(model)
            self.model = self.model.to(device)
        else:
            self.model = model.to(device)
        
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def prepare_dataset(self, data_path: str, split: str = 'train') -> Tuple[DataLoader, List[str]]:
        dataset = datasets.ImageFolder(
            os.path.join(data_path, split),
            transform=self.transform
        )
        
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True
        )
        
        return loader, dataset.classes

    @torch.no_grad()
    def extract_features(self, loader: DataLoader, save_path: str):
        self.model.eval()
        
        # 获取第一批数据来确定特征维度
        first_batch = next(iter(loader))
        first_images = first_batch[0].to(self.device)
        
        if isinstance(self.model.module if isinstance(self.model, nn.DataParallel) 
                    else self.model, AutoModel):
            # DINO
            outputs = self.model(first_images, output_hidden_states=True)
            feature_dim = outputs.last_hidden_state.shape[-1]
        else:
            # CLIP
            outputs = self.model(first_images, output_hidden_states=True)
            feature_dim = outputs.last_hidden_state.shape[-1]
        
        # 创建 HDF5 文件
        with h5py.File(save_path, 'w') as f:
            total_samples = len(loader.dataset)
            
            if isinstance(self.model.module if isinstance(self.model, nn.DataParallel) 
                        else self.model, AutoModel):
                # DINO 特征
                f.create_dataset('cls_features', shape=(total_samples, feature_dim), dtype='float32')
                f.create_dataset('mean_patch_features', shape=(total_samples, feature_dim), dtype='float32')
            else:
                # CLIP 特征
                f.create_dataset('last_hidden_cls', shape=(total_samples, feature_dim), dtype='float32')
                f.create_dataset('penultimate_cls', shape=(total_samples, feature_dim), dtype='float32')
                f.create_dataset('last_hidden_mean_patch', shape=(total_samples, feature_dim), dtype='float32')
                f.create_dataset('penultimate_mean_patch', shape=(total_samples, feature_dim), dtype='float32')
                f.create_dataset('pooler_output', shape=(total_samples, feature_dim), dtype='float32')
            
            f.create_dataset('targets', shape=(total_samples,), dtype='int64')
            
            # 处理所有批次
            current_index = 0
            for images, targets in tqdm(loader, desc=f"提取 {self.model_name} 特征"):
                images = images.to(self.device)
                batch_size = images.size(0)
                
                if isinstance(self.model.module if isinstance(self.model, nn.DataParallel) 
                            else self.model, AutoModel):
                    # DINO
                    outputs = self.model(images, output_hidden_states=True)
                    last_hidden = outputs.last_hidden_state
                    cls_token = last_hidden[:, 0]
                    patch_tokens = last_hidden[:, 1:]
                    mean_patch = torch.mean(patch_tokens, dim=1)
                    
                    f['cls_features'][current_index:current_index + batch_size] = cls_token.cpu().numpy()
                    f['mean_patch_features'][current_index:current_index + batch_size] = mean_patch.cpu().numpy()
                    
                else:
                    # CLIP
                    outputs = self.model(images, output_hidden_states=True)
                    
                    # 最后一层特征
                    last_hidden = outputs.last_hidden_state
                    last_hidden_cls = last_hidden[:, 0]
                    last_hidden_patches = last_hidden[:, 1:]
                    last_hidden_mean_patch = torch.mean(last_hidden_patches, dim=1)
                    
                    # 倒数第二层特征
                    penultimate = outputs.hidden_states[-2]
                    penultimate_cls = penultimate[:, 0]
                    penultimate_patches = penultimate[:, 1:]
                    penultimate_mean_patch = torch.mean(penultimate_patches, dim=1)
                    
                    # Pooler 输出
                    pooler = outputs.pooler_output
                    
                    # 保存所有特征
                    f['last_hidden_cls'][current_index:current_index + batch_size] = last_hidden_cls.cpu().numpy()
                    f['penultimate_cls'][current_index:current_index + batch_size] = penultimate_cls.cpu().numpy()
                    f['last_hidden_mean_patch'][current_index:current_index + batch_size] = last_hidden_mean_patch.cpu().numpy()
                    f['penultimate_mean_patch'][current_index:current_index + batch_size] = penultimate_mean_patch.cpu().numpy()
                    f['pooler_output'][current_index:current_index + batch_size] = pooler.cpu().numpy()
                
                f['targets'][current_index:current_index + batch_size] = targets.numpy()
                current_index += batch_size
