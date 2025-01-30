import os
import hydra
from omegaconf import DictConfig
from transformers import AutoModel, CLIPVisionModel
from feature_extractor import FeatureExtractor, DatasetLoader, DatasetConfig
from rich.console import Console
from typing import Dict

console = Console()

def extract_features(cfg: DictConfig) -> Dict[str, str]:
    """提取特征的主函数
    
    Args:
        cfg: 配置对象
        
    Returns:
        Dict[str, str]: 包含训练集和验证集特征路径的字典
    """
    # 创建数据集配置
    dataset_config = DatasetConfig(
        dataset_name=cfg.data.dataset_name,
        data_path=cfg.data.data_path,
        batch_size=cfg.extractor.batch_size,
        num_workers=cfg.extractor.num_workers,
        max_samples=cfg.extractor.get('max_samples', None)
    )
    
    # 加载模型
    if cfg.model.name.lower() == "clip":
        model = CLIPVisionModel.from_pretrained(cfg.model.pretrained)
    else:
        model = AutoModel.from_pretrained(cfg.model.pretrained)
    
    # 创建数据集加载器
    dataset_loader = DatasetLoader(dataset_config)
    
    # 创建特征提取器
    device = cfg.extractor.device
    extractor = FeatureExtractor(model, device)
    
    feature_paths = {}
    
    # 处理训练集和验证集
    for split in ["train", "val"]:
        # 加载数据集
        data_loader, class_to_idx = dataset_loader.load_dataset(split)
        
        # 构建输出路径
        output_path = os.path.join(
            cfg.data.cache_dir,
            f"{cfg.model.name.lower()}_{split}_features.h5"
        )
        
        # 提取特征
        extractor.extract_features(
            data_loader=data_loader,
            output_path=output_path,
            class_to_idx=class_to_idx
        )
        
        feature_paths[f"{split}_features"] = output_path
    
    return feature_paths

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """主函数"""
    extract_features(cfg)

if __name__ == "__main__":
    main()