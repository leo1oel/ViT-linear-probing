import os
import hydra
from omegaconf import DictConfig
from typing import Dict
from trainer import train_and_evaluate

def linear_probe(cfg: DictConfig, feature_paths: Dict[str, str]) -> Dict:
    """线性探测训练的主函数
    
    Args:
        cfg: 配置对象
        feature_paths: 包含训练集和验证集特征路径的字典
        
    Returns:
        Dict: 训练结果和评估指标
    """
    model_name = cfg.model.name
    
    # 训练和评估
    metrics = train_and_evaluate(
        model_name=model_name,
        features_path=feature_paths["train_features"],
        val_features_path=feature_paths["val_features"],
        config=cfg
    )
    
    return metrics

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main function for linear probe training"""
    model_name = cfg.model.name
    
    # 构建特征路径
    feature_paths = {
        "train_features": os.path.join(
            cfg.data.cache_dir,
            f"{model_name.lower()}_train_features.h5"
        ),
        "val_features": os.path.join(
            cfg.data.cache_dir,
            f"{model_name.lower()}_val_features.h5"
        )
    }
    
    # 运行线性探测
    linear_probe(cfg, feature_paths)

if __name__ == "__main__":
    main()
