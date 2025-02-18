import os
import hydra
from omegaconf import DictConfig
from typing import Dict
from trainer import train_and_evaluate
from pathlib import Path

def linear_probe(cfg: DictConfig, feature_paths: Dict[str, str]) -> Dict:
    """Main function for linear probing training
    
    Args:
        cfg: Configuration object
        feature_paths: Dictionary containing paths to train and validation features
        
    Returns:
        Dict: Training results and evaluation metrics
    """
    model_name = cfg.model.name
    
    # Train and evaluate
    metrics = train_and_evaluate(
        model_name=model_name,
        features_path=feature_paths["train_features"],
        val_features_path=feature_paths["val_features"],
        config=cfg
    )
    
    return metrics

current_dir = Path(__file__).resolve().parent
config_path = current_dir.parent / "conf"

@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg: DictConfig):
    """Main function for linear probe training"""
    model_name = cfg.model.name
    
    cache_dir = Path(cfg.data.cache_dir)
    feature_paths = {
        "train_features": cache_dir / f"{cfg.model.name.lower()}_train_features.h5",
        "val_features": cache_dir / f"{cfg.model.name.lower()}_val_features.h5"
    }
    
    # Run linear probing
    linear_probe(cfg, feature_paths)

if __name__ == "__main__":
    main()
