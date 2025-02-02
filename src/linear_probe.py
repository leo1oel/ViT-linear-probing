import os
import hydra
from omegaconf import DictConfig
from typing import Dict
from trainer import train_and_evaluate

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

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(os.path.dirname(current_dir), "conf")
@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(cfg: DictConfig):
    """Main function for linear probe training"""
    model_name = cfg.model.name
    
    # Build feature paths
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
    
    # Run linear probing
    linear_probe(cfg, feature_paths)

if __name__ == "__main__":
    main()
