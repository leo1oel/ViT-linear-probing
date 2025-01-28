#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig
import wandb
from extract_features import extract_features
from linear_probe import linear_probe

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    # Initialize wandb if enabled
    if cfg.get("use_wandb", False):
        wandb.init(
            project=cfg.get("wandb_project", "vision-model-evaluation"),
            name=cfg.get("wandb_run_name", None),
            config=dict(cfg)
        )
    
    # Extract features
    features_path = extract_features(cfg)
    
    # Perform linear probing
    metrics = linear_probe(cfg, features_path)
    
    # Log metrics to wandb if enabled
    if cfg.get("use_wandb", False):
        wandb.log(metrics)
        wandb.finish()
    
    return metrics

if __name__ == "__main__":
    main()
