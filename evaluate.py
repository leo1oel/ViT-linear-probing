#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig
import wandb
from extract_features import extract_features
from linear_probe import linear_probe
import os

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    if cfg.probe.wandb.use_wandb: 
        os.environ["WANDB_API_KEY"] = cfg.probe.wandb.key
        os.environ["WANDB_START_METHOD"] = "thread"

    # Extract features
    features_path = extract_features(cfg)
    
    # Perform linear probing
    metrics = linear_probe(cfg, features_path)
    
    # Log metrics to wandb if enabled
    if cfg.probe.wandb.use_wandb:  
        wandb.finish()
    
    return metrics

if __name__ == "__main__":
    main()
