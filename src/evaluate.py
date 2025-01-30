#!/usr/bin/env python3
import hydra
from omegaconf import DictConfig
from extract_features import extract_features
from linear_probe import linear_probe
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(os.path.dirname(current_dir), "conf")
@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(cfg: DictConfig):

    features_path = extract_features(cfg)

    metrics = linear_probe(cfg, features_path)

    return metrics

if __name__ == "__main__":
    main()
