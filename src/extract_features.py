import os
import hydra
from omegaconf import DictConfig
from transformers import AutoModel, CLIPVisionModel
from feature_extractor import FeatureExtractor, DatasetLoader, DatasetConfig
from rich.console import Console
from rich.panel import Panel
from typing import Dict

console = Console()

def extract_features(cfg: DictConfig) -> Dict[str, str]:
    """Main function for feature extraction
    
    Args:
        cfg: Configuration object containing:
            - data: dataset configuration
            - model: model configuration
            - extractor: feature extraction configuration including:
                - splits: list of splits to extract features from (e.g., ["train"], ["val"], or ["train", "val"])
        
    Returns:
        Dict[str, str]: Dictionary containing paths to extracted features for each requested split
    """
    # Create dataset configuration
    dataset_config = DatasetConfig(
        dataset_name=cfg.data.dataset_name,
        data_path=cfg.data.data_path,
        batch_size=cfg.extractor.batch_size,
        num_workers=cfg.extractor.num_workers,
        max_samples=cfg.extractor.get('max_samples', None)
    )
    
    # Load model
    if "clip" in cfg.model.name.lower():
        model = CLIPVisionModel.from_pretrained(cfg.model.pretrained)
        console.print(Panel("CLIP model loaded", style="bold red"))
    elif "mae" in cfg.model.name.lower():
        model = AutoModel.from_pretrained(cfg.model.pretrained, mask_ratio=0.0)
        console.print(Panel("MAE model loaded with mask ratio set to 0.0"), style="bold red")
    else:
        model = AutoModel.from_pretrained(cfg.model.pretrained)
        console.print(Panel("Auto model loaded"), style="bold red")
    
    # Create dataset loader
    dataset_loader = DatasetLoader(dataset_config)
    
    # Create feature extractor
    device = cfg.extractor.device
    extractor = FeatureExtractor(model, device)
    
    feature_paths = {}
    
    # Get splits to process from config
    splits = cfg.extractor.splits
    
    # Process requested splits
    for split in splits:
        # Load dataset
        data_loader, class_to_idx = dataset_loader.load_dataset(split)
        
        # Build output path
        output_path = os.path.join(
            cfg.data.cache_dir,
            f"{cfg.model.name.lower()}_{split}_features.h5"
        )
        
        # Extract features
        extractor.extract_features(
            data_loader=data_loader,
            output_path=output_path,
            class_to_idx=class_to_idx
        )
        
        feature_paths[f"{split}_features"] = output_path
    
    return feature_paths

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(os.path.dirname(current_dir), "conf")
@hydra.main(version_base=None, config_path=config_path, config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point"""
    extract_features(cfg)

if __name__ == "__main__":
    main()