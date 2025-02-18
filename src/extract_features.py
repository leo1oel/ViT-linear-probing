from pathlib import Path
import hydra
from omegaconf import DictConfig
from transformers import AutoModel, CLIPVisionModel
from feature_extractor import FeatureExtractor, DatasetLoader, DatasetConfig
from rich.console import Console
from rich.panel import Panel
from typing import Dict

console = Console()

def extract_features(cfg: DictConfig) -> Dict[str, str]:
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
    
    dataset_loader = DatasetLoader(dataset_config)
    
    extractor = FeatureExtractor(model, cfg.extractor.device)
    
    feature_paths = {}
    
    # Process requested splits
    for split in cfg.extractor.splits:
        # Load dataset
        data_loader, class_to_idx = dataset_loader.load_dataset(split)
        
        # Build output path
        output_path = Path(cfg.data.cache_dir) / f"{cfg.model.name.lower()}_{split}_features.h5"
        
        # Extract features
        extractor.extract_features(
            data_loader=data_loader,
            output_path=str(output_path),
            class_to_idx=class_to_idx
        )
        
        feature_paths[f"{split}_features"] = str(output_path)
    
    return feature_paths

current_dir = Path(__file__).resolve().parent
config_path = current_dir.parent / "conf"

@hydra.main(version_base=None, config_path=str(config_path), config_name="config")
def main(cfg: DictConfig) -> None:
    extract_features(cfg)

if __name__ == "__main__":
    main()