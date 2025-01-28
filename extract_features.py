import os
import hydra
from omegaconf import DictConfig
from transformers import AutoModel, CLIPVisionModel
from feature_extractor import FeatureExtractor
from config import Config

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    config = Config(**cfg)
    os.makedirs(config.data.cache_dir, exist_ok=True)
    
    print("加载预训练模型...")
    if config.model.name.lower() == 'clip':
        model = CLIPVisionModel.from_pretrained(config.model.pretrained)
    else:  # DINO
        model = AutoModel.from_pretrained(config.model.pretrained)
    
    extractor = FeatureExtractor(
        config.model.name,
        model,
        device=config.extractor.device,
        batch_size=config.extractor.batch_size
    )
    
    for split in ['train', 'val']:
        print(f"\n处理 {config.model.name} {split} 集...")
        loader, classes = extractor.prepare_dataset(config.data.data_path, split)
        
        save_path = os.path.join(config.data.cache_dir, 
                                f'{config.model.name.lower()}_{split}_features.h5')
        extractor.extract_features(loader, save_path)
        print(f"特征已保存到 {save_path}")

if __name__ == "__main__":
    main()
