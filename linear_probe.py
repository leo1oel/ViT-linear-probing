import os
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """Main function for linear probe training"""
    model_name = cfg.model.name
    
    # 构建特征路径
    train_features = os.path.join(
        cfg.data.data_path,
        f"{model_name}_train_features.h5"
    )
    val_features = os.path.join(
        cfg.data.data_path,
        f"{model_name}_val_features.h5"
    )
    
    # 训练和评估
    from trainer import train_and_evaluate
    train_and_evaluate(
        model_name=model_name,
        features_path=train_features,
        val_features_path=val_features,
        config=cfg
    )

if __name__ == "__main__":
    main()
