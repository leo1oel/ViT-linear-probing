import os
import sys
import hydra
import torch
import deepspeed
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from typing import Dict
from trainer import train_and_evaluate
import h5py

def setup_distributed():
    """设置分布式训练环境"""
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if torch.distributed.is_initialized():
        return local_rank

    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed()
    
    return local_rank

def linear_probe(cfg: DictConfig, feature_paths: Dict[str, str]) -> Dict:
    model_name = cfg.model.name
    
    # 计算每个 epoch 的步数并更新到配置中
    train_samples = 0
    with h5py.File(feature_paths["train_features"], 'r') as f:
        train_samples = len(f['targets'])
    
    steps_per_epoch = train_samples // (cfg.probe.batch_size * int(os.environ.get("WORLD_SIZE", 1)))
    OmegaConf.update(cfg, "steps_per_epoch", steps_per_epoch, merge=True)
    
    # 训练和评估
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
def main(cfg: DictConfig) -> None:
    """Linear probe 训练的主函数"""
    # 从环境变量获取 local_rank
    local_rank = setup_distributed()
    
    # 设置特征文件路径
    cache_dir = Path(cfg.data.cache_dir)
    feature_paths = {
        "train_features": cache_dir / f"{cfg.model.name.lower()}_train_features.h5",
        "val_features": cache_dir / f"{cfg.model.name.lower()}_val_features.h5"
    }
    
    # 确保特征文件存在
    for path in feature_paths.values():
        if not path.exists():
            raise FileNotFoundError(f"找不到特征文件：{path}")
    
    # 运行 linear probing
    try:
        metrics = linear_probe(cfg, feature_paths)
        if local_rank == 0:  # 只在主进程打印结果
            print("\n最终评估结果:")
            for k, v in metrics.items():
                print(f"{k}: {v}")
    except Exception as e:
        import traceback
        print(f"错误：{str(e)}")
        print(traceback.format_exc())
        raise e

if __name__ == "__main__":
    # 处理 DeepSpeed 参数
    args = []
    for arg in sys.argv[1:]:
        if not arg.startswith("--local_rank"):
            args.append(arg)
    sys.argv = [sys.argv[0]] + args
    main()