from dataclasses import dataclass
from typing import Optional
from omegaconf import MISSING

@dataclass
class DataConfig:
    data_path: str = MISSING
    cache_dir: str = MISSING

@dataclass
class ExtractorConfig:
    batch_size: int = 256
    device: str = "cuda"
    num_workers: int = 8

@dataclass
class WandbConfig:
    project: str = "clip-dino-linear-probe"
    key: Optional[str] = None

@dataclass
class ProbeConfig:
    batch_size: int = 512
    lr: float = 1e-4
    epochs: int = 100
    weight_decay: float = 1e-3
    save_dir: str = "results"
    wandb: WandbConfig = WandbConfig()

@dataclass
class ModelConfig:
    name: str = MISSING
    pretrained: str = MISSING

@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    extractor: ExtractorConfig
    probe: ProbeConfig
