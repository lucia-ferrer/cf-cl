from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Optional, Union

from torch import nn
from torchvision.transforms import v2

TransformLike = Union[nn.Module, v2.Transform, Callable[[Any], Any]]


@dataclass
class ConfigBenchmark:
    n_experiences: int
    return_task_id: Optional[bool] = None
    train_transform: Optional[TransformLike] = None
    eval_transform: Optional[TransformLike] = None

    def to_kwargs(self):
        cfg = asdict(self)
        # remove 'self' and drop None values
        return {
            k: v
            for k, v in cfg.items()
            if k not in ("self", "n_experiences") and v is not None
        }


@dataclass
class ConfigModel:
    img_channels: int  # 1 or 3
    img_size: int  # 28, 32, 64
    latent_dim: int = 64
    categorical_dim: int = 0
    arch: str = "CnnVAE"  # e.g., "MultiBandVAE"
    activation : str = "sigmoid"
    dropout: float = 0.2
    base_ch: int = 32
    lr: float = 1e-3
    optimizer: str = "Adam" # or "SGD"
    

@dataclass
class ConfigOptim:
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 0.0


@dataclass
class ConfigStrategy:
    train_mb_size: int = 128
    eval_mb_size: int = 512
    train_epochs: int = 30
    device: str = "cuda"  # or "cpu"


@dataclass
class Config:
    name: str
    benchmark: ConfigBenchmark
    model: ConfigModel
    train: ConfigStrategy
    
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        return self.dataset.name


__all__ = ["Config", "ConfigBenchmark", "ConfigModel", "ConfigStrategy"]
