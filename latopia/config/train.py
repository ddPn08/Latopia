from typing import *

from .base import BaseConfig


class TrainConfig(BaseConfig):
    pretrained_model_path: Optional[str] = None
    pretrained_discriminator_path: Optional[str] = None
    resume_model_path: Optional[str] = None
    output_name: str = "model"
    output_dir: str = "output"
    cache_in_gpu: bool = False
    save_as: Literal["pt", "safetensors"] = "safetensors"
    save_state: bool = True
    sampling_rate: int = 40000
    seed: Optional[int] = None
    learning_rate: float = 1e-4
    batch_size: int = 16
    mixed_precision: Optional[Literal["fp16", "bf16"]] = None
    max_train_epoch: int = 30
    save_every_n_epoch: int = 10
    betas: List[float] = (0.8, 0.99)
    eps: float = 1e-9
    lr_decay: float = 0.999875
