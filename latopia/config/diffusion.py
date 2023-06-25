from typing import *

from .base import BaseConfig
from .dataset import DatasetConfig
from .train import TrainConfig


class DiffusionModelConfig(BaseConfig):
    emb_channels: int = 768
    spk_embed_dim: int = 1
    use_pitch_aug: bool = False
    n_layers: int = 20
    n_chans: int = 512
    n_hidden: int = 256
    k_step_max: int = 1000


class DiffusionDatasetConfig(DatasetConfig):
    pass


class DiffusionTrainConfig(TrainConfig):
    vocoder_type: Literal["nsf-hifigan", "nsf-hifigan-log10"] = "nsf-hifigan"
    vocoder_dir: str = "./models/vocoders/nsf_hifigan"
