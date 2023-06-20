from typing import *

from .base import BaseConfig


class ViTsGeneratorConfig(BaseConfig):
    inter_channels: int = 192
    hidden_channels: int = 192
    filter_channels: int = 768
    n_heads: int = 2
    n_layers: int = 6
    kernel_size: int = 3
    p_dropout: float = 0
    resblock: bool = True
    resblock_kernel_sizes: List[int] = (3, 7, 11)
    resblock_dilation_sizes: List[List[int]] = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    upsample_rates: List[int] = (10, 10, 2, 2)
    upsample_initial_channel: int = 512
    upsample_kernel_sizes: List[int] = (16, 16, 4, 4)
    use_spectral_norm: bool = False
    gin_channels: int = 256
    emb_channels: int = 768
    spk_embed_dim: int = 109


class ViTsDiscriminatorConfig(BaseConfig):
    use_spectral_norm: bool = False
    periods: List[int] = [2, 3, 5, 7, 11, 17, 23, 37]


class ViTsDatasetConfig(BaseConfig):
    max_wav_value: int = 32768.0
    sampling_rate: int = 40000
    filter_length: int = 2048
    hop_length: int = 400
    win_length: int = 2048
    n_mel_channels: int = 125
    mel_fmin: float = 0.0
    mel_fmax: float = None


class ViTsTrainConfig(BaseConfig):
    segment_size: int = 12800
    init_lr_ratio: int = 1
    warmup_epochs: int = 0
    c_mel: int = 45
    c_kl: float = 1.0
    dataset: ViTsDatasetConfig = ViTsDatasetConfig()


class ViTsConfig(BaseConfig):
    generator: ViTsGeneratorConfig = ViTsGeneratorConfig()
    discriminator: ViTsDiscriminatorConfig = ViTsDiscriminatorConfig()
    train: ViTsTrainConfig = ViTsTrainConfig()
