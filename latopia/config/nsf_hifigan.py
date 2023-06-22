from typing import *

from .base import BaseConfig


class DistConfig(BaseConfig):
    dist_backend: str
    dist_url: str
    world_size: int


class NsfHifiganConfig(BaseConfig):
    resblock: str
    num_gpus: int
    batch_size: int
    learning_rate: float
    adam_b1: float
    adam_b2: float
    lr_decay: float
    seed: int
    upsample_rates: List[int]
    upsample_kernel_sizes: List[int]
    upsample_initial_channel: int
    resblock_kernel_sizes: List[int]
    resblock_dilation_sizes: List[List[int]]
    discriminator_periods: List[int]
    segment_size: int
    num_mels: int
    num_freq: int
    n_fft: int
    hop_size: int
    win_size: int
    sampling_rate: int
    fmin: int
    fmax: int
    fmax_for_loss: Any
    num_workers: int
    dist_config: DistConfig
