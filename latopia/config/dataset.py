from typing import *

from .base import BaseConfig


class DatasetSubsetConfig(BaseConfig):
    data_dir: str
    speaker_id: str = 0
    recursive: bool = False
    extensions: List[str] = [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".wma"]
    normalize: bool = True


class DatasetConfig(BaseConfig):
    write_mute: bool = True
    slice: bool = True
    sampling_rate: int = 40000
    hop_length: int = 512
    f0_max: int = 1100.0
    f0_min: int = 50.0
    f0_mel_max: Optional[int] = None
    f0_mel_min: Optional[int] = 0
    subsets: List[DatasetSubsetConfig]
