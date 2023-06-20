from typing import *

from .base import BaseConfig


class DatasetSubsetConfig(BaseConfig):
    data_dir: str
    speaker_id: str = 0
    recursive: bool = False
    extensions: List[str] = [".wav", ".flac", ".mp3", ".ogg", ".m4a", ".wma"]
    normalize: bool = True


class DatasetConfig(BaseConfig):
    subsets: List[DatasetSubsetConfig]
