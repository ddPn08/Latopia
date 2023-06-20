from typing import *

from .base import BaseConfig
from .vits import ViTsConfig


class ModelConfig(BaseConfig):
    vits: Optional[ViTsConfig]
