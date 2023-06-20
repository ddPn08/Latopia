from typing import *

import torch

from latopia.config.dataset import DatasetConfig
from latopia.config.train import TrainConfig
from latopia.config.vits import ViTsConfig
from latopia.vits.train import train


def train_vits(
    config_path: str,
    dataset_config_path: str,
    vits_config_path: Optional[str] = None,
    device: Union[str, List[str]] = "cpu",
):
    config = TrainConfig.parse_auto(config_path)
    dataset_config = DatasetConfig.parse_auto(dataset_config_path)
    if vits_config_path is not None:
        vits_config = ViTsConfig.parse_auto(vits_config_path)
    else:
        vits_config = ViTsConfig()

    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, list):
        device = [torch.device(d) for d in device]

    train(
        device,
        config,
        dataset_config,
        vits_config,
    )


def run():
    return {"vits": train_vits}
