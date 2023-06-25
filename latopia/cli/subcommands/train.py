import os
from typing import *

import torch

from latopia.config.diffusion import (
    DiffusionDatasetConfig,
    DiffusionModelConfig,
    DiffusionTrainConfig,
)
from latopia.config.train import TrainConfig
from latopia.config.vits import ViTsDatasetConfig, ViTsModelConfig, ViTsTrainConfig
from latopia.diffusion.train import train as run_diff_train
from latopia.vits.train import train as run_vits_train


def parse_config(
    config_path: str,
    config_class: Type[TrainConfig] = TrainConfig,
    dataset_config_path: Optional[str] = None,
    model_config_path: Optional[str] = None,
):
    config = config_class.parse_auto(config_path)
    config_dir = os.path.dirname(config_path)
    if config.model_config is not None and model_config_path is None:
        model_config_path = (
            config.model_config
            if os.path.isabs(config.model_config)
            else os.path.join(config_dir, config.model_config)
        )
    if config.dataset_config is not None and dataset_config_path is None:
        dataset_config_path = (
            config.dataset_config
            if os.path.isabs(config.dataset_config)
            else os.path.join(config_dir, config.dataset_config)
        )
    return config, dataset_config_path, model_config_path


def train_vits(
    config_path: str,
    dataset_config_path: Optional[str] = None,
    model_config_path: Optional[str] = None,
    device: Union[str, List[str]] = "cpu",
):
    config, dataset_config_path, model_config_path = parse_config(
        config_path, ViTsTrainConfig, dataset_config_path, model_config_path
    )
    dataset_config = ViTsDatasetConfig.parse_auto(dataset_config_path)
    if model_config_path is not None:
        vits_config = ViTsModelConfig.parse_auto(model_config_path)
    else:
        vits_config = ViTsModelConfig()

    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, list):
        device = [torch.device(d) for d in device]

    run_vits_train(
        device,
        config,
        dataset_config,
        vits_config,
    )


def train_diff(
    config_path: str,
    dataset_config_path: str,
    model_config_path: Optional[str] = None,
    device: Union[str, List[str]] = "cpu",
):
    config, dataset_config_path, model_config_path = parse_config(
        config_path, DiffusionTrainConfig, dataset_config_path, model_config_path
    )
    dataset_config = DiffusionDatasetConfig.parse_auto(dataset_config_path)
    if model_config_path is not None:
        model_config = DiffusionModelConfig.parse_auto(model_config_path)
    else:
        model_config = DiffusionModelConfig()

    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, list):
        device = [torch.device(d) for d in device]

    run_diff_train(
        device,
        config,
        dataset_config,
        model_config,
    )


def run():
    return {"vits": train_vits, "diff": train_diff}
