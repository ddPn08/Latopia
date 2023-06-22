from typing import *

import toml

from latopia.config.dataset import DatasetConfig
from latopia.dataset.base import AudioDataset
from latopia.dataset.preprocess import PreProcessor
from latopia.f0_extractor import F0_METHODS_TYPE
from latopia.logger import set_logger

logger = set_logger(__name__)


def all(
    config_path: str,
    target_sr: int,
    max_workers: int = 1,
    write_mute: bool = True,
    slice: bool = True,
    f0_method: F0_METHODS_TYPE = "crepe",
    crepe_model: str = "tiny",
    hop_length: int = 160,
    f0_max: int = 1100.0,
    f0_min: int = 50.0,
    f0_mel_max: Optional[int] = None,
    f0_mel_min: Optional[int] = None,
    encoder_path: str = "./models/encoders/checkpoint_best_legacy_500.pt",
    encoder_channels: int = 768,
    encoder_output_layer: int = 12,
    device: str = "cpu",
):
    r"""
    Preprocess all files in the dataset.

    Args:
        # Common arguments
        config_path (str): Path to the dataset config file.
        target_sr (int): Target sample rate.
        max_workers (int, optional): Number of workers.

        # write_wave arguments
        slice (bool, optional): Slice the audio file into small pieces.

        # extract_f0 arguments
        f0_method (str, optional): F0 extraction method.
        crepe_model (str, optional): Crepe model type.
        hop_length (int, optional): Hop length for f0 extraction.
        f0_max (int, optional): Maximum f0 value.
        f0_min (int, optional): Minimum f0 value.
        f0_mel_max (int, optional): Maximum f0 value in mel scale.
        f0_mel_min (int, optional): Minimum f0 value in mel scale.

        # extract_features arguments
        encoder_path (str, optional): Path to the encoder model.
        encoder_channels (int, optional): Number of channels of the encoder model.
        encoder_output_layer (int, optional): Output layer of the encoder model.
        device (str, optional): Device to run the encoder model.
    """
    config = DatasetConfig.parse_obj(toml.load(config_path))
    dataset = AudioDataset(config)
    preprocessor = PreProcessor(dataset)
    logger.info("Writing wave files...")
    preprocessor.write_wave(
        target_sr,
        slice=slice,
        write_mute=write_mute,
        max_workers=max_workers,
    )
    logger.info("Extracting f0...")
    preprocessor.extract_f0(
        f0_method,
        crepe_model=crepe_model,
        hop_length=hop_length,
        f0_max=f0_max,
        f0_min=f0_min,
        f0_mel_max=f0_mel_max,
        f0_mel_min=f0_mel_min,
        max_workers=max_workers,
    )
    logger.info("Extracting features...")
    preprocessor.extract_features(
        encoder_path,
        encoder_channels=encoder_channels,
        encoder_output_layer=encoder_output_layer,
        device=device,
    )


def run():
    return {
        "all": all,
    }
