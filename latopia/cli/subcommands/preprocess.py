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
    max_workers: int = 1,
    f0_method: F0_METHODS_TYPE = "crepe",
    crepe_model: str = "tiny",
    encoder_path: str = "./models/encoders/checkpoint_best_legacy_500.pt",
    encoder_output_layer: int = 12,
    device: str = "cpu",
    vocoder_path: str = "./models/vocoders/nsf_hifigan",
    vocoder_type: str = "nsf-hifigan",
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
        config.sampling_rate,
        slice=config.slice,
        write_mute=config.write_mute,
        max_workers=max_workers,
    )
    logger.info("Extracting volume...")
    preprocessor.extract_volume(
        sampling_rate=config.sampling_rate,
        hop_length=config.hop_length,
        max_workers=max_workers,
    )
    logger.info("Extracting f0...")
    preprocessor.extract_f0(
        f0_method,
        crepe_model=crepe_model,
        hop_length=config.hop_length,
        f0_max=config.f0_max,
        f0_min=config.f0_min,
        f0_mel_max=config.f0_mel_max,
        f0_mel_min=config.f0_mel_min,
        max_workers=max_workers,
    )
    logger.info("Extracting features...")
    preprocessor.extract_features(
        encoder_path,
        encoder_output_layer=encoder_output_layer,
        hop_length=config.hop_length,
        device=device,
    )
    logger.info("Extracting mel spectrogram...")
    preprocessor.extract_mel(
        vocoder_path,
        sampling_rate=config.sampling_rate,
        max_workers=max_workers,
        vocoder_type=vocoder_type,
        device=device,
    )


def run():
    return {
        "all": all,
    }
