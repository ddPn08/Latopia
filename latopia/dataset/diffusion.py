from typing import *

import librosa
import numpy as np
import torch.nn.functional as F
import torch.utils.data

from latopia.config.diffusion import DiffusionDatasetConfig
from latopia.dataset.base import AudioDataset

from .vits import ViTsAudioDataSubset


class DiffusionAudioDataSubset(ViTsAudioDataSubset):
    VOLUME_DIR_NAME = "volume"
    MEL_DIR_NAME = "mel"

    DATA_TYPES = [
        *ViTsAudioDataSubset.DATA_TYPES,
        ("volume", VOLUME_DIR_NAME, ".npy"),
        ("mel", MEL_DIR_NAME, ".npy"),
    ]


class DiffusionAudioDataset(AudioDataset):
    __subset_class__ = DiffusionAudioDataSubset

    def __init__(self, config: DiffusionDatasetConfig):
        super().__init__(config)
        self.config = config

    def __getitem__(self, idx):
        subset, rel_idx = self.get_rel_idx(idx)

        data = subset.get(rel_idx)

        speaker_id = torch.LongTensor([int(subset.config.speaker_id)])

        audio, _ = librosa.load(data["waves"], self.config.sampling_rate)
        audio = torch.from_numpy(audio).float().unsqueeze(0)
        duration = librosa.get_duration(
            filename=data["waves"], sr=self.config.sampling_rate
        )

        frame_resolution = self.config.hop_length / self.config.sampling_rate
        start_frame = int(0 / frame_resolution)
        units_frame_len = int(duration / frame_resolution)
        n_frames = int(audio.size(-1) // self.config.hop_length + 1)

        mel = torch.from_numpy(np.load(data["mel"]))
        features = torch.from_numpy(np.load(data["features"]))
        f0_nsf = torch.from_numpy(np.load(data["f0_nsf"])).float().unsqueeze(-1)
        volume = torch.from_numpy(np.load(data["volume"])).float().unsqueeze(-1)

        features = features.transpose(1, 2)
        features = F.interpolate(features, size=int(n_frames), mode="nearest")
        features = features.transpose(1, 2).squeeze(0)

        mel = mel[start_frame : start_frame + units_frame_len]
        features = features[start_frame : start_frame + units_frame_len]
        f0_nsf = f0_nsf[start_frame : start_frame + units_frame_len]
        volume = volume[start_frame : start_frame + units_frame_len]

        example = {}

        example["duration"] = duration
        example["mel"] = mel
        example["features"] = features
        example["f0"] = f0_nsf
        example["volume"] = volume
        example["speaker_id"] = speaker_id

        return example


class DiffusionAudioCollate:
    def __call__(self, batch):
        example = {}

        example["duration"] = torch.FloatTensor([e["duration"] for e in batch])
        example["mel"] = torch.nn.utils.rnn.pad_sequence(
            [e["mel"] for e in batch], batch_first=True
        )
        example["features"] = torch.nn.utils.rnn.pad_sequence(
            [e["features"] for e in batch], batch_first=True
        )
        example["f0"] = torch.nn.utils.rnn.pad_sequence(
            [e["f0"] for e in batch], batch_first=True
        )
        example["volume"] = torch.nn.utils.rnn.pad_sequence(
            [e["volume"] for e in batch], batch_first=True
        )
        example["speaker_id"] = torch.nn.utils.rnn.pad_sequence(
            [e["speaker_id"] for e in batch], batch_first=True
        )

        return example
