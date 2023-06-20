import os
from typing import *

import numpy as np
import torch.utils.data
from scipy.io.wavfile import read

from latopia import mel_extractor
from latopia.config.dataset import DatasetConfig
from latopia.config.vits import ViTsDatasetConfig
from latopia.dataset.base import AudioDataset


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


class ViTsAudioDataset(AudioDataset):
    def __init__(self, config: DatasetConfig, data_config: ViTsDatasetConfig):
        super().__init__(config)

        self.max_wav_value = data_config.max_wav_value
        self.sampling_rate = data_config.sampling_rate
        self.filter_length = data_config.filter_length
        self.hop_length = data_config.hop_length
        self.win_length = data_config.win_length
        self.sampling_rate = data_config.sampling_rate

    def __getitem__(self, idx):
        subset = self.subsets[self.idx_subset_map[idx]]
        prev_subset = (
            None
            if self.idx_subset_map[idx] == 0
            else self.subsets[self.idx_subset_map[idx] - 1]
        )
        rel_idx = idx if prev_subset is None else idx - len(prev_subset.files)

        file = subset.get_waves()[rel_idx]
        f0 = subset.get_f0()[rel_idx]
        f0_nsf = subset.get_f0_nsf()[rel_idx]
        features = subset.get_features()[rel_idx]

        speaker_id = torch.LongTensor([int(subset.config.speaker_id)])

        mel, audio = self.get_audio(file)

        f0 = np.load(f0)
        f0_nsf = np.load(f0_nsf)
        features = np.load(features)

        features = np.repeat(features, 2, axis=0)
        n_num = min(features.shape[0], 900)

        features = torch.FloatTensor(features[:n_num, :])
        f0 = torch.LongTensor(f0[:n_num])
        f0_nsf = torch.FloatTensor(f0_nsf[:n_num])

        len_features, len_mel = features.size()[0], mel.size()[-1]

        if len_features != len_mel:
            len_min = min(len_features, len_mel)
            len_audio = len_min * self.hop_length
            mel = mel[:, :len_min]
            audio = audio[:, :len_audio]

            features = features[:len_min, :]
            f0 = f0[:len_min]
            f0_nsf = f0_nsf[:len_min]

        example = {}
        example["features"] = features
        example["audio"] = audio
        example["mel"] = mel
        example["f0"] = f0
        example["f0_nsf"] = f0_nsf
        example["speaker_id"] = speaker_id

        return example

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate
                )
            )
        # audio_norm = audio / self.max_wav_value
        audio_norm = audio.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = mel_extractor.spectrogram_torch(
                audio_norm,
                self.filter_length,
                self.hop_length,
                self.win_length,
                center=False,
            )
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        return spec, audio_norm


class TextAudioCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x["mel"].size(1) for x in batch]), dim=0, descending=True
        )

        max_mel_len = max([x["mel"].size(1) for x in batch])
        max_wave_len = max([x["audio"].size(1) for x in batch])
        mel_lengths = torch.LongTensor(len(batch))
        audio_lengths = torch.LongTensor(len(batch))
        mel_padded = torch.FloatTensor(len(batch), batch[0]["mel"].size(0), max_mel_len)
        audio_padded = torch.FloatTensor(len(batch), 1, max_wave_len)
        mel_padded.zero_()
        audio_padded.zero_()

        max_phone_len = max([x["features"].size(0) for x in batch])
        features_lengths = torch.LongTensor(len(batch))
        features_padded = torch.FloatTensor(
            len(batch), max_phone_len, batch[0]["features"].shape[1]
        )
        f0_padded = torch.LongTensor(len(batch), max_phone_len)
        f0_nsf_padded = torch.FloatTensor(len(batch), max_phone_len)
        features_padded.zero_()
        f0_padded.zero_()
        f0_nsf_padded.zero_()
        speaker_id = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            mel = row["mel"]
            mel_padded[i, :, : mel.size(1)] = mel
            mel_lengths[i] = mel.size(1)

            audio = row["audio"]
            audio_padded[i, :, : audio.size(1)] = audio
            audio_lengths[i] = audio.size(1)

            features = row["features"]
            features_padded[i, : features.size(0), :] = features
            features_lengths[i] = features.size(0)

            f0 = row["f0"]
            f0_padded[i, : f0.size(0)] = f0
            f0_nsf = row["f0_nsf"]
            f0_nsf_padded[i, : f0_nsf.size(0)] = f0_nsf

            speaker_id[i] = row["speaker_id"]

        example = {}
        example["features"] = features_padded
        example["features_lengths"] = features_lengths
        example["mel"] = mel_padded
        example["mel_lengths"] = mel_lengths
        example["audio"] = audio_padded
        example["audio_lengths"] = audio_lengths
        example["f0"] = f0_padded
        example["f0_nsf"] = f0_nsf_padded
        example["speaker_id"] = speaker_id

        return example
