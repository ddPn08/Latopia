import os
from typing import *

import numpy as np
import torch.utils.data
from scipy.io.wavfile import read

from latopia import mel_extractor
from latopia.config.vits import ViTsDatasetConfig

from .base import AudioDataset, AudioDataSubset


def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


class ViTsAudioDataSubset(AudioDataSubset):
    F0_DIR_NAME = "f0"
    F0_NSF_DIR_NAME = "f0_nsf"
    FEATURES_DIR_NAME = "features"

    DATA_TYPES = [
        *AudioDataSubset.DATA_TYPES,
        ("f0", F0_DIR_NAME, ".npy"),
        ("f0_nsf", F0_NSF_DIR_NAME, ".npy"),
        ("features", FEATURES_DIR_NAME, ".npy"),
    ]


class ViTsAudioDataset(AudioDataset):
    __subset_class__ = ViTsAudioDataSubset

    def __init__(self, config: ViTsDatasetConfig):
        super().__init__(config)
        self.config = config

    def __getitem__(self, idx):
        subset, rel_idx = self.get_rel_idx(idx)

        data = subset.get(rel_idx)

        speaker_id = torch.LongTensor([int(subset.config.speaker_id)])

        _, _audio = read(data["waves"])
        spec, audio = self.get_audio(data["waves"])

        f0 = np.load(data["f0"])
        f0_nsf = np.load(data["f0_nsf"])

        features = np.load(data["features"]).squeeze(0)
        features = np.repeat(features, 2, axis=0)
        n_num = min(features.shape[0], 900)
        features = torch.FloatTensor(features[:n_num, :])

        f0 = torch.LongTensor(f0[:n_num])
        f0_nsf = torch.FloatTensor(f0_nsf[:n_num])

        len_features, len_spec = features.size()[0], spec.size()[-1]

        if len_features != len_spec:
            len_min = min(len_features, len_spec)
            len_audio = len_min * self.config.hop_length
            spec = spec[:, :len_min]
            audio = audio[:, :len_audio]

            features = features[:len_min, :]
            f0 = f0[:len_min]
            f0_nsf = f0_nsf[:len_min]

        example = {}
        example["features"] = features
        example["audio"] = audio
        example["spec"] = spec
        example["f0"] = f0
        example["f0_nsf"] = f0_nsf
        example["speaker_id"] = speaker_id

        return example

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.config.sampling_rate:
            raise ValueError(
                "{} SR doesn't match target {} SR".format(
                    sampling_rate, self.config.sampling_rate
                )
            )
        audio_norm = audio / self.config.max_wav_value
        audio_norm = audio.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = mel_extractor.spectrogram_torch(
                audio_norm,
                self.config.filter_length,
                self.config.hop_length,
                self.config.win_length,
                center=False,
            )
            spec = spec.squeeze(0)
            torch.save(spec, spec_filename, _use_new_zipfile_serialization=False)
        return spec, audio_norm


class ViTsAudioCollate:
    """Zero-pads model inputs and targets"""

    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x["spec"].size(1) for x in batch]), dim=0, descending=True
        )

        max_spec_len = max([x["spec"].size(1) for x in batch])
        max_wave_len = max([x["audio"].size(1) for x in batch])
        max_phone_len = max([x["features"].size(0) for x in batch])

        spec_lengths = torch.LongTensor(len(batch))
        audio_lengths = torch.LongTensor(len(batch))
        features_lengths = torch.LongTensor(len(batch))

        spec_padded = torch.FloatTensor(
            len(batch), batch[0]["spec"].size(0), max_spec_len
        )
        audio_padded = torch.FloatTensor(len(batch), 1, max_wave_len)
        f0_padded = torch.LongTensor(len(batch), max_phone_len)
        f0_nsf_padded = torch.FloatTensor(len(batch), max_phone_len)
        features_padded = torch.FloatTensor(
            len(batch), max_phone_len, batch[0]["features"].shape[1]
        )

        spec_padded.zero_()
        audio_padded.zero_()
        f0_padded.zero_()
        f0_nsf_padded.zero_()
        features_padded.zero_()

        speaker_id = torch.LongTensor(len(batch))

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            spec = row["spec"]
            spec_padded[i, :, : spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

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
        example["spec"] = spec_padded
        example["spec_lengths"] = spec_lengths
        example["audio"] = audio_padded
        example["audio_lengths"] = audio_lengths
        example["f0"] = f0_padded
        example["f0_nsf"] = f0_nsf_padded
        example["speaker_id"] = speaker_id

        return example


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.

    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """

    def __init__(
        self,
        dataset: ViTsAudioDataset,
        batch_size: int,
        boundaries: List[int],
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries

        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas

    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, -1, -1):  #
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i + 1)

        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (
                total_batch_size - (len_bucket % total_batch_size)
            ) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = (
                ids_bucket
                + ids_bucket * (rem // len_bucket)
                + ids_bucket[: (rem % len_bucket)]
            )

            # subsample
            ids_bucket = ids_bucket[self.rank :: self.num_replicas]

            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [
                    bucket[idx]
                    for idx in ids_bucket[
                        j * self.batch_size : (j + 1) * self.batch_size
                    ]
                ]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches

        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)

    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1

        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid + 1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size
