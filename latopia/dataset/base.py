import glob
import os
from typing import *

import torch.utils.data

from latopia.config.dataset import DatasetConfig, DatasetSubsetConfig


class AudioDataSubset:
    PROCESSED_DIR = ".latopia"
    WAVES_DIR_NAME = "waves"
    WAVES_16K_DIR_NAME = "waves_16k"

    DATA_TYPES = [
        ("waves", WAVES_DIR_NAME, ".wav"),
        ("waves_16k", WAVES_16K_DIR_NAME, ".wav"),
    ]

    def __init__(self, config: DatasetSubsetConfig, sampling_rate: int = 40000):
        self.config = config
        self.sampling_rate = sampling_rate

        self.files = []
        for ext in self.config.extensions:
            if self.config.recursive:
                glob_path = f"{self.config.data_dir}/**/*{ext}"
            else:
                glob_path = f"{self.config.data_dir}/*{ext}"
            for file in glob.glob(glob_path, recursive=self.config.recursive):
                self.files.append(file)

        self.files.sort()

        if len(self.get_waves()) > 0:
            self.files = self.get_waves()

    def get_processed_dir(self):
        return os.path.join(
            self.config.data_dir, self.__class__.PROCESSED_DIR, str(self.sampling_rate)
        )

    def get(self, idx: int):
        data = {}
        for data_type, dir, ext in self.__class__.DATA_TYPES:
            data[data_type] = self.__get_files(data_type, dir, ext)[idx]

        return data

    def __get_files(self, var_name: str, dir_name: str, ext: str = ".npy"):
        if hasattr(self, var_name) and getattr(self, var_name):
            return getattr(self, var_name)
        dir_path = os.path.join(
            self.config.data_dir,
            self.__class__.PROCESSED_DIR,
            str(self.sampling_rate),
            dir_name,
        )
        if not os.path.exists(dir_path):
            return []
        setattr(
            self, var_name, sorted(glob.glob(f"{dir_path}/**/*{ext}", recursive=True))
        )
        return getattr(self, var_name)

    def get_waves(self):
        return self.__get_files("waves", self.__class__.WAVES_DIR_NAME, ext=".wav")

    def get_16k_waves(self):
        return self.__get_files(
            "waves_16k", self.__class__.WAVES_16K_DIR_NAME, ext=".wav"
        )


class AudioDataset(torch.utils.data.Dataset):
    __subset_class__ = AudioDataSubset

    def __init__(self, config: DatasetConfig):
        self.config = config

        self.subsets: List[AudioDataSubset] = []
        for subset_config in self.config.subsets:
            self.subsets.append(
                self.__class__.__subset_class__(subset_config, config.sampling_rate)
            )

        self.length = sum([len(subset.get_waves()) for subset in self.subsets])

        self.idx_subset_map = {}

        global_idx = 0
        for subset_idx, subset in enumerate(self.subsets):
            for idx in range(len(subset.files)):
                self.idx_subset_map[global_idx + idx] = subset_idx
            global_idx += len(subset.files)

        self.lengths = []
        for subset in self.subsets:
            for file in subset.files:
                self.lengths.append(
                    os.path.getsize(file) // (3 * self.config.hop_length)
                )

    def get_rel_idx(self, idx):
        subset = self.subsets[self.idx_subset_map[idx]]
        prev_subset = (
            None
            if self.idx_subset_map[idx] == 0
            else self.subsets[self.idx_subset_map[idx] - 1]
        )
        rel_idx = idx if prev_subset is None else idx - len(prev_subset.files)
        return subset, rel_idx

    def __len__(self):
        return self.length
