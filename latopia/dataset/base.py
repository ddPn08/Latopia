import glob
import os
from typing import *

import torch.utils.data

from latopia.config.dataset import DatasetConfig, DatasetSubsetConfig


class AudioDataSubset:
    PROCESSED_DIR = ".latopia"
    WAVES_DIR_NAME = "waves"
    WAVES_16K_DIR_NAME = "waves_16k"
    F0_DIR_NAME = "f0"
    F0_NSF_DIR_NAME = "f0_nsf"
    FEATURES_DIR_NAME = "features"

    def __init__(self, config: DatasetSubsetConfig):
        self.config = config

        self.files = []
        for ext in self.config.extensions:
            if self.config.recursive:
                glob_path = f"{self.config.data_dir}/**/*{ext}"
            else:
                glob_path = f"{self.config.data_dir}/*{ext}"
            for file in glob.glob(glob_path, recursive=self.config.recursive):
                self.files.append(file)

        self.files.sort()

        self.waves: List[str] = None
        self.waves_16k: List[str] = None
        self.f0: List[str] = None
        self.f0_nsf: List[str] = None
        self.features: List[str] = None

        if len(self.get_waves()) > 0:
            self.files = self.get_waves()

    def get_processed_dir(self):
        return os.path.join(self.config.data_dir, AudioDataSubset.PROCESSED_DIR)

    def __get_files(self, var_name: str, dir_name: str, ext: str = ".npy"):
        if hasattr(self, var_name) and getattr(self, var_name):
            return getattr(self, var_name)
        dir_path = os.path.join(
            self.config.data_dir,
            AudioDataSubset.PROCESSED_DIR,
            dir_name,
        )
        if not os.path.exists(dir_path):
            return []
        setattr(
            self, var_name, sorted(glob.glob(f"{dir_path}/**/*{ext}", recursive=True))
        )
        return getattr(self, var_name)

    def get_waves(self):
        return self.__get_files("waves", AudioDataSubset.WAVES_DIR_NAME, ext=".wav")

    def get_16k_waves(self):
        return self.__get_files(
            "waves_16k", AudioDataSubset.WAVES_16K_DIR_NAME, ext=".wav"
        )

    def get_f0(self):
        return self.__get_files("f0", AudioDataSubset.F0_DIR_NAME)

    def get_f0_nsf(self):
        return self.__get_files("f0_nsf", AudioDataSubset.F0_NSF_DIR_NAME)

    def get_features(self):
        return self.__get_files("features", AudioDataSubset.FEATURES_DIR_NAME)


class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, config: DatasetConfig):
        self.config = config

        self.subsets: List[AudioDataSubset] = []
        for subset_config in self.config.subsets:
            self.subsets.append(AudioDataSubset(subset_config))

        self.length = sum([len(subset.get_waves()) for subset in self.subsets])

        self.idx_subset_map = {}
        for subset_idx, subset in enumerate(self.subsets):
            for idx in range(len(subset.files)):
                self.idx_subset_map[idx] = subset_idx

    def __len__(self):
        return self.length + (1 if self.config.add_mute else 0)
