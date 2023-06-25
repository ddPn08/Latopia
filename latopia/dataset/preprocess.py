import multiprocessing as mp
import os
import time
from multiprocessing.sharedctypes import Synchronized
from typing import *

import librosa
import numpy as np
import scipy.signal as signal
import torch
import tqdm
from pydantic import BaseModel
from scipy.io import wavfile

from latopia import f0_extractor, feature_extractor, volume_extractor
from latopia.audio_slicer import Slicer
from latopia.config.dataset import DatasetSubsetConfig
from latopia.diffusion.vocoder import Vocoder
from latopia.logger import set_logger

from .base import AudioDataset
from .diffusion import DiffusionAudioDataSubset

logger = set_logger(__name__)


class WriteWaveTask(BaseModel):
    target_sr: int
    slice: bool = True
    write_mute: bool = True
    subset_config: List[DatasetSubsetConfig]


class F0ExtractTask(BaseModel):
    f0_method: f0_extractor.F0_METHODS_TYPE
    max_workers: int = 1
    crepe_model: str = "tiny"
    sampling_rate: int = 16000
    hop_length: int = 160
    f0_max: int = 1100.0
    f0_min: int = 50.0
    f0_mel_max: Optional[int] = None
    f0_mel_min: Optional[int] = None


class FeatureExtractTask(BaseModel):
    sampling_rate: int = 16000
    hop_length: int = 160
    encoder_path: str
    encoder_output_layer: Literal[9, 12] = 12
    device: str = "cpu"
    subset_config: List[DatasetSubsetConfig]


class MelExtractTask(BaseModel):
    sampling_rate: int = 40000
    device: str = "cpu"
    vocoder_path: str
    vocoder_type: Literal["nsf-hifigan", "nsf-hifigan-log10"] = "nsf-hifigan"


class VolumeExtractTask(BaseModel):
    sampling_rate: int = 16000
    hop_length: int = 160


def load_audio(path: str, sr: Optional[int] = None):
    audio, sr = librosa.load(path, sr=sr)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    return audio, sr


def write_wave_runner(
    progress: Synchronized, files: List[Tuple[int, str, str]], task_raw: Dict
):
    task = WriteWaveTask.parse_obj(task_raw)
    slicer = Slicer(
        sr=task.target_sr,
        threshold=-42,
        min_length=1500,
        min_interval=400,
        hop_size=15,
        max_sil_kept=500,
    )

    per = 3.7
    overlap = 0.3
    tail = per + overlap
    max = 0.95
    alpha = 0.8

    bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=task.target_sr)

    for subset_idx, file, dir in files:
        subset_config = task.subset_config[subset_idx]
        waves_dir = os.path.join(dir, DiffusionAudioDataSubset.WAVES_DIR_NAME)
        waves_16k_dir = os.path.join(dir, DiffusionAudioDataSubset.WAVES_16K_DIR_NAME)
        os.makedirs(waves_dir, exist_ok=True)
        os.makedirs(waves_16k_dir, exist_ok=True)

        def write_wave(tmp_audio: np.ndarray, filename: str):
            if subset_config.normalize:
                tmp_audio = (tmp_audio / np.abs(tmp_audio).max() * (max * alpha)) + (
                    1 - alpha
                ) * tmp_audio
            else:
                # clip level to max (cause sometimes when floating point decoding)
                audio_min = np.min(tmp_audio)
                if audio_min < -max:
                    tmp_audio = tmp_audio / -audio_min * max
                audio_max = np.max(tmp_audio)
                if audio_max > max:
                    tmp_audio = tmp_audio / audio_max * max

            wavfile.write(
                os.path.join(waves_dir, f"{filename}.wav"),
                task.target_sr,
                tmp_audio.astype(np.float32),
            )

            tmp_audio = librosa.resample(
                tmp_audio, orig_sr=task.target_sr, target_sr=16000, res_type="soxr_vhq"
            )
            wavfile.write(
                os.path.join(waves_16k_dir, f"{filename}.wav"),
                16000,
                tmp_audio.astype(np.float32),
            )

        audio, sr = load_audio(file, task.target_sr)
        audio = signal.lfilter(bh, ah, audio)
        if task.slice:
            for audio in slicer.slice(audio):
                i = 0
                while 1:
                    start = int(task.target_sr * (per - overlap) * i)
                    i += 1
                    if len(audio[start:]) > tail * task.target_sr:
                        tmp_audio = audio[start : start + int(per * task.target_sr)]
                        write_wave(
                            tmp_audio,
                            f"{os.path.splitext(os.path.basename(file))[0]}_{i}",
                        )
                    else:
                        tmp_audio = audio[start:]
                        break
                write_wave(
                    tmp_audio, f"{os.path.splitext(os.path.basename(file))[0]}_{i}"
                )
        else:
            write_wave(audio, os.path.splitext(os.path.basename(file))[0])

        progress.value += 1

        mute_filepath = os.path.join(waves_dir, "__mute.wav")
        mute_16k_filepath = os.path.join(waves_16k_dir, "__mute.wav")

        if task.write_mute:
            if not os.path.exists(mute_filepath):
                wavfile.write(
                    mute_filepath,
                    task.target_sr,
                    np.zeros(task.target_sr * 3).astype(np.float32),
                )
            if not os.path.exists(mute_16k_filepath):
                wavfile.write(
                    mute_16k_filepath,
                    16000,
                    np.zeros(16000 * 3).astype(np.float32),
                )


def f0_extract_runner(
    progress: Synchronized, files: List[Tuple[str, str]], task_raw: Dict
):
    task = F0ExtractTask.parse_obj(task_raw)

    for file, dir in files:
        f0_dir = os.path.join(dir, DiffusionAudioDataSubset.F0_DIR_NAME)
        f0_nsf_dir = os.path.join(dir, DiffusionAudioDataSubset.F0_NSF_DIR_NAME)
        os.makedirs(f0_dir, exist_ok=True)
        os.makedirs(f0_nsf_dir, exist_ok=True)

        audio, sr = load_audio(file)

        n_frames = int(len(audio) // task.hop_length) + 1
        start_frame = int(0 * task.sampling_rate / task.hop_length)
        real_silence_front = start_frame * task.hop_length / task.sampling_rate
        audio = audio[int(np.round(real_silence_front * task.sampling_rate)) :]

        def pad(f0):
            if "crepe" in task.f0_method:
                f0 = np.array(
                    [
                        f0[
                            int(
                                min(
                                    int(
                                        np.round(
                                            n
                                            * task.hop_length
                                            / task.sampling_rate
                                            / 0.005
                                        )
                                    ),
                                    len(f0) - 1,
                                )
                            )
                        ]
                        for n in range(n_frames - start_frame)
                    ]
                )
                f0 = np.pad(f0, (start_frame, 0))
            else:
                f0 = np.pad(
                    f0.astype(np.float32),
                    (start_frame, n_frames - len(f0) - start_frame),
                )

            return f0

        f0 = f0_extractor.compute(
            audio,
            method=task.f0_method,
            sr=sr,
            hop=task.hop_length,
            max=task.f0_max,
            min=task.f0_min,
        )
        f0 = pad(f0)
        np.save(
            os.path.join(f0_nsf_dir, f"{os.path.basename(file)}.npy"),
            f0,
            allow_pickle=False,
        )
        coarse = f0_extractor.course(f0, 256)
        coarse = pad(coarse)
        np.save(
            os.path.join(f0_dir, f"{os.path.basename(file)}.npy"),
            coarse,
            allow_pickle=False,
        )

        progress.value += 1


def feature_extract_runner(
    progress: Synchronized, files: List[Tuple[int, str, str]], task_raw: Dict
):
    task = FeatureExtractTask.parse_obj(task_raw)
    device = torch.device(task.device)
    encoder = feature_extractor.load_encoder(task.encoder_path, device)[0]

    for subset_idx, file, dir in files:
        subset_config = task.subset_config[subset_idx]
        features_dir = os.path.join(dir, DiffusionAudioDataSubset.FEATURES_DIR_NAME)
        os.makedirs(features_dir, exist_ok=True)

        audio, sr = load_audio(file)
        audio = torch.from_numpy(audio).float().unsqueeze(0).to(device)

        features = feature_extractor.extract(
            audio,
            sr,
            device,
            encoder,
            task.encoder_output_layer,
            subset_config.normalize,
        )

        if np.isnan(features).sum() != 0:
            logger.warning(f"{file} contains nan, skip")
        else:
            np.save(
                os.path.join(
                    features_dir, f"{os.path.splitext(os.path.basename(file))[0]}.npy"
                ),
                features,
                allow_pickle=False,
            )
        progress.value += 1


def extract_mel_spectrogram_runner(
    progress: Synchronized, files: List[Tuple[str, str]], task_raw: Dict
):
    task = MelExtractTask.parse_obj(task_raw)
    device = torch.device(task.device)
    vocoder = Vocoder(task.vocoder_type, task.vocoder_path, device)
    for file, dir in files:
        mel_dir = os.path.join(dir, DiffusionAudioDataSubset.MEL_DIR_NAME)
        os.makedirs(mel_dir, exist_ok=True)
        audio, sr = load_audio(file, task.sampling_rate)
        audio = torch.from_numpy(audio).float().unsqueeze(0).to(device)
        mel = vocoder.extract(audio, task.sampling_rate)
        mel = mel.squeeze().to("cpu").numpy()
        np.save(
            os.path.join(mel_dir, f"{os.path.splitext(os.path.basename(file))[0]}.npy"),
            mel,
            allow_pickle=False,
        )
        progress.value += 1


def extract_volume_runner(
    progress: Synchronized, files: List[Tuple[str, str]], task_raw: Dict
):
    task = VolumeExtractTask.parse_obj(task_raw)
    for file, dir in files:
        volumes_dir = os.path.join(dir, DiffusionAudioDataSubset.VOLUME_DIR_NAME)
        os.makedirs(volumes_dir, exist_ok=True)
        audio, sr = load_audio(file, task.sampling_rate)
        volume = volume_extractor.extract_volume(audio, hop_size=task.hop_length)
        np.save(
            os.path.join(
                volumes_dir, f"{os.path.splitext(os.path.basename(file))[0]}.npy"
            ),
            volume,
            allow_pickle=False,
        )
        progress.value += 1


def mp_progress_bar(progress: Synchronized, total: int):
    bar = tqdm.tqdm(total=total)
    while progress.value < total:
        bar.n = progress.value
        bar.refresh()
        time.sleep(0.1)
    bar.n = progress.value
    bar.refresh()
    bar.close()


class PreProcessor:
    def __init__(self, dataset: AudioDataset) -> None:
        self.dataset = dataset

    def run_progress_proc(
        self,
        target: Callable,
        files: List[Tuple[str, str]],
        args: List[Any],
        max_workers: int = 1,
    ):
        progress = mp.Value("i", 0)
        processes: List[mp.Process] = []
        for i in range(max_workers):
            ps = mp.Process(
                target=target,
                args=(progress, files[i::max_workers], *args),
            )
            processes.append(ps)
            ps.start()

        progress_ps = mp.Process(
            target=mp_progress_bar,
            args=(
                progress,
                len(files),
            ),
        )
        progress_ps.start()

        for ps in processes:
            ps.join()

        progress_ps.kill()

    def write_wave(
        self,
        target_sr: int,
        slice: bool = True,
        write_mute: bool = True,
        max_workers: int = 1,
    ):
        files = []
        for i, subset in enumerate(self.dataset.subsets):
            for file in subset.files:
                files.append((i, file, subset.get_processed_dir()))

        self.run_progress_proc(
            write_wave_runner,
            files,
            (
                WriteWaveTask(
                    target_sr=target_sr,
                    slice=slice,
                    write_mute=write_mute,
                    subset_config=[subset.config for subset in self.dataset.subsets],
                ).dict(),
            ),
            max_workers,
        )

    def extract_f0(
        self,
        f0_method: f0_extractor.F0_METHODS_TYPE,
        max_workers: int = 1,
        crepe_model: str = "tiny",
        hop_length: int = 160,
        f0_max: int = 1100.0,
        f0_min: int = 50.0,
        f0_mel_max: Optional[int] = None,
        f0_mel_min: Optional[int] = None,
    ):
        files = []
        for subset in self.dataset.subsets:
            for file in subset.get_waves():
                files.append((file, subset.get_processed_dir()))

        if "crepe" in f0_method:
            max_workers = 1

        self.run_progress_proc(
            f0_extract_runner,
            files,
            (
                F0ExtractTask(
                    f0_method=f0_method,
                    crepe_model=crepe_model,
                    hop_length=hop_length,
                    f0_max=f0_max,
                    f0_min=f0_min,
                    f0_mel_max=f0_mel_max,
                    f0_mel_min=f0_mel_min,
                ).dict(),
            ),
            max_workers,
        )

    def extract_features(
        self,
        encoder_path: str,
        hop_length: int = 160,
        encoder_output_layer: int = 12,
        device: Literal["cpu", "cuda"] = "cpu",
    ):
        files = []
        for i, subset in enumerate(self.dataset.subsets):
            for file in subset.get_waves():
                files.append((i, file, subset.get_processed_dir()))

        self.run_progress_proc(
            feature_extract_runner,
            files,
            (
                FeatureExtractTask(
                    hop_length=hop_length,
                    encoder_path=encoder_path,
                    encoder_output_layer=encoder_output_layer,
                    device=device,
                    subset_config=[subset.config for subset in self.dataset.subsets],
                ).dict(),
            ),
            1,
        )

    def extract_volume(
        self,
        sampling_rate: int,
        hop_length: int = 160,
        max_workers: int = 1,
    ):
        files = []
        for subset in self.dataset.subsets:
            for file in subset.get_waves():
                files.append((file, subset.get_processed_dir()))

        self.run_progress_proc(
            extract_volume_runner,
            files,
            (
                VolumeExtractTask(
                    sampling_rate=sampling_rate,
                    hop_length=hop_length,
                ).dict(),
            ),
            max_workers,
        )

    def extract_mel(
        self,
        vocoder_path: str,
        sampling_rate: int,
        vocoder_type: Literal["nsf-hifigan", "nsf-hifigan-log10"] = "nsf-hifigan",
        device: str = "cpu",
        max_workers: int = 1,
    ):
        files = []
        for subset in self.dataset.subsets:
            for file in subset.get_waves():
                files.append((file, subset.get_processed_dir()))

        self.run_progress_proc(
            extract_mel_spectrogram_runner,
            files,
            (
                MelExtractTask(
                    vocoder_path=vocoder_path,
                    sampling_rate=sampling_rate,
                    vocoder_type=vocoder_type,
                    device=device,
                ).dict(),
            ),
            max_workers,
        )
