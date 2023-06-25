from typing import *

import numpy as np
import pyworld
import torch
import torchcrepe
from torchaudio.transforms import Resample

from .device import get_optimal_torch_device
from .logger import set_logger

logger = set_logger(__name__)

F0_METHODS = ["dio", "harvest", "crepe"]
F0_METHODS_TYPE = Literal["dio", "harvest", "crepe"]

CREPE_SAMPLING_RATE = 16000

CREPE_RESAMPLE_KERNEL = {}


def compute(
    audio: np.ndarray,
    method: F0_METHODS_TYPE = "crepe",
    sr: int = 40000,
    hop: int = 160,
    max: int = 1100.0,
    min: int = 50.0,
    crepe_model: str = "tiny",
):
    resample_kernel = lambda x: x
    if "crepe" in method and sr != CREPE_SAMPLING_RATE:
        key_str = str(sr)
        if key_str not in CREPE_RESAMPLE_KERNEL:
            CREPE_RESAMPLE_KERNEL[key_str] = Resample(
                sr, CREPE_SAMPLING_RATE, lowpass_filter_width=128
            )
        resample_kernel = CREPE_RESAMPLE_KERNEL[key_str].to(get_optimal_torch_device())
    if method == "dio":
        f0, t = pyworld.harvest(
            audio.astype(np.double),
            fs=sr,
            f0_ceil=max,
            f0_floor=min,
            frame_period=1000 * hop / sr,
        )
        f0 = pyworld.stonemask(audio.astype(np.double), f0, t, sr)
    elif method == "harvest":
        f0, t = pyworld.harvest(
            audio.astype(np.double),
            fs=sr,
            f0_ceil=max,
            f0_floor=min,
            frame_period=1000 * hop / sr,
        )
    elif method == "crepe":
        batch_size = 512
        torch_device = get_optimal_torch_device()
        audio = resample_kernel(torch.FloatTensor(audio).unsqueeze(0).to(torch_device))
        f0, pd = torchcrepe.predict(
            audio,
            CREPE_SAMPLING_RATE,
            hop,
            min,
            max,
            crepe_model,
            batch_size=batch_size,
            device=torch_device,
            pad=True,
            return_periodicity=True,
        )
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        f0 = f0[0].cpu().numpy()
        f0 = f0[1:]  # Get rid of extra first frame

    return f0


def course(
    f0: np.ndarray,
    bin: int,
    max: int = 1100.0,
    min: int = 50.0,
    mel_max: Optional[int] = None,
    mel_min: Optional[int] = None,
):
    mel_max = mel_max or 1127 * np.log(1 + max / 700)
    mel_min = mel_min or 1127 * np.log(1 + min / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - mel_min) * (bin - 2) / (
        mel_max - mel_min
    ) + 1

    f0_mel[f0_mel <= 1] = 1
    f0_mel[f0_mel > bin - 1] = bin - 1
    f0_coarse = np.rint(f0_mel).astype(np.int64)
    # assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
    #     f0_coarse.max(),
    #     f0_coarse.min(),
    # )
    return f0_coarse
