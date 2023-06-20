from typing import *

import numpy as np
import pyworld
import torch
import torchcrepe

from .device import get_optimal_torch_device
from .logger import set_logger

logger = set_logger(__name__)

F0_METHODS = ["dio", "harvest", "mangio-crepe", "crepe"]
F0_METHODS_TYPE = Literal["dio", "harvest", "mangio-crepe", "crepe"]


def compute(
    audio: np.ndarray,
    method: F0_METHODS_TYPE = "crepe",
    sr: int = 16000,
    hop: int = 160,
    max: int = 1100.0,
    min: int = 50.0,
    crepe_model: str = "tiny",
):
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
        f0 = pyworld.stonemask(audio.astype(np.double), f0, t, sr)
    elif method == "mangio-crepe":
        audio = audio.astype(
            np.float32
        )  # fixes the F.conv2D exception. We needed to convert double to float.
        audio /= np.quantile(np.abs(audio), 0.999)
        torch_device = get_optimal_torch_device()
        audio = torch.from_numpy(audio).to(torch_device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()
        logger.info(f"Initiating prediction with a crepe_hop_length of: {hop}")
        pitch: torch.Tensor = torchcrepe.predict(
            audio,
            sr,
            hop,
            min,
            max,
            crepe_model,
            batch_size=hop * 2,
            device=torch_device,
            pad=True,
        )
        p_len = audio.shape[0] // hop
        # Resize the pitch for final f0
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )
        f0 = np.nan_to_num(target)
        f0 = f0[1:]  # Get rid of extra first frame
    elif method == "crepe":
        batch_size = 512
        torch_device = get_optimal_torch_device()
        audio = torch.tensor(np.copy(audio))[None].float()
        f0, pd = torchcrepe.predict(
            audio,
            sr,
            hop,
            min,
            max,
            crepe_model,
            batch_size=batch_size,
            device=torch_device,
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
    f0_coarse = np.rint(f0_mel).astype(np.int32)
    assert f0_coarse.max() <= 255 and f0_coarse.min() >= 1, (
        f0_coarse.max(),
        f0_coarse.min(),
    )
    return f0_coarse
