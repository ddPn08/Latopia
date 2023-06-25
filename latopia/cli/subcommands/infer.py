from typing import *

import librosa
import soundfile as sf
import torch

from latopia.f0_extractor import F0_METHODS_TYPE
from latopia.pipelines.diffusion import DiffusionSvcPipeline
from latopia.pipelines.vits import ViTsPipeline
from latopia.utils import get_torch_dtype, load_audio


def infer_vits(
    audio_input: str,
    output_path: str,
    pretrained_model_path: str,
    encoder_model_path: str = "./models/encoders/checkpoint_best_legacy_500.pt",
    f0_method: F0_METHODS_TYPE = "crepe",
    transpose: int = 0,
    device: str = "cpu",
    torch_dtype: str = "float32",
):
    device = torch.device(device)
    torch_dtype = get_torch_dtype(torch_dtype)
    pipeline = ViTsPipeline(
        pretrained_model_path,
        encoder_model_path,
        device=device,
        torch_dtype=torch_dtype,
    )

    audio = load_audio(audio_input, 16000)
    audio_opt = pipeline(audio, f0_method=f0_method, transpose=transpose)

    sf.write(output_path, audio_opt, pipeline.model.sampling_rate)


def infer_diff(
    audio_input: str,
    output_path: str,
    pretrained_model_path: str,
    vocoder_model_dir: str = "./models/vocoders/nsf_hifigan",
    encoder_model_path: str = "./models/encoders/checkpoint_best_legacy_500.pt",
    vocoder_type: Literal["nsf-hifigan", "nsf-hifigan-log10"] = "nsf-hifigan",
    f0_method: F0_METHODS_TYPE = "crepe",
    speaker_id: int = 0,
    k_step: int = 100,
    speedup: int = 10,
    sampling_method: Literal["dpm-solver", "unipc"] = "dpm-solver",
    transpose: int = 0,
    threshold: int = -60,
    threshold_for_split: int = -40,
    min_len: int = 5000,
    f0_max: int = 1100.0,
    f0_min: int = 50.0,
    device: str = "cpu",
    torch_dtype: str = "float32",
):
    device = torch.device(device)
    torch_dtype = get_torch_dtype(torch_dtype)
    pipeline = DiffusionSvcPipeline(
        pretrained_model_path,
        vocoder_model_dir,
        encoder_model_path,
        vocoder_type=vocoder_type,
        device=device,
        torch_dtype=torch_dtype,
    )

    audio, sr = librosa.load(audio_input, sr=None)
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    audio_opt = pipeline(
        audio,
        sr,
        f0_method=f0_method,
        speaker_id=speaker_id,
        k_step=k_step,
        speedup=speedup,
        sampling_method=sampling_method,
        transpose=transpose,
        threshold=threshold,
        threshold_for_split=threshold_for_split,
        min_len=min_len,
        f0_max=f0_max,
        f0_min=f0_min,
    )

    sf.write(output_path, audio_opt, pipeline.model.sampling_rate)


def run():
    return {"vits": infer_vits, "diff": infer_diff}
