from typing import *

import numpy as np
import scipy.signal as signal
import torch
import torch.nn.functional as F

from latopia import f0_extractor, feature_extractor
from latopia.utils import clear_vram
from latopia.vits.models import load_generator


class ViTsPipeline:
    def __init__(
        self,
        pretrained_model_path: str,
        encoder_model_path: str,
        target_sampling_rate: int = 40000,
        device: torch.device = torch.device("cpu"),
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.device = device
        self.dtype = torch_dtype

        if device.type == "cuda":
            vram = torch.cuda.get_device_properties(device).total_memory / 1024**3
        else:
            vram = None

        if vram is not None and vram <= 4:
            self.x_pad = 1
            self.x_query = 5
            self.x_center = 30
            self.x_max = 32
        elif vram is not None and vram <= 5:
            self.x_pad = 1
            self.x_query = 6
            self.x_center = 38
            self.x_max = 41
        else:
            self.x_pad = 3
            self.x_query = 10
            self.x_center = 60
            self.x_max = 65

        self.sr = 16000  # hubert input sample rate
        self.window = 160  # hubert input window
        self.t_pad = self.sr * self.x_pad  # padding time for each utterance
        self.t_pad_tgt = target_sampling_rate * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query  # query time before and after query point
        self.t_center = self.sr * self.x_center  # query cut point position
        self.t_max = self.sr * self.x_max  # max time for no query

        self.encoder, _ = feature_extractor.load_encoder(encoder_model_path, device)
        self.encoder = self.encoder.to(device, torch_dtype)

        self.model = load_generator(pretrained_model_path, mode="infer")
        self.model.eval().to(device, torch_dtype)

    def _convert(
        self,
        speaker_id: torch.LongTensor,
        audio: np.ndarray,
        f0: torch.FloatTensor,
        f0_nsf: torch.LongTensor,
    ):
        feats = feature_extractor.extract(
            audio,
            self.device,
            self.encoder,
            768,
            12,
            normalize=False,
            return_type="torch",
            dtype=self.dtype,
        )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)

        p_len = audio.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            f0 = f0[:, :p_len]
            f0_nsf = f0_nsf[:, :p_len]
        p_len = torch.tensor([p_len], device=self.device).long()
        result = (
            (self.model.infer(feats, p_len, f0, f0_nsf, speaker_id)[0][0, 0] * 32768)
            .data.cpu()
            .float()
            .numpy()
            .astype(np.int16)
        )
        del feats, p_len
        clear_vram()
        return result

    @torch.inference_mode()
    def __call__(
        self,
        audio: np.ndarray,
        f0_method: f0_extractor.F0_METHODS_TYPE,
        transpose: int = 0,
        speaker_id: int = 0,
    ):
        bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)
        audio = signal.filtfilt(bh, ah, audio)

        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")

        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )

        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window

        speaker_id = torch.tensor(speaker_id, device=self.device).unsqueeze(0).long()

        f0_nsf = f0_extractor.compute(
            audio_pad,
            f0_method,
            self.sr,
        )
        f0_nsf *= pow(2, transpose / 12)
        f0 = f0_extractor.course(f0_nsf, 256)

        f0 = f0[:p_len]
        f0_nsf = f0_nsf[:p_len]
        if self.device.type == "mps":
            f0_nsf = f0_nsf.astype(np.float32)

        f0 = torch.tensor(f0, device=self.device).unsqueeze(0).long()
        f0_nsf = torch.tensor(f0_nsf, device=self.device).unsqueeze(0).float()

        audio_opt = []
        s = 0
        t = None

        for t in opt_ts:
            t = t // self.window * self.window
            audio_opt.append(
                self._convert(
                    speaker_id,
                    audio_pad[s : t + self.t_pad2 + self.window],
                    f0[:, s // self.window : (t + self.t_pad2) // self.window],
                    f0_nsf[:, s // self.window : (t + self.t_pad2) // self.window],
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
            s = t

        audio_opt.append(
            self._convert(
                speaker_id,
                audio_pad[t:],
                f0[:, t // self.window :] if t is not None else f0,
                f0_nsf[:, t // self.window :] if t is not None else f0_nsf,
            )[self.t_pad_tgt : -self.t_pad_tgt]
        )

        audio_opt = np.concatenate(audio_opt)

        del f0, f0_nsf, speaker_id
        clear_vram()

        return audio_opt
