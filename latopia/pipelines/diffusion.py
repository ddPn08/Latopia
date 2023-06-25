from typing import *

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.transforms import Resample

from latopia import f0_extractor, feature_extractor, volume_extractor
from latopia.audio_slicer import Slicer
from latopia.diffusion.slicer import split
from latopia.diffusion.unit2mel import load_model
from latopia.diffusion.vocoder import Vocoder


def cross_fade(a: np.ndarray, b: np.ndarray, idx: int):
    result = np.zeros(idx + b.shape[0])
    fade_len = a.shape[0] - idx
    np.copyto(dst=result[:idx], src=a[:idx])
    k = np.linspace(0, 1.0, num=fade_len, endpoint=True)
    result[idx : a.shape[0]] = (1 - k) * a[idx:] + k * b[:fade_len]
    np.copyto(dst=result[a.shape[0] :], src=b[fade_len:])
    return result


class DiffusionSvcPipeline:
    def __init__(
        self,
        pretrained_model_path: str,
        vocoder_model_dir: str,
        encoder_model_path: str,
        vocoder_type: Literal["nsf-hifigan", "nsf-hifigan-log10"] = "nsf-hifigan",
        device: Union[str, torch.device] = torch.device("cpu"),
        torch_dtype: torch.dtype = torch.float32,
    ):
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.dtype = torch_dtype
        self.vocoder = Vocoder(vocoder_type, vocoder_model_dir, device=self.device)
        self.vocoder.vocoder = self.vocoder.vocoder.to(self.dtype)
        self.model = load_model(pretrained_model_path, self.vocoder).to(
            self.device, self.dtype
        )
        self.encoder = feature_extractor.load_encoder(encoder_model_path, self.device)[
            0
        ].to(self.dtype)

        self.resample_kernels = {}

    @torch.no_grad()
    def mel2wav(self, mel, f0, start_frame=0):
        if start_frame == 0:
            return self.vocoder.infer(mel, f0)
        else:  # for realtime speedup
            mel = mel[:, start_frame:, :]
            f0 = f0[:, start_frame:, :]
            out_wav = self.vocoder.infer(mel, f0)
            return F.pad(out_wav, (start_frame * self.vocoder.vocoder_hop_size, 0))

    @torch.no_grad()
    def __call__(
        self,
        audio: np.ndarray,
        sampling_rate: int,
        f0_method: f0_extractor.F0_METHODS_TYPE = "crepe",
        speaker_id: int = 0,
        k_step: int = 100,
        speedup: int = 10,
        sampling_method: Literal["dpm-solver", "unipc"] = "dpm-solver",
        transpose: int = 0,
        threshold: int = -60,
        threshold_for_split=-40,
        min_len=5000,
        f0_max: int = 1100.0,
        f0_min: int = 50.0,
    ):
        hop_size = self.model.hop_length * sampling_rate / self.model.sampling_rate
        segments = split(
            audio,
            sampling_rate,
            hop_size,
            db_thresh=threshold_for_split,
            min_len=min_len,
        )

        speaker_id = torch.LongTensor(np.array([[int(speaker_id)]])).to(self.device)

        n_frames = int(len(audio) // hop_size) + 1
        start_frame = int(0 * sampling_rate / hop_size)
        real_silence_front = start_frame * hop_size / sampling_rate

        audio = audio[int(np.round(real_silence_front * sampling_rate)) :]

        def pad(f0):
            if "crepe" in f0_method:
                f0 = np.array(
                    [
                        f0[
                            int(
                                min(
                                    int(np.round(n * hop_size / sampling_rate / 0.005)),
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
                    f0.astype("float"),
                    (start_frame, n_frames - len(f0) - start_frame),
                )

            return f0

        f0 = f0_extractor.compute(
            audio,
            f0_method,
            sampling_rate,
            80 if "crepe" in f0_method else hop_size,
            max=f0_max,
            min=f0_min,
        )
        f0 = pad(f0)
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
        f0[f0 < f0_min] = f0_min
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        f0 = f0 * 2 ** (float(transpose) / 12)

        volume = volume_extractor.extract_volume(
            audio,
            sampling_rate,
            self.model.hop_length,
            block_size=self.model.hop_length,
            model_sampling_rate=self.model.sampling_rate,
        )
        mask = volume_extractor.get_mask_from_volume(
            volume,
            float(threshold),
            block_size=self.model.hop_length,
            device=self.device,
        )
        volume = (
            torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        )

        if k_step is not None:
            assert 0 < int(k_step) <= 1000
            k_step = int(k_step)
            audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            gt_spec = self.vocoder.extract(audio_t, self.model.sampling_rate)
            gt_spec = torch.cat((gt_spec, gt_spec[:, -1:, :]), 1)
        else:
            gt_spec = None

        result = np.zeros(0)
        current_length = 0

        for segment in segments:
            start_frame = segment[0]
            seg_input = (
                torch.from_numpy(segment[1]).float().unsqueeze(0).to(self.device)
            )
            # seg_units = self.units_encoder.encode(seg_input, sr, hop_size)
            seg_units = feature_extractor.extract(
                seg_input,
                sampling_rate,
                self.device,
                self.encoder,
                12,
                return_type="torch",
            )

            n_frames = int(seg_input.size(-1) // hop_size + 1)
            seg_units = seg_units.transpose(1, 2)
            seg_units = F.interpolate(
                seg_units, size=int(n_frames), mode="nearest"
            ).transpose(1, 2)

            seg_f0 = f0[:, start_frame : start_frame + seg_units.size(1), :]
            seg_volume = volume[:, start_frame : start_frame + seg_units.size(1), :]
            if gt_spec is not None:
                seg_gt_spec = gt_spec[
                    :, start_frame : start_frame + seg_units.size(1), :
                ]
            else:
                seg_gt_spec = None

            out_mel = self.model(
                seg_units,
                seg_f0,
                seg_volume,
                spk_id=speaker_id,
                gt_spec=seg_gt_spec,
                infer=True,
                infer_speedup=speedup,
                method=sampling_method,
                k_step=k_step,
            )
            seg_output = self.mel2wav(out_mel, seg_f0)
            _left = start_frame * self.model.hop_length
            _right = (start_frame + seg_units.size(1)) * self.model.hop_length
            seg_output *= mask[:, _left:_right]
            seg_output = seg_output.squeeze().cpu().numpy()
            silent_length = round(start_frame * self.model.hop_length) - current_length
            if silent_length >= 0:
                result = np.append(result, np.zeros(silent_length))
                result = np.append(result, seg_output)
            else:
                result = cross_fade(result, seg_output, current_length + silent_length)
            current_length = current_length + silent_length + len(seg_output)

        return result
