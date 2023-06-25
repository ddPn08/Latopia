import numpy as np
import torch
import torch.nn.functional as F


def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = F.interpolate(
        torch.cat((signal, signal[:, :, -1:]), 2),
        size=signal.shape[-1] * factor + 1,
        mode="linear",
        align_corners=True,
    )
    signal = signal[:, :, :-1]
    return signal.permute(0, 2, 1)


class VolumeExtractor:
    def __init__(self, hop_size=512, block_size=None, model_sampling_rate=None):
        self.block_size = block_size
        self.model_sampling_rate = model_sampling_rate
        self.hop_size = hop_size
        if (self.block_size is not None) or (self.model_sampling_rate is not None):
            assert (self.block_size is not None) and (
                self.model_sampling_rate is not None
            )
            self.hop_size_follow_input = True
        else:
            self.hop_size_follow_input = False

    def extract(self, audio, sr=None):  # audio: 1d numpy array
        if sr is not None:
            assert self.hop_size_follow_input
            self.hop_size = self.block_size * sr / self.model_sampling_rate
        n_frames = int(len(audio) // self.hop_size) + 1
        audio2 = audio**2
        audio2 = np.pad(
            audio2,
            (int(self.hop_size // 2), int((self.hop_size + 1) // 2)),
            mode="reflect",
        )
        volume = np.array(
            [
                np.mean(audio2[int(n * self.hop_size) : int((n + 1) * self.hop_size)])
                for n in range(n_frames)
            ]
        )
        volume = np.sqrt(volume)
        """
        if isinstance(audio, torch.Tensor):
            n_frames = int(audio.size(-1) // self.hop_size) + 1
            audio2 = audio ** 2
            audio2 = torch.nn.functional.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)),
                                             mode='reflect')
            audio_frame = torch.nn.functional.unfold(audio2[:, None, None, :], (1, int(self.hop_size)),
                                                     stride=int(self.hop_size))[:, :, :n_frames]
            volume = audio_frame.mean(dim=1)[0]
            volume = torch.sqrt(volume).squeeze().cpu().numpy()
        else:
            n_frames = int(len(audio) // self.hop_size) + 1
            audio2 = audio ** 2
            audio2 = np.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode='reflect')
            volume = np.array(
                [np.mean(audio2[int(n * self.hop_size): int((n + 1) * self.hop_size)]) for n in range(n_frames)])
            volume = np.sqrt(volume)
        """
        return volume

    def get_mask_from_volume(self, volume, threhold=-60.0, device="cpu"):
        mask = (volume > 10 ** (float(threhold) / 20)).astype("float")
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.block_size).squeeze(-1)
        return mask


def extract_volume(
    audio: np.ndarray,
    sr: int = None,
    hop_size: int = 160,
    block_size=None,
    model_sampling_rate=None,
):
    if (block_size is not None) or (model_sampling_rate is not None):
        assert (block_size is not None) and (model_sampling_rate is not None)
        hop_size_follow_input = True
    else:
        hop_size_follow_input = False
    # audio: 1d numpy array
    if sr is not None:
        assert hop_size_follow_input
        hop_size = block_size * sr / model_sampling_rate
    n_frames = int(len(audio) // hop_size) + 1
    audio = audio**2
    audio = np.pad(
        audio,
        (int(hop_size // 2), int((hop_size + 1) // 2)),
        mode="reflect",
    )
    volume = np.array(
        [
            np.mean(audio[int(n * hop_size) : int((n + 1) * hop_size)])
            for n in range(n_frames)
        ]
    )
    volume = np.sqrt(volume)
    return volume


def get_mask_from_volume(volume, threhold=-60.0, block_size=512, device="cpu"):
    mask = (volume > 10 ** (float(threhold) / 20)).astype("float")
    mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
    mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
    mask = torch.from_numpy(mask).float().to(device).unsqueeze(-1).unsqueeze(0)
    mask = upsample(mask, block_size).squeeze(-1)
    return mask
