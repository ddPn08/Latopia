import torch
from librosa.filters import mel as librosa_mel_fn

MAX_WAV_VALUE = 32768.0


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    return dynamic_range_compression_torch(magnitudes)


def spectral_de_normalize_torch(magnitudes):
    return dynamic_range_decompression_torch(magnitudes)


hann_window = {}


def spectrogram_torch(
    audio: torch.Tensor,
    filter_length: int,
    hop_size: int,
    win_size: int,
    center: bool = False,
):
    global hann_window
    if torch.min(audio) < -1.07:
        print("min value is ", torch.min(audio))
    if torch.max(audio) > 1.07:
        print("max value is ", torch.max(audio))

    dtype_device = str(audio.dtype) + "_" + str(audio.device)
    wnsize_dtype_device = str(win_size) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(
            dtype=audio.dtype, device=audio.device
        )

    audio = torch.nn.functional.pad(
        audio.unsqueeze(1),
        (int((filter_length - hop_size) / 2), int((filter_length - hop_size) / 2)),
        mode="reflect",
    )
    audio = audio.squeeze(1)

    # mps does not support torch.stft.
    if audio.device.type == "mps":
        i = audio.cpu()
        win = hann_window[wnsize_dtype_device].cpu()
    else:
        i = audio
        win = hann_window[wnsize_dtype_device]
    spec = torch.stft(
        i,
        filter_length,
        hop_length=hop_size,
        win_length=win_size,
        window=win,
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
        return_complex=False,
    ).to(device=audio.device)

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


mel_basis = {}


def spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax):
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(
            sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax
        )
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(
            dtype=spec.dtype, device=spec.device
        )
    melspec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    melspec = spectral_normalize_torch(melspec)
    return melspec


def mel_spectrogram_torch(
    y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False
):
    """Convert waveform into Mel-frequency Log-amplitude spectrogram.

    Args:
        y       :: (B, T)           - Waveforms
    Returns:
        melspec :: (B, Freq, Frame) - Mel-frequency Log-amplitude spectrogram
    """
    # Linear-frequency Linear-amplitude spectrogram :: (B, T) -> (B, Freq, Frame)
    spec = spectrogram_torch(y, n_fft, hop_size, win_size, center)

    # Mel-frequency Log-amplitude spectrogram :: (B, Freq, Frame) -> (B, Freq=num_mels, Frame)
    melspec = spec_to_mel_torch(spec, n_fft, num_mels, sampling_rate, fmin, fmax)

    return melspec
