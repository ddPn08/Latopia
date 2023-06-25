from typing import *

import numpy as np
import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils
from fairseq.models.hubert import HubertModel
from torchaudio.transforms import Resample

from latopia.utils import clear_vram


def load_encoder(filepath: str, device) -> Tuple[HubertModel, Dict]:
    models, cfg, _ = checkpoint_utils.load_model_ensemble_and_task(
        [filepath],
        suffix="",
    )
    encoder_model = models[0].eval().to(device)

    return encoder_model, cfg


RESAMPLING_KERNEL = {}
SAMPLING_RATE = 16000


def extract(
    feats: Union[np.ndarray, torch.Tensor],
    sampling_rate: int,
    device: torch.device,
    encoder: HubertModel,
    encoder_output_layer: int,
    normalize: bool = True,
    return_type: Literal["numpy", "torch"] = "numpy",
    dtype: torch.dtype = torch.float32,
):
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats).to(device, dtype)

    if sampling_rate != SAMPLING_RATE:
        key_str = str(sampling_rate)
        if key_str not in RESAMPLING_KERNEL:
            RESAMPLING_KERNEL[key_str] = Resample(
                sampling_rate, SAMPLING_RATE, lowpass_filter_width=128
            ).to(device)
        feats = RESAMPLING_KERNEL[key_str].to(device, dtype)(feats)

    if feats.size(-1) < 400:
        feats = F.pad(feats, (0, 400 - feats.size(-1)))

    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    padding_mask = torch.BoolTensor(feats.shape).to(device).fill_(False)
    inputs = {
        "source": feats,
        "padding_mask": padding_mask,
        "output_layer": encoder_output_layer,
    }

    with torch.no_grad():
        logits = encoder.extract_features(**inputs)
        feats = logits[0]

    del padding_mask
    clear_vram()

    if return_type == "numpy":
        return feats.cpu().numpy()
    elif return_type == "torch":
        return feats
