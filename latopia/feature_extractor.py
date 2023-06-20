from typing import *

import numpy as np
import torch
import torch.nn.functional as F
from fairseq import checkpoint_utils
from fairseq.models.hubert import HubertModel

from latopia.utils import clear_vram


def load_encoder(filepath: str, device) -> Tuple[HubertModel, Dict]:
    models, cfg, _ = checkpoint_utils.load_model_ensemble_and_task(
        [filepath],
        suffix="",
    )
    encoder_model = models[0].eval().to(device)

    return encoder_model, cfg


def extract(
    audio: np.ndarray,
    device: torch.device,
    encoder: HubertModel,
    encoder_channels: int,
    encoder_output_layer: int,
    normalize: bool = True,
    return_type: Literal["numpy", "torch"] = "numpy",
    dtype: torch.dtype = torch.float32,
):
    feats = torch.from_numpy(audio).to(device, dtype)
    if feats.dim() == 2:
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
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
        if encoder_channels == 768:
            feats = logits[0]
        else:
            feats = encoder.final_proj(logits[0])

    del padding_mask
    clear_vram()

    if return_type == "numpy":
        return feats.squeeze(0).float().cpu().numpy()
    elif return_type == "torch":
        return feats
