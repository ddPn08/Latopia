import json
import os
from typing import *

import numpy as np
import safetensors.torch
import torch
import torch.nn as nn

from latopia.config.diffusion import DiffusionModelConfig
from latopia.utils import read_safetensors_metadata

from .diffusion import GaussianDiffusion
from .vocoder import Vocoder
from .wavnet import WaveNet


class Unit2Mel(nn.Module):
    def __init__(
        self,
        input_channel: int,
        n_spk: int,
        use_pitch_aug: bool = False,
        out_dims: int = 128,
        n_layers: int = 20,
        n_chans: int = 384,
        n_hidden: int = 256,
        hop_length: int = 512,
        sampling_rate: int = 44100,
    ):
        super().__init__()
        self.unit_embed = nn.Linear(input_channel, n_hidden)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, n_hidden)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None

        self.n_spk = n_spk

        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)

        self.hop_length = hop_length
        self.sampling_rate = sampling_rate

        # diffusion
        self.decoder = GaussianDiffusion(
            WaveNet(out_dims, n_layers, n_chans, n_hidden), out_dims=out_dims
        )

    def forward(
        self,
        units,
        f0,
        volume,
        spk_id=None,
        spk_mix_dict=None,
        aug_shift=None,
        gt_spec=None,
        infer=True,
        infer_speedup=10,
        method="dpm-solver",
        k_step=None,
        use_tqdm=True,
    ):
        """
        input:
            B x n_frames x n_unit
        return:
            dict of B x n_frames x feat
        """

        x = (
            self.unit_embed(units)
            + self.f0_embed((1 + f0 / 700).log())
            + self.volume_embed(volume)
        )
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    x = x + v * self.spk_embed(spk_id_torch - 1)
            else:
                x = x + self.spk_embed(spk_id)
        if self.aug_shift_embed is not None and aug_shift is not None:
            x = x + self.aug_shift_embed(aug_shift / 5)

        x = self.decoder(
            x,
            gt_spec=gt_spec,
            infer=infer,
            infer_speedup=infer_speedup,
            method=method,
            k_step=k_step,
            use_tqdm=use_tqdm,
        )

        return x


def save_model(
    filepath: str,
    model: Unit2Mel,
    config: DiffusionModelConfig,
    metadata: Dict[str, Any],
    save_as: Literal["pt", "safetensors"] = "safetensors",
):
    metadata = {
        **metadata,
        "_config": json.dumps(
            {
                "config": config.json(),
                "hop_length": model.hop_length,
                "sampling_rate": model.sampling_rate,
            }
        ),
    }
    filepath = filepath + f".{save_as}"
    if save_as == "safetensors":
        safetensors.torch.save_file(
            model.state_dict(),
            filepath,
            metadata=metadata,
        )
    elif save_as == "pt":
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metadata": metadata,
            },
            filepath,
        )


def load_model(filepath: str, vocoder: Vocoder):
    if os.path.splitext(filepath)[1] == ".safetensors":
        state_dict = safetensors.torch.load_file(filepath)
        metadata = read_safetensors_metadata(filepath)
    else:
        data = torch.load(filepath)
        metadata = data["metadata"]
        state_dict = data["state_dict"]

    synthesizer_config = json.loads(metadata["_config"])
    config = DiffusionModelConfig.parse_json(synthesizer_config["config"])

    model = Unit2Mel(
        config.emb_channels,
        config.spk_embed_dim,
        config.use_pitch_aug,
        vocoder.dimension,
        config.n_layers,
        config.n_chans,
        config.n_hidden,
        synthesizer_config["hop_length"],
        synthesizer_config["sampling_rate"],
    )
    model.load_state_dict(state_dict)
    return model
