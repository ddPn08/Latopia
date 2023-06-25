import json
import math
import os
from typing import *

import safetensors.torch
import torch
from torch import nn
from torch.nn import Conv1d, Conv2d
from torch.nn import functional as F
from torch.nn.utils import spectral_norm, weight_norm

from latopia import torch_utils
from latopia.config.vits import ViTsDiscriminatorConfig, ViTsGeneratorConfig
from latopia.utils import read_safetensors_metadata

from . import attentions, modules
from .vocoder import Vocoder


class TextEncoder(nn.Module):
    def __init__(
        self,
        out_channels: int,
        hidden_channels: int,
        filter_channels: int,
        emb_channels: int,
        n_heads: int,
        n_layers: int,
        kernel_size: int,
        p_dropout: int,
        f0: bool = True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.emb_channels = emb_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.emb_phone = nn.Linear(emb_channels, hidden_channels)
        self.lrelu = nn.LeakyReLU(0.1, inplace=True)
        if f0 == True:
            self.emb_pitch = nn.Embedding(256, hidden_channels)  # pitch 256
        self.encoder = attentions.Encoder(
            hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, phone, pitch, lengths):
        if pitch == None:
            x = self.emb_phone(phone)
        else:
            x = self.emb_phone(phone) + self.emb_pitch(pitch)
        x = x * math.sqrt(self.hidden_channels)  # [b, t, h]
        x = self.lrelu(x)
        x = torch.transpose(x, 1, -1)  # [b, h, t]
        x_mask = torch.unsqueeze(torch_utils.sequence_mask(lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.encoder(x * x_mask, x_mask)
        stats = self.proj(x) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return m, logs, x_mask


class ResidualCouplingBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        n_flows: int = 4,
        gin_channels: int = 0,
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()


class PosteriorEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        kernel_size: int,
        dilation_rate: int,
        n_layers: int,
        gin_channels=0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            gin_channels=gin_channels,
        )
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(torch_utils.sequence_mask(x_lengths, x.size(2)), 1).to(
            x.dtype
        )
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class ViTsSynthesizer(nn.Module):
    def __init__(
        self,
        config: ViTsGeneratorConfig,
        mel_channels: int,
        segment_size: int,
        sampling_rate: int,
    ):
        super().__init__()
        self.mel_channels = mel_channels
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        self.config = config
        self.enc_p = TextEncoder(
            config.inter_channels,
            config.hidden_channels,
            config.filter_channels,
            config.emb_channels,
            config.n_heads,
            config.n_layers,
            config.kernel_size,
            config.p_dropout,
        )
        self.dec = Vocoder(
            config.inter_channels,
            config.resblock,
            config.resblock_kernel_sizes,
            config.resblock_dilation_sizes,
            config.upsample_rates,
            config.upsample_initial_channel,
            config.upsample_kernel_sizes,
            sampling_rate=sampling_rate,
            gin_channels=config.gin_channels,
        )
        self.enc_q = PosteriorEncoder(
            mel_channels,
            config.inter_channels,
            config.hidden_channels,
            5,
            1,
            16,
            gin_channels=config.gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            config.inter_channels,
            config.hidden_channels,
            5,
            1,
            3,
            gin_channels=config.gin_channels,
        )
        self.emb_g = nn.Embedding(config.spk_embed_dim, config.gin_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def forward(self, features, features_lengths, f0, f0_nsf, y, y_lengths, ds):
        m_p, logs_p, x_mask = self.enc_p(features, f0, features_lengths)

        g = self.emb_g(ds).unsqueeze(-1)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
        z_p = self.flow(z, y_mask, g=g)
        z_slice, ids_slice = torch_utils.rand_slice_segments(
            z, y_lengths, self.segment_size
        )
        f0_nsf = torch_utils.slice_segments_single_dim(
            f0_nsf, ids_slice, self.segment_size
        )
        o = self.dec(z_slice, f0_nsf, g=g)
        result = {
            "o": o,
            "ids_slice": ids_slice,
            "x_mask": x_mask,
            "y_mask": y_mask,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "m_q": m_q,
            "logs_q": logs_q,
        }
        # return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
        return result

    def infer(self, features, features_lengths, f0, f0_nsf, speaker_id, max_len=None):
        g = self.emb_g(speaker_id).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(features, f0, features_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        o = self.dec((z * x_mask)[:, :, :max_len], f0_nsf, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(
                    Conv2d(
                        1,
                        32,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(torch_utils.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        32,
                        128,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(torch_utils.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        128,
                        512,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(torch_utils.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        512,
                        1024,
                        (kernel_size, 1),
                        (stride, 1),
                        padding=(torch_utils.get_padding(kernel_size, 1), 0),
                    )
                ),
                norm_f(
                    Conv2d(
                        1024,
                        1024,
                        (kernel_size, 1),
                        1,
                        padding=(torch_utils.get_padding(kernel_size, 1), 0),
                    )
                ),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, config: ViTsDiscriminatorConfig):
        super(MultiPeriodDiscriminator, self).__init__()
        self.config = config

        discs = [DiscriminatorS(use_spectral_norm=config.use_spectral_norm)]
        discs = discs + [
            DiscriminatorP(i, use_spectral_norm=config.use_spectral_norm)
            for i in config.periods
        ]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []  #
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        result = {
            "y_d_rs": y_d_rs,
            "y_d_gs": y_d_gs,
            "fmap_rs": fmap_rs,
            "fmap_gs": fmap_gs,
        }

        # return y_d_rs, y_d_gs, fmap_rs, fmap_gs
        return result


def save_generator(
    dir: str,
    filename: str,
    model: ViTsSynthesizer,
    config: ViTsGeneratorConfig,
    metadata: Dict[str, Any],
    save_as: Literal["pt", "safetensors"] = "safetensors",
):
    filepath = os.path.join(dir, f"{filename}.g.{save_as}")
    metadata = {
        **metadata,
        "_config": json.dumps(
            {
                "config": config.json(),
                "mel_channels": model.mel_channels,
                "segment_size": model.segment_size,
                "sampling_rate": model.sampling_rate,
            }
        ),
    }
    if save_as == "safetensors":
        safetensors.torch.save_file(
            model.state_dict(),
            filepath,
            metadata=metadata,
        )
    else:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metadata": metadata,
            },
            filepath,
        )


def save_discriminator(
    dir: str,
    filename: str,
    model: MultiPeriodDiscriminator,
    config: ViTsDiscriminatorConfig,
    metadata: Dict[str, Any],
    save_as: Literal["pt", "safetensors"] = "safetensors",
):
    filepath = os.path.join(dir, f"{filename}.d.{save_as}")
    metadata = {
        **metadata,
        "_config": json.dumps(
            {
                "config": config.json(),
            }
        ),
    }
    if save_as == "safetensors":
        safetensors.torch.save_file(
            model.state_dict(),
            filepath,
            metadata=metadata,
        )
    else:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "metadata": metadata,
            },
            filepath,
        )


def load_generator(
    filepath: str,
    mode: Literal["infer", "train"] = "train",
):
    if os.path.splitext(filepath)[1] == ".safetensors":
        state_dict = safetensors.torch.load_file(filepath)
        metadata = read_safetensors_metadata(filepath)
    else:
        data = torch.load(filepath)
        metadata = data["metadata"]
        state_dict = data["state_dict"]

    synthesizer_config = json.loads(metadata["_config"])
    synthesizer_config["config"] = ViTsGeneratorConfig.parse_json(
        synthesizer_config["config"]
    )

    model = ViTsSynthesizer(
        config=synthesizer_config["config"],
        mel_channels=synthesizer_config["mel_channels"],
        segment_size=synthesizer_config["segment_size"],
        sampling_rate=synthesizer_config["sampling_rate"],
    )
    if mode == "infer":
        del model.enc_q
        model.load_state_dict(state_dict, strict=False)
        model.eval()
    else:
        model.load_state_dict(state_dict)

    return model


def load_discriminator(
    filepath: str,
):
    if os.path.splitext(filepath)[1] == ".safetensors":
        state_dict = safetensors.torch.load_file(filepath)
        metadata = read_safetensors_metadata(filepath)
    else:
        data = torch.load(filepath)
        metadata = data["metadata"]
        state_dict = data["state_dict"]

    discriminator_config = json.loads(metadata["_config"])
    discriminator_config["config"] = ViTsDiscriminatorConfig.parse_obj(
        discriminator_config["config"]
    )

    model = MultiPeriodDiscriminator(
        config=discriminator_config["config"],
    )
    model.load_state_dict(state_dict)

    return model
