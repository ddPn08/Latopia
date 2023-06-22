import gc
import os
from typing import *

import safetensors.torch
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from latopia.config.dataset import DatasetConfig
from latopia.config.train import TrainConfig
from latopia.config.vits import ViTsConfig
from latopia.dataset.vits import (
    DistributedBucketSampler,
    TextAudioCollate,
    ViTsAudioDataset,
)
from latopia.mel_extractor import mel_spectrogram_torch, spec_to_mel_torch
from latopia.utils import find_empty_port, get_torch_dtype, read_safetensors_metadata
from latopia.vits.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from latopia.vits.models import (
    MultiPeriodDiscriminator,
    ViTsSynthesizer,
    save_discriminator,
    save_generator,
)

from latopia import torch_utils


def train(
    device: Union[List[torch.device], torch.device],
    config: TrainConfig,
    dataset_config: DatasetConfig,
    vits: ViTsConfig,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_empty_port())

    if type(device) == list:
        if len(device) <= 1:
            device = device[0]

    os.makedirs(config.output_dir, exist_ok=True)
    config.write_toml(os.path.join(config.output_dir, "config.toml"))
    dataset_config.write_toml(os.path.join(config.output_dir, "dataset.toml"))
    vits.write_toml(os.path.join(config.output_dir, "vits.toml"))

    if type(device) == torch.device:
        train_runner(
            device,
            0,
            1,
            config,
            dataset_config,
            vits,
        )
    else:
        processes = []
        for i, d in enumerate(device):
            assert d.type == "cuda", "Only cuda devices are supported when using DDP."
            ps = mp.Process(
                target=train_runner,
                args=(
                    d,
                    i,
                    len(device),
                    config,
                    dataset_config,
                    vits,
                ),
            )
            ps.start()
            processes.append(ps)

        for ps in processes:
            ps.join()


def train_runner(
    device: torch.device,
    rank: int,
    world_size: int,
    config: TrainConfig,
    dataset_config: DatasetConfig,
    vits: ViTsConfig,
):
    is_multi_process = world_size > 1
    is_main_process = rank == 0
    global_step = 0
    mixed_precision = get_torch_dtype(config.mixed_precision)

    checkpoint_dir = os.path.join(config.output_dir, "checkpoints")
    state_dir = os.path.join(config.output_dir, "states")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()

    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo", init_method="env://", rank=rank, world_size=world_size
        )

    if is_multi_process:
        torch.cuda.set_device(rank)

    if config.seed is None:
        config.seed = int(torch.randint(0, 2**32, (1,)).item())

    torch.manual_seed(config.seed)

    collate_fn = TextAudioCollate()

    dataset = ViTsAudioDataset(dataset_config, vits.train.dataset)
    train_sampler = DistributedBucketSampler(
        dataset,
        config.batch_size * world_size,
        [100, 200, 300, 400, 500, 600, 700, 800, 900],
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    data_loader = DataLoader(
        dataset,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
        persistent_workers=True,
        prefetch_factor=8,
    )

    net_g = ViTsSynthesizer(
        vits.generator,
        vits.train.dataset.filter_length // 2 + 1,
        vits.train.segment_size // vits.train.dataset.hop_length,
        sampling_rate=config.sampling_rate,
    ).to(device=device)
    net_d = MultiPeriodDiscriminator(vits.discriminator).to(device=device)

    if is_multi_process:
        net_g = DDP(net_g, device_ids=[rank])
        net_d = DDP(net_d, device_ids=[rank])

    optimizer_g = torch.optim.AdamW(
        net_g.parameters(),
        config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )
    optimizer_d = torch.optim.AdamW(
        net_d.parameters(),
        config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )

    def load_model(filepath: str):
        ext = os.path.splitext(filepath)[1]
        if ext == ".safetensors":
            state_dict = safetensors.torch.load_file(filepath)
            metadata = read_safetensors_metadata(filepath)
        else:
            state_dict = torch.load(filepath)
            metadata = state_dict.pop("metadata") if "metadata" in state_dict else {}
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

        return state_dict, metadata

    epoch = 0

    resume = None

    if config.resume_model_path is not None:
        resume, metadata = load_model(config.resume_model_path)
        epoch = int(metadata["epoch"])

    if resume is not None:
        optimizer_g.load_state_dict(resume["optimizer_g"])
        optimizer_d.load_state_dict(resume["optimizer_d"])

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_g, gamma=config.lr_decay, last_epoch=epoch - 1
    )
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_d, gamma=config.lr_decay, last_epoch=epoch - 1
    )

    if resume is not None:
        scheduler_g.load_state_dict(resume["scheduler_g"])
        scheduler_d.load_state_dict(resume["scheduler_d"])

    scaler = GradScaler(enabled=mixed_precision is not None)

    net_d, net_g = (
        net_d.module if type(net_d) == DDP else net_d,
        net_g.module if type(net_g) == DDP else net_g,
    )

    if config.pretrained_model_path is not None:
        state_dict, metadata = load_model(config.pretrained_model_path)
        net_g.load_state_dict(state_dict)
    if config.pretrained_discriminator_path is not None:
        state_dict, metadata = load_model(config.pretrained_discriminator_path)
        net_d.load_state_dict(state_dict)

    cache = []
    progress_bar = tqdm.tqdm(
        range((config.max_train_epoch - epoch + 1) * len(data_loader))
    )
    progress_bar.set_postfix(epoch=epoch)
    step = -1

    for epoch in range(epoch, config.max_train_epoch + 1):
        net_g.train()
        net_d.train()

        use_cache = len(cache) == len(data_loader)
        data = cache if use_cache else enumerate(data_loader)

        def save_model(
            filename=f"{config.output_name}-{epoch}",
        ):
            metadata = {
                "epoch": f"{epoch}",
            }
            save_generator(
                checkpoint_dir,
                filename,
                net_g,
                vits.generator,
                metadata,
                save_as=config.save_as,
            )
            save_discriminator(
                checkpoint_dir,
                filename,
                net_d,
                vits.discriminator,
                metadata,
                save_as=config.save_as,
            )

            if config.save_state:
                torch.save(
                    {
                        "state_dict": {
                            "optimizer_g": optimizer_g.state_dict(),
                            "optimizer_d": optimizer_d.state_dict(),
                            "scheduler_g": scheduler_g.state_dict(),
                            "scheduler_d": scheduler_d.state_dict(),
                        },
                        "metadata": metadata,
                    },
                    os.path.join(state_dir, f"{epoch}.state.ckpt"),
                )

        if is_main_process:
            lr = optimizer_g.param_groups[0]["lr"]

        for i, batch in data:
            step += 1
            progress_bar.update(1)

            if not use_cache:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device=device, non_blocking=True)

            if config.cache_in_gpu:
                cache.append((i, batch))

            with autocast(enabled=mixed_precision is not None, dtype=mixed_precision):
                g_result = net_g(
                    batch["features"],
                    batch["features_lengths"],
                    batch["f0"],
                    batch["f0_nsf"],
                    batch["mel"],
                    batch["mel_lengths"],
                    batch["speaker_id"],
                )

                mel = spec_to_mel_torch(
                    batch["mel"],
                    vits.train.dataset.filter_length,
                    vits.train.dataset.hop_length,
                    vits.train.dataset.sampling_rate,
                    vits.train.dataset.mel_fmin,
                    vits.train.dataset.mel_fmax,
                )
                y_mel = torch_utils.slice_segments(
                    mel,
                    g_result["ids_slice"],
                    vits.train.segment_size // vits.train.dataset.hop_length,
                )
                y_hat_mel = mel_spectrogram_torch(
                    g_result["o"].float().squeeze(1),
                    vits.train.dataset.filter_length,
                    vits.train.dataset.n_mel_channels,
                    vits.train.dataset.sampling_rate,
                    vits.train.dataset.hop_length,
                    vits.train.dataset.win_length,
                    vits.train.dataset.mel_fmin,
                    vits.train.dataset.mel_fmax,
                )

                audio = torch_utils.slice_segments(
                    batch["audio"],
                    g_result["ids_slice"] * vits.train.dataset.hop_length,
                    vits.train.segment_size,
                )

                d_result = net_d(audio, g_result["o"].detach())
                with autocast(enabled=False):
                    loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                        d_result["y_d_rs"], d_result["y_d_gs"]
                    )

            optimizer_d.zero_grad()
            scaler.scale(loss_disc).backward()
            scaler.unscale_(optimizer_d)
            scaler.step(optimizer_d)

            with autocast(enabled=mixed_precision is not None, dtype=mixed_precision):
                d_result_2 = net_d(audio, g_result["o"])
                with autocast(enabled=False):
                    loss_mel = F.l1_loss(y_mel, y_hat_mel) * vits.train.c_mel
                    loss_kl = kl_loss(
                        g_result["z_p"],
                        g_result["logs_q"],
                        g_result["m_p"],
                        g_result["logs_p"],
                        g_result["y_mask"],
                    )
                    loss_fm = feature_loss(d_result_2["fmap_rs"], d_result_2["fmap_gs"])
                    loss_gen, losses_gen = generator_loss(d_result_2["y_d_gs"])
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl

            optimizer_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optimizer_g)
            scaler.step(optimizer_g)
            scaler.update()

            if is_main_process:
                progress_bar.set_postfix(
                    epoch=epoch,
                    loss_g=float(loss_gen_all) if loss_gen_all is not None else 0.0,
                    loss_d=float(loss_disc) if loss_disc is not None else 0.0,
                    lr=float(lr) if lr is not None else 0.0,
                    use_cache=use_cache,
                )

            global_step += 1

        scheduler_g.step()
        scheduler_d.step()

        if is_main_process:
            if (
                config.save_every_n_epoch > 0
                and epoch != 0
                and epoch % config.save_every_n_epoch == 0
                and epoch != config.max_train_epoch
            ):
                save_model()

    if is_main_process:
        save_model(
            filename=config.output_name,
        )
