import gc
import os
from typing import *

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tqdm
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from latopia.config.dataset import DatasetConfig
from latopia.config.diffusion import (
    DiffusionDatasetConfig,
    DiffusionModelConfig,
    DiffusionTrainConfig,
)
from latopia.dataset.diffusion import DiffusionAudioCollate, DiffusionAudioDataset
from latopia.diffusion.unit2mel import Unit2Mel, save_model
from latopia.diffusion.vocoder import Vocoder
from latopia.utils import find_empty_port, get_torch_dtype, load_model


def train(
    device: Union[List[torch.device], torch.device],
    config: DiffusionTrainConfig,
    dataset_config: DatasetConfig,
    diff_config: DiffusionModelConfig,
):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(find_empty_port())

    os.makedirs(config.output_dir, exist_ok=True)
    config.write_toml(os.path.join(config.output_dir, "config.toml"))
    dataset_config.write_toml(os.path.join(config.output_dir, "dataset.toml"))
    diff_config.write_toml(os.path.join(config.output_dir, "diffusion.toml"))

    if type(device) == torch.device:
        train_runner(
            device,
            0,
            1,
            config,
            dataset_config,
            diff_config,
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
                    diff_config,
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
    config: DiffusionTrainConfig,
    dataset_config: DiffusionDatasetConfig,
    model_config: DiffusionModelConfig,
):
    is_multi_process = world_size > 1
    is_main_process = rank == 0
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

    dataset = DiffusionAudioDataset(dataset_config)
    collate_fn = DiffusionAudioCollate()
    data_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=1,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    vocoder = Vocoder(config.vocoder_type, config.vocoder_dir, device)

    model = Unit2Mel(
        model_config.emb_channels,
        model_config.spk_embed_dim,
        model_config.use_pitch_aug,
        vocoder.dimension,
        model_config.n_layers,
        model_config.n_chans,
        model_config.n_hidden,
    ).to(device)

    if is_multi_process:
        model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )

    epoch = 0

    resume = None

    if config.resume_model_path is not None:
        resume, metadata = load_model(config.resume_model_path)
        epoch = int(metadata["epoch"])

    if resume is not None:
        optimizer.load_state_dict(resume["optimizer"])

    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=config.lr_decay, last_epoch=epoch - 1
    )
    if resume is not None:
        lr_scheduler.load_state_dict(resume["scheduler"])

    scaler = GradScaler(enabled=mixed_precision is not None)

    model = model.module if type(model) == DDP else model

    if config.pretrained_model_path is not None:
        state_dict, metadata = load_model(config.pretrained_model_path)
        model.load_state_dict(state_dict, strict=False)

    cache = []
    if is_main_process:
        progress_bar = tqdm.tqdm(
            range((config.max_train_epoch - epoch + 1) * len(data_loader))
        )
        progress_bar.set_postfix(epoch=epoch)
    step = -1

    model.train()

    for epoch in range(epoch, config.max_train_epoch + 1):
        use_cache = len(cache) == len(data_loader)
        data = cache if use_cache else enumerate(data_loader)

        def save(
            filename=f"{config.output_name}-{epoch}",
        ):
            metadata = {
                "epoch": f"{epoch}",
            }

            save_model(
                os.path.join(checkpoint_dir, filename),
                model,
                model_config,
                metadata,
                config.save_as,
            )

            if config.save_state:
                torch.save(
                    {
                        "state_dict": {
                            "optimizer": optimizer.state_dict(),
                            "scheduler": lr_scheduler.state_dict(),
                        },
                        "metadata": metadata,
                    },
                    os.path.join(state_dir, f"{epoch}.state.ckpt"),
                )

        for i, batch in data:
            step += 1
            if is_main_process:
                progress_bar.update(1)

            if not use_cache:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device=device, non_blocking=True)

            if config.cache_in_gpu:
                cache.append((i, batch))

            optimizer.zero_grad()

            with autocast(enabled=mixed_precision is not None, dtype=mixed_precision):
                loss = model(
                    batch["features"],
                    batch["f0"],
                    batch["volume"],
                    batch["speaker_id"],
                    None,
                    gt_spec=batch["mel"],
                    infer=False,
                    k_step=model_config.k_step_max,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            lr_scheduler.step()

            if is_main_process:
                lr = optimizer.param_groups[0]["lr"]
                progress_bar.set_postfix(
                    epoch=epoch,
                    loss=f"{float(loss):.4f}",
                    lr=f"{lr:.6f}",
                    use_cache=use_cache,
                )

        if is_main_process:
            if (
                config.save_every_n_epoch > 0
                and epoch != 0
                and epoch % config.save_every_n_epoch == 0
                and epoch != config.max_train_epoch
            ):
                save()

    if is_main_process:
        save(
            filename=config.output_name,
        )
