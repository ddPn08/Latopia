import asyncio
import os
from typing import *

import torch
from reactpy import component, hooks, html

from latopia.config.dataset import DatasetConfig, DatasetSubsetConfig
from latopia.config.train import TrainConfig
from latopia.config.vits import ViTsConfig
from latopia.vits.train import train
from latopia.webui.components.radio import LabeledRadio
from latopia.webui.context import ROOT_DIR, context, opts
from latopia.webui.models import create_model_path, is_discriminator
from latopia.webui.utils import labeled


def sort():
    return 0


def title():
    return "Train"


sampling_rate_map = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


@component
def ui():
    ctx, set_ctx = hooks.use_context(context)
    generators = hooks.use_memo(
        lambda: [ckpt for ckpt in ctx["vits_models"] if not is_discriminator(ckpt)]
    )
    discriminators = hooks.use_memo(
        lambda: [ckpt for ckpt in ctx["vits_models"] if is_discriminator(ckpt)]
    )

    device, set_device = hooks.use_state("cuda")
    generator, set_generator = hooks.use_state(
        generators[0] if len(generators) > 0 else None
    )
    discriminator, set_discriminator = hooks.use_state(
        discriminators[0] if len(discriminators) > 0 else None
    )
    resume_from, set_resume_from = hooks.use_state(None)
    output_name, set_output_name = hooks.use_state("vits-model")
    output_dir, set_output_dir = hooks.use_state(
        os.path.join(opts.vits_model_dir, "vits-model")
    )
    dataset_dir, set_dataset_dir = hooks.use_state("")
    save_as, set_save_as = hooks.use_state("safetensors")
    save_state, set_save_state = hooks.use_state(True)
    sampling_rate, set_sampling_rate = hooks.use_state("40k")
    seed, set_seed = hooks.use_state(-1)
    learning_rate, set_learning_rate = hooks.use_state(1e-4)
    batch_size, set_batch_size = hooks.use_state(4)
    max_train_epoch, set_max_train_epoch = hooks.use_state(30)
    save_every_n_epoch, set_save_every_n_epoch = hooks.use_state(5)
    cache_in_gpu, set_cache_in_gpu = hooks.use_state(False)

    running, set_running = hooks.use_state(False)

    async def run_train(e):
        set_running(True)
        loop = asyncio.get_event_loop()

        config = TrainConfig(
            pretrained_model_path=create_model_path(generator),
            pretrained_discriminator_path=create_model_path(discriminator),
            resume_model_path=resume_from,
            output_name=output_name,
            output_dir=output_dir,
            save_as=save_as,
            save_state=save_state,
            sampling_rate=sampling_rate_map[sampling_rate],
            seed=int(seed),
            learning_rate=float(learning_rate),
            batch_size=int(batch_size),
            max_train_epoch=int(max_train_epoch),
            save_every_n_epoch=int(save_every_n_epoch),
            cache_in_gpu=cache_in_gpu,
        )
        dataset_config = DatasetConfig(
            subsets=[
                DatasetSubsetConfig(
                    data_dir=dataset_dir,
                )
            ]
        )
        vits_config = ViTsConfig.parse_toml(
            os.path.join(ROOT_DIR, "configs", "vits", f"{sampling_rate}.toml")
        )
        await loop.run_in_executor(
            None, train, torch.device(device), config, dataset_config, vits_config
        )
        set_running(False)

    return html.div(
        {
            "class": "m-4 w-full flex flex-col gap-4",
        },
        html.div(
            {"class": "flex flex-row gap-8"},
            labeled(
                html.div(
                    *[
                        LabeledRadio(name, device, set_device)
                        for name in ["cuda", "cpu", "mps"]
                    ]
                ),
                "Device",
            ),
            labeled(
                html.select(
                    {
                        "class": "select select-bordered w-full max-w-xs",
                        "value": generator,
                        "on_change": lambda e: set_generator(e["target"]["value"]),
                    },
                    *[html.option(ckpt) for ckpt in generators],
                ),
                "Pretrained generator",
            ),
            labeled(
                html.select(
                    {
                        "class": "select select-bordered w-full max-w-xs",
                        "value": discriminator,
                        "on_change": lambda e: set_discriminator(e["target"]["value"]),
                    },
                    *[html.option(ckpt) for ckpt in discriminators],
                ),
                "Pretrained discriminator",
            ),
            labeled(
                html.input(
                    {
                        "class": "input input-bordered",
                        "type": "text",
                        "value": resume_from,
                        "on_change": lambda e: set_resume_from(e["target"]["value"]),
                    }
                ),
                "Resume from",
            ),
        ),
        html.div(
            {"class": "flex flex-row gap-8 w-full"},
            labeled(
                html.input(
                    {
                        "class": "input input-bordered",
                        "type": "text",
                        "value": output_name,
                        "on_change": lambda e: set_output_name(e["target"]["value"]),
                    }
                ),
                "Output name",
                attributes={"class": "w-1/5"},
            ),
            labeled(
                html.input(
                    {
                        "class": "input input-bordered w-full",
                        "type": "text",
                        "value": output_dir,
                        "on_change": lambda e: set_output_dir(e["target"]["value"]),
                    }
                ),
                "Output directory",
                attributes={"class": "w-2/5"},
            ),
            labeled(
                html.input(
                    {
                        "class": "input input-bordered w-full",
                        "type": "text",
                        "value": dataset_dir,
                        "on_change": lambda e: set_dataset_dir(e["target"]["value"]),
                    }
                ),
                "Dataset directory",
                attributes={"class": "w-2/5"},
            ),
        ),
        html.div(
            {"class": "flex flex-row gap-8 w-full"},
            labeled(
                html.div(
                    *[
                        LabeledRadio(name, save_as, set_save_as)
                        for name in ["pt", "safetensors"]
                    ]
                ),
                "Save as",
            ),
            labeled(
                html.input(
                    {
                        "class": "checkbox",
                        "type": "checkbox",
                        "checked": save_state,
                        "on_change": lambda e: set_save_state(
                            e["target"]["value"] == "on"
                        ),
                    }
                ),
                "Save state",
                horizontal=True,
            ),
            labeled(
                html.div(
                    *[
                        LabeledRadio(name, sampling_rate, set_sampling_rate)
                        for name in ["32k", "40k", "48k"]
                    ]
                ),
                "Sampling rate",
            ),
            labeled(
                html.input(
                    {
                        "class": "input input-bordered",
                        "type": "nubmer",
                        "value": seed,
                        "on_change": lambda e: set_seed((e["target"]["value"])),
                    }
                ),
                "Seed",
            ),
            labeled(
                html.input(
                    {
                        "class": "input input-bordered",
                        "type": "nubmer",
                        "value": batch_size,
                        "on_change": lambda e: set_batch_size(e["target"]["value"]),
                    }
                ),
                "Batch size",
            ),
            labeled(
                html.input(
                    {
                        "class": "input input-bordered",
                        "type": "text",
                        "value": learning_rate,
                        "on_change": lambda e: set_learning_rate(e["target"]["value"]),
                    }
                ),
                "Learning rate",
            ),
            labeled(
                html.input(
                    {
                        "class": "input input-bordered",
                        "type": "nubmer",
                        "value": max_train_epoch,
                        "on_change": lambda e: set_max_train_epoch(
                            e["target"]["value"]
                        ),
                    }
                ),
                "Max train epoch",
            ),
            labeled(
                html.input(
                    {
                        "class": "input input-bordered",
                        "type": "nubmer",
                        "value": save_every_n_epoch,
                        "on_change": lambda e: set_save_every_n_epoch(
                            e["target"]["value"]
                        ),
                    }
                ),
                "Save every n epoch",
            ),
            labeled(
                html.input(
                    {
                        "class": "checkbox",
                        "type": "checkbox",
                        "checked": cache_in_gpu,
                        "on_change": lambda e: set_cache_in_gpu(
                            e["target"]["value"] == "on"
                        ),
                    }
                ),
                "Cache in GPU",
                horizontal=True,
            ),
        ),
        html.button(
            {"class": "btn btn-primary", "on_click": run_train, "disabled": running},
            html.span({"class": "loading loading-spinner"}) if running else "Train",
        ),
    )
