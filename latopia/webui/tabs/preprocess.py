import asyncio
import multiprocessing as mp
from typing import *

from reactpy import component, hooks, html

from latopia.config.dataset import DatasetConfig, DatasetSubsetConfig
from latopia.dataset.base import AudioDataset
from latopia.dataset.preprocess import PreProcessor
from latopia.f0_extractor import F0_METHODS_TYPE
from latopia.logger import set_logger
from latopia.webui.components.radio import LabeledRadio
from latopia.webui.utils import labeled

logger = set_logger(__name__)


def preprocess(
    dataset_dir: str,
    target_sr: int,
    max_workers: int = 1,
    slice: bool = True,
    f0_method: F0_METHODS_TYPE = "crepe",
    crepe_model: str = "tiny",
    hop_length: int = 160,
    f0_max: int = 1100.0,
    f0_min: int = 50.0,
    f0_mel_max: Optional[int] = None,
    f0_mel_min: Optional[int] = None,
    encoder_path: str = "./models/encoders/checkpoint_best_legacy_500.pt",
    encoder_channels: int = 768,
    encoder_output_layer: int = 12,
    device: str = "cpu",
):
    config = DatasetConfig(subsets=[DatasetSubsetConfig(data_dir=dataset_dir)])
    dataset = AudioDataset(config)
    preprocessor = PreProcessor(dataset)
    logger.info("Writing wave files...")
    preprocessor.write_wave(
        target_sr,
        slice=slice,
        max_workers=max_workers,
    )
    logger.info("Extracting f0...")
    preprocessor.extract_f0(
        f0_method,
        crepe_model=crepe_model,
        hop_length=hop_length,
        f0_max=f0_max,
        f0_min=f0_min,
        f0_mel_max=f0_mel_max,
        f0_mel_min=f0_mel_min,
        max_workers=max_workers,
    )
    logger.info("Extracting features...")
    preprocessor.extract_features(
        encoder_path,
        encoder_channels=encoder_channels,
        encoder_output_layer=encoder_output_layer,
        device=device,
    )


def sort():
    return 0


def title():
    return "PreProcess"


@component
def ui():
    dataset_dir, set_dataset_dir = hooks.use_state("")
    target_sr, set_target_sr = hooks.use_state("40000")
    max_workers, set_max_workers = hooks.use_state("1")
    slice, set_slice = hooks.use_state(True)
    f0_method, set_f0_method = hooks.use_state("crepe")
    crepe_model, set_crepe_model = hooks.use_state("tiny")

    running, set_running = hooks.use_state(False)

    async def run_preprocess(e):
        set_running(True)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            preprocess,
            dataset_dir,
            int(target_sr),
            int(max_workers),
            slice,
            f0_method,
            crepe_model,
        )
        set_running(False)

    return html.div(
        {
            "class": "m-4 flex flex-col gap-4",
        },
        html.div(
            {"class": "flex flex-row gap-8"},
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
                attributes={"class": "w-full"},
            ),
        ),
        html.div(
            {"class": "flex flex-row gap-8"},
            labeled(
                html.input(
                    {
                        "class": "input input-bordered",
                        "type": "nubmer",
                        "value": target_sr,
                        "on_change": lambda e: set_target_sr((e["target"]["value"])),
                    }
                ),
                "Target sampling rate",
            ),
            labeled(
                html.div(
                    {
                        "class": "w-full max-w-xs",
                    },
                    html.input(
                        {
                            "class": "range range-xs",
                            "type": "range",
                            "min": 1,
                            "max": mp.cpu_count(),
                            "value": max_workers,
                            "step": 1,
                            "on_change": lambda e: set_max_workers(
                                e["target"]["value"]
                            ),
                        }
                    ),
                    html.input(
                        {
                            "class": "input input-bordered input-sm w-full mt-4",
                            "value": max_workers,
                            "type": "number",
                            "disabled": True,
                        }
                    ),
                ),
                "Max workers",
            ),
            labeled(
                html.input(
                    {
                        "class": "checkbox",
                        "type": "checkbox",
                        "checked": slice,
                        "on_change": lambda e: set_slice(e["target"]["value"] == "on"),
                    }
                ),
                "Slice",
                horizontal=True,
            ),
            labeled(
                html.div(
                    *[
                        LabeledRadio(name, f0_method, set_f0_method)
                        for name in ["dio", "harvest", "crepe", "mangio-crepe"]
                    ]
                ),
                "F0 Method",
            ),
            labeled(
                html.div(
                    *[
                        LabeledRadio(name, crepe_model, set_crepe_model)
                        for name in ["tiny", "full"]
                    ]
                ),
                "Crepe model",
            ),
        ),
        html.button(
            {
                "class": "btn btn-primary",
                "on_click": run_preprocess,
                "disabled": running,
            },
            html.span({"class": "loading loading-spinner"})
            if running
            else "Preprocess",
        ),
    )
