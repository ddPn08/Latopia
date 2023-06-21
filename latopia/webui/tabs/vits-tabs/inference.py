import asyncio
from typing import *

from pydub import AudioSegment
from reactpy import component, hooks, html

from latopia.utils import load_audio
from latopia.webui.audio import save_audio
from latopia.webui.components.radio import LabeledRadio
from latopia.webui.context import INFER_INPUT_FILE, context
from latopia.webui.models import create_vits, is_discriminator
from latopia.webui.utils import labeled


def sort():
    return 0


def title():
    return "Inference"


def infer(
    checkpoint: str,
    transpose: int,
    f0_extractor: str,
    encoder_model: str,
):
    pipe = create_vits(checkpoint, encoder_model)
    audio = load_audio(INFER_INPUT_FILE["path"], 16000)
    result = pipe(audio, f0_extractor, transpose)
    result = AudioSegment(
        result,
        frame_rate=pipe.model.sampling_rate,
        sample_width=2,
        channels=1,
    )
    filepath = save_audio(result, INFER_INPUT_FILE["filename"])
    return filepath


@component
def ui():
    ctx, set_ctx = hooks.use_context(context)
    generators = hooks.use_memo(
        lambda: [ckpt for ckpt in ctx["vits_models"] if not is_discriminator(ckpt)]
    )
    checkpoint, set_checkpoint = hooks.use_state(
        generators[0] if len(generators) > 0 else None
    )
    source_file, set_source_file = hooks.use_state(None)
    transpose, set_transpose = hooks.use_state(0)
    f0_extractor, set_f0_extractor = hooks.use_state("harvest")
    encoder_model, set_encoder_model = hooks.use_state(
        ctx["encoders"][0] if len(ctx["encoders"]) > 0 else None
    )
    output_audio, set_output_audio = hooks.use_state(None)
    running, set_running = hooks.use_state(False)

    async def run_infer(e):
        set_output_audio(None)
        set_running(True)
        loop = asyncio.get_event_loop()
        filepath = await loop.run_in_executor(
            None, infer, checkpoint, transpose, f0_extractor, encoder_model
        )
        set_running(False)
        set_output_audio(filepath)

    return html.div(
        {
            "class": "m-4 flex flex-col gap-4",
        },
        html.div(
            {"class": "flex flex-row gap-8"},
            labeled(
                html.select(
                    {
                        "class": "select select-bordered w-full max-w-xs",
                        "value": checkpoint,
                        "on_change": lambda e: set_checkpoint(e["target"]["value"]),
                    },
                    *[html.option(ckpt) for ckpt in generators],
                ),
                "Checkpoint",
            ),
            labeled(
                html._(
                    html.input(
                        {
                            "class": "latopia-audio-input file-input w-full max-w-xs mb-2",
                            "type": "file",
                            "value": source_file,
                            "on_change": lambda e: set_source_file(
                                e["target"]["value"]
                            ),
                        }
                    ),
                ),
                "Source Audio",
            ),
        ),
        html.div(
            {"class": "flex flex-row gap-8"},
            labeled(
                html.div(
                    {
                        "class": "w-full max-w-xs",
                    },
                    html.input(
                        {
                            "class": "range range-xs",
                            "type": "range",
                            "min": -20,
                            "max": 20,
                            "value": transpose,
                            "step": 1,
                            "on_change": lambda e: set_transpose(
                                int(e["target"]["value"])
                            ),
                        }
                    ),
                    html.input(
                        {
                            "class": "input input-bordered input-sm w-full mt-4",
                            "value": transpose,
                            "type": "number",
                            "on_change": lambda e: set_transpose(
                                int(e["target"]["value"])
                            ),
                        }
                    ),
                ),
                "Transpose",
            ),
            labeled(
                html.div(
                    *[
                        LabeledRadio(name, f0_extractor, set_f0_extractor)
                        for name in ["dio", "harvest", "crepe", "mangio-crepe"]
                    ]
                ),
                "F0 Extractor",
            ),
            labeled(
                html.select(
                    {
                        "class": "select w-full max-w-xs",
                        "value": encoder_model,
                        "on_change": lambda e: set_encoder_model(e["target"]["value"]),
                    },
                    *[html.option(encoder) for encoder in ctx["encoders"]],
                ),
                "Encoder model",
            ),
        ),
        html.button(
            {
                "class": "latopia-infer-button btn btn-primary",
                "disabled": running,
                "on_click": run_infer,
            },
            html.span({"class": "loading loading-spinner"}) if running else "Inference",
        ),
        html.audio(
            {
                "class": "w-full",
                "src": f"/api/output-audio/{output_audio}" if output_audio else None,
                "controls": True,
            }
        )
        if output_audio
        else (),
    )
