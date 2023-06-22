import os
from typing import *

from pydantic import BaseModel
from reactpy import create_context

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INFER_INPUT_FILE = {
    "filename": None,
    "path": None,
}


class Options(BaseModel):
    vits_model_dir: str = os.path.join(ROOT_DIR, "models", "checkpoints", "vits")
    encoder_dir: str = os.path.join(ROOT_DIR, "models", "encoders")
    output_dir: str = os.path.join(ROOT_DIR, "output")
    device: str = "cuda"


opts = Options()


class ContextData(TypedDict):
    vits_models: List[str]
    encoders: List[str]


def set_context(ctx: ContextData):
    pass


context = create_context(tuple([ContextData(), set_context]))


def setup(**kwargs):
    for k, v in kwargs.items():
        if v is not None:
            setattr(opts, k, v)
