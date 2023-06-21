import os
from typing import *

from reactpy import component, hooks, html

from .components.tabs import Tabs
from .context import ContextData, context, opts
from .models import glob_models
from .utils import get_tabs

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TABS_DIR = os.path.join(ROOT_DIR, "webui", "tabs")


@component
def Main():
    tabs = get_tabs(TABS_DIR, "latopia.webui.tabs")
    ctx, set_ctx = hooks.use_state(
        ContextData(
            vits_models=glob_models(opts.vits_model_dir),
            encoders=glob_models(opts.encoder_dir),
        )
    )
    return context(
        Tabs(tabs, {"class": "h-full m-8"}),
        html.div({"class": "display-none", "id": "latopia-initialized"}),
        value=(ctx, set_ctx),
    )
