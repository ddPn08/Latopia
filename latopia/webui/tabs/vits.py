import os
from typing import *

from reactpy import component

from ..components.tabs import Tabs
from ..utils import get_tabs


def sort():
    return 1


def title():
    return "ViTs"


@component
def ui():
    tabs = get_tabs(
        os.path.join(os.path.dirname(__file__), "vits-tabs"),
        "latopia.webui.tabs.vits-tabs",
    )
    return Tabs(tabs, {"class": ""})
