import importlib
import os

from reactpy import html
from reactpy.core.types import VdomChildren

OUT_FILE_FORMAT = "{filename}-{index}.{ext}"


def get_tabs(dir: str, root_module: str):
    tabs = []
    for file in os.listdir(dir):
        if not file.endswith(".py"):
            continue
        module_name = file[:-3]
        tab = importlib.import_module(f"{root_module}.{module_name}")
        if (
            not hasattr(tab, "title")
            or not hasattr(tab, "ui")
            or not hasattr(tab, "sort")
        ):
            continue
        tabs.append(tab)
    tabs = sorted(tabs, key=lambda x: x.sort())
    return tabs


def labeled(children: VdomChildren, label: str, horizontal=False, attributes={}):
    className = (
        "flex flex-row items-center justify-center gap-2"
        if horizontal
        else "flex flex-col"
    )
    return html.div(
        {**attributes, "class": f"{className} " + attributes.get("class", "")},
        html.label(
            {"class": "label"},
            html.span({"class": "label-text font-bold"}, label),
        ),
        children,
    )
