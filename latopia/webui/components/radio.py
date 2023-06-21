from typing import *

from reactpy import component, html


@component
def LabeledRadio(label: str, value: str, set_value: Callable):
    return html.label(
        {"class": "label cursor-pointer"},
        html.span({"class": "label-text"}, label),
        html.input(
            {
                "class": "radio",
                "type": "radio",
                "checked": value == label,
                "on_click": lambda _: set_value(label),
            }
        ),
    )
