from typing import *

from reactpy import html

from latopia.webui.utils import labeled


def AudioInput(
    source_file: str,
    set_source_file: Callable[[str], None],
):
    return (
        labeled(
            html._(
                html.input(
                    {
                        "class": "latopia-audio-input file-input w-full max-w-xs mb-2",
                        "type": "file",
                        "value": source_file,
                        "on_change": lambda e: set_source_file(e["target"]["value"]),
                    }
                ),
            ),
            "Source Audio",
        ),
    )
