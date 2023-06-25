import json
from typing import *

from fastapi import FastAPI
from reactpy import component, html
from reactpy.backend.fastapi import Options, configure
from uvicorn import run

from . import context
from .api import register
from .main import Main

TAILWIND_CONFIG = {
    "darkMode": "class",
    "daisyui": {
        "themes": ["night"],
    },
}


def head():
    return (
        html.title("Latopia"),
        html.link(
            {
                "href": "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap",
                "rel": "stylesheet",
            }
        ),
        html.link(
            {
                "href": "https://cdn.jsdelivr.net/npm/daisyui@3.1.0/dist/full.css",
                "rel": "stylesheet",
            }
        ),
        html.script({"src": "https://cdn.tailwindcss.com"}, ""),
        html.script(
            f"""
            tailwind.config = {json.dumps(TAILWIND_CONFIG)}
            """
        ),
        html.script({"src": "/javascripts/index.js", "type": "module"}, ""),
    )


@component
def Root():
    return html.div(Main())


def __hijack_reactpy():
    import reactpy.backend.starlette as backend
    from starlette.requests import Request
    from starlette.responses import HTMLResponse

    def _make_index_route(options: Options):
        index_html = backend.read_client_index_html(options)
        index_html = index_html.replace(
            '<html lang="en">', '<html data-theme="night" lang="en">'
        )

        async def serve_index(request: Request) -> HTMLResponse:
            return HTMLResponse(index_html)

        return serve_index

    backend._make_index_route = _make_index_route


app = FastAPI()
register(app)
__hijack_reactpy()
configure(app, Root, options=Options(head=head()))


def launch(
    checkpoints_dir: Optional[str] = None,
    encoder_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    host: Optional[str] = None,
    port: int = 8000,
):
    context.setup(
        checkpoints_dir=checkpoints_dir,
        encoder_dir=encoder_dir,
        output_dir=output_dir,
    )
    run(app, host=host, port=port)
