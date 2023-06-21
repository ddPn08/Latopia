from fastapi import FastAPI
from fastapi.responses import FileResponse

from . import infer, javascripts


def audio():
    return FileResponse(
        path="/home/ddpn08/data/ai/Latopia/input.wav", media_type="audio/wav"
    )


def register(app: FastAPI):
    app.get("/api/audio")(audio)
    infer.register(app)
    javascripts.register(app)
