import shutil
import tempfile

from fastapi import FastAPI, Request, UploadFile

from latopia.webui import audio, context
from latopia.webui.responses.audio_response import AudioResponse


def infer_input_audio_upload(file: UploadFile):
    out_path = tempfile.mktemp()
    with open(out_path, "wb") as out_file:
        shutil.copyfileobj(file.file, out_file)
    context.INFER_INPUT_FILE["filename"] = file.filename
    context.INFER_INPUT_FILE["path"] = out_path
    return {"status": "ok"}


def get_output_audio(request: Request, filepath: str):
    return AudioResponse(
        path=audio.get_output_audio_path(filepath),
        headers=request.headers,
        media_type="audio/wav",
    )


def register(app: FastAPI):
    app.post("/api/infer/input-file-upload")(infer_input_audio_upload)
    app.get("/api/output-audio/{filepath:path}")(get_output_audio)
