import glob
import os
import re

import numpy as np
import soundfile as sf

from .context import opts


def list_output_audios():
    files = glob.glob(os.path.join(opts.output_dir, "*"))
    for file in files:
        match = re.search(r"-(\d+)\..*", os.path.basename(file))
        if match:
            yield file


def get_output_audio_path(filename: str):
    return os.path.join(opts.output_dir, filename)


def save_audio(audio: np.ndarray, sampling_rate: int, filename: str, format="wav"):
    os.makedirs(opts.output_dir, exist_ok=True)
    index = len(list(list_output_audios())) + 1
    filename = os.path.splitext(filename)[0]
    filepath = os.path.join(opts.output_dir, f"{filename}-{index}.{format}")
    sf.write(filepath, audio, sampling_rate)
    audio.export(filepath, format=format)
    return os.path.relpath(filepath, opts.output_dir).replace("\\", "/")
