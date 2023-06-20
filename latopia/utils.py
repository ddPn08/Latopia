import gc
import json
import mmap
import socket
import subprocess

import numpy as np
import torch

str_prec_to_torch = {
    "fp16": torch.float16,
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def clear_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def get_torch_dtype(dtype):
    if isinstance(dtype, str):
        dtype = str_prec_to_torch[dtype]
    return dtype


def read_safetensors_metadata(filename):
    with open(filename, mode="r", encoding="utf8") as file_obj:
        with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as m:
            header = m.read(8)
            n = int.from_bytes(header, "little")
            metadata_bytes = m.read(n)
            metadata = json.loads(metadata_bytes)

    return metadata.get("__metadata__", {})


def find_empty_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port


def load_audio(file: str, sr):
    # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "f32le",
        "-ac",
        "1",
        "-acodec",
        "pcm_f32le",
        "-ar",
        str(sr),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.float32).flatten()
