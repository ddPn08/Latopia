import glob
import os

from latopia.pipelines.vits import ViTsPipeline

from .context import opts

MODEL_EXTENSIONS = [".pt", ".pth", ".safetensors"]

model = {"key": None, "model": None}


def is_discriminator(model: str):
    basename = os.path.splitext(os.path.basename(model))[0]
    return basename.endswith(".d")


def glob_models(dir: str, filter_discriminator: bool = False):
    files = []
    for ext in MODEL_EXTENSIONS:
        for file in glob.glob(os.path.join(dir, f"**/*{ext}"), recursive=True):
            if filter_discriminator and is_discriminator(file):
                continue
            files.append(file)
    files = sorted([os.path.relpath(file, dir).replace("\\", "/") for file in files])
    return files


def create_model_path(model: str):
    return os.path.join(opts.vits_model_dir, model)


def create_vits(generator: str, encoder: str) -> ViTsPipeline:
    generator = os.path.join(opts.vits_model_dir, generator)
    encoder = os.path.join(opts.encoder_dir, encoder)
    key = f"{generator}+{encoder}"
    if model["key"] == key:
        return model["model"]

    model["key"] = key
    model["model"] = ViTsPipeline(generator, encoder, device=opts.device)
    return model["model"]
