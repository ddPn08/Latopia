import torch
from pydub import AudioSegment

from latopia.f0_extractor import F0_METHODS_TYPE
from latopia.pipelines.vits import ViTsPipeline
from latopia.utils import get_torch_dtype, load_audio


def infer_vits(
    audio_input: str,
    output_path: str,
    pretrained_model_path: str,
    encoder_model_path: str,
    f0_method: F0_METHODS_TYPE = "crepe",
    transpose: int = 0,
    device: str = "cpu",
    torch_dtype: str = "float32",
):
    device = torch.device(device)
    torch_dtype = get_torch_dtype(torch_dtype)
    pipeline = ViTsPipeline(
        pretrained_model_path,
        encoder_model_path,
        device=device,
        torch_dtype=torch_dtype,
    )

    audio = load_audio(audio_input, 16000)
    audio_opt = pipeline(audio, f0_method=f0_method, transpose=transpose)

    audio = AudioSegment(
        audio_opt,
        frame_rate=pipeline.model.sampling_rate,
        sample_width=2,
        channels=1,
    )
    audio.export(output_path, format="wav")


def run():
    return {"vits": infer_vits}
