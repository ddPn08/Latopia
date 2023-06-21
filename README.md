> **Note**
> This project is currently in development.

# Latopia

Speech AI training and inference tools

## Installation

```bash
git clone https://github.com/ddPn08/Latopia.git
cd Latopia
python -m venv venv
source venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python setup.py develop
```

## WebUI

```bash
latopia webui
```

## Cli Usage

### Vits

#### Training

1. Create config files

```toml
# ./configs/zundamon/config.toml
pretrained_model_path = "./models/checkpoints/vits/pretrained-40k.g.safetensors"
pretrained_discriminator_path = "./models/checkpoints/vits/pretrained-40k.d.safetensors"
cache_in_gpu = true
output_name = "zundamon"
output_dir = "./output"
mixed_precision = "fp16"
batch_size = 32
save_every_n_epoch = 5
```

```toml
# ./configs/zundamon/dataset.toml
[[subsets]]
data_dir = "./datasets/zundamon_voice_data/emotion/normal"
```

2. Preprocess dataset

```bash
latopia preprocess all ./configs/zundamon/dataset.toml 40000 --max_workers 8 --device cuda --f0_method harvest
```

3. Train

```bash
latopia train vits ./configs/zundamon/config.toml ./configs/zundamon/dataset.toml  --vits_config_path ./configs/vits/40k.toml --device cuda
```

#### Inference

```bash
latopia infer vits ./input.wav output.wav ./output/checkpoints/zundamon-30-G.safetensors ./models/encoder/checkpoint_best_legacy_500.pt --device cuda --torch_dtype fp16 --f0_method harvest
```
