# DAGPID-Adapter / S1S2-LC8

**Downstream Adaptation of Remote Sensing Foundation Models for Optical-SAR Land-Cover Segmentation**

Code for **DAGPID-Adapter: Downstream Adaptation of Remote Sensing Foundation Models for Optical-SAR Land-Cover Segmentation**. This repository provides training and inference for 18 frozen remote sensing foundation model (RSFM) variants from 13 model families on S1S2-LC8, using 100% or 10% of the training labels. All variants share the DAGPID-Adapter downstream framework, and their settings are consolidated in `config.yaml`.

## Setup

Training and inference use the same Linux/CUDA environment. The manuscript experiments use NVIDIA RTX A6000 GPUs; each training run uses one GPU.

| Component | Requirement |
| --- | --- |
| Operating system | Linux |
| Python | 3.10 or newer; the installation example below uses 3.11 |
| GPU | NVIDIA GPU with CUDA and BF16 support for the default configuration |
| PyTorch | CUDA build, `>=2.6,<3` |
| State-space extensions | `mamba-ssm>=2.2` with CUDA selective scan, and `causal-conv1d>=1.4` |
| Other packages | See `requirements.txt` and the selected encoder's upstream dependencies |

Clone the repository and create an environment:

```bash
git clone https://github.com/zhangjiayu608/DHN-S1S2-LC8.git
cd DHN-S1S2-LC8
conda create -n dagpid python=3.11 -y
conda activate dagpid
python -m pip install --upgrade pip setuptools wheel packaging ninja
```

Install a CUDA-enabled PyTorch build compatible with your GPU driver, following the [official PyTorch instructions](https://pytorch.org/get-started/locally/). For example, PyTorch 2.6.0 with the matching torchvision package is available for CUDA 12.6:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu126
MAMBA_KEEP_CUDA_BUILD=TRUE python -m pip install --no-build-isolation -r requirements.txt
```

The CUDA version above is an installation example. Match `mamba-ssm` and `causal-conv1d` to the installed PyTorch/CUDA environment. `MAMBA_KEEP_CUDA_BUILD=TRUE` enables the CUDA selective-scan extension in current Mamba releases; see the [Mamba installation guide](https://github.com/state-spaces/mamba#installation). Building extensions from source also requires a compatible CUDA toolkit (`nvcc`) and C++ compiler. Install any additional dependencies required by the selected encoder's upstream project.

Check CUDA and the Mamba forward/backward path before starting a full run:

```bash
python - <<'PY'
import torch
from mamba_ssm import Mamba

assert torch.cuda.is_available(), "A CUDA-enabled PyTorch installation is required."
assert torch.cuda.is_bf16_supported(), "The default configuration uses BF16."
model = Mamba(d_model=32, d_state=16, d_conv=4, expand=2).cuda()
x = torch.randn(1, 16, 32, device="cuda", requires_grad=True)
model(x).mean().backward()
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch:", torch.__version__, "CUDA:", torch.version.cuda)
print("Mamba CUDA forward/backward: OK")
PY
```

## Data

Place the prepared S1S2-LC8 TIFF files under a data root with this layout:

```text
S1S2_LC8_Robust/
  train/{s1,s2,lc}/
  val/{s1,s2,lc}/
  test/{s1,s2,lc}/
```
Download the prepared S1S2-LC8 dataset from [Baidu Netdisk](https://pan.baidu.com/s/1sHuSP4NE0IOBJyyOGwaX_A?pwd=f3dw).

- Extraction code: `f3dw`
- Archive: `S1S2-LC8.zip` (18.82 GB)

Extract the archive and set `--data-root` to the extracted
`S1S2_LC8_Robust` directory.

The files in `metadata/` contain the fixed sample paths, class histograms, image fingerprints, and training-image normalization statistics. The loader reads the compressed CSV files directly. Use the prepared benchmark files matching these manifests; original source archives require the benchmark's preprocessing and file naming.

| Partition | Image pairs |
| --- | ---: |
| Training, 100% labels | 45,000 |
| Training, fixed 10% subset | 4,500 |
| Validation | 9,994 |
| Test | 21,399 |

The 10% subset reproduces the seed-13 main comparison. Both label budgets use the same validation/test splits; CROMA uses the same full-training-image normalization statistics under both budgets. Class weights are computed from the selected training labels. LC8 class IDs 0–7 are forest/tree cover, shrubland, grassland, wetland/flooded vegetation, cropland, built-up, barren/bare ground, and water; 255 is ignored. Keep all files in `metadata/`: the compressed CSV files are sample manifests, not training logs.

## Pretrained models

Set `--assets-root` to a directory containing `weights/` and `third_party/`. `config.yaml` specifies every required relative path. Obtain encoder code and weights from the corresponding upstream releases and preserve their directory layouts:

| Model family | Variants | Upstream release | Code directory under `third_party/` |
| --- | --- | --- | --- |
| AnySat | Base | [AnySat](https://github.com/gastruc/anysat) | `AnySat-main/` |
| CopernicusFM | Base, Large | [Copernicus-FM](https://huggingface.co/wangyi111/Copernicus-FM) | `Copernicus-FM/Copernicus-FM/` |
| CROMA | Base, Large | [CROMA](https://github.com/antofuller/CROMA) | `CROMA-main/` |
| FoMo | Base | [FoMo-Bench](https://arxiv.org/abs/2312.10114) | `FoMo-Bench-main/` |
| Galileo | Base | [Galileo](https://huggingface.co/nasaharvest/galileo) | `galileo-main/` |
| MaRS | Base | [MaRS](https://rsidea.whu.edu.cn/mars.htm) | Implemented by the local wrapper |
| Panopticon | ViT-B/14 | [Panopticon](https://huggingface.co/lewaldm/panopticon) | `panopticon-main/` |
| RAMEN | Base | [RAMEN](https://huggingface.co/nicolashoudre/RAMEN) | `RAMEN-main/pangaea-bench/` |
| SkySense | S1/S2 | [SkySense checkpoints](https://www.notion.so/SkySense-Checkpoints-a7fcff6ce29a4647a08c7fe416910509) | Implemented by the local wrapper |
| SkySense++ | S1/S2 | [SkySense++](https://github.com/kang-wu/SkySensePlusPlus) | Implemented by the local wrapper |
| SoftCon | ViT-B/14 | [SoftCon](https://huggingface.co/wangyi111/softcon) | `softcon-main/` |
| TerraMind | Tiny, Small, Base, Large | [TerraTorch](https://github.com/torchgeo/terratorch) | `terratorch-main/` |
| TESSERA | v1.1 MPC | [TESSERA](https://github.com/ucam-eo/tessera) | `tessera-master/tessera-master/tessera_infer_QAT/src/` |

All variants also use the shared DINOv3-SAT ViT-L/16 encoder. Place its Hugging Face model directory, including `config.json` and `model.safetensors`, at `third_party/dinov3-vitl16-pretrain-sat493m/`; see [DINOv3](https://github.com/facebookresearch/dinov3). DINOv3 is additional to the 18 compared variants.

Weights named `*_encoder_only`, `*_enc`, or `*_teacher` refer to the encoder state used by the wrappers. Match the state structure and filenames in `config.yaml` when preparing an upstream checkpoint. Preserve the original licenses and attribution of each upstream model.

## Train

List the model identifiers:

```bash
python train.py --list-models
```

Run either label budget:

```bash
python train.py --model croma_base --labels 100 --gpu 0 --data-root /path/to/S1S2_LC8_Robust --assets-root /path/to/assets
python train.py --model croma_base --labels 10 --gpu 0 --data-root /path/to/S1S2_LC8_Robust --assets-root /path/to/assets
```

All model settings and both label budgets are defined in **`config.yaml`**. Shared training settings are 100 epochs, batch size 64, BF16, AdamW with weight decay 0.05, learning rate 7.5e-5, five warmup epochs, and cosine decay to 1.5e-6. The default seed is 13; the released 10% subset uses this seed. RSFM and DINOv3 encoders remain frozen. Each run uses one GPU; independent runs can use different GPUs.

Outputs are written to `outputs/<model>/labels_<budget>/seed_<seed>/`, including `best.pt`, `last.pt`, the resolved configuration, and epoch metrics. Validation mIoU selects `best.pt`. Resume with the same model, label budget, seed, and paths:

```bash
python train.py --model croma_base --labels 100 --gpu 0 --data-root /path/to/S1S2_LC8_Robust --assets-root /path/to/assets --resume outputs/croma_base/labels_100/seed_13/last.pt
```

Use `--workers` to adjust data-loader workers, or `--output-dir` to select a run directory. Data and asset defaults can also be set in `config.yaml`; `DAGPID_ASSETS` is accepted as an asset-root environment variable.

## Inference and evaluation

Run inference and compute the seven reported metrics on the complete test split:

```bash
python evaluate.py --checkpoint outputs/croma_base/labels_100/seed_13/best.pt --gpu 0 --data-root /path/to/S1S2_LC8_Robust --assets-root /path/to/assets
```

Evaluation loads the checkpoint's saved model configuration and writes `test/metrics.json`, containing mIoU, OA, mAcc, mF1, mPrecision, Cohen's kappa, BF1, classwise metrics, and the confusion matrix. Metric values are fractions; multiply by 100 for the paper's reporting scale. Use `--split val` for validation. Compact checkpoints reference their original frozen encoder assets, which must be available for loading. The entry point evaluates prepared image pairs; it does not export prediction mosaics. Per-epoch training logs omit boundary computation by default; obtain the reported BF1 from this evaluation command.

## Repository

```text
DHN-S1S2-LC8/
  train.py             # Main training entry point
  evaluate.py          # Inference and seven-metric evaluation
  config.yaml          # Shared settings and all 18 RSFM variants
  requirements.txt     # Common Python and CUDA-extension dependencies
  dagpid/              # Data loading, models, losses, metrics, training loop
  metadata/            # Fixed splits, class mapping, statistics, checksums
```
