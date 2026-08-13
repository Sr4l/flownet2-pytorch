# FlowNet2-Pytorch

PyTorch implementation of FlowNet 2.0 optical flow estimation (NVIDIA). Ported from the original Ubuntu 16.04 / PyTorch 0.4 stack to modern Python (3.10+).

## Setup

```bash
bash install.sh          # creates .venv (Python 3.10), installs root deps via uv
source .venv/bin/activate
```

`install.sh` only installs the root `pyproject.toml` package. It does **not** build the `networks/*_package/` CUDA extensions.

### Custom layers

The three `*_package/` directories (`correlation`, `resample2d`, `channelnorm`) each ship with both `.cu`/`.cc` CUDA kernels and pure-PyTorch fallbacks. **The Python imports always use the PyTorch fallbacks** — the compiled CUDA code is dead code and never loaded at runtime. No CUDA compilation step is needed.

## Usage

### Inference
```bash
python main.py --inference --model FlowNet2 --save_flow \
  --inference_dataset MpiSintelClean \
  --inference_dataset_root /path/to/dataset \
  --resume /path/to/checkpoint.pth.tar
```

For fp16 inference, add `--fp16`.

### Training
```bash
python main.py --batch_size 8 --model FlowNet2 --loss=L1Loss --optimizer=Adam --optimizer_lr=1e-4 \
  --training_dataset FlyingChairs --training_dataset_root /path/to/data \
  --validation_dataset MpiSintelClean --validation_dataset_root /path/to/data
```

### Quick pair inference

`run_a_pair.py` has **stale imports** (`from Networks.FlowNet2`) and hardcoded paths — it won't run as-is. Use `main.py --inference` with `ImagesFromFolder` instead.

## Architectures (models.py)

| Model | Description |
|---|---|
| `FlowNet2` | Full: C → S → S → SD → Fusion (162M params) |
| `FlowNet2C` | Correlation-only; dual-stream conv + correlation tensor + decoder |
| `FlowNet2S` | Single-stream; 6-channel concat (RGB×2) → conv → decoder |
| `FlowNet2SD` | Single-direction; like S but deeper (twin conv blocks, inter-level convs in decoder) |
| `FlowNet2CS` | C + S fused |
| `FlowNet2CSS` | C + S + S fused |

Each model requires `args` with `.rgb_max` (default 255.) and `.fp16` (default False).

## Arguments

`main.py` uses dynamic argument generation via `tools.add_arguments_for_module()`. Parameters are namespaced: `--model_paramname=value`, `--loss_startScale=4`, `--optimizer_lr=1e-4`, `--training_dataset_root=/path`, etc. Run `python main.py --help` for the full list.

The effective batch size is `batch_size × number_gpus`.

## Checkpoints

Pre-trained weights (Caffe→PyTorch converted) are on Google Drive (links in README.md). `--resume` accepts a `.pth.tar` path. `convert.py` requires Python 2.7 + caffe; not needed for normal use.

## Datasets (datasets.py)

Available: `FlyingChairs`, `FlyingThingsClean`, `FlyingThingsFinal`, `MpiSintelClean`, `MpiSintelFinal`, `ChairsSDHomTrain`, `ChairsSDHomTest`, `ImagesFromFolder`.

Images must be divisible by 64; the dataloader auto-adjusts `inference_size` (default `-1,-1`) to the largest valid dimension.

## Gotchas

- No tests, CI, lint, or typecheck exist in this repo.
- Output goes to `./work/` by default (`--save`). TensorBoard logs: `./work/train/` and `./work/validation/`.
- `main.py` calls `os.chdir()` to its own directory at startup — relative paths resolve from the repo root.
- The pure-PyTorch Correlation and Resample2d fallbacks are slower than the original CUDA kernels. Expect lower throughput.