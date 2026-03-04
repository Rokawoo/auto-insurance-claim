# Auto Insurance Claim

> [!CAUTION]
> This project was DEVELOPED by BASEDCODED & LAINPILLED DEVS

# Setup

Requires: NVIDIA driver 570+, Conda, CUDA 12.8+ system toolkit.

## 1. Create env + install PyTorch

```bash
conda create -n auto-claim python=3.12 -y
conda activate claim
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
```

> [!NOTE]
> If using Nvidia Blackwell Arch+/error `sm_120 is not compatible`, use nightly instead:

```bash
pip uninstall torch torchvision -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Verify

`
python -c "import torch, cv2, ultralytics; print('torch', torch.__version__, ' cuda', torch.cuda.is_available(), ' gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); print('opencv', cv2.__version__, ' ultralytics', ultralytics.__version__)"
`

You should see `cuda True` and your GPU name.
If it says `+cpu` in the torch version, you installed the wrong wheel, redo step 1.

## Notes

- Don't `conda install` any ML packages, only use conda for Python itself. Everything else through pip.
- `requirements.txt` excludes torch on purpose, it must come from the CUDA index URL above.