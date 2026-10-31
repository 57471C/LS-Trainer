# LS-Trainer

Desktop app for training YOLO models on your own labeled datasets and exporting them to ONNX.

Built for the Lean Studio / LS-Vision workflow. Point it at a Label Studio (or YOLO-format) export, run a training job, and get a usable model without leaving the app.

## Features

- Train YOLO11 and YOLO26 models for object detection or instance segmentation
- Segmentation weights: `yolo11s-seg.pt`, `yolo11m-seg.pt`, `yolo26s-seg.pt` (`task="segment"`)
- Bench defaults for a fixed top-down camera: `degrees` 180, `flipud` 0.5, `scale` 0.05, `close_mosaic` 10
- Image sizes 640, 960, and 1280 px
- Warns when a segmentation run is pointed at plain bounding-box labels
- Live training log
- Auto-export the finished model to ONNX at the selected image size (`dynamic=False`, `simplify=True`)
- Open the run folder or copy the weights and ONNX paths
- Runs the local Python environment (Ultralytics) as a sidecar

## Windows Installer & Post-Install Setup

The Windows NSIS installer configures the Python training environment automatically:
- **Automatic Setup Hook**: During installation, an NSIS hook automatically executes `setup-venv.bat` via `nsExec::ExecToLog` on the finish page, keeping all output visible in the installer log.
- **Dedicated Virtual Environment**: `setup-venv.bat` creates a `.venv` directory directly adjacent to the installed `app.exe` (using `%~dp0`), upgrades `pip`, and installs `ultralytics` and `rfdetr`. If a compatible system Python is not found on PATH, it prompts to download and install Python 3.12.
- **GPU Detection Priority (Intel Arc XPU &rarr; NVIDIA CUDA &rarr; CPU)**: Before installing ML packages, `setup-venv.bat` (and the in-app GPU check/repair) probes `Win32_VideoController` and `nvidia-smi`:
  1. **Intel Arc / XPU**: Installs the official Intel Arc XPU PyTorch build (`https://download.pytorch.org/whl/xpu`) and avoids pulling CUDA packages.
  2. **NVIDIA GPU**: Installs the matching CUDA 12.8 (`cu128`, Blackwell / RTX 50-series) or CUDA 12.4 (`cu124`) PyTorch build.
  3. **CPU Fallback**: Installs the lightweight CPU PyTorch wheel (`https://download.pytorch.org/whl/cpu`) when neither Intel Arc nor NVIDIA hardware is present.
- **Hardware Verification**: At the conclusion of setup, it verifies `torch.xpu.is_available()` and `torch.cuda.is_available()` and prints the active training accelerator. During RF-DETR training, `torch.xpu.is_available()` is preferred first (`xpu`), followed by `torch.cuda.is_available()` (`cuda`), with `cpu` as the final fallback.
- **Low-Friction**: No extra user steps are needed after clicking Finish—the application is ready to run immediately.

## Linux Setup (Linux Mint, Ubuntu, Debian)

For Linux systems (including single or multi-GPU Intel Arc Pro workstations):

```bash
chmod +x setup-linux.sh
./setup-linux.sh
```

The script automatically detects:
1. **Intel Arc / XPU GPUs** (via `lspci`, `xpu-smi`, and DRM sysfs), pulling the official PyTorch XPU build (`https://download.pytorch.org/whl/xpu`).
2. **NVIDIA GPUs** (via `nvidia-smi`), installing CUDA 12.8 or 12.4 builds.
3. **CPU Fallback** when no discrete GPU is found.

### Intel Arc Compute Prerequisites (Linux Mint / Ubuntu)
Before training on Intel Arc GPUs under Linux:
- Install compute runtime and Level Zero packages:
  ```bash
  sudo apt update && sudo apt install -y intel-opencl-icd intel-level-zero-gpu level-zero
  ```
- Ensure your user belongs to the `render` and `video` groups to access `/dev/dri/renderD*` devices:
  ```bash
  sudo usermod -aG render,video $USER
  ```
  *(Log out and log back in for group membership to take effect)*


## Stack

- Tauri 2
- Svelte + TypeScript + Vite
- Python sidecar (Ultralytics YOLO)

## Development

### Prerequisites

- Node.js
- Rust
- Python 3.10+ (with a virtual environment)

### Setup

```bash
npm install
```

#### Python environment

Python 3.10–3.12 is recommended.  
Python 3.13+ can work but may have slower wheel support for some Ultralytics dependencies.

```powershell
# Create the virtual environment in the project root
python -m venv .venv

# Activate it (Windows)
.\.venv\Scripts\Activate.ps1

# Upgrade pip (recommended)
python -m pip install --upgrade pip

# Install Ultralytics
pip install ultralytics
```

If you later add a `requirements.txt`, use:

```powershell
pip install -r requirements.txt
```

The app looks for this `.venv` when it runs training and ONNX export.

### Run

```bash
npm run tauri dev
```

The app expects a YOLO-format dataset archive (`.tar`, `.tar.gz`, or `.tgz`, including a Label Studio export).  
Detection runs are written under `runs/detect`. Segmentation runs are written under `runs/segment`.  
When auto-export is on, a successful run writes `best.onnx` next to `best.pt`. Copy that path for LS-Vision.

## Project status

Version 0.1.4. Usable for local detection and instance-segmentation training, plus ONNX export.
