# LS-Trainer

Desktop app for training YOLO models on your own labeled datasets and exporting them to ONNX.

Built for the Lean Studio / LS-Vision workflow. Point it at a Label Studio (or YOLO-format) export, run a training job, and get a usable model without leaving the app.

## Features

- Train YOLO models from a local dataset (`data.yaml` + images)
- Live training log
- Export finished model to ONNX
- Open the run folder or copy the weights path
- Runs the local Python environment (Ultralytics) as a sidecar

## Windows Installer & Post-Install Setup

The Windows NSIS installer configures the Python training environment automatically:
- **Automatic Setup Hook**: During installation, an NSIS hook automatically executes `setup-venv.bat` via `nsExec::ExecToLog` on the finish page, keeping all output visible in the installer log.
- **Dedicated Virtual Environment**: `setup-venv.bat` creates a `.venv` directory directly adjacent to the installed `app.exe` (using `%~dp0`), upgrades `pip`, and installs `ultralytics`. If a compatible system Python is not found on PATH, it prompts to download and install Python 3.12.
- **CUDA / GPU Check**: At the conclusion of setup, it runs `python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"` and prints a clear note indicating whether training can use GPU acceleration or will fall back to CPU.
- **Low-Friction**: No extra user steps are needed after clicking Finish—the application is ready to run immediately.

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

The app expects a YOLO-format dataset (the zip export from Label Studio works).  
After a successful run you can convert the model to ONNX and copy the path for use in LS-Vision.

## Project status

Early but usable.  
Current focus is reliable local training + ONNX export.  
Other base models and dataset formats can be added later.
