# LS-Trainer

Desktop app for training YOLO models on your own labeled datasets and exporting them to ONNX.

Built for the Lean Studio / LS-Vision workflow. Point it at a Label Studio (or YOLO-format) export, run a training job, and get a usable model without leaving the app.

## Features

- Train YOLO models from a local dataset (`data.yaml` + images)
- Live training log
- Export finished model to ONNX
- Open the run folder or copy the weights path
- Runs the local Python environment (Ultralytics) as a sidecar

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

Create and activate the Python environment used for training:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows
pip install -r requirements.txt       # if present
# or
pip install ultralytics
```

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
