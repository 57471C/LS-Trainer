#!/usr/bin/env bash
# ==============================================================================
# LS-Trainer: Linux Python Environment Setup
#
# Detects system Python (3.10-3.12), creates or updates .venv, and installs
# PyTorch with hardware acceleration priority:
#   1. Intel Arc / XPU (Official Intel Arc build: https://download.pytorch.org/whl/xpu)
#   2. NVIDIA CUDA (CUDA 12.8 / 12.4 build)
#   3. CPU Fallback (https://download.pytorch.org/whl/cpu)
#
# Followed by Ultralytics YOLO, RF-DETR, and ONNX training dependencies.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
VENV_PYTHON="${VENV_DIR}/bin/python"

AUTO_YES=0
for arg in "$@"; do
    case "$arg" in
        -y|--yes|--unattended)
            AUTO_YES=1
            ;;
        -h|--help)
            echo "Usage: ./setup-linux.sh [-y|--yes|--unattended]"
            echo "Sets up the Python virtual environment (.venv) for LS-Trainer on Linux."
            exit 0
            ;;
    esac
done

echo "========================================================"
echo "  LS-Trainer: Linux Environment Setup"
echo "========================================================"
echo "[*] Working directory: ${SCRIPT_DIR}"
echo "[*] Target virtualenv: ${VENV_DIR}"
echo

# ------------------------------------------------------------------------------
# 1. Detect System Python (3.10 - 3.12 recommended)
# ------------------------------------------------------------------------------
find_system_python() {
    for candidate in python3.12 python3.11 python3.10 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1; then
            if "$candidate" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >/dev/null 2>&1; then
                echo "$candidate"
                return 0
            fi
        fi
    done
    return 1
}

echo "[*] Detecting system Python..."
if ! SYSTEM_PYTHON="$(find_system_python)"; then
    echo "========================================================"
    echo " ERROR: Python 3.10 or higher is required but was not found."
    echo
    echo " On Ubuntu / Linux Mint / Debian, install Python with:"
    echo "   sudo apt update && sudo apt install -y python3 python3-venv python3-pip"
    echo
    echo " On Fedora:"
    echo "   sudo dnf install -y python3 python3-pip"
    echo
    echo " On Arch Linux:"
    echo "   sudo pacman -S python python-pip"
    echo "========================================================"
    exit 1
fi

PY_VER_STR="$("$SYSTEM_PYTHON" --version 2>&1)"
echo "[*] Found system Python: ${SYSTEM_PYTHON} (${PY_VER_STR})"

# ------------------------------------------------------------------------------
# 2. Check or Create Virtual Environment
# ------------------------------------------------------------------------------
if [ -x "${VENV_PYTHON}" ]; then
    echo "[*] Existing virtual environment found at ${VENV_DIR}."
else
    echo "[*] Creating virtual environment at ${VENV_DIR}..."
    if ! "$SYSTEM_PYTHON" -m venv "${VENV_DIR}" 2>/dev/null; then
        echo "========================================================"
        echo " ERROR: Failed to create virtual environment."
        echo " Ensure the python3-venv package is installed:"
        echo "   sudo apt install -y python3-venv"
        echo "========================================================"
        exit 1
    fi
fi

if [ ! -x "${VENV_PYTHON}" ]; then
    echo "========================================================"
    echo " ERROR: Virtual environment executable not found at:"
    echo "   ${VENV_PYTHON}"
    echo "========================================================"
    exit 1
fi

# Upgrade pip
echo "[*] Upgrading pip in virtual environment..."
"${VENV_PYTHON}" -m pip install --upgrade pip >/dev/null 2>&1 || true

# ------------------------------------------------------------------------------
# 3. Hardware GPU Detection (Intel Arc XPU -> NVIDIA CUDA -> CPU fallback)
# ------------------------------------------------------------------------------
echo
echo "[*] Detecting GPU hardware (Intel Arc XPU / NVIDIA CUDA / CPU fallback)..."

DETECTED_TYPE="cpu"
TARGET_WHEEL="cpu"
TARGET_INDEX_URL="https://download.pytorch.org/whl/cpu"
GPU_DISPLAY_NAME="CPU-only"
INTEL_ARC_FOUND=0
NVIDIA_FOUND=0
NV_DRIVER_MAJOR=0

# A. Probe Intel Arc / XPU via lspci
if command -v lspci >/dev/null 2>&1; then
    LSPCI_OUTPUT="$(lspci 2>/dev/null || true)"
    # Look for Intel VGA/3D/display controllers with Arc, DG2, Battlemage, Alchemist, Xe, or Pro
    INTEL_MATCH="$(echo "$LSPCI_OUTPUT" | grep -iE 'vga|3d|display' | grep -i 'intel' | grep -iE 'arc|dg2|battlemage|alchemist|xe|pro a|pro b|graphics|device 56' | head -n 1 || true)"
    if [ -n "$INTEL_MATCH" ]; then
        INTEL_ARC_FOUND=1
        GPU_DISPLAY_NAME="$(echo "$INTEL_MATCH" | sed -E 's/^[^:]+:[[:space:]]*//; s/Intel Corporation[[:space:]]*//I; s/\(rev [0-9a-f]+\)//I' | xargs)"
        if [ -z "$GPU_DISPLAY_NAME" ]; then
            GPU_DISPLAY_NAME="Intel Arc Graphics"
        fi
    fi
fi

# B. Probe Intel Arc / XPU via xpu-smi if installed
if [ "$INTEL_ARC_FOUND" -eq 0 ] && command -v xpu-smi >/dev/null 2>&1; then
    XPU_NAME="$(xpu-smi discovery 2>/dev/null | grep -i 'Device Name:' | head -n 1 | sed 's/.*Device Name:[[:space:]]*//' | xargs || true)"
    if [ -n "$XPU_NAME" ]; then
        INTEL_ARC_FOUND=1
        GPU_DISPLAY_NAME="$XPU_NAME"
    fi
fi

# C. Probe Linux sysfs DRM devices for Intel vendor 0x8086
if [ "$INTEL_ARC_FOUND" -eq 0 ] && [ -d /sys/class/drm ]; then
    for card_vendor in /sys/class/drm/card*/device/vendor; do
        if [ -f "$card_vendor" ]; then
            VENDOR_HEX="$(cat "$card_vendor" 2>/dev/null | tr '[:upper:]' '[:lower:]' | xargs || true)"
            if [ "$VENDOR_HEX" = "0x8086" ]; then
                DRIVER_LINK="$(dirname "$card_vendor")/driver"
                if [ -L "$DRIVER_LINK" ]; then
                    DRV_NAME="$(basename "$(readlink -f "$DRIVER_LINK")" 2>/dev/null || true)"
                    if [ "$DRV_NAME" = "i915" ] || [ "$DRV_NAME" = "xe" ]; then
                        INTEL_ARC_FOUND=1
                        GPU_DISPLAY_NAME="Intel Arc GPU (${DRV_NAME})"
                        break
                    fi
                fi
            fi
        fi
    done
fi

if [ "$INTEL_ARC_FOUND" -eq 1 ]; then
    DETECTED_TYPE="xpu"
    TARGET_WHEEL="xpu"
    TARGET_INDEX_URL="https://download.pytorch.org/whl/xpu"
    echo "[*] Detected Hardware: Intel Arc / XPU -> ${GPU_DISPLAY_NAME}"
else
    # D. Probe NVIDIA CUDA via nvidia-smi
    if command -v nvidia-smi >/dev/null 2>&1; then
        SMI_LINE="$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
        if [ -n "$SMI_LINE" ]; then
            NVIDIA_FOUND=1
            NV_NAME="$(echo "$SMI_LINE" | cut -d',' -f1 | xargs)"
            NV_DRV="$(echo "$SMI_LINE" | cut -d',' -f2 | xargs)"
            GPU_DISPLAY_NAME="${NV_NAME} (Driver ${NV_DRV})"
            NV_DRIVER_MAJOR="$(echo "$NV_DRV" | cut -d'.' -f1 | tr -dc '0-9')"
            NV_NAME_LOWER="$(echo "$NV_NAME" | tr '[:upper:]' '[:lower:]')"

            if echo "$NV_NAME_LOWER" | grep -qE 'rtx 50|rtx50|blackwell|gb10|gb20|b100|b200' || [ "${NV_DRIVER_MAJOR:-0}" -ge 560 ]; then
                DETECTED_TYPE="cuda"
                TARGET_WHEEL="cu128"
                TARGET_INDEX_URL="https://download.pytorch.org/whl/cu128"
            else
                DETECTED_TYPE="cuda"
                TARGET_WHEEL="cu124"
                TARGET_INDEX_URL="https://download.pytorch.org/whl/cu124"
            fi
            echo "[*] Detected Hardware: NVIDIA CUDA (${TARGET_WHEEL}) -> ${GPU_DISPLAY_NAME}"
        fi
    fi

    if [ "$NVIDIA_FOUND" -eq 0 ]; then
        DETECTED_TYPE="cpu"
        TARGET_WHEEL="cpu"
        TARGET_INDEX_URL="https://download.pytorch.org/whl/cpu"
        echo "[*] No discrete Intel Arc or NVIDIA GPU detected -> selecting CPU fallback"
    fi
fi

# ------------------------------------------------------------------------------
# 4. Install / Verify PyTorch Wheel
# ------------------------------------------------------------------------------
MARKER_FILE="${VENV_DIR}/.ls_trainer_wheel"

NEED_TORCH_INSTALL=1
if "${VENV_PYTHON}" -c "import torch" >/dev/null 2>&1; then
    VERIFY_RESULT="$("${VENV_PYTHON}" -c '
import sys, torch
tw = sys.argv[1]
ver = getattr(torch, "__version__", "")
xpu_ok = False
if hasattr(torch, "xpu") and torch.xpu.is_available():
    try:
        _ = torch.zeros(1, device="xpu") + 1
        torch.xpu.synchronize()
        xpu_ok = True
    except Exception:
        xpu_ok = False
cuda_ok = False
if torch.cuda.is_available():
    try:
        _ = torch.zeros(1, device="cuda") + 1
        torch.cuda.synchronize()
        cuda_ok = True
    except Exception:
        cuda_ok = False

if tw == "xpu" and xpu_ok and "+xpu" in ver:
    print("READY")
elif tw.startswith("cu") and cuda_ok and ("+" + tw) in ver:
    print("READY")
elif tw == "cpu" and ver and "+cu" not in ver and "+xpu" not in ver:
    print("READY")
else:
    print("MISMATCH")
' "${TARGET_WHEEL}" 2>/dev/null || echo "ERROR")"

    if [ "$VERIFY_RESULT" = "READY" ]; then
        echo "[*] Matching PyTorch build (${TARGET_WHEEL}) is already installed and functional in .venv."
        NEED_TORCH_INSTALL=0
    fi
fi

if [ "$NEED_TORCH_INSTALL" -eq 1 ]; then
    echo
    echo "[*] Installing PyTorch (${TARGET_WHEEL}) from ${TARGET_INDEX_URL}..."
    "${VENV_PYTHON}" -m pip install torch torchvision torchaudio --index-url "${TARGET_INDEX_URL}"
    echo "${TARGET_WHEEL}" > "${MARKER_FILE}"
fi

# ------------------------------------------------------------------------------
# 5. Install Ultralytics and RF-DETR Dependencies
# ------------------------------------------------------------------------------
echo
echo "[*] Installing Ultralytics YOLO..."
"${VENV_PYTHON}" -m pip install ultralytics

echo
echo "[*] Installing RF-DETR and COCO/ONNX training dependencies..."
"${VENV_PYTHON}" -m pip install rfdetr pytorch_lightning "torchmetrics[detection]<1.9.0,>=1.8.2" vernier faster-coco-eval hotcoco pycocotools scipy peft onnx onnxruntime

# ------------------------------------------------------------------------------
# 6. Verify Environment & Accelerators
# ------------------------------------------------------------------------------
echo
echo "========================================================"
echo "  Verifying Environment & Training Accelerators"
echo "========================================================"

"${VENV_PYTHON}" -c '
import sys, os, torch

print(f"[*] Python version:    {sys.version.split()[0]}")
print(f"[*] PyTorch version:   {torch.__version__}")

xpu_avail = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
cuda_avail = bool(torch.cuda.is_available())

print(f"[*] Intel Arc XPU:     {xpu_avail}")
print(f"[*] NVIDIA CUDA:       {cuda_avail}")

active_device = "CPU (No accelerator)"
if xpu_avail:
    cnt = torch.xpu.device_count()
    dev_names = []
    for i in range(cnt):
        try:
            dev_names.append(torch.xpu.get_device_name(i))
        except Exception:
            dev_names.append(f"XPU:{i}")
    active_device = f"Intel Arc XPU ({cnt} device(s): { \", \".join(dev_names) })"
    # Kernel test
    try:
        t = torch.zeros(2, 2, device="xpu") + 1
        torch.xpu.synchronize()
        print(f"[*] XPU Kernel test:   PASSED (device={t.device})")
    except Exception as e:
        print(f"[!] XPU Kernel test:   FAILED ({e})")
elif cuda_avail:
    cnt = torch.cuda.device_count()
    dev_names = [torch.cuda.get_device_name(i) for i in range(cnt)]
    active_device = f"NVIDIA CUDA ({cnt} device(s): { \", \".join(dev_names) })"
    # Kernel test
    try:
        t = torch.zeros(2, 2, device="cuda") + 1
        torch.cuda.synchronize()
        print(f"[*] CUDA Kernel test:  PASSED (device={t.device})")
    except Exception as e:
        print(f"[!] CUDA Kernel test:  FAILED ({e})")

print(f"[*] Active device:     {active_device}")
'

# ------------------------------------------------------------------------------
# 7. Linux-specific Intel Arc Tips (permissions / level-zero)
# ------------------------------------------------------------------------------
if [ "$DETECTED_TYPE" = "xpu" ]; then
    echo
    echo "========================================================"
    echo "  Intel Arc on Linux - System Configuration Tips"
    echo "========================================================"
    # Check render group
    if ! id -nG "$USER" 2>/dev/null | grep -qw "render"; then
        echo "[!] Note: Current user '$USER' is not in the 'render' group."
        echo "    To ensure access to Intel Arc render nodes (/dev/dri/renderD*):"
        echo "      sudo usermod -aG render,video $USER"
        echo "    (log out and back in for group changes to take effect)"
    fi
    # Check Level Zero loader
    if [ ! -f /usr/lib/x86_64-linux-gnu/libze_loader.so.1 ] && [ ! -f /usr/lib64/libze_loader.so.1 ] && ! ldconfig -p 2>/dev/null | grep -q libze_loader; then
        echo "[!] Note: Level Zero loader (libze_loader.so) was not found in system paths."
        echo "    On Linux Mint / Ubuntu, install compute runtime packages with:"
        echo "      sudo apt update"
        echo "      sudo apt install -y intel-opencl-icd intel-level-zero-gpu level-zero"
    fi
fi

echo
echo "========================================================"
echo "  Setup Complete! LS-Trainer Python environment is ready."
echo "========================================================"
