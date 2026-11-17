from collections import deque
import ast
import json
import math
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


class TrainingControlState:
    def __init__(self):
        self._lock = threading.Lock()
        self._resume_event = threading.Event()
        self._resume_event.set()
        self.paused = False
        self.checkpoint_requested = False
        self.stop_requested = False

    def request_pause(self):
        with self._lock:
            self.paused = True
            self._resume_event.clear()
        print("[LS-Trainer] Training paused.", flush=True)

    def request_resume(self):
        with self._lock:
            self.paused = False
            self._resume_event.set()
        print("[LS-Trainer] Training resumed.", flush=True)

    def request_checkpoint(self):
        with self._lock:
            self.checkpoint_requested = True

    def consume_checkpoint_request(self):
        with self._lock:
            req = self.checkpoint_requested
            self.checkpoint_requested = False
            return req

    def request_stop(self):
        with self._lock:
            self.stop_requested = True
            self.paused = False
            self._resume_event.set()

    def is_stop_requested(self, stop_file=None):
        with self._lock:
            if self.stop_requested:
                return True
        if stop_file and os.path.exists(stop_file):
            return True
        return False

    def wait_if_paused(self, stop_file=None):
        while not self._resume_event.is_set():
            if self.is_stop_requested(stop_file):
                return
            self._resume_event.wait(timeout=0.2)


def _iter_stdin_control_lines():
    """Yield control lines from piped stdin without holding Python's sys.stdin buffer lock."""
    if sys.stdin is None:
        return
    try:
        fd = sys.stdin.fileno()
    except Exception:
        return
    try:
        if os.isatty(fd):
            return
    except Exception:
        pass

    buf = b""
    if os.name == "nt":
        import ctypes
        import ctypes.wintypes as wintypes
        import msvcrt

        handle = msvcrt.get_osfhandle(fd)
        kernel32 = ctypes.windll.kernel32
        avail = wintypes.DWORD(0)
        while True:
            ok = kernel32.PeekNamedPipe(handle, None, 0, None, ctypes.byref(avail), None)
            if not ok:
                break
            if avail.value == 0:
                time.sleep(0.05)
                continue
            chunk = os.read(fd, min(4096, avail.value))
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line.decode("utf-8", errors="replace")
    else:
        import select

        while True:
            rlist, _, _ = select.select([fd], [], [], 0.1)
            if not rlist:
                continue
            chunk = os.read(fd, 4096)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                yield line.decode("utf-8", errors="replace")

    if buf.strip():
        yield buf.decode("utf-8", errors="replace")


def start_stdin_control_listener(ctrl_state):
    def _listener():
        try:
            for raw_line in _iter_stdin_control_lines():
                cmd = raw_line.strip().upper()
                if not cmd:
                    continue
                if cmd == "PAUSE":
                    ctrl_state.request_pause()
                elif cmd == "RESUME":
                    ctrl_state.request_resume()
                elif cmd == "CHECKPOINT":
                    ctrl_state.request_checkpoint()
                elif cmd == "STOP":
                    ctrl_state.request_stop()
                    break
        except Exception:
            pass

    t = threading.Thread(target=_listener, name="ls-trainer-stdin-control", daemon=True)
    t.start()
    return t


def _safe_float(val, default=0.0):
    if val is None:
        return default
    if hasattr(val, "item"):
        try:
            val = val.item()
        except Exception:
            pass
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except Exception:
        return default


def _safe_opt_float(val):
    if val is None:
        return None
    if hasattr(val, "item"):
        try:
            val = val.item()
        except Exception:
            pass
    try:
        f = float(val)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def query_gpu_telemetry():
    try:
        import torch
    except Exception:
        return []

    gpus = []
    try:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            count = int(torch.xpu.device_count())
            for idx in range(count):
                name = "Intel Arc XPU"
                try:
                    name = str(torch.xpu.get_device_name(idx))
                except Exception:
                    pass
                used_bytes = 0
                total_bytes = 0
                if hasattr(torch.xpu, "mem_get_info"):
                    try:
                        free_b, dev_total_b = torch.xpu.mem_get_info(idx)
                        if dev_total_b and dev_total_b > 0:
                            total_bytes = int(dev_total_b)
                            used_bytes = max(0, total_bytes - int(free_b))
                    except Exception:
                        pass
                try:
                    proc_used = max(
                        int(torch.xpu.memory_reserved(idx)),
                        int(torch.xpu.memory_allocated(idx)),
                    )
                    used_bytes = max(used_bytes, proc_used)
                except Exception:
                    pass
                if total_bytes <= 0:
                    try:
                        props = torch.xpu.get_device_properties(idx)
                        total_bytes = int(
                            getattr(props, "total_memory", 0)
                            or getattr(props, "max_mem_alloc_size", 0)
                            or 0
                        )
                    except Exception:
                        pass
                used_gb = round(used_bytes / (1024.0 ** 3), 3)
                total_gb = round(total_bytes / (1024.0 ** 3), 3)
                util = round((used_gb / total_gb) * 100.0, 1) if total_gb > 0 else 0.0
                gpus.append({
                    "index": idx,
                    "name": name,
                    "vram_used_gb": _safe_float(used_gb),
                    "vram_total_gb": _safe_float(total_gb),
                    "util_percent": _safe_float(util),
                })
        elif torch.cuda.is_available():
            count = int(torch.cuda.device_count())
            for idx in range(count):
                name = "CUDA GPU"
                try:
                    name = str(torch.cuda.get_device_name(idx))
                except Exception:
                    pass
                used_bytes = 0
                total_bytes = 0
                try:
                    used_bytes = max(
                        int(torch.cuda.memory_reserved(idx)),
                        int(torch.cuda.memory_allocated(idx)),
                    )
                except Exception:
                    pass
                try:
                    props = torch.cuda.get_device_properties(idx)
                    total_bytes = int(getattr(props, "total_memory", 0) or 0)
                except Exception:
                    pass
                used_gb = round(used_bytes / (1024.0 ** 3), 3)
                total_gb = round(total_bytes / (1024.0 ** 3), 3)
                util = 0.0
                try:
                    if hasattr(torch.cuda, "utilization"):
                        util = float(torch.cuda.utilization(idx))
                except Exception:
                    util = round((used_gb / total_gb) * 100.0, 1) if total_gb > 0 else 0.0
                gpus.append({
                    "index": idx,
                    "name": name,
                    "vram_used_gb": _safe_float(used_gb),
                    "vram_total_gb": _safe_float(total_gb),
                    "util_percent": _safe_float(util),
                })
    except Exception:
        return []
    return gpus


def emit_metrics(
    epoch,
    total_epochs,
    step,
    total_steps,
    train_loss,
    val_loss=None,
    map50=None,
    map50_95=None,
    lr=0.0,
    samples_per_sec=0.0,
    eta_secs=0.0,
):
    payload = {
        "epoch": max(0, int(epoch)),
        "total_epochs": max(0, int(total_epochs)),
        "step": max(0, int(step)),
        "total_steps": max(0, int(total_steps)),
        "train_loss": _safe_float(train_loss),
        "val_loss": _safe_opt_float(val_loss),
        "map50": _safe_opt_float(map50),
        "map50_95": _safe_opt_float(map50_95),
        "lr": _safe_float(lr),
        "samples_per_sec": _safe_float(samples_per_sec),
        "eta_secs": _safe_float(eta_secs),
        "gpus": query_gpu_telemetry(),
    }
    try:
        out_stream = getattr(sys, "__stdout__", None) or sys.stdout
        print(f"\n@@LS_METRICS@@{json.dumps(payload, allow_nan=False)}", file=out_stream, flush=True)
    except Exception:
        pass


def format_duration_hm(total_seconds):
    sec = max(0, round(total_seconds))
    total_mins = sec // 60
    hours = total_mins // 60
    mins = total_mins % 60
    return f"{hours}h {mins:02d}m"


def format_epoch_rate(sec_per_epoch):
    if sec_per_epoch is None or sec_per_epoch <= 0:
        return ""
    if sec_per_epoch < 60:
        return f"~{max(1, round(sec_per_epoch))}s/epoch"
    total_mins = round(sec_per_epoch / 60)
    hours = total_mins // 60
    mins = total_mins % 60
    if hours > 0:
        return f"~{hours}h {mins:02d}m/epoch"
    return f"~{mins}m/epoch"


def format_time_remaining(sec_left):
    if sec_left is None or sec_left <= 0:
        return ""
    if sec_left < 45:
        return f"~{max(1, round(sec_left))}s left"
    total_mins = round(sec_left / 60)
    hours = total_mins // 60
    mins = total_mins % 60
    if hours > 0:
        return f"~{hours}h {mins:02d}m left"
    return f"~{mins}m left"


class ProgressTracker:
    def __init__(self, total_epochs=1):
        self.start_time = time.monotonic()
        self.epoch_start_time = None
        self.recent_epoch_durations = deque(maxlen=5)
        self.current_epoch = 0
        self.total_epochs = max(1, int(total_epochs))

    def start_epoch(self, epoch, total_epochs):
        self.current_epoch = int(epoch)
        self.total_epochs = max(1, int(total_epochs))
        self.epoch_start_time = time.monotonic()

    def finish_epoch(self):
        if self.epoch_start_time is not None:
            dur = time.monotonic() - self.epoch_start_time
            if dur > 0.02:
                self.recent_epoch_durations.append(dur)
            self.epoch_start_time = None

    def elapsed_secs(self):
        return max(0.0, time.monotonic() - self.start_time)

    def avg_epoch_secs(self):
        if not self.recent_epoch_durations:
            return None
        return sum(self.recent_epoch_durations) / len(self.recent_epoch_durations)

    def timing_summary(self, completed_epochs=None):
        if completed_epochs is None:
            completed_epochs = max(0, self.current_epoch - 1)
        parts = [f"elapsed {format_duration_hm(self.elapsed_secs())}"]
        avg = self.avg_epoch_secs()
        if avg is not None and avg > 0:
            rate = format_epoch_rate(avg)
            if rate:
                parts.append(rate)
            remaining = max(0, self.total_epochs - completed_epochs)
            if remaining == 0:
                parts.append("Complete")
            else:
                eta = format_time_remaining(remaining * avg)
                if eta:
                    parts.append(eta)
        else:
            parts.append("estimating...")
        return " | ".join(parts)


class CliSpinner:
    FRAMES = ("|", "/", "-", "\\")

    def __init__(self, tracker):
        self.tracker = tracker
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._active = False
        self._paused = False
        self._last_len = 0
        self._frame_idx = 0
        self._is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._thread = None

    def start(self):
        if not self._is_tty or self._thread is not None:
            return
        self._stop_event.clear()
        self._active = True
        self._thread = threading.Thread(
            target=self._run, name="ls-trainer-cli-spinner", daemon=True
        )
        self._thread.start()

    def _clear_locked(self):
        if self._is_tty and self._last_len > 0:
            sys.stdout.write("\r" + (" " * self._last_len) + "\r")
            sys.stdout.flush()
            self._last_len = 0

    def print_line(self, msg):
        with self._lock:
            self._clear_locked()
            print(msg, flush=True)

    def _run(self):
        while not self._stop_event.wait(0.25):
            with self._lock:
                if not self._active or self._paused or self.tracker.current_epoch <= 0:
                    continue
                frame = self.FRAMES[self._frame_idx % len(self.FRAMES)]
                self._frame_idx += 1
                summary = self.tracker.timing_summary()
                line = (
                    f"[LS-Trainer] {frame} Epoch {self.tracker.current_epoch}/{self.tracker.total_epochs} "
                    f"running... {summary}"
                )
                pad = " " * max(0, self._last_len - len(line))
                sys.stdout.write(f"\r{line}{pad}")
                sys.stdout.flush()
                self._last_len = len(line)

    def stop(self):
        with self._lock:
            self._active = False
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            self._clear_locked()


def _parse_arg_value(args, prefix):
    for a in args:
        if a.startswith(prefix):
            return a[len(prefix):]
    return None


def _is_rank_zero():
    try:
        return int(os.environ.get("RANK", "0")) == 0
    except Exception:
        return True


# Module-level runtime state so Ultralytics callbacks remain top-level functions
# with no closure over threading.Lock/Event (which breaks cloudpickle in DDP).
_RUNTIME = {}


def _ensure_runtime_for_trainer(trainer):
    if "state" not in _RUNTIME:
        epochs = int(getattr(trainer, "epochs", _RUNTIME.get("configured_epochs", 1)))
        _RUNTIME["state"] = {
            "step": 0,
            "last_step_time": time.time(),
            "ema_step_dt": None,
            "last_train_loss": 0.0,
            "last_lr": 0.0,
            "last_sps": 0.0,
        }
        _RUNTIME["configured_epochs"] = epochs
        _RUNTIME["progress_tracker"] = ProgressTracker(epochs)
    return _RUNTIME["state"], _RUNTIME["progress_tracker"]


def _resolve_stop_file(trainer):
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir:
        return os.path.join(str(save_dir), ".stop_requested")
    return _RUNTIME.get("default_stop_file")


def _extract_yolo_train_loss(trainer, state):
    tloss = getattr(trainer, "tloss", None)
    if tloss is not None:
        try:
            if hasattr(tloss, "sum"):
                return _safe_float(tloss.sum())
            return _safe_float(tloss)
        except Exception:
            pass
    loss = getattr(trainer, "loss", None)
    if loss is not None:
        return _safe_float(loss)
    return state["last_train_loss"]


def _extract_yolo_lr(trainer, state):
    opt = getattr(trainer, "optimizer", None)
    if opt is not None and getattr(opt, "param_groups", None):
        try:
            return _safe_float(opt.param_groups[0].get("lr", 0.0))
        except Exception:
            pass
    return state["last_lr"]


def on_train_epoch_start(trainer):
    if not _is_rank_zero():
        return
    state, progress_tracker = _ensure_runtime_for_trainer(trainer)
    ctrl_state = _RUNTIME.get("ctrl_state")
    configured_epochs = _RUNTIME.get("configured_epochs", 1)
    state["step"] = 0
    state["last_step_time"] = time.time()
    total_epochs = max(1, int(getattr(trainer, "epochs", configured_epochs)))
    epoch = min(total_epochs, int(getattr(trainer, "epoch", 0)) + 1)
    progress_tracker.start_epoch(epoch, total_epochs)
    stop_file = _resolve_stop_file(trainer)
    if ctrl_state is not None:
        ctrl_state.wait_if_paused(stop_file)
        if ctrl_state.is_stop_requested(stop_file):
            trainer.stop = True
            raise KeyboardInterrupt("Stop requested by user")
    elif stop_file and os.path.exists(stop_file):
        trainer.stop = True
        raise KeyboardInterrupt("Stop requested by user")


def on_train_batch_end(trainer):
    if not _is_rank_zero():
        return
    state, _ = _ensure_runtime_for_trainer(trainer)
    ctrl_state = _RUNTIME.get("ctrl_state")
    configured_batch = _RUNTIME.get("configured_batch", 4)

    now = time.time()
    dt = max(1e-6, now - state["last_step_time"])
    state["last_step_time"] = now
    if state["ema_step_dt"] is None:
        state["ema_step_dt"] = dt
    else:
        state["ema_step_dt"] = 0.85 * state["ema_step_dt"] + 0.15 * dt

    state["step"] += 1
    step = state["step"]
    train_loader = getattr(trainer, "train_loader", None)
    try:
        total_steps = len(train_loader) if train_loader is not None else step
    except Exception:
        total_steps = step
    total_steps = max(step, int(total_steps))

    total_epochs = max(1, int(getattr(trainer, "epochs", 1)))
    epoch = min(total_epochs, int(getattr(trainer, "epoch", 0)) + 1)
    batch_sz = int(getattr(trainer, "batch_size", configured_batch) or configured_batch)
    sps = batch_sz / dt
    state["last_sps"] = sps

    done_global = (epoch - 1) * total_steps + step
    total_global = max(done_global, total_epochs * total_steps)
    rem_steps = max(0, total_global - done_global)
    eta_secs = rem_steps * (state["ema_step_dt"] or dt)

    train_loss = _extract_yolo_train_loss(trainer, state)
    lr = _extract_yolo_lr(trainer, state)
    state["last_train_loss"] = train_loss
    state["last_lr"] = lr

    emit_metrics(
        epoch=epoch,
        total_epochs=total_epochs,
        step=step,
        total_steps=total_steps,
        train_loss=train_loss,
        val_loss=None,
        map50=None,
        map50_95=None,
        lr=lr,
        samples_per_sec=sps,
        eta_secs=eta_secs,
    )

    stop_file = _resolve_stop_file(trainer)
    if ctrl_state is not None:
        if ctrl_state.consume_checkpoint_request():
            try:
                if hasattr(trainer, "save_model"):
                    trainer.save_model()
                print("[LS-Trainer] Immediate checkpoint saved.", flush=True)
            except Exception as exc:
                print(f"[LS-Trainer Warning] Immediate checkpoint save failed: {exc}", flush=True)
        ctrl_state.wait_if_paused(stop_file)
        if ctrl_state.is_stop_requested(stop_file):
            trainer.stop = True
            try:
                if hasattr(trainer, "save_model"):
                    trainer.save_model()
            except Exception:
                pass
            raise KeyboardInterrupt("Stop requested by user")
    elif stop_file and os.path.exists(stop_file):
        trainer.stop = True
        try:
            if hasattr(trainer, "save_model"):
                trainer.save_model()
        except Exception:
            pass
        raise KeyboardInterrupt("Stop requested by user")


def on_fit_epoch_end(trainer):
    if not _is_rank_zero():
        return
    state, progress_tracker = _ensure_runtime_for_trainer(trainer)
    cli_spinner = _RUNTIME.get("cli_spinner")

    total_epochs = max(1, int(getattr(trainer, "epochs", 1)))
    epoch = min(total_epochs, int(getattr(trainer, "epoch", 0)) + 1)
    train_loader = getattr(trainer, "train_loader", None)
    try:
        total_steps = len(train_loader) if train_loader is not None else state["step"]
    except Exception:
        total_steps = state["step"]
    total_steps = max(1, int(total_steps))

    metrics = getattr(trainer, "metrics", None) or {}
    m50 = None
    for k in ("metrics/mAP50(B)", "metrics/mAP50(M)"):
        v = _safe_opt_float(metrics.get(k))
        if v is not None and (m50 is None or v > m50):
            m50 = v

    m50_95 = None
    for k in ("metrics/mAP50-95(B)", "metrics/mAP50-95(M)"):
        v = _safe_opt_float(metrics.get(k))
        if v is not None and (m50_95 is None or v > m50_95):
            m50_95 = v

    val_losses = [
        _safe_opt_float(v)
        for k, v in metrics.items()
        if str(k).startswith("val/") and str(k).endswith("loss")
    ]
    val_losses = [v for v in val_losses if v is not None]
    val_loss = sum(val_losses) if val_losses else None

    rem_epochs = max(0, total_epochs - epoch)
    avg_dt = state["ema_step_dt"] or 0.0
    eta_secs = rem_epochs * total_steps * avg_dt

    emit_metrics(
        epoch=epoch,
        total_epochs=total_epochs,
        step=total_steps,
        total_steps=total_steps,
        train_loss=state["last_train_loss"],
        val_loss=val_loss,
        map50=m50,
        map50_95=m50_95,
        lr=state["last_lr"],
        samples_per_sec=state["last_sps"],
        eta_secs=eta_secs,
    )
    progress_tracker.finish_epoch()
    timing = progress_tracker.timing_summary(epoch)
    m50_str = f"{m50:.4f}" if m50 is not None else "0.0000"
    m95_str = f"{m50_95:.4f}" if m50_95 is not None else "0.0000"
    msg = (
        f"[LS-Trainer Metrics] Epoch {epoch}/{total_epochs} | mAP50={m50_str} | mAP50-95={m95_str} | {timing}"
    )
    if cli_spinner is not None:
        cli_spinner.print_line(msg)
    else:
        print(msg, flush=True)


def normalize_device(device_str, arch="yolo"):
    if not device_str:
        return None
    # This list conversion is YOLO/Ultralytics-specific; RF-DETR and RT-DETR keep the original string.
    if arch == "yolo" and not device_str.strip().startswith("[") and "xpu" in device_str and "," in device_str:
        parts = [p.strip() for p in device_str.split(",") if p.strip()]
        normalized = []
        for p in parts:
            if p.isdigit():
                normalized.append(f"xpu:{p}")
            else:
                normalized.append(p)
        return normalized
    return device_str


_STRING_TRAIN_KEYS = frozenset({
    "data", "project", "name", "model", "task", "cfg", "pretrained", "optimizer",
})

# Set from the main thread's signal handler. A plain flag avoids taking the
# stdin-control lock inside the handler.
_STOP_REQUESTED = {"flag": False}


def _is_xpu_device_token(token):
    text = str(token).strip().lower()
    return text == "xpu" or text.startswith("xpu:")


def _parse_device_list_literal(text):
    stripped = str(text).strip()
    if not (stripped.startswith("[") and stripped.endswith("]")):
        return None
    try:
        value = ast.literal_eval(stripped)
    except (SyntaxError, ValueError):
        return None
    if isinstance(value, (list, tuple)) and value and all(isinstance(item, str) for item in value):
        return [str(item) for item in value]
    return None


def yolo_multi_xpu_devices(device_str, arch="yolo"):
    """Return a multi-XPU device list for YOLO, or None when the normal launcher should run.

    RF-DETR and RT-DETR keep the original device string. A single XPU, CUDA list, or CPU
    request also stays on the in-process Ultralytics entrypoint.
    """
    if arch != "yolo" or device_str is None:
        return None
    text = str(device_str).strip()
    if not text:
        return None
    if text.startswith("["):
        devices = _parse_device_list_literal(text)
    else:
        normalized = normalize_device(text, arch)
        devices = normalized if isinstance(normalized, list) else None
    if not devices or len(devices) < 2:
        return None
    if not all(_is_xpu_device_token(device) for device in devices):
        return None
    return devices


def standalone_multi_xpu_devices_from_args(cli_args, arch="yolo"):
    device_arg = _parse_arg_value(cli_args, "device=") or ""
    return yolo_multi_xpu_devices(device_arg, arch)


def _coerce_cli_value(key, raw):
    text = str(raw).strip()
    if key in _STRING_TRAIN_KEYS:
        return text
    lower = text.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in ("none", "null"):
        return None
    if text.isdigit() or (text[:1] in "+-" and text[1:].isdigit()):
        return int(text)
    try:
        return float(text)
    except ValueError:
        return text


def _train_kwargs_from_cli(cli_args, device_list):
    kwargs = {}
    for arg in cli_args:
        if "=" not in arg:
            continue
        key, raw = arg.split("=", 1)
        if key in ("model", "device"):
            continue
        kwargs[key] = _coerce_cli_value(key, raw)
    kwargs["device"] = [str(device) for device in device_list]
    return kwargs


MULTI_XPU_METRICS_PATH = "/tmp/ls-trainer-metrics.json"

# Callback source inlined into the standalone script. It must stay free of this
# module's control-thread locks: Ultralytics cloudpickles model callbacks into DDP workers.
_STANDALONE_METRICS_PY = r'''
import json
import time

_LS_METRICS_PATH = "/tmp/ls-trainer-metrics.json"
_LS_XPU_COUNT = @@XPU_COUNT@@
_LS_METRICS_INTERVAL = 5.0


def _ls_remove_metrics_file():
    for suffix in ("", ".tmp"):
        try:
            os.remove(_LS_METRICS_PATH + suffix)
        except OSError:
            pass


def _ls_rank0():
    for key in ("RANK", "LOCAL_RANK"):
        raw = os.environ.get(key)
        if raw is None or raw == "":
            continue
        try:
            rank = int(raw)
        except (TypeError, ValueError):
            return False
        if rank not in (-1, 0):
            return False
    return True


def _ls_finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _ls_metric(metrics, keys):
    if not isinstance(metrics, dict):
        return None
    found = None
    for key in keys:
        number = _ls_finite(metrics.get(key))
        if number is None:
            continue
        if found is None or number > found:
            found = number
    return found


def _ls_xe_idle_paths(xpu):
    cached = getattr(_ls_xe_idle_paths, "value", None)
    if cached is not None:
        return cached
    cards = []
    try:
        entries = os.listdir("/sys/class/drm")
    except OSError:
        entries = []
    for name in entries:
        if not name.startswith("card") or not name[4:].isdigit():
            continue
        idle = "/sys/class/drm/" + name + "/device/tile0/gt0/gtidle/idle_residency_ms"
        if not os.path.isfile(idle):
            continue
        bus = None
        try:
            bdf = os.path.basename(os.path.realpath("/sys/class/drm/" + name + "/device"))
            bus = int(bdf.split(":")[1], 16)
        except (OSError, ValueError, IndexError):
            bus = None
        cards.append((int(name[4:]), bus, idle))
    cards.sort()
    paths = []
    for index in range(int(_LS_XPU_COUNT)):
        bus = None
        try:
            uuid = str(getattr(xpu.get_device_properties(index), "uuid", ""))
            bus = int(uuid.split("-")[3][:2], 16)
        except Exception:
            bus = None
        chosen = None
        if bus is not None:
            for _card_index, card_bus, idle in cards:
                if card_bus == bus:
                    chosen = idle
                    break
        if chosen is None and index < len(cards):
            chosen = cards[index][2]
        paths.append(chosen)
    _ls_xe_idle_paths.value = paths
    return paths


def _ls_read_int_file(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _ls_fill_xe_util(xpu, busy):
    # torch.xpu.utilization needs Level Zero engine groups. These Arc B70s do not
    # expose them, so the render-engine idle counter is the reading that moves.
    try:
        paths = _ls_xe_idle_paths(xpu)
    except Exception:
        return busy
    need = [
        index
        for index, value in enumerate(busy)
        if value is None and index < len(paths) and paths[index]
    ]
    if not need:
        return busy
    started = time.monotonic()
    before = {index: _ls_read_int_file(paths[index]) for index in need}
    time.sleep(0.1)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    if elapsed_ms <= 0:
        return busy
    for index in need:
        idle0 = before.get(index)
        idle1 = _ls_read_int_file(paths[index])
        if idle0 is None or idle1 is None or idle1 < idle0:
            continue
        pct = 100.0 * (1.0 - ((idle1 - idle0) / elapsed_ms))
        if pct < 0.0:
            pct = 0.0
        elif pct > 100.0:
            pct = 100.0
        busy[index] = int(round(pct))
    return busy


def _ls_device_stats():
    vram_mb = []
    util_pct = []
    gpu_index = []
    try:
        import torch
        xpu = getattr(torch, "xpu", None)
    except Exception:
        return vram_mb, util_pct, gpu_index
    if xpu is None:
        return vram_mb, util_pct, gpu_index
    for index in range(int(_LS_XPU_COUNT)):
        allocated_mb = None
        info_mb = None
        try:
            allocated_mb = int(xpu.memory_allocated(index) / (1024 * 1024))
        except Exception:
            allocated_mb = None
        # memory_allocated is per-process, so another DDP rank's card stays at 0.
        # mem_get_info is device-wide and visible from rank 0.
        try:
            free_bytes, total_bytes = xpu.mem_get_info(index)
            info_mb = int(max(0, int(total_bytes) - int(free_bytes)) / (1024 * 1024))
        except Exception:
            info_mb = None
        if allocated_mb is None and info_mb is None:
            used = None
        else:
            used = max(value for value in (allocated_mb, info_mb) if value is not None)
        busy = None
        try:
            busy = int(round(float(xpu.utilization(index))))
        except Exception:
            busy = None
        vram_mb.append(used)
        util_pct.append(busy)
        gpu_index.append(index)
    if any(value is None for value in util_pct):
        _ls_fill_xe_util(xpu, util_pct)
    kept_vram = []
    kept_util = []
    kept_index = []
    for index, used, busy in zip(gpu_index, vram_mb, util_pct):
        if used is None and busy is None:
            continue
        kept_index.append(index)
        kept_vram.append(used)
        kept_util.append(busy)
    return kept_vram, kept_util, kept_index


def _ls_write_metrics(trainer, force=False):
    try:
        if not _ls_rank0():
            return
        now = time.time()
        last = getattr(_ls_write_metrics, "last_write", 0.0)
        if (not force) and last and (now - last) < _LS_METRICS_INTERVAL:
            return
        epoch = getattr(_ls_write_metrics, "epoch", 0)
        raw_epoch = getattr(trainer, "epoch", None) if trainer is not None else None
        if raw_epoch is not None:
            try:
                epoch = int(raw_epoch) + 1
            except (TypeError, ValueError):
                pass
            total = getattr(trainer, "epochs", None)
            try:
                total = int(total) if total is not None else 0
                if total > 0 and epoch > total:
                    epoch = total
            except (TypeError, ValueError):
                pass
            _ls_write_metrics.epoch = epoch
        metrics = getattr(trainer, "metrics", None) if trainer is not None else None
        map50 = _ls_metric(metrics, ("metrics/mAP50(B)", "metrics/mAP50(M)"))
        map95 = _ls_metric(metrics, ("metrics/mAP50-95(B)", "metrics/mAP50-95(M)"))
        if map50 is not None:
            prev50 = getattr(_ls_write_metrics, "best50", None)
            best50 = map50 if prev50 is None else max(prev50, map50)
            _ls_write_metrics.best50 = best50
        else:
            best50 = getattr(_ls_write_metrics, "best50", None)
        if map95 is not None:
            headline = map95
        elif map50 is not None:
            headline = map50
        else:
            headline = getattr(_ls_write_metrics, "headline", None)
        if map95 is not None or map50 is not None:
            _ls_write_metrics.headline = headline
            prev_best = getattr(_ls_write_metrics, "best_headline", None)
            best_headline = headline if prev_best is None else max(prev_best, headline)
            _ls_write_metrics.best_headline = best_headline
        else:
            best_headline = getattr(_ls_write_metrics, "best_headline", None)
        vram_mb, util_pct, gpu_index = _ls_device_stats()
        payload = {
            "epoch": int(epoch),
            "best_map": best_headline,
            "map": headline,
            "map50": map50 if map50 is not None else getattr(_ls_write_metrics, "map50", None),
            "best_map50": best50,
            "map50_95": map95 if map95 is not None else getattr(_ls_write_metrics, "map95", None),
            "vram_mb": vram_mb,
            "util_pct": util_pct,
            "gpu_index": gpu_index,
        }
        if map50 is not None:
            _ls_write_metrics.map50 = map50
        if map95 is not None:
            _ls_write_metrics.map95 = map95
        text = json.dumps(payload, allow_nan=False) + "\n"
        tmp = _LS_METRICS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, _LS_METRICS_PATH)
        _ls_write_metrics.last_write = now
    except Exception:
        return


def _ls_on_train_batch_end(trainer):
    _ls_write_metrics(trainer, force=False)


def _ls_on_val_batch_end(validator):
    _ls_write_metrics(validator, force=False)


def _ls_on_fit_epoch_end(trainer):
    _ls_write_metrics(trainer, force=True)


def _ls_on_train_end(trainer):
    _ls_write_metrics(trainer, force=True)
'''


def _remove_multi_xpu_metrics_file():
    for suffix in ("", ".tmp"):
        try:
            os.remove(MULTI_XPU_METRICS_PATH + suffix)
        except OSError:
            pass


def build_standalone_yolo_script(cli_args, device_list):
    """Short fresh-process trainer. It must not import this module or touch its locks."""
    model = _parse_arg_value(cli_args, "model=") or ""
    train_kwargs = _train_kwargs_from_cli(cli_args, device_list)
    metrics = _STANDALONE_METRICS_PY.replace("@@XPU_COUNT@@", str(len(device_list)))
    return (
        "import os\n"
        "import signal\n"
        "\n"
        "# Die with the parent if the TUI is killed before it can signal this process.\n"
        "if os.name == 'posix':\n"
        "    try:\n"
        "        import ctypes\n"
        "        ctypes.CDLL(None).prctl(1, int(signal.SIGTERM))\n"
        "    except Exception:\n"
        "        pass\n"
        "\n"
        f"{metrics}"
        "from ultralytics import YOLO\n"
        "\n"
        "if __name__ == '__main__':\n"
        f"    model = YOLO({model!r})\n"
        "    model.add_callback('on_train_batch_end', _ls_on_train_batch_end)\n"
        "    model.add_callback('on_val_batch_end', _ls_on_val_batch_end)\n"
        "    model.add_callback('on_fit_epoch_end', _ls_on_fit_epoch_end)\n"
        "    model.add_callback('on_train_end', _ls_on_train_end)\n"
        "    try:\n"
        f"        model.train(**{train_kwargs!r})\n"
        "    finally:\n"
        "        _ls_remove_metrics_file()\n"
    )


def _terminate_process_group(proc):
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.terminate()
        except Exception:
            pass


def _stop_child_process(proc):
    _terminate_process_group(proc)
    try:
        proc.wait(timeout=8)
        return
    except subprocess.TimeoutExpired:
        pass
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.kill()
            except Exception:
                pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def _install_stop_signals():
    def _handle(_signum, _frame):
        _STOP_REQUESTED["flag"] = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _handle)
        except Exception:
            pass


def launch_standalone_yolo_multi_xpu(cli_args, device_list):
    """Run model.train in a new interpreter so Ultralytics DDP does not pickle this process.

    The TUI parents this module with a piped stdin and a control thread. Those locks are not
    picklable, and Ultralytics `generate_ddp_file` cloudpickles the trainer from whichever
    process calls `model.train`. The child starts with stdin closed and no inherited control state.
    """
    _STOP_REQUESTED["flag"] = False
    _remove_multi_xpu_metrics_file()
    script = build_standalone_yolo_script(cli_args, device_list)
    script_fd, script_path = tempfile.mkstemp(prefix="ls-trainer-yolo-xpu-", suffix=".py")
    os.close(script_fd)
    proc = None
    reader = None
    try:
        with open(script_path, "w", encoding="utf-8") as handle:
            handle.write(script)
        print(
            "[LS-Trainer] Multi-XPU YOLO standalone script "
            f"{script_path} device={device_list}",
            flush=True,
        )
        print(
            f"[LS-Trainer] Multi-XPU metrics file {MULTI_XPU_METRICS_PATH}",
            flush=True,
        )
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("PYTHONIOENCODING", "utf-8")
        proc = subprocess.Popen(
            [sys.executable, "-u", script_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        ctrl_state = TrainingControlState()
        start_stdin_control_listener(ctrl_state)
        _install_stop_signals()
        project_dir = _parse_arg_value(cli_args, "project=")
        run_name = _parse_arg_value(cli_args, "name=")
        stop_file = (
            os.path.join(project_dir, run_name, ".stop_requested")
            if (project_dir and run_name)
            else None
        )

        def _reader():
            try:
                if proc.stdout is None:
                    return
                for line in proc.stdout:
                    sys.stdout.write(line)
                    sys.stdout.flush()
            except Exception:
                pass

        reader = threading.Thread(target=_reader, name="ls-trainer-xpu-stdout", daemon=True)
        reader.start()

        announced_stop = False
        while proc.poll() is None:
            stop_now = _STOP_REQUESTED["flag"] or ctrl_state.is_stop_requested(stop_file)
            if stop_now:
                if not announced_stop:
                    announced_stop = True
                    print(
                        "[LS-Trainer] Stop requested; terminating multi-XPU training.",
                        flush=True,
                    )
                _stop_child_process(proc)
                break
            try:
                time.sleep(0.2)
            except InterruptedError:
                _STOP_REQUESTED["flag"] = True

        code = proc.wait()
        return 1 if code is None else code
    finally:
        if proc is not None and proc.poll() is None:
            _stop_child_process(proc)
        if reader is not None:
            reader.join(timeout=2)
        if proc is not None and proc.stdout is not None:
            try:
                proc.stdout.close()
            except Exception:
                pass
        try:
            os.remove(script_path)
        except OSError:
            pass
        _remove_multi_xpu_metrics_file()


def main():
    cli_args = sys.argv[1:]
    multi_xpu = standalone_multi_xpu_devices_from_args(cli_args, arch="yolo")
    if multi_xpu:
        device_arg = _parse_arg_value(cli_args, "device=") or ""
        print(
            f"[LS-Trainer] Multi-GPU: {device_arg} ({len(multi_xpu)} devices)",
            flush=True,
        )
        code = launch_standalone_yolo_multi_xpu(cli_args, multi_xpu)
        if code:
            sys.exit(code if code > 0 else 128 + abs(code))
        return

    ctrl_state = TrainingControlState()
    start_stdin_control_listener(ctrl_state)

    project_dir = _parse_arg_value(cli_args, "project=")
    run_name = _parse_arg_value(cli_args, "name=")
    batch_arg = _parse_arg_value(cli_args, "batch=")
    device_arg = _parse_arg_value(cli_args, "device=") or ""
    if "," in device_arg:
        dev_parts = [p.strip() for p in device_arg.split(",") if p.strip()]
        if len(dev_parts) > 1:
            print(f"[LS-Trainer] Multi-GPU: {device_arg} ({len(dev_parts)} devices)", flush=True)

    try:
        configured_batch = max(1, int(batch_arg)) if batch_arg else 4
    except Exception:
        configured_batch = 4

    default_stop_file = (
        os.path.join(project_dir, run_name, ".stop_requested")
        if (project_dir and run_name)
        else None
    )

    state = {
        "step": 0,
        "last_step_time": time.time(),
        "ema_step_dt": None,
        "last_train_loss": 0.0,
        "last_lr": 0.0,
        "last_sps": 0.0,
    }
    epochs_arg = _parse_arg_value(cli_args, "epochs=")
    try:
        configured_epochs = max(1, int(epochs_arg)) if epochs_arg else 1
    except Exception:
        configured_epochs = 1
    progress_tracker = ProgressTracker(configured_epochs)
    cli_spinner = CliSpinner(progress_tracker)

    _RUNTIME.clear()
    _RUNTIME.update({
        "ctrl_state": ctrl_state,
        "progress_tracker": progress_tracker,
        "cli_spinner": cli_spinner,
        "state": state,
        "configured_epochs": configured_epochs,
        "configured_batch": configured_batch,
        "default_stop_file": default_stop_file,
    })

    try:
        from ultralytics.utils import callbacks as ul_callbacks
        from ultralytics.utils.callbacks import base as ul_cb_base

        for target_dict in (
            getattr(ul_callbacks, "default_callbacks", None),
            getattr(ul_cb_base, "default_callbacks", None),
        ):
            if isinstance(target_dict, dict):
                for event_name, fn in (
                    ("on_train_epoch_start", on_train_epoch_start),
                    ("on_train_batch_end", on_train_batch_end),
                    ("on_fit_epoch_end", on_fit_epoch_end),
                ):
                    cb_list = target_dict.setdefault(event_name, [])
                    if fn not in cb_list:
                        cb_list.append(fn)
    except Exception:
        pass

    normalized_cli_args = []
    for arg in cli_args:
        if arg.startswith("device="):
            raw_dev = arg[len("device="):]
            norm_dev = normalize_device(raw_dev, "yolo")
            if isinstance(norm_dev, list):
                dev_literal = "[" + ",".join(f"'{d}'" for d in norm_dev) + "]"
                normalized_cli_args.append(f"device={dev_literal}")
            elif norm_dev is not None:
                normalized_cli_args.append(f"device={norm_dev}")
        else:
            normalized_cli_args.append(arg)

    sys.argv = ["yolo"] + normalized_cli_args
    cli_spinner.start()
    try:
        from ultralytics.cfg import entrypoint
        entrypoint()
    except KeyboardInterrupt:
        cli_spinner.stop()
        print("[LS-Trainer] Training interrupted by user.", flush=True)
    finally:
        cli_spinner.stop()


def run_self_checks():
    assert yolo_multi_xpu_devices("xpu:0,1", "yolo") == ["xpu:0", "xpu:1"]
    assert yolo_multi_xpu_devices("xpu:0, xpu:1", "yolo") == ["xpu:0", "xpu:1"]
    assert yolo_multi_xpu_devices("['xpu:0','xpu:1']", "yolo") == ["xpu:0", "xpu:1"]
    assert yolo_multi_xpu_devices("xpu", "yolo") is None
    assert yolo_multi_xpu_devices("xpu:0", "yolo") is None
    assert yolo_multi_xpu_devices("0,1", "yolo") is None
    assert yolo_multi_xpu_devices("cuda", "yolo") is None
    assert yolo_multi_xpu_devices("cpu", "yolo") is None
    assert yolo_multi_xpu_devices("xpu:0,1", "rf-detr") is None
    assert yolo_multi_xpu_devices("xpu:0,1", "rt-detr") is None
    assert standalone_multi_xpu_devices_from_args(["device=xpu:0"], "yolo") is None
    assert standalone_multi_xpu_devices_from_args(["device=xpu:0,1"], "yolo") == ["xpu:0", "xpu:1"]
    assert standalone_multi_xpu_devices_from_args(
        ["device=['xpu:0','xpu:1']"], "yolo"
    ) == ["xpu:0", "xpu:1"]

    script = build_standalone_yolo_script(
        [
            "detect",
            "train",
            "task=detect",
            "data=/tmp/data.yaml",
            "model=/tmp/yolo11n.pt",
            "epochs=1",
            "batch=2",
            "imgsz=64",
            "workers=0",
            "device=['xpu:0','xpu:1']",
            "amp=False",
            "mosaic=0",
            "scale=0.05",
            "degrees=180",
            "name=run",
            "project=/tmp/runs",
        ],
        ["xpu:0", "xpu:1"],
    )
    compile(script, "<standalone>", "exec")
    assert "model.train(**" in script
    assert "['xpu:0', 'xpu:1']" in script
    assert "YOLO('/tmp/yolo11n.pt')" in script
    assert "'epochs': 1" in script
    assert "'amp': False" in script
    assert "'scale': 0.05" in script
    assert "'degrees': 180" in script
    assert "TrainingControlState" not in script
    assert "threading" not in script
    assert "stdin" not in script
    assert "_RUNTIME" not in script
    assert "/tmp/ls-trainer-metrics.json" in script
    assert "model.add_callback(" in script
    assert "memory_allocated" in script
    assert "utilization" in script
    assert "os.replace" in script

    namespace = {"__name__": "ls_trainer_metrics_check"}
    exec(compile(script, "<standalone>", "exec"), namespace)
    import cloudpickle

    for callback_name in (
        "_ls_on_train_batch_end",
        "_ls_on_val_batch_end",
        "_ls_on_fit_epoch_end",
        "_ls_on_train_end",
    ):
        cloudpickle.dumps(namespace[callback_name])

    class _FakeTrainer:
        epoch = 3
        epochs = 4
        metrics = {
            "metrics/mAP50(B)": 0.81,
            "metrics/mAP50-95(B)": 0.55,
        }

    saved_rank = os.environ.pop("RANK", None)
    saved_local_rank = os.environ.pop("LOCAL_RANK", None)
    try:
        namespace["_ls_on_fit_epoch_end"](_FakeTrainer())
        with open(MULTI_XPU_METRICS_PATH, encoding="utf-8") as handle:
            payload = json.load(handle)
        assert payload["epoch"] == 4
        assert payload["map"] == 0.55
        assert payload["best_map"] == 0.55
        assert payload["map50"] == 0.81
        assert isinstance(payload["vram_mb"], list)
        assert isinstance(payload["util_pct"], list)
        assert isinstance(payload["gpu_index"], list)
    finally:
        if saved_rank is not None:
            os.environ["RANK"] = saved_rank
        if saved_local_rank is not None:
            os.environ["LOCAL_RANK"] = saved_local_rank
        _remove_multi_xpu_metrics_file()
    print("yolo multi-xpu self-checks ok")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    if os.environ.get("LS_TRAINER_YOLO_SELFTEST") == "1":
        run_self_checks()
    else:
        try:
            main()
        except Exception:
            traceback.print_exc()
            sys.exit(1)
