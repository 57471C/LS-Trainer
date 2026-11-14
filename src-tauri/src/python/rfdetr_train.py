from collections import deque
import json
import math
import multiprocessing
import os
import sys
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
                try:
                    used_bytes = max(
                        int(torch.xpu.memory_reserved(idx)),
                        int(torch.xpu.memory_allocated(idx)),
                    )
                except Exception:
                    pass
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
        print(f"@@LS_METRICS@@{json.dumps(payload, allow_nan=False)}", file=out_stream, flush=True)
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

    def set_paused(self, paused):
        with self._lock:
            self._paused = paused
            if paused:
                self._clear_locked()

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


def detect_num_workers(cfg):
    if "num_workers" in cfg and cfg["num_workers"] is not None:
        return max(0, min(8, int(cfg["num_workers"])))
    phys = None
    try:
        import psutil
        phys = psutil.cpu_count(logical=False)
    except Exception:
        phys = None
    logical = os.cpu_count() or 2
    base_cores = phys if (phys is not None and phys > 0) else max(1, logical // 2)
    return min(8, max(1, base_cores - 1))


def select_training_device(requested):
    import torch
    req = str(requested or "auto").strip().lower()
    xpu_avail = bool(hasattr(torch, "xpu") and torch.xpu.is_available())
    cuda_avail = bool(torch.cuda.is_available())

    if req == "cpu":
        print(
            "[LS-Trainer] RF-DETR device selection: CPU requested -> device=cpu (accelerator=cpu, devices=1)",
            flush=True,
        )
        return "cpu"

    if req in ("", "auto", "gpu", "0", "xpu", "xpu:0", "cuda", "cuda:0"):
        if xpu_avail and req not in ("cuda", "cuda:0"):
            dev_name = "Intel Arc XPU"
            dev_count = 1
            try:
                dev_count = torch.xpu.device_count()
                dev_name = torch.xpu.get_device_name(0)
            except Exception:
                pass
            count_info = f" ({dev_count} device(s) detected)" if dev_count > 1 else ""
            print(
                f"[LS-Trainer] RF-DETR device selection: torch.xpu.is_available()=True -> using Intel Arc XPU device=xpu ({dev_name}{count_info})",
                flush=True,
            )
            return "xpu"
        if cuda_avail:
            dev_name = "CUDA GPU"
            try:
                dev_name = torch.cuda.get_device_name(0)
            except Exception:
                pass
            print(
                f"[LS-Trainer] RF-DETR device selection: torch.cuda.is_available()=True -> using NVIDIA CUDA device=cuda ({dev_name})",
                flush=True,
            )
            return "cuda"
        print(
            "[LS-Trainer] RF-DETR device selection: neither Intel Arc XPU nor CUDA is available -> falling back to device=cpu",
            flush=True,
        )
        return "cpu"

    if req.startswith("xpu"):
        dev_name = "Intel Arc XPU"
        try:
            idx = int(req.split(":")[1]) if ":" in req else 0
            dev_name = torch.xpu.get_device_name(idx)
        except Exception:
            pass
        print(
            f"[LS-Trainer] RF-DETR device selection: using specified Intel Arc device={requested} ({dev_name})",
            flush=True,
        )
        return str(requested)

    print(f"[LS-Trainer] RF-DETR device selection: using custom device={requested}", flush=True)
    return str(requested)


def main():
    cfg_path = sys.argv[1]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    ctrl_state = TrainingControlState()
    start_stdin_control_listener(ctrl_state)

    models_dir = cfg["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.chdir(models_dir)

    try:
        import torch_linear_assignment
    except ImportError:
        print(
            "[LS-Trainer Error] Missing required dependency 'torch-linear-assignment' needed for RF-DETR Hungarian matching. "
            "Please install it in your environment: pip install torch-linear-assignment",
            flush=True,
        )
        sys.exit(1)

    import torch
    import rfdetr
    import rfdetr.detr as rf_detr_mod
    import rfdetr.training as rf_training
    import rfdetr.training.trainer as rf_trainer_mod
    from pytorch_lightning import Callback
    from pytorch_lightning.accelerators import Accelerator
    from pytorch_lightning.strategies import SingleDeviceStrategy

    try:
        import rfdetr.utilities.console as rf_console
        from rich.console import Console
        rf_console._get_rich_console = lambda _trainer: Console(
            force_terminal=False,
            legacy_windows=False,
            color_system=None,
        )
    except Exception:
        pass

    class_name = cfg["rfdetr_class"]
    dataset_dir = cfg["dataset_dir"]
    output_dir = cfg["output_dir"]
    epochs = int(cfg["epochs"])
    batch_size = max(1, int(cfg["batch_size"]))
    resolution = int(cfg["resolution"])
    device = select_training_device(cfg.get("device", "auto"))
    pretrain_weights = cfg["pretrain_weights"]
    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    num_workers = detect_num_workers(cfg)

    os.makedirs(output_dir, exist_ok=True)
    stop_file = os.path.join(output_dir, ".stop_requested")

    class XPUAccelerator(Accelerator):
        """PyTorch Lightning Accelerator for Intel Arc / XPU devices."""
        def setup_device(self, dev):
            if dev.type != "xpu":
                raise ValueError(f"Expected xpu device, got {dev}")
            torch.xpu.set_device(dev)

        def get_device_stats(self, dev):
            return {}

        def teardown(self):
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                torch.xpu.empty_cache()

        @staticmethod
        def parse_devices(devices):
            return [torch.device("xpu", 0)]

        @staticmethod
        def get_parallel_devices(devices):
            return [torch.device("xpu", 0)]

        @staticmethod
        def auto_device_count():
            return torch.xpu.device_count() if (hasattr(torch, "xpu") and torch.xpu.is_available()) else 0

        @staticmethod
        def is_available():
            return bool(hasattr(torch, "xpu") and torch.xpu.is_available())

        @staticmethod
        def name():
            return "xpu"

    orig_resolve_device_kwargs = getattr(rf_detr_mod.RFDETR, "_resolve_trainer_device_kwargs", None)
    if orig_resolve_device_kwargs is not None:
        @staticmethod
        def patched_resolve_device_kwargs(dev, *args, **kwargs):
            if dev is not None and str(dev).strip().lower().startswith("xpu"):
                return "auto", 1
            return orig_resolve_device_kwargs(dev, *args, **kwargs)
        rf_detr_mod.RFDETR._resolve_trainer_device_kwargs = patched_resolve_device_kwargs

    best_state = {"map50": 0.0, "map50_95": 0.0}
    progress_tracker = ProgressTracker(epochs)
    cli_spinner = CliSpinner(progress_tracker)
    step_state = {
        "last_time": time.time(),
        "ema_dt": None,
        "last_loss": 0.0,
        "last_lr": 0.0,
        "last_sps": 0.0,
        "last_total_steps": 1,
    }

    def to_float(v):
        return _safe_opt_float(v)

    class LSTrainerProgressCallback(Callback):
        def on_fit_start(self, trainer, pl_module):
            dm = getattr(trainer, "datamodule", None)
            if dm is not None and hasattr(dm, "_num_workers"):
                dm._num_workers = num_workers
            root_dev = getattr(getattr(trainer, "strategy", None), "root_device", device)
            accel_name = type(getattr(trainer, "accelerator", None)).__name__
            print(
                f"[LS-Trainer] Lightning Trainer active: accelerator={accel_name}, root_device={root_dev}, train/val num_workers={num_workers}",
                flush=True,
            )

        def _check_stop_requested(self, trainer):
            ctrl_state.wait_if_paused(stop_file)
            if ctrl_state.is_stop_requested(stop_file):
                if not os.path.exists(stop_file):
                    try:
                        with open(stop_file, "w", encoding="utf-8") as sf:
                            sf.write("stop")
                    except Exception:
                        pass
                print("[LS-Trainer] Stop requested by user. Finishing current step...", flush=True)
                trainer.should_stop = True
                return True
            return False

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            now = time.time()
            dt = max(1e-6, now - step_state["last_time"])
            step_state["last_time"] = now
            if step_state["ema_dt"] is None:
                step_state["ema_dt"] = dt
            else:
                step_state["ema_dt"] = 0.85 * step_state["ema_dt"] + 0.15 * dt

            ep = int(trainer.current_epoch) + 1
            total_ep = int(trainer.max_epochs)
            step = int(batch_idx) + 1
            raw_total = getattr(trainer, "num_training_batches", 0)
            try:
                total_steps = max(step, int(raw_total)) if raw_total and math.isfinite(float(raw_total)) else step
            except Exception:
                total_steps = step
            step_state["last_total_steps"] = total_steps

            loss_val = None
            if isinstance(outputs, dict) and "loss" in outputs:
                loss_val = _safe_opt_float(outputs.get("loss"))
            elif outputs is not None:
                loss_val = _safe_opt_float(outputs)
            if loss_val is None:
                cb_metrics = getattr(trainer, "callback_metrics", None) or {}
                for k in ("train/loss", "train_loss", "loss"):
                    loss_val = _safe_opt_float(cb_metrics.get(k))
                    if loss_val is not None:
                        break
            train_loss = loss_val if loss_val is not None else step_state["last_loss"]
            step_state["last_loss"] = train_loss

            lr = step_state["last_lr"]
            try:
                opts = getattr(trainer, "optimizers", None) or []
                if opts and getattr(opts[0], "param_groups", None):
                    lr = _safe_float(opts[0].param_groups[0].get("lr", lr), lr)
            except Exception:
                pass
            step_state["last_lr"] = lr

            sps = batch_size / dt
            step_state["last_sps"] = sps
            done_global = (ep - 1) * total_steps + step
            total_global = max(done_global, total_ep * total_steps)
            rem_steps = max(0, total_global - done_global)
            eta_secs = rem_steps * (step_state["ema_dt"] or dt)

            emit_metrics(
                epoch=ep,
                total_epochs=total_ep,
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

            if ctrl_state.consume_checkpoint_request():
                try:
                    ckpt_path = os.path.join(output_dir, "checkpoint.pth")
                    trainer.save_checkpoint(ckpt_path)
                    print(f"[LS-Trainer] Immediate checkpoint saved to: {ckpt_path}", flush=True)
                except Exception as exc:
                    print(f"[LS-Trainer Warning] Immediate checkpoint save failed: {exc}", flush=True)

            self._check_stop_requested(trainer)

        def on_train_epoch_end(self, trainer, pl_module):
            progress_tracker.finish_epoch()
            self._check_stop_requested(trainer)

        def on_train_epoch_start(self, trainer, pl_module):
            step_state["last_time"] = time.time()
            self._check_stop_requested(trainer)
            ep = int(trainer.current_epoch) + 1
            total = int(trainer.max_epochs)
            progress_tracker.start_epoch(ep, total)
            timing = progress_tracker.timing_summary(max(0, ep - 1))
            cli_spinner.print_line(
                f"Epoch {ep}/{total} - RF-DETR ({class_name}) training on {device}... | {timing}"
            )

        def on_validation_epoch_end(self, trainer, pl_module):
            if getattr(trainer, "sanity_checking", False):
                return
            ep = int(trainer.current_epoch) + 1
            total = int(trainer.max_epochs)
            metrics = trainer.callback_metrics or {}

            m50 = None
            for k in ("val/segm_mAP_50", "val/ema_segm_mAP_50", "val/mAP_50", "val/ema_mAP_50"):
                val = to_float(metrics.get(k))
                if val is not None and (m50 is None or val > m50):
                    m50 = val

            m50_95 = None
            for k in ("val/segm_mAP_50_95", "val/ema_segm_mAP_50_95", "val/mAP_50_95", "val/ema_mAP_50_95"):
                val = to_float(metrics.get(k))
                if val is not None and (m50_95 is None or val > m50_95):
                    m50_95 = val

            val_loss = None
            for k in ("val/loss", "val_loss"):
                v = to_float(metrics.get(k))
                if v is not None:
                    val_loss = v
                    break

            if m50 is not None and m50 > best_state["map50"]:
                best_state["map50"] = m50
            if m50_95 is not None and m50_95 > best_state["map50_95"]:
                best_state["map50_95"] = m50_95

            m50_disp = m50 if m50 is not None else best_state["map50"]
            m95_disp = m50_95 if m50_95 is not None else best_state["map50_95"]
            progress_tracker.finish_epoch()
            timing = progress_tracker.timing_summary(ep)
            cli_spinner.print_line(
                f"[LS-Trainer Metrics] Epoch {ep}/{total} | mAP50={m50_disp:.4f} | mAP50-95={m95_disp:.4f} | {timing}"
            )

            total_steps = max(1, int(step_state["last_total_steps"]))
            rem_epochs = max(0, total - ep)
            eta_secs = rem_epochs * total_steps * (step_state["ema_dt"] or 0.0)
            emit_metrics(
                epoch=ep,
                total_epochs=total,
                step=total_steps,
                total_steps=total_steps,
                train_loss=step_state["last_loss"],
                val_loss=val_loss,
                map50=m50_disp,
                map50_95=m95_disp,
                lr=step_state["last_lr"],
                samples_per_sec=step_state["last_sps"],
                eta_secs=eta_secs,
            )

    orig_build_trainer = rf_training.build_trainer

    def hooked_build_trainer(*args, **kw):
        if str(device).lower().startswith("xpu"):
            dev_idx = 0
            if ":" in str(device):
                try:
                    dev_idx = int(str(device).split(":")[1])
                except Exception:
                    dev_idx = 0
            kw["accelerator"] = "auto"
            kw["devices"] = 1
            kw["strategy"] = SingleDeviceStrategy(
                device=torch.device("xpu", dev_idx),
                accelerator=XPUAccelerator(),
            )
        elif str(device).lower().startswith("cuda"):
            dev_idx = 0
            if ":" in str(device):
                try:
                    dev_idx = int(str(device).split(":")[1])
                except Exception:
                    dev_idx = 0
            kw["accelerator"] = "cuda"
            kw["devices"] = [dev_idx]
        elif str(device).lower() == "cpu":
            kw["accelerator"] = "cpu"
            kw["devices"] = 1
        trainer = orig_build_trainer(*args, **kw)
        trainer.callbacks.append(LSTrainerProgressCallback())
        return trainer

    rf_training.build_trainer = hooked_build_trainer
    rf_trainer_mod.build_trainer = hooked_build_trainer

    eval_backend = "faster_coco_eval"
    for candidate in ("vernier", "hotcoco", "faster_coco_eval"):
        try:
            __import__(candidate)
            eval_backend = candidate
            break
        except ImportError:
            continue

    model_cls = getattr(rfdetr, class_name)
    print(
        f"[LS-Trainer] Initializing {class_name} (resolution={resolution}, pretrain_weights={pretrain_weights}, device={device}, eval_backend={eval_backend}, num_workers={num_workers})...",
        flush=True,
    )
    model = model_cls(resolution=resolution, pretrain_weights=pretrain_weights, device=device)

    train_kwargs = {
        "dataset_dir": dataset_dir,
        "output_dir": output_dir,
        "epochs": epochs,
        "batch_size": batch_size,
        "resolution": resolution,
        "device": device,
        "num_workers": num_workers,
        "run_test": False,
        "progress_bar": False,
        "tensorboard": False,
        "eval_backend": eval_backend,
    }
    if freeze_backbone:
        train_kwargs["lr_encoder"] = 0.0

    print(f"Epoch 1/{epochs} - Starting RF-DETR training loop on device={device} (num_workers={num_workers})...", flush=True)
    cli_spinner.start()
    try:
        model.train(**train_kwargs)
    except KeyboardInterrupt:
        cli_spinner.stop()
        print("[LS-Trainer] Training interrupted by user.", flush=True)
    finally:
        cli_spinner.stop()

    if ctrl_state.is_stop_requested(stop_file):
        print(f"[LS-Trainer] RF-DETR training stopped by user. Checkpoints preserved at: {output_dir}", flush=True)

    metrics_file = os.path.join(output_dir, "ls_trainer_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as mf:
        json.dump(best_state, mf, indent=2)

    print(
        f"[LS-Trainer Metrics] Final | mAP50={best_state['map50']:.4f} | mAP50-95={best_state['map50_95']:.4f} | elapsed {format_duration_hm(progress_tracker.elapsed_secs())}",
        flush=True,
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        sys.exit(1)
