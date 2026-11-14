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


def start_stdin_control_listener(ctrl_state):
    def _listener():
        try:
            for raw_line in sys.stdin:
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


def main():
    cli_args = sys.argv[1:]
    ctrl_state = TrainingControlState()
    start_stdin_control_listener(ctrl_state)

    project_dir = _parse_arg_value(cli_args, "project=")
    run_name = _parse_arg_value(cli_args, "name=")
    batch_arg = _parse_arg_value(cli_args, "batch=")
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

    def _resolve_stop_file(trainer):
        save_dir = getattr(trainer, "save_dir", None)
        if save_dir:
            return os.path.join(str(save_dir), ".stop_requested")
        return default_stop_file

    def _extract_yolo_train_loss(trainer):
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

    def _extract_yolo_lr(trainer):
        opt = getattr(trainer, "optimizer", None)
        if opt is not None and getattr(opt, "param_groups", None):
            try:
                return _safe_float(opt.param_groups[0].get("lr", 0.0))
            except Exception:
                pass
        return state["last_lr"]

    def on_train_epoch_start(trainer):
        state["step"] = 0
        state["last_step_time"] = time.time()
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(getattr(trainer, "epochs", configured_epochs))
        progress_tracker.start_epoch(epoch, total_epochs)
        stop_file = _resolve_stop_file(trainer)
        ctrl_state.wait_if_paused(stop_file)
        if ctrl_state.is_stop_requested(stop_file):
            trainer.stop = True
            raise KeyboardInterrupt("Stop requested by user")

    def on_train_batch_end(trainer):
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

        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(getattr(trainer, "epochs", 1))
        batch_sz = int(getattr(trainer, "batch_size", configured_batch) or configured_batch)
        sps = batch_sz / dt
        state["last_sps"] = sps

        done_global = (epoch - 1) * total_steps + step
        total_global = max(done_global, total_epochs * total_steps)
        rem_steps = max(0, total_global - done_global)
        eta_secs = rem_steps * (state["ema_step_dt"] or dt)

        train_loss = _extract_yolo_train_loss(trainer)
        lr = _extract_yolo_lr(trainer)
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

        if ctrl_state.consume_checkpoint_request():
            try:
                if hasattr(trainer, "save_model"):
                    trainer.save_model()
                print("[LS-Trainer] Immediate checkpoint saved.", flush=True)
            except Exception as exc:
                print(f"[LS-Trainer Warning] Immediate checkpoint save failed: {exc}", flush=True)

        stop_file = _resolve_stop_file(trainer)
        ctrl_state.wait_if_paused(stop_file)
        if ctrl_state.is_stop_requested(stop_file):
            trainer.stop = True
            try:
                if hasattr(trainer, "save_model"):
                    trainer.save_model()
            except Exception:
                pass
            raise KeyboardInterrupt("Stop requested by user")

    def on_fit_epoch_end(trainer):
        epoch = int(getattr(trainer, "epoch", 0)) + 1
        total_epochs = int(getattr(trainer, "epochs", 1))
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
        cli_spinner.print_line(
            f"[LS-Trainer Metrics] Epoch {epoch}/{total_epochs} | mAP50={m50_str} | mAP50-95={m95_str} | {timing}"
        )

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

    sys.argv = ["yolo"] + cli_args
    cli_spinner.start()
    try:
        from ultralytics.cfg import entrypoint
        entrypoint()
    except KeyboardInterrupt:
        cli_spinner.stop()
        print("[LS-Trainer] Training interrupted by user.", flush=True)
    finally:
        cli_spinner.stop()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
