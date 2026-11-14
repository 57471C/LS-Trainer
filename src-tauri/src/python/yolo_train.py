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
    try:
        from ultralytics.cfg import entrypoint
        entrypoint()
    except KeyboardInterrupt:
        print("[LS-Trainer] Training interrupted by user.", flush=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
