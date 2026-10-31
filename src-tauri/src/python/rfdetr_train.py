import json
import multiprocessing
import os
import sys
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

    models_dir = cfg["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    os.chdir(models_dir)

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

    def to_float(v):
        if v is None:
            return None
        if hasattr(v, "item"):
            try:
                return float(v.item())
            except Exception:
                pass
        try:
            return float(v)
        except Exception:
            return None

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

        def on_train_epoch_start(self, trainer, pl_module):
            ep = int(trainer.current_epoch) + 1
            total = int(trainer.max_epochs)
            print(f"Epoch {ep}/{total} - RF-DETR ({class_name}) training on {device}...", flush=True)

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

            if m50 is not None and m50 > best_state["map50"]:
                best_state["map50"] = m50
            if m50_95 is not None and m50_95 > best_state["map50_95"]:
                best_state["map50_95"] = m50_95

            m50_disp = m50 if m50 is not None else best_state["map50"]
            m95_disp = m50_95 if m50_95 is not None else best_state["map50_95"]
            print(
                f"[LS-Trainer Metrics] Epoch {ep}/{total} | mAP50={m50_disp:.4f} | mAP50-95={m95_disp:.4f}",
                flush=True,
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
    model.train(**train_kwargs)

    metrics_file = os.path.join(output_dir, "ls_trainer_metrics.json")
    with open(metrics_file, "w", encoding="utf-8") as mf:
        json.dump(best_state, mf, indent=2)

    print(
        f"[LS-Trainer Metrics] Final | mAP50={best_state['map50']:.4f} | mAP50-95={best_state['map50_95']:.4f}",
        flush=True,
    )

if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        sys.exit(1)
