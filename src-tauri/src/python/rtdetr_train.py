import json
import os
import sys
import time
import traceback
from pathlib import Path
from PIL import Image

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

import torch
if hasattr(torch, "xpu") and torch.xpu.is_available():
    _orig_nonzero = torch.nonzero
    def _safe_nonzero(t, *a, **k):
        if hasattr(t, "is_xpu") and t.is_xpu:
            return _orig_nonzero(t.cpu(), *a, **k).to(t.device)
        return _orig_nonzero(t, *a, **k)
    torch.nonzero = _safe_nonzero
    try:
        import transformers.models.rt_detr_v2.modeling_rt_detr_v2 as rtdetr_v2_mod
        rtdetr_v2_mod.torch.nonzero = _safe_nonzero
    except Exception:
        pass

    class SafeEmbeddingFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, indices, weight):
            ctx.save_for_backward(indices)
            ctx.num_embeddings = weight.shape[0]
            ctx.embedding_dim = weight.shape[1]
            return weight[indices]

        @staticmethod
        def backward(ctx, grad_output):
            indices, = ctx.saved_tensors
            flat_indices = indices.reshape(-1).long()
            flat_grad = grad_output.reshape(-1, ctx.embedding_dim)
            grad_weight = torch.zeros((ctx.num_embeddings, ctx.embedding_dim), dtype=grad_output.dtype, device=grad_output.device)
            grad_weight.index_add_(0, flat_indices, flat_grad)
            return None, grad_weight

    _orig_f_emb = torch.nn.functional.embedding
    def _safe_f_emb(input, weight, *args, **kwargs):
        if hasattr(weight, "is_xpu") and weight.is_xpu:
            return SafeEmbeddingFn.apply(input, weight)
        return _orig_f_emb(input, weight, *args, **kwargs)

    torch.nn.functional.embedding = _safe_f_emb
    try:
        import torch.nn.modules.sparse as sparse_mod
        sparse_mod.F.embedding = _safe_f_emb
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
            "[LS-Trainer] RT-DETR device selection: CPU requested -> device=cpu",
            flush=True,
        )
        return "cpu", "CPU"

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
                f"[LS-Trainer] RT-DETR device selection: torch.xpu.is_available()=True -> using Intel Arc XPU device=xpu ({dev_name}{count_info})",
                flush=True,
            )
            return "xpu", dev_name
        if cuda_avail:
            dev_name = "CUDA GPU"
            try:
                dev_name = torch.cuda.get_device_name(0)
            except Exception:
                pass
            print(
                f"[LS-Trainer] RT-DETR device selection: torch.cuda.is_available()=True -> using NVIDIA CUDA device=cuda ({dev_name})",
                flush=True,
            )
            return "cuda", dev_name
        print(
            "[LS-Trainer] RT-DETR device selection: neither Intel Arc XPU nor CUDA is available -> falling back to device=cpu",
            flush=True,
        )
        return "cpu", "CPU"

    if req.startswith("xpu"):
        dev_name = "Intel Arc XPU"
        try:
            idx = int(req.split(":")[1]) if ":" in req else 0
            dev_name = torch.xpu.get_device_name(idx)
        except Exception:
            pass
        print(
            f"[LS-Trainer] RT-DETR device selection: using specified Intel Arc device={requested} ({dev_name})",
            flush=True,
        )
        return req, dev_name

    if req.startswith("cuda"):
        dev_name = "CUDA GPU"
        try:
            idx = int(req.split(":")[1]) if ":" in req else 0
            dev_name = torch.cuda.get_device_name(idx)
        except Exception:
            pass
        print(
            f"[LS-Trainer] RT-DETR device selection: using specified CUDA device={requested} ({dev_name})",
            flush=True,
        )
        return req, dev_name

    print(f"[LS-Trainer] RT-DETR device selection: using custom device={requested}", flush=True)
    return req, requested

def find_annotation_file(split_dir):
    candidates = [
        "_annotations.coco.json",
        "annotations.coco.json",
        "instances.json",
        "annotations.json",
        "instances_default.json",
    ]
    for c in candidates:
        p = os.path.join(split_dir, c)
        if os.path.isfile(p):
            return p
    for item in os.listdir(split_dir):
        if item.endswith(".json") and "annotation" in item.lower():
            return os.path.join(split_dir, item)
    return None

class RTDetrCocoDataset:
    def __init__(self, split_dir, ann_file, resolution=640, cat_id_to_idx=None, is_train=True):
        import torch
        import torchvision.transforms.functional as TF

        self.split_dir = split_dir
        self.resolution = resolution
        self.is_train = is_train

        with open(ann_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.categories = data.get("categories", [])
        if cat_id_to_idx is None:
            sorted_cats = sorted(self.categories, key=lambda c: c["id"])
            self.cat_id_to_idx = {c["id"]: i for i, c in enumerate(sorted_cats)}
            self.class_names = [c["name"] for c in sorted_cats]
        else:
            self.cat_id_to_idx = cat_id_to_idx
            self.class_names = [c["name"] for c in sorted(self.categories, key=lambda c: self.cat_id_to_idx.get(c["id"], 0))]

        self.images = {img["id"]: img for img in data.get("images", [])}
        self.img_ids = list(self.images.keys())

        # Group annotations by image_id
        self.img_to_anns = {}
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            if img_id not in self.img_to_anns:
                self.img_to_anns[img_id] = []
            self.img_to_anns[img_id].append(ann)

        # Precompute mean and std normalization tensors
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        import torch
        import torchvision.transforms.functional as TF

        img_id = self.img_ids[idx]
        img_info = self.images[img_id]
        img_filename = img_info["file_name"]

        # Support subdirectories or direct image file
        img_path = os.path.join(self.split_dir, img_filename)
        if not os.path.isfile(img_path):
            base_filename = os.path.basename(img_filename)
            alt_path = os.path.join(self.split_dir, base_filename)
            if os.path.isfile(alt_path):
                img_path = alt_path

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize image to target resolution
        img_resized = img.resize((self.resolution, self.resolution), Image.BILINEAR)
        img_tensor = TF.to_tensor(img_resized)  # (3, H, W) in [0.0, 1.0]
        img_tensor = (img_tensor - self.mean) / self.std

        raw_anns = self.img_to_anns.get(img_id, [])
        boxes = []
        class_labels = []

        for ann in raw_anns:
            cat_id = ann.get("category_id")
            if cat_id not in self.cat_id_to_idx:
                continue
            bbox = ann.get("bbox", [])
            if len(bbox) != 4:
                continue
            x, y, w, h = bbox
            if w <= 0 or h <= 0 or orig_w <= 0 or orig_h <= 0:
                continue

            # Convert to normalized [cx, cy, norm_w, norm_h]
            cx = (x + w / 2.0) / float(orig_w)
            cy = (y + h / 2.0) / float(orig_h)
            nw = w / float(orig_w)
            nh = h / float(orig_h)

            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.0, min(1.0, nw))
            nh = max(0.0, min(1.0, nh))

            boxes.append([cx, cy, nw, nh])
            class_labels.append(self.cat_id_to_idx[cat_id])

        if len(boxes) == 0:
            target_boxes = torch.zeros((0, 4), dtype=torch.float32)
            target_labels = torch.zeros((0,), dtype=torch.long)
        else:
            target_boxes = torch.tensor(boxes, dtype=torch.float32)
            target_labels = torch.tensor(class_labels, dtype=torch.long)

        target = {
            "boxes": target_boxes,
            "class_labels": target_labels,
            "image_id": img_id,
            "orig_size": (orig_h, orig_w),
        }
        return img_tensor, target

def collate_fn(batch):
    import torch
    pixel_values = torch.stack([item[0] for item in batch], dim=0)
    labels = [item[1] for item in batch]
    return pixel_values, labels

def evaluate_coco(model, val_loader, raw_val_ann_file, device, amp=False):
    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    model.eval()
    results = []

    with torch.no_grad():
        for pixel_values, targets in val_loader:
            pixel_values = pixel_values.to(device)
            dev_type = "xpu" if device.startswith("xpu") else ("cuda" if device.startswith("cuda") else "cpu")
            with torch.autocast(device_type=dev_type, enabled=amp and dev_type != "cpu"):
                outputs = model(pixel_values=pixel_values)

            # outputs.logits: [batch, 300, num_classes]
            # outputs.pred_boxes: [batch, 300, 4] in normalized [cx, cy, w, h]
            logits = outputs.logits
            pred_boxes = outputs.pred_boxes
            scores = logits.sigmoid()

            for i in range(len(targets)):
                target = targets[i]
                img_id = target["image_id"]
                orig_h, orig_w = target["orig_size"]

                img_scores = scores[i].detach().cpu()  # [300, num_classes]
                img_boxes = pred_boxes[i].detach().cpu()  # [300, 4]

                max_scores, class_indices = img_scores.max(dim=-1)

                # Keep detections with score > 0.05 for evaluation
                mask = max_scores > 0.05
                sel_scores = max_scores[mask].tolist()
                sel_classes = class_indices[mask].tolist()
                sel_boxes = img_boxes[mask]

                for s, c_idx, b in zip(sel_scores, sel_classes, sel_boxes):
                    cx, cy, nw, nh = b.tolist()
                    abs_w = nw * orig_w
                    abs_h = nh * orig_h
                    abs_x = (cx * orig_w) - (abs_w / 2.0)
                    abs_y = (cy * orig_h) - (abs_h / 2.0)

                    # Map contiguous class index back to original COCO category id
                    cat_id = val_loader.dataset.categories[c_idx]["id"] if c_idx < len(val_loader.dataset.categories) else c_idx

                    results.append({
                        "image_id": img_id,
                        "category_id": cat_id,
                        "bbox": [round(abs_x, 2), round(abs_y, 2), round(abs_w, 2), round(abs_h, 2)],
                        "score": round(float(s), 4),
                    })

    if not results:
        return 0.0, 0.0

    try:
        coco_gt = COCO(raw_val_ann_file)
        coco_dt = coco_gt.loadRes(results)
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        # Suppress verbose evaluation stdout
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")
        try:
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
        finally:
            sys.stdout.close()
            sys.stdout = old_stdout

        map50_95 = float(coco_eval.stats[0])
        map50 = float(coco_eval.stats[1])
        return max(0.0, map50), max(0.0, map50_95)
    except Exception as e:
        print(f"[LS-Trainer Warning] COCO evaluation error: {e}", flush=True)
        return 0.0, 0.0

def main():
    if len(sys.argv) < 2:
        print("[LS-Trainer Error] Missing configuration argument.", flush=True)
        sys.exit(1)

    cfg_path = sys.argv[1]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    models_dir = cfg.get("models_dir", "models")
    os.makedirs(models_dir, exist_ok=True)

    dataset_dir = cfg["dataset_dir"]
    output_dir = cfg["output_dir"]
    epochs = int(cfg.get("epochs", 50))
    batch_size = max(1, int(cfg.get("batch_size", 4)))
    resolution = int(cfg.get("resolution", 640))
    pretrain_weights = cfg.get("pretrain_weights", "PekingU/rtdetr_v2_r18vd")
    freeze_backbone = bool(cfg.get("freeze_backbone", False))
    amp = bool(cfg.get("amp", True))
    num_workers = detect_num_workers(cfg)
    device_req, device_name = select_training_device(cfg.get("device", "auto"))

    os.makedirs(output_dir, exist_ok=True)

    # Locate dataset splits
    train_dir = os.path.join(dataset_dir, "train")
    valid_dir = os.path.join(dataset_dir, "valid")
    if not os.path.isdir(valid_dir):
        valid_dir = os.path.join(dataset_dir, "val")
    if not os.path.isdir(valid_dir):
        valid_dir = os.path.join(dataset_dir, "test")
    if not os.path.isdir(valid_dir):
        valid_dir = train_dir

    train_ann = find_annotation_file(train_dir)
    valid_ann = find_annotation_file(valid_dir)
    if not train_ann:
        print(f"[LS-Trainer Error] Could not find COCO annotations in: {train_dir}", flush=True)
        sys.exit(1)
    if not valid_ann:
        valid_ann = train_ann
        valid_dir = train_dir

    import torch
    from torch.utils.data import DataLoader
    from transformers import RTDetrV2ForObjectDetection

    torch_device = torch.device(device_req)

    # Build datasets
    train_ds = RTDetrCocoDataset(train_dir, train_ann, resolution=resolution, is_train=True)
    val_ds = RTDetrCocoDataset(
        valid_dir,
        valid_ann,
        resolution=resolution,
        cat_id_to_idx=train_ds.cat_id_to_idx,
        is_train=False,
    )

    class_names = train_ds.class_names
    num_classes = max(1, len(class_names))

    print(
        f"[LS-Trainer] RT-DETR training started: model=RT-DETR v2-R18 ({pretrain_weights}) | device={device_req} ({device_name}) | resolution={resolution} | epochs={epochs} | batch={batch_size} | workers={num_workers} | classes={num_classes} | amp={amp}",
        flush=True,
    )
    print(f"[LS-Trainer] Dataset: train={len(train_ds)} images | val={len(val_ds)} images | classes={class_names}", flush=True)

    # Pin memory when on GPU/XPU
    pin_mem = device_req != "cpu"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_mem,
    )

    # Initialize model
    print(f"[LS-Trainer] Loading base checkpoint '{pretrain_weights}' with {num_classes} classes...", flush=True)
    if os.path.isfile(pretrain_weights):
        # Continue from custom checkpoint
        model = RTDetrV2ForObjectDetection.from_pretrained(
            "PekingU/rtdetr_v2_r18vd",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        try:
            ckpt_data = torch.load(pretrain_weights, map_location="cpu")
            state_dict = ckpt_data.get("state_dict", ckpt_data)
            model.load_state_dict(state_dict, strict=False)
            print(f"[LS-Trainer] Restored weights from checkpoint: {pretrain_weights}", flush=True)
        except Exception as e:
            print(f"[LS-Trainer Warning] Failed to load custom checkpoint ({e}), continuing with pretrained weights", flush=True)
    else:
        model = RTDetrV2ForObjectDetection.from_pretrained(
            pretrain_weights,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    if freeze_backbone and hasattr(model.model, "backbone"):
        for param in model.model.backbone.parameters():
            param.requires_grad = False
        print("[LS-Trainer] RT-DETR: Backbone parameters frozen.", flush=True)

    model.to(torch_device)

    # Optimizer and LR scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-6)

    best_map50 = 0.0
    best_map50_95 = 0.0
    dev_type = "xpu" if device_req.startswith("xpu") else ("cuda" if device_req.startswith("cuda") else "cpu")

    weights_dir = os.path.join(output_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        print(f"Epoch {epoch}/{epochs} - RT-DETR (v2-R18) training on {device_req}...", flush=True)

        model.train()
        total_loss = 0.0
        num_batches = 0

        for pixel_values, labels in train_loader:
            pixel_values = pixel_values.to(torch_device)
            # Move targets to device
            device_labels = []
            for lab in labels:
                device_labels.append({
                    "class_labels": lab["class_labels"].to(torch_device),
                    "boxes": lab["boxes"].to(torch_device),
                })

            optimizer.zero_grad()
            with torch.autocast(device_type=dev_type, enabled=amp and dev_type != "cpu"):
                outputs = model(pixel_values=pixel_values, labels=device_labels)
                loss = outputs.loss
            print(f"[LS-Trainer] Step forward complete, loss={float(loss.detach().item()):.4f}", flush=True)

            loss.backward()
            print("[LS-Trainer] Step backward complete", flush=True)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.1)
            optimizer.step()
            print("[LS-Trainer] Optimizer step complete", flush=True)

            total_loss += float(loss.detach().item())
            num_batches += 1

        lr_scheduler.step()
        avg_loss = total_loss / max(1, num_batches)
        dur = time.time() - epoch_start_time

        # Validation
        print("[LS-Trainer] Running COCO validation evaluation...", flush=True)
        map50, map50_95 = evaluate_coco(model, val_loader, valid_ann, device_req, amp=amp)
        print(f"[LS-Trainer] Evaluation complete: map50={map50:.4f}, map50-95={map50_95:.4f}", flush=True)

        is_best = map50 >= best_map50
        if is_best:
            best_map50 = map50
            best_map50_95 = map50_95

        # Checkpoint saving
        state_dict_cpu = {k: v.cpu() for k, v in model.state_dict().items()}
        ckpt_payload = {
            "epoch": epoch,
            "state_dict": state_dict_cpu,
            "class_names": class_names,
            "map50": map50,
            "map50_95": map50_95,
        }

        last_path = os.path.join(weights_dir, "last.pth")
        torch.save(ckpt_payload, last_path)
        # Also save directly in output_dir for easy location
        torch.save(ckpt_payload, os.path.join(output_dir, "last.pth"))

        if is_best:
            best_path = os.path.join(weights_dir, "best.pth")
            torch.save(ckpt_payload, best_path)
            torch.save(ckpt_payload, os.path.join(output_dir, "best.pth"))

        # Save run metrics and training config
        with open(os.path.join(output_dir, "ls_trainer_metrics.json"), "w", encoding="utf-8") as mf:
            json.dump({"map50": best_map50, "map50_95": best_map50_95}, mf, indent=2)

        metrics_str = f"loss: {avg_loss:.4f} | mAP50={map50:.4f} | mAP50-95={map50_95:.4f}"
        print(f"[LS-Trainer Epoch] Epoch {epoch}/{epochs} finished in {dur:.1f}s - {metrics_str}", flush=True)
        print(f"Epoch {epoch}/{epochs} finished", flush=True)
        print(f"[LS-Trainer Metrics] Epoch {epoch}/{epochs} | {metrics_str}", flush=True)

    # Save final training_config.json
    final_config = {
        "model_family": "RT-DETR",
        "model_name": "RT-DETR v2-R18",
        "pretrain_weights": pretrain_weights,
        "num_classes": num_classes,
        "class_names": class_names,
        "resolution": resolution,
        "task": "detect",
        "epochs": epochs,
        "best_map50": best_map50,
        "best_map50_95": best_map50_95,
    }
    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(final_config, f, indent=2)

    print(
        f"[LS-Trainer] RT-DETR training completed successfully! Results saved to: {output_dir}",
        flush=True,
    )

if __name__ == "__main__":
    main()
