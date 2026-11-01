import json
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

def embed_rtdetr_onnx_metadata(onnx_file, resolution, class_names, task="detect", model_name="RT-DETR v2-R18"):
    import onnx
    m = onnx.load(str(onnx_file))
    props = {p.key: p.value for p in m.metadata_props}
    props["ls_model_type"] = "rtdetr"
    props["ls_model_family"] = "RT-DETR"
    props["ls_model_name"] = str(model_name)
    props["task"] = str(task)
    props["imgsz"] = f"[{resolution}, {resolution}]"

    if class_names:
        names_dict = {i: name for i, name in enumerate(class_names)}
        props["names"] = str(names_dict)
        props["class_names"] = json.dumps(class_names)

    del m.metadata_props[:]
    for k, v in props.items():
        entry = m.metadata_props.add()
        entry.key = str(k)
        entry.value = str(v)
    onnx.save(m, str(onnx_file))
    print(
        f'[LS-Trainer] Embedded ONNX metadata: ls_model_type="rtdetr" (ls_model_family="RT-DETR", model={model_name}, task={task}, imgsz={resolution})',
        flush=True,
    )

def main():
    if len(sys.argv) < 2:
        print("[LS-Trainer Error] Missing configuration argument for RT-DETR export.", flush=True)
        sys.exit(1)

    cfg_path = sys.argv[1]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    checkpoint_path = cfg["checkpoint_path"]
    output_dir = cfg["output_dir"]
    resolution = int(cfg.get("resolution", 640))
    dynamic_batch = bool(cfg.get("dynamic", False))
    simplify = bool(cfg.get("simplify", True))
    task = cfg.get("task", "detect")
    model_name = cfg.get("model_name", "RT-DETR v2-R18")
    base_weights = cfg.get("base_weights", "PekingU/rtdetr_v2_r18vd")

    os.makedirs(output_dir, exist_ok=True)

    # Resolve class names and model weights from training config if available
    class_names = ["object"]
    search_dirs = [output_dir, os.path.dirname(checkpoint_path), os.path.dirname(os.path.dirname(checkpoint_path))]
    for d in search_dirs:
        tc_path = os.path.join(d, "training_config.json")
        if os.path.isfile(tc_path):
            try:
                with open(tc_path, "r", encoding="utf-8") as tf:
                    tc = json.load(tf)
                    if isinstance(tc.get("class_names"), list) and tc["class_names"]:
                        class_names = [str(c) for c in tc["class_names"]]
                    if tc.get("model_name"):
                        model_name = str(tc["model_name"])
                    if tc.get("base_weights"):
                        base_weights = str(tc["base_weights"])
                    elif tc.get("pretrain_weights") and not str(tc["pretrain_weights"]).endswith((".pt", ".pth")):
                        base_weights = str(tc["pretrain_weights"])
                    break
            except Exception:
                pass

    import torch
    from transformers import RTDetrV2ForObjectDetection

    class RTDetrExportWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images):
            out = self.model(pixel_values=images)
            return out.logits, out.pred_boxes

    num_classes = max(1, len(class_names))
    model = RTDetrV2ForObjectDetection.from_pretrained(
        base_weights,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    if os.path.isfile(checkpoint_path):
        try:
            ckpt_data = torch.load(checkpoint_path, map_location="cpu")
            state_dict = ckpt_data.get("state_dict", ckpt_data)
            model.load_state_dict(state_dict, strict=False)
            print(f"[LS-Trainer] Loaded checkpoint weights: {checkpoint_path}", flush=True)
            if "class_names" in ckpt_data and isinstance(ckpt_data["class_names"], list):
                class_names = ckpt_data["class_names"]
        except Exception as e:
            print(f"[LS-Trainer Warning] Could not load checkpoint weights directly: {e}", flush=True)

    model.eval()
    wrapper = RTDetrExportWrapper(model)

    dummy_input = torch.randn(1, 3, resolution, resolution)
    onnx_file = os.path.join(output_dir, "best.onnx")

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch"},
            "logits": {0: "batch"},
            "pred_boxes": {0: "batch"},
        }

    print(
        f"[LS-Trainer] Exporting {model_name} to ONNX: shape=({resolution}, {resolution}), dynamic={dynamic_batch}, simplify={simplify}...",
        flush=True,
    )

    torch.onnx.export(
        wrapper,
        (dummy_input,),
        onnx_file,
        input_names=["images"],
        output_names=["logits", "pred_boxes"],
        dynamic_axes=dynamic_axes,
        opset_version=17,
        dynamo=False,
    )

    if simplify:
        try:
            import onnxsim
            import onnx
            model_onnx = onnx.load(onnx_file)
            model_simp, check = onnxsim.simplify(model_onnx)
            if check:
                onnx.save(model_simp, onnx_file)
                print("[LS-Trainer] Successfully simplified ONNX model with onnxsim.", flush=True)
        except Exception as e:
            # Non-fatal if onnxsim is not installed or fails
            pass

    embed_rtdetr_onnx_metadata(onnx_file, resolution, class_names, task=task, model_name=model_name)

    # Also place in weights/ subdirectory if present
    weights_dir = os.path.join(output_dir, "weights")
    if os.path.isdir(weights_dir):
        weights_onnx = os.path.join(weights_dir, "best.onnx")
        try:
            import shutil
            shutil.copy2(onnx_file, weights_onnx)
        except Exception:
            pass

    print(f"[LS-Trainer] Exported RT-DETR ONNX model to: {onnx_file}", flush=True)

if __name__ == "__main__":
    main()

