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

def embed_rfdetr_onnx_metadata(onnx_file, class_name, task, resolution, output_dir, model):
    import onnx
    m = onnx.load(str(onnx_file))
    props = {p.key: p.value for p in m.metadata_props}
    props["ls_model_type"] = "rfdetr"
    props["ls_model_family"] = "RF-DETR"
    props["ls_model_name"] = str(class_name)
    props["task"] = str(task)
    props["imgsz"] = f"[{resolution}, {resolution}]"

    class_names = None
    tc_path = os.path.join(output_dir, "training_config.json")
    if os.path.isfile(tc_path):
        try:
            with open(tc_path, "r", encoding="utf-8") as tf:
                tc = json.load(tf)
                if isinstance(tc.get("class_names"), list) and tc["class_names"]:
                    class_names = [str(c) for c in tc["class_names"]]
        except Exception:
            pass
    if not class_names:
        raw_names = getattr(model, "class_names", None)
        if isinstance(raw_names, dict):
            class_names = [str(v) for _, v in sorted(raw_names.items())]
        elif isinstance(raw_names, (list, tuple)) and raw_names:
            class_names = [str(c) for c in raw_names]

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
        f'[LS-Trainer] Embedded ONNX metadata: ls_model_type="rfdetr" (ls_model_family="RF-DETR", model={class_name}, task={task}, imgsz={resolution})',
        flush=True,
    )

def main():
    cfg_path = sys.argv[1]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    import rfdetr

    class_name = cfg["rfdetr_class"]
    ckpt_path = cfg["checkpoint_path"]
    output_dir = cfg["output_dir"]
    resolution = int(cfg["resolution"])
    dynamic_batch = bool(cfg["dynamic"])
    device = cfg["device"]
    task = cfg.get("task", "detect")

    os.makedirs(output_dir, exist_ok=True)
    model_cls = getattr(rfdetr, class_name)
    if hasattr(model_cls, "from_checkpoint"):
        model = model_cls.from_checkpoint(
            ckpt_path,
            resolution=resolution,
            device=device,
            trust_checkpoint=True,
        )
    else:
        model = model_cls(
            pretrain_weights=ckpt_path,
            resolution=resolution,
            device=device,
        )
    onnx_file = model.export(
        output_dir=output_dir,
        opset_version=17,
        verbose=False,
        shape=(resolution, resolution),
        dynamic_batch=dynamic_batch,
        output_name="best",
    )
    embed_rfdetr_onnx_metadata(onnx_file, class_name, task, resolution, output_dir, model)
    print(f"[LS-Trainer] Exported RF-DETR ONNX model to: {onnx_file}", flush=True)

if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
