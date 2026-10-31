import sys, onnx
onnx_path, model_type, model_family = sys.argv[1], sys.argv[2], sys.argv[3]
m = onnx.load(onnx_path)
props = {p.key: p.value for p in m.metadata_props}
props["ls_model_type"] = model_type
props["ls_model_family"] = model_family
del m.metadata_props[:]
for k, v in props.items():
    entry = m.metadata_props.add()
    entry.key = str(k)
    entry.value = str(v)
onnx.save(m, onnx_path)
