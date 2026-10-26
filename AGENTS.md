# AGENTS.md

Guidance for AI coding agents working on LS-Trainer. Version 0.1.4.

## Project context

LS-Trainer is a Tauri 2 + Svelte desktop app for training Ultralytics YOLO models on local datasets and exporting them to ONNX for LS-Vision. The UI is `src/App.svelte`. Training, export, GPU repair, and dataset checks are Tauri commands in `src-tauri/src/lib.rs`. Python runs from the project `.venv` (`yolo.exe` on Windows).

## Training tasks

- Task is `detect` or `segment`. A weight whose name contains `-seg` always trains as `segment`, even if the task selector says detection.
- The Ultralytics command is `yolo <task> train task=<task> ...`. Detection results go to `runs/detect`. Segmentation results go to `runs/segment`.
- Fresh segmentation weights are `yolo11s-seg.pt`, `yolo11m-seg.pt`, and `yolo26s-seg.pt`. Missing weights are downloaded into `models/` at train time. Do not commit `*.pt` files.
- Segmentation labels are polygon lines with more than 5 numeric values. A line with exactly 5 values is a bounding box. Selecting a segmentation model must warn when the dataset has boxes, mixed labels, or no polygons. The warning does not block training. The same check is logged when the run starts.

## Bench defaults

Keep these as the initial form values and the Bench preset unless the user asks to change them:

- `degrees` 180
- `flipud` 0.5 (range 0.0–1.0)
- `scale` 0.05
- `close_mosaic` 10
- `mosaic` 0
- Image size options stay 640, 960, and 1280. Do not remove 1280.

Off, Light, and Medium presets are separate from Bench. Changing Bench must not silently rewrite those presets.

## ONNX export

Auto-export defaults to on. After a successful train (`message` contains "complete"), export `best.pt` with the image size selected for that run, `dynamic=False`, and `simplify=True`. Show the `.onnx` path in the completion summary. A manual re-export uses the same arguments. Do not hardcode `imgsz=640`.

## Version bump

Keep these in sync when the version changes:

- `package.json`
- `src-tauri/Cargo.toml`
- `src-tauri/Cargo.lock` package `app` only
- `src-tauri/tauri.conf.json`
- the version badge in `src/App.svelte`

Do not bump unrelated crate versions in `Cargo.lock`.

## Do not commit

`runs/`, `datasets/`, `models/`, `*.pt`, `.venv/`, `src-tauri/target/`, and `node_modules/` are gitignored. Leave them out of commits. Windows display paths use backslashes. `explorer.exe` must be given backslashes.
