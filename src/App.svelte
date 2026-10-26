<script lang="ts">
import { Channel, invoke, isTauri } from "@tauri-apps/api/core";
import { onMount } from "svelte";

type ModelMode = "continue" | "fresh";
type TrainTask = "detect" | "segment";
type ImageSize = 640 | 960 | 1280;
type YoloModel =
	| "yolo11n.pt"
	| "yolo11s.pt"
	| "yolo11m.pt"
	| "yolo11l.pt"
	| "yolo11x.pt"
	| "yolo26n.pt"
	| "yolo26s.pt"
	| "yolo26m.pt"
	| "yolo26l.pt"
	| "yolo26x.pt"
	| "yolo11s-seg.pt"
	| "yolo11m-seg.pt"
	| "yolo26s-seg.pt";
type AugPreset = "off" | "light" | "medium" | "bench" | "custom";

interface TrainingResponse {
	success: boolean;
	results_dir: string;
	message: string;
}

interface ExportResponse {
	success: boolean;
	onnx_path: string;
	message: string;
}

interface GpuInfo {
	cuda_available: boolean;
	cpu_fallback_confirmed: boolean;
	device_name: string | null;
	driver_version: string | null;
	recommended_wheel: string;
	cuda_label: string;
	torch_version: string | null;
	needs_repair: boolean;
	status_summary: string;
}

interface LabelCheckResult {
	polygons_found: boolean;
	bboxes_found: boolean;
	files_scanned: number;
	lines_scanned: number;
	warning: string | null;
}

const freshModelOptions: {
	id: YoloModel;
	family: "YOLO11" | "YOLO26";
	task: TrainTask;
	label: string;
	desc: string;
}[] = [
	{
		id: "yolo11n.pt",
		family: "YOLO11",
		task: "detect",
		label: "YOLO11n",
		desc: "Nano (fastest, lightweight)",
	},
	{
		id: "yolo11s.pt",
		family: "YOLO11",
		task: "detect",
		label: "YOLO11s",
		desc: "Small (balanced fast)",
	},
	{
		id: "yolo11m.pt",
		family: "YOLO11",
		task: "detect",
		label: "YOLO11m",
		desc: "Medium (default, high accuracy)",
	},
	{
		id: "yolo11l.pt",
		family: "YOLO11",
		task: "detect",
		label: "YOLO11l",
		desc: "Large (extra accuracy)",
	},
	{
		id: "yolo11x.pt",
		family: "YOLO11",
		task: "detect",
		label: "YOLO11x",
		desc: "XLarge (maximum capacity)",
	},
	{
		id: "yolo26n.pt",
		family: "YOLO26",
		task: "detect",
		label: "YOLO26n",
		desc: "Nano (next-gen lightweight)",
	},
	{
		id: "yolo26s.pt",
		family: "YOLO26",
		task: "detect",
		label: "YOLO26s",
		desc: "Small (next-gen balanced)",
	},
	{
		id: "yolo26m.pt",
		family: "YOLO26",
		task: "detect",
		label: "YOLO26m",
		desc: "Medium (next-gen high accuracy)",
	},
	{
		id: "yolo26l.pt",
		family: "YOLO26",
		task: "detect",
		label: "YOLO26l",
		desc: "Large (next-gen extra accuracy)",
	},
	{
		id: "yolo26x.pt",
		family: "YOLO26",
		task: "detect",
		label: "YOLO26x",
		desc: "XLarge (next-gen maximum capacity)",
	},
	{
		id: "yolo11s-seg.pt",
		family: "YOLO11",
		task: "segment",
		label: "YOLO11s-seg",
		desc: "Small instance segmentation",
	},
	{
		id: "yolo11m-seg.pt",
		family: "YOLO11",
		task: "segment",
		label: "YOLO11m-seg",
		desc: "Medium instance segmentation",
	},
	{
		id: "yolo26s-seg.pt",
		family: "YOLO26",
		task: "segment",
		label: "YOLO26s-seg",
		desc: "Small next-gen segmentation",
	},
];

let datasetPath = $state<string>("");
let datasetFileName = $state<string>("");
let modelMode = $state<ModelMode>("fresh");
let existingModelPath = $state<string>("");
let existingModelFileName = $state<string>("");
let freshModel = $state<YoloModel>("yolo11m.pt");
let trainTask = $state<TrainTask>("detect");
let freezeBackbone = $state<boolean>(false);
const freezeLayersCount = 10;
let runName = $state<string>("");
let epochs = $state<number>(60);
let batch = $state<number>(4);
let imageSize = $state<ImageSize>(960);
let device = $state<string>("0");
let isCustomDevice = $state<boolean>(false);
let customDevice = $state<string>("");
let gpuInfo = $state<GpuInfo | null>(null);
let isCheckingGpu = $state<boolean>(true);
let isRepairingCuda = $state<boolean>(false);

// Training & Augmentation controls
let amp = $state<boolean>(true);
let mosaic = $state<number>(0.0);
let mixup = $state<number>(0.0);
let copyPaste = $state<boolean>(false);
let degrees = $state<number>(180);
let flipud = $state<number>(0.5);
let translate = $state<number>(0.1);
let scale = $state<number>(0.05);
let closeMosaic = $state<number>(10);
let autoExportOnnx = $state<boolean>(true);
let labelWarning = $state<string | null>(null);
let labelCheckNote = $state<string | null>(null);
let labelCheckPending = $state<boolean>(false);

let activeTooltip = $state<string | null>(null);

const activeAugPreset = $derived.by<AugPreset>(() => {
	if (
		mosaic === 0.0 &&
		mixup === 0.0 &&
		!copyPaste &&
		degrees === 180 &&
		flipud === 0.5 &&
		translate === 0.1 &&
		scale === 0.05 &&
		closeMosaic === 10
	) {
		return "bench";
	}
	if (
		mosaic === 0.0 &&
		mixup === 0.0 &&
		!copyPaste &&
		degrees === 0.0 &&
		flipud === 0.0 &&
		translate === 0.0 &&
		scale === 0.0
	) {
		return "off";
	}
	if (
		mosaic === 0.0 &&
		mixup === 0.0 &&
		!copyPaste &&
		degrees === 0.0 &&
		flipud === 0.0 &&
		translate === 0.1 &&
		scale === 0.2
	) {
		return "light";
	}
	if (
		mosaic === 0.5 &&
		mixup === 0.1 &&
		!copyPaste &&
		degrees === 5.0 &&
		flipud === 0.0 &&
		translate === 0.1 &&
		scale === 0.4
	) {
		return "medium";
	}
	return "custom";
});

function applyAugPreset(preset: "off" | "light" | "medium" | "bench") {
	if (isTraining) return;
	if (preset === "off") {
		mosaic = 0.0;
		mixup = 0.0;
		copyPaste = false;
		degrees = 0.0;
		flipud = 0.0;
		translate = 0.0;
		scale = 0.0;
	} else if (preset === "light") {
		mosaic = 0.0;
		mixup = 0.0;
		copyPaste = false;
		degrees = 0.0;
		flipud = 0.0;
		translate = 0.1;
		scale = 0.2;
	} else if (preset === "medium") {
		mosaic = 0.5;
		mixup = 0.1;
		copyPaste = false;
		degrees = 5.0;
		flipud = 0.0;
		translate = 0.1;
		scale = 0.4;
	} else if (preset === "bench") {
		mosaic = 0.0;
		mixup = 0.0;
		copyPaste = false;
		degrees = 180;
		flipud = 0.5;
		translate = 0.1;
		scale = 0.05;
		closeMosaic = 10;
	}
}

function selectTrainTask(task: TrainTask) {
	if (isTraining) return;
	trainTask = task;
	const current = freshModelOptions.find((o) => o.id === freshModel);
	if (!current || current.task !== task) {
		freshModel = task === "segment" ? "yolo11s-seg.pt" : "yolo11m.pt";
	}
}

const activeModelLabel = $derived.by(() => {
	if (modelMode === "continue") {
		return existingModelFileName.trim() || "Custom .pt";
	}
	const found = freshModelOptions.find((o) => o.id === freshModel);
	return found ? found.label : freshModel;
});

const activeFreezeLabel = $derived(
	freezeBackbone ? `freeze=${freezeLayersCount}` : "unfrozen",
);

const activeAugSummary = $derived(
	`mosaic=${mosaic} · mixup=${mixup} · copy_paste=${copyPaste ? "on" : "off"} · deg=${degrees} · flipud=${flipud} · trans=${translate} · scale=${scale} · close_mosaic=${closeMosaic}`,
);

const effectiveModelName = $derived(
	modelMode === "continue"
		? existingModelFileName || existingModelPath
		: freshModel,
);

const isSegmentationRun = $derived(
	trainTask === "segment" || /-seg/i.test(effectiveModelName),
);

const visibleModelOptions = $derived(
	freshModelOptions.filter((o) => o.task === trainTask),
);

const activeTaskLabel = $derived(isSegmentationRun ? "segment" : "detect");

const activeRunConfigBanner = $derived(
	`${activeTaskLabel} · ${activeModelLabel} · ${activeFreezeLabel} · ${activeAugSummary}`,
);

// Epoch countdown & rolling ETA tracking
let currentEpoch = $state<number>(0);
let totalEpochs = $state<number>(0);
let currentEpochStartTime = $state<number | null>(null);
let recentEpochDurations = $state<number[]>([]);
let firstEpochEstimatedSec = $state<number | null>(null);

const remainingEpochs = $derived(Math.max(0, totalEpochs - currentEpoch));
const rollingAvgEpochSec = $derived(
	recentEpochDurations.length > 0
		? recentEpochDurations.reduce((sum, d) => sum + d, 0) /
				recentEpochDurations.length
		: firstEpochEstimatedSec,
);
const epochEtaLabel = $derived.by(() => {
	if (currentEpoch <= 0 || totalEpochs <= 0) return "";
	if (!isTraining && remainingEpochs === 0) return "Complete";
	if (rollingAvgEpochSec === null || rollingAvgEpochSec <= 0) {
		return "estimating…";
	}
	const secLeft = remainingEpochs * rollingAvgEpochSec;
	if (secLeft < 45) {
		return `~${Math.max(1, Math.round(secLeft))} sec left`;
	}
	const mins = Math.max(1, Math.round(secLeft / 60));
	return `~${mins} min left`;
});
const epochCountdownText = $derived(
	currentEpoch > 0 && totalEpochs > 0
		? `Epoch ${currentEpoch} / ${totalEpochs}   (${remainingEpochs} remaining)   ${epochEtaLabel}`
		: "",
);
const epochProgressPercent = $derived(
	totalEpochs > 0
		? Math.min(100, Math.round((currentEpoch / totalEpochs) * 100))
		: 0,
);

function shortenGpuName(name: string): string {
	return name
		.replace(/^NVIDIA GeForce\s+/i, "")
		.replace(/^NVIDIA\s+/i, "")
		.trim();
}

function processIncomingLogLine(line: string) {
	if (line.startsWith("[LS-Trainer Status] ")) {
		statusMessage = line.replace("[LS-Trainer Status] ", "").trim();
		return;
	}

	// Detect Ultralytics epoch progress row e.g. "     12/80      2.41G ..." or "Epoch 12/80"
	const yoloRowMatch =
		line.match(/^\s*(\d+)\s*\/\s*(\d+)\s+\S+\s+\d+(?:\.\d+)?/) ||
		line.match(/^\s*(\d+)\s*\/\s*(\d+)\s+.*?\d+%\|/) ||
		line.match(/epochs?\s*\(?(\d+)\s*\/\s*(\d+)\)?/i);

	if (yoloRowMatch) {
		const ep = Number(yoloRowMatch[1]);
		const tot = Number(yoloRowMatch[2]);
		if (ep >= 1 && tot >= 1 && ep <= tot) {
			const now = Date.now();
			if (ep !== currentEpoch) {
				if (
					currentEpoch > 0 &&
					currentEpochStartTime !== null &&
					ep > currentEpoch
				) {
					const elapsedSec =
						(now - currentEpochStartTime) / (1000 * (ep - currentEpoch));
					if (elapsedSec > 0.02) {
						recentEpochDurations = [
							...recentEpochDurations.slice(-4),
							elapsedSec,
						];
					}
				}
				currentEpoch = ep;
				totalEpochs = tot;
				currentEpochStartTime = now;
			}

			// Parse tqdm [MM:SS<MM:SS] for first-epoch ETA before a full epoch transition completes
			const tqdmTimeMatch = line.match(/\[(\d+):(\d+)<(\d+):(\d+)/);
			if (tqdmTimeMatch) {
				const elap = Number(tqdmTimeMatch[1]) * 60 + Number(tqdmTimeMatch[2]);
				const rem = Number(tqdmTimeMatch[3]) * 60 + Number(tqdmTimeMatch[4]);
				if (elap + rem > 0) {
					firstEpochEstimatedSec = elap + rem;
				}
			}

			// Replace in-place if the previous log line is an incomplete tqdm bar for the same epoch
			const isTqdmBar = /\d+%\|/.test(line);
			if (isTqdmBar && logs.length > 0) {
				const prev = logs[logs.length - 1];
				const prevMatch = prev.match(/^\s*(\d+)\s*\/\s*(\d+)\s+/);
				if (
					prevMatch &&
					Number(prevMatch[1]) === ep &&
					Number(prevMatch[2]) === tot &&
					!/100%\|/.test(prev)
				) {
					logs[logs.length - 1] = line;
					return;
				}
			}
		}
	}

	logs = [...logs, line];
}

async function handleRepairCuda() {
	if (isTraining || isRepairingCuda) {
		return;
	}
	isRepairingCuda = true;
	isLogPanelOpen = true;

	if (gpuInfo?.device_name) {
		statusMessage = `Installing PyTorch for ${shortenGpuName(gpuInfo.device_name)} (${gpuInfo.cuda_label})…`;
	} else {
		statusMessage = "Detecting GPU and installing matching PyTorch wheel…";
	}

	if (isTauri()) {
		try {
			const onLog = new Channel<string>();
			onLog.onmessage = (message) => {
				processIncomingLogLine(message);
			};
			const updated = await invoke<GpuInfo>("repair_cuda", { onLog });
			gpuInfo = updated;
			device = updated.cuda_available ? "0" : "cpu";
			statusMessage = updated.status_summary;
		} catch (err) {
			const errMsg = err instanceof Error ? err.message : String(err);
			logs = [...logs, `[LS-Trainer Error] ${errMsg}`];
			statusMessage = "CUDA repair encountered an error";
		} finally {
			isRepairingCuda = false;
		}
	} else {
		logs = [
			...logs,
			"[LS-Trainer] Simulated CUDA repair in Browser Mode",
			"[LS-Trainer] Verified PyTorch environment.",
		];
		statusMessage = "Ready";
		isRepairingCuda = false;
	}
}

onMount(async () => {
	if (isTauri()) {
		isCheckingGpu = true;
		statusMessage = "Checking GPU and PyTorch CUDA environment…";
		try {
			gpuInfo = await invoke<GpuInfo>("get_gpu_info");
			if (gpuInfo.cuda_available) {
				device = "0";
			} else {
				device = "cpu";
			}
			statusMessage = gpuInfo.status_summary || "Ready";
			isCheckingGpu = false;

			if (gpuInfo.needs_repair) {
				await handleRepairCuda();
			}
		} catch (e) {
			console.error("Failed to query GPU info:", e);
			statusMessage = "Ready";
			isCheckingGpu = false;
		}
	} else {
		isCheckingGpu = false;
	}
});

$effect(() => {
	const path = datasetPath.trim() || datasetFileName.trim();
	const seg = isSegmentationRun;
	if (!seg || !path) {
		labelWarning = null;
		labelCheckNote = null;
		labelCheckPending = false;
		return;
	}
	if (!isTauri()) {
		labelWarning = null;
		labelCheckPending = false;
		labelCheckNote =
			"Polygon check runs in the desktop app when a dataset archive is selected.";
		return;
	}

	let cancelled = false;
	labelCheckPending = true;
	labelWarning = null;
	labelCheckNote = null;
	invoke<LabelCheckResult>("inspect_dataset_labels", { datasetPath: path })
		.then((result) => {
			if (cancelled) return;
			labelCheckPending = false;
			if (result.warning) {
				labelWarning = result.warning;
				labelCheckNote = null;
			} else if (result.polygons_found) {
				labelCheckNote = `Polygon annotations detected (${result.files_scanned} label files).`;
			} else {
				labelWarning =
					"No polygon annotations were found. Instance segmentation needs label lines with more than 5 values.";
			}
		})
		.catch((err) => {
			if (cancelled) return;
			labelCheckPending = false;
			labelWarning = err instanceof Error ? err.message : String(err);
		});

	return () => {
		cancelled = true;
	};
});

let isTraining = $state<boolean>(false);
let logs = $state<string[]>([]);
let resultsDir = $state<string | null>(null);
let statusMessage = $state<string>("Ready");
let hasCopiedLogs = $state<boolean>(false);
let hasCopiedResultsPath = $state<boolean>(false);
let hasCopiedOnnxPath = $state<boolean>(false);
let isExportingOnnx = $state<boolean>(false);
let onnxPath = $state<string | null>(null);
let onnxError = $state<string | null>(null);

let isLogPanelOpen = $state<boolean>(false);
let logPanelWidth = $state<number>(540);
let isDragging = $state<boolean>(false);

let datasetInputRef: HTMLInputElement | undefined = $state();
let modelInputRef: HTMLInputElement | undefined = $state();
let logContainerRef: HTMLDivElement | undefined = $state();

const isDatasetSelected = $derived(
	Boolean(datasetPath.trim() || datasetFileName.trim()),
);
const isModelSelected = $derived(
	modelMode === "continue"
		? Boolean(existingModelPath.trim() || existingModelFileName.trim())
		: Boolean(freshModel),
);
const isTorchReady = $derived(
	!isCheckingGpu &&
		!isRepairingCuda &&
		(gpuInfo === null
			? !isTauri()
			: Boolean(gpuInfo.cuda_available || gpuInfo.cpu_fallback_confirmed)),
);
const isFormValid = $derived(
	isDatasetSelected && isModelSelected && !isTraining && isTorchReady,
);

$effect(() => {
	if (logs.length > 0 && logContainerRef) {
		logContainerRef.scrollTop = logContainerRef.scrollHeight;
	}
});

function startResize(e: MouseEvent) {
	e.preventDefault();
	isDragging = true;
	const dragStartX = e.clientX;
	const dragStartWidth = logPanelWidth;

	const handleMouseMove = (moveEvent: MouseEvent) => {
		const delta = dragStartX - moveEvent.clientX;
		const minWidth = 320;
		const maxWidth = Math.max(minWidth, window.innerWidth - 380);
		logPanelWidth = Math.min(
			maxWidth,
			Math.max(minWidth, dragStartWidth + delta),
		);
	};

	const handleMouseUp = () => {
		isDragging = false;
		window.removeEventListener("mousemove", handleMouseMove);
		window.removeEventListener("mouseup", handleMouseUp);
		document.body.style.cursor = "";
		document.body.style.userSelect = "";
	};

	window.addEventListener("mousemove", handleMouseMove);
	window.addEventListener("mouseup", handleMouseUp);
	document.body.style.cursor = "col-resize";
	document.body.style.userSelect = "none";
}

function normalizePath(path: string | null | undefined): string {
	if (!path) return "";
	const trimmed = path.trim();
	const isWindows =
		(typeof navigator !== "undefined" &&
			(/windows|win32|win64/i.test(navigator.userAgent) ||
				navigator.platform?.startsWith("Win"))) ||
		/^[a-zA-Z]:[\\/]/.test(trimmed);

	if (isWindows) {
		let win = trimmed.replace(/\//g, "\\");
		const isUnc = win.startsWith("\\\\");
		if (isUnc) {
			win = `\\\\${win.slice(2).replace(/\\+/g, "\\")}`;
		} else {
			win = win.replace(/\\+/g, "\\");
		}
		return win;
	}

	return trimmed.replace(/\\/g, "/").replace(/\/+/g, "/");
}

async function copyLogsToClipboard() {
	if (logs.length === 0) {
		return;
	}
	try {
		await navigator.clipboard.writeText(logs.join("\n"));
		hasCopiedLogs = true;
		setTimeout(() => {
			hasCopiedLogs = false;
		}, 2000);
	} catch (err) {
		console.warn("Failed to copy logs to clipboard:", err);
	}
}

async function handlePickDataset() {
	if (isTauri()) {
		try {
			const picked = await invoke<string | null>("pick_dataset_file");
			if (picked) {
				const norm = normalizePath(picked);
				datasetPath = norm;
				const cleanName = norm.replace(/\\/g, "/").split("/").pop();
				datasetFileName = cleanName || norm;
			}
			return;
		} catch (err) {
			console.warn("Native file picker unavailable, falling back:", err);
		}
	}
	datasetInputRef?.click();
}

function handleDatasetFileInputChange(event: Event) {
	const target = event.currentTarget as HTMLInputElement;
	const file = target.files?.[0];
	if (file) {
		datasetFileName = file.name;
		const raw = (file as unknown as { path?: string }).path || file.name;
		datasetPath = normalizePath(raw);
	}
}

function clearDataset() {
	datasetFileName = "";
	datasetPath = "";
	if (datasetInputRef) {
		datasetInputRef.value = "";
	}
}

async function handlePickExistingModel() {
	if (isTauri()) {
		try {
			const picked = await invoke<string | null>("pick_model_file");
			if (picked) {
				const norm = normalizePath(picked);
				existingModelPath = norm;
				const cleanName = norm.replace(/\\/g, "/").split("/").pop();
				existingModelFileName = cleanName || norm;
				if (/-seg/i.test(existingModelFileName)) {
					selectTrainTask("segment");
				}
			}
			return;
		} catch (err) {
			console.warn("Native file picker unavailable, falling back:", err);
		}
	}
	modelInputRef?.click();
}

function handleExistingModelFileInputChange(event: Event) {
	const target = event.currentTarget as HTMLInputElement;
	const file = target.files?.[0];
	if (file) {
		existingModelFileName = file.name;
		const raw = (file as unknown as { path?: string }).path || file.name;
		existingModelPath = normalizePath(raw);
		if (/-seg/i.test(file.name)) {
			selectTrainTask("segment");
		}
	}
}

function clearExistingModel() {
	existingModelFileName = "";
	existingModelPath = "";
	if (modelInputRef) {
		modelInputRef.value = "";
	}
}

async function handleStartTraining() {
	if (!isFormValid || isTraining) {
		return;
	}

	isTraining = true;
	isLogPanelOpen = true;
	logs = [];
	resultsDir = null;
	onnxPath = null;
	onnxError = null;
	hasCopiedResultsPath = false;
	hasCopiedOnnxPath = false;
	currentEpoch = 0;
	totalEpochs = Number(epochs);
	currentEpochStartTime = null;
	recentEpochDurations = [];
	firstEpochEstimatedSec = null;
	statusMessage = "Training in progress...";

	const effectiveRunName = runName.trim() || "BB-Parts-v3";
	const effectiveDataset = datasetPath.trim() || datasetFileName.trim();
	const effectiveModel =
		modelMode === "continue"
			? existingModelPath.trim() || existingModelFileName.trim()
			: freshModel;

	const effectiveCopyPaste = copyPaste ? 0.3 : 0.0;
	const taskName = isSegmentationRun ? "segment" : "detect";
	const requestPayload = {
		dataset_path: effectiveDataset,
		model_choice: effectiveModel,
		run_name: effectiveRunName,
		epochs: Number(epochs),
		batch: Number(batch),
		imgsz: Number(imageSize),
		device,
		amp: Boolean(amp),
		freeze_backbone: Boolean(freezeBackbone),
		freeze_layers: freezeLayersCount,
		mosaic: Number(mosaic),
		mixup: Number(mixup),
		copy_paste: effectiveCopyPaste,
		degrees: Number(degrees),
		flipud: Number(flipud),
		translate: Number(translate),
		scale: Number(scale),
		close_mosaic: Number(closeMosaic),
		task: taskName,
	};

	console.log("LS-Trainer Starting Training:", requestPayload);

	if (isTauri()) {
		try {
			const onLog = new Channel<string>();
			onLog.onmessage = (message) => {
				processIncomingLogLine(message);
			};

			const response = await invoke<TrainingResponse>("start_training", {
				onLog,
				request: requestPayload,
			});

			if (response.success && totalEpochs > 0) {
				currentEpoch = totalEpochs;
			}
			resultsDir = normalizePath(response.results_dir);
			if (
				response.success &&
				autoExportOnnx &&
				/complete/i.test(response.message)
			) {
				statusMessage = "Training complete — exporting ONNX…";
				await handleExportToOnnx();
			}
			statusMessage = response.success
				? onnxPath
					? `Finished — ${normalizePath(onnxPath)}`
					: `Finished — ${normalizePath(response.results_dir)}`
				: "Process finished with status warnings";
		} catch (err) {
			const errMsg = err instanceof Error ? err.message : String(err);
			logs = [...logs, `[LS-Trainer Error] ${errMsg}`];
			statusMessage = "Training encountered an error";
		} finally {
			isTraining = false;
		}
	} else {
		// Browser development preview simulation
		const freezeArgStr = freezeBackbone ? ` freeze=${freezeLayersCount}` : "";
		logs = [
			"[LS-Trainer] Running outside Tauri webview (Browser Mode)",
			`[LS-Trainer] Simulated training run '${effectiveRunName}'`,
			`[LS-Trainer] Ultralytics task="${taskName}"`,
			`[LS-Trainer] Active config: ${activeRunConfigBanner}`,
			`[LS-Trainer] Command: yolo ${taskName} train task=${taskName} data="${effectiveDataset}" model="${effectiveModel}" name="${effectiveRunName}" epochs=${epochs} batch=${batch} imgsz=${imageSize} device="${device}" amp=${amp}${freezeArgStr} mosaic=${mosaic} close_mosaic=${closeMosaic} mixup=${mixup} copy_paste=${effectiveCopyPaste} degrees=${degrees} flipud=${flipud} translate=${translate} scale=${scale}`,
			"[LS-Trainer] Unpacking archive simulation complete.",
			`[LS-Trainer] Starting ${activeModelLabel} epochs (1/${epochs})...`,
			"[LS-Trainer] Training finished successfully.",
		];
		currentEpoch = Number(epochs);
		totalEpochs = Number(epochs);
		resultsDir = normalizePath(
			`C:/Scripts/LS-Trainer/runs/${taskName}/${effectiveRunName}/`,
		);
		if (autoExportOnnx) {
			statusMessage = "Training complete — exporting ONNX…";
			await handleExportToOnnx();
		}
		statusMessage = onnxPath
			? `Finished — ${normalizePath(onnxPath)}`
			: `Finished — ${normalizePath(resultsDir)}`;
		isTraining = false;
	}
}

async function handleOpenFolder(folderPath: string) {
	if (!folderPath) {
		return;
	}
	if (isTauri()) {
		try {
			await invoke("open_folder", { path: folderPath });
		} catch (err) {
			console.warn("Failed to open folder:", err);
		}
	} else {
		console.log("Open folder requested for:", folderPath);
	}
}

async function handleCopyResultsPath() {
	if (!resultsDir) {
		return;
	}
	try {
		await navigator.clipboard.writeText(normalizePath(resultsDir));
		hasCopiedResultsPath = true;
		setTimeout(() => {
			hasCopiedResultsPath = false;
		}, 2000);
	} catch (err) {
		console.warn("Failed to copy path:", err);
	}
}

async function handleCopyOnnxPath() {
	if (!onnxPath) {
		return;
	}
	try {
		await navigator.clipboard.writeText(normalizePath(onnxPath));
		hasCopiedOnnxPath = true;
		setTimeout(() => {
			hasCopiedOnnxPath = false;
		}, 2000);
	} catch (err) {
		console.warn("Failed to copy ONNX path:", err);
	}
}

async function handleExportToOnnx() {
	if (!resultsDir || isExportingOnnx) {
		return;
	}
	isExportingOnnx = true;
	onnxError = null;
	isLogPanelOpen = true;

	if (isTauri()) {
		try {
			const onLog = new Channel<string>();
			onLog.onmessage = (message) => {
				logs = [...logs, message];
			};

			const response = await invoke<ExportResponse>("export_to_onnx", {
				onLog,
				request: {
					results_dir: resultsDir,
					imgsz: Number(imageSize),
					dynamic: false,
					simplify: true,
				},
			});

			if (response.success && response.onnx_path) {
				onnxPath = normalizePath(response.onnx_path);
			} else {
				onnxError = response.message || "Failed to export ONNX model";
			}
		} catch (err) {
			const errMsg = err instanceof Error ? err.message : String(err);
			onnxError = errMsg;
			logs = [...logs, `[LS-Trainer Error] ${errMsg}`];
		} finally {
			isExportingOnnx = false;
		}
	} else {
		// Browser development preview simulation
		logs = [
			...logs,
			`[LS-Trainer] model.export(format="onnx", imgsz=${imageSize}, dynamic=False, simplify=True)`,
			`[LS-Trainer] Export completed successfully: ${normalizePath(resultsDir)}weights\\best.onnx`,
			"[LS-Trainer] Note: This model is optimized and ready for inference in LS-Vision.",
		];
		onnxPath = normalizePath(`${resultsDir}weights/best.onnx`);
		isExportingOnnx = false;
	}
}
</script>

<svelte:window
	onclick={() => {
		activeTooltip = null;
	}}
	onkeydown={(e) => {
		if (e.key === "Escape") activeTooltip = null;
	}}
/>

<div class="flex h-screen w-screen flex-col overflow-hidden bg-zinc-950 font-sans text-zinc-100 antialiased selection:bg-amber-500/30 selection:text-amber-200">
	<!-- Top Bar -->
	<header class="flex h-14 shrink-0 items-center justify-between border-b border-zinc-800/80 bg-zinc-900/90 px-6 backdrop-blur-md z-10">
		<div class="flex items-center gap-3">
			<img
				src="/LS-Trainer.svg"
				alt="LS-Trainer Logo"
				class="h-7 w-7 rounded-md select-none"
			/>
			<div>
				<h1 class="text-sm font-bold tracking-wider text-zinc-100 uppercase">LS-Trainer</h1>
			</div>
		</div>
		<div class="flex items-center gap-2.5">
			{#if logs.length > 0 || isTraining}
				<button
					type="button"
					onclick={() => {
						isLogPanelOpen = !isLogPanelOpen;
					}}
					class={`flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition ${
						isLogPanelOpen
							? "border-amber-500/50 bg-amber-500/10 text-amber-300"
							: "border-zinc-700 bg-zinc-800/80 text-zinc-300 hover:bg-zinc-700 hover:text-white"
					}`}
				>
					<span class={`h-1.5 w-1.5 rounded-full ${isTraining ? "bg-amber-400 animate-pulse" : "bg-emerald-500"}`}></span>
					<span>{isLogPanelOpen ? "Hide Log" : "Show Log"}</span>
					<span class="font-mono text-[10px] text-zinc-400">({logs.length})</span>
				</button>
			{/if}
			<span class="rounded border border-zinc-700/60 bg-zinc-800/80 px-2.5 py-0.5 font-mono text-xs text-zinc-400">
				v0.1.4
			</span>
		</div>
	</header>

	<!-- Main Workspace Split -->
	<div class="flex flex-1 overflow-hidden relative">
		<!-- Left: Form Area (Scrollable) -->
		<div class="flex flex-1 justify-center overflow-y-auto px-4 pt-4 pb-8 sm:px-8 sm:pt-6 sm:pb-10 overscroll-none">
			<form
				onsubmit={(e) => {
					e.preventDefault();
					handleStartTraining();
				}}
				class="w-full max-w-4xl space-y-6 rounded-xl border border-zinc-800/90 bg-zinc-900/50 p-6 shadow-2xl backdrop-blur-sm sm:p-8"
			>
				<!-- Section 1: Dataset -->
				<section class="space-y-2.5">
					<div class="flex items-center justify-between">
						<label for="dataset-file-input" class="text-xs font-semibold tracking-wider text-zinc-300 uppercase">
							Dataset
						</label>
						<span class="text-xs text-zinc-500">Supported: .tar, .tar.gz</span>
					</div>

					<input
						id="dataset-file-input"
						bind:this={datasetInputRef}
						type="file"
						accept=".tar,.tar.gz,application/x-tar,application/gzip,application/x-gzip"
						onchange={handleDatasetFileInputChange}
						class="hidden"
					/>

					{#if !datasetFileName}
						<button
							type="button"
							onclick={handlePickDataset}
							disabled={isTraining}
							class="group flex w-full flex-col items-center justify-center rounded-lg border-2 border-dashed border-zinc-700/80 bg-zinc-950/40 p-5 text-center transition-all hover:border-amber-500/70 hover:bg-zinc-900/60 focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500/40 disabled:opacity-50"
						>
							<svg
								class="mb-2 h-6 w-6 text-zinc-400 transition-colors group-hover:text-amber-400"
								aria-hidden="true"
								fill="none"
								stroke="currentColor"
								viewBox="0 0 24 24"
							>
								<path
									stroke-linecap="round"
									stroke-linejoin="round"
									stroke-width="1.75"
									d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
								/>
							</svg>
							<p class="text-sm font-medium text-zinc-200">
								Choose dataset archive
							</p>
							<p class="mt-0.5 text-xs text-zinc-500">
								Select a .tar or .tar.gz file from local disk
							</p>
						</button>
					{:else}
						<div class="flex items-center justify-between rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 px-4">
							<div class="flex items-center gap-3 min-w-0">
								<svg
									class="h-5 w-5 shrink-0 text-amber-400"
									aria-hidden="true"
									fill="none"
									stroke="currentColor"
									viewBox="0 0 24 24"
								>
									<path
										stroke-linecap="round"
										stroke-linejoin="round"
										stroke-width="2"
										d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
									/>
								</svg>
								<div class="min-w-0">
									<p class="truncate text-sm font-medium text-zinc-100" title={normalizePath(datasetPath) || datasetFileName}>
										{datasetFileName}
									</p>
									<p class="truncate text-xs text-amber-400/90" title={normalizePath(datasetPath) || datasetFileName}>
										{normalizePath(datasetPath) || "Archive selected"}
									</p>
								</div>
							</div>
							<div class="flex items-center gap-2 pl-3 shrink-0">
								<button
									type="button"
									onclick={handlePickDataset}
									disabled={isTraining}
									class="rounded bg-zinc-800 px-2.5 py-1 text-xs font-medium text-zinc-300 transition hover:bg-zinc-700 hover:text-white disabled:opacity-50"
								>
									Change
								</button>
								<button
									type="button"
									onclick={clearDataset}
									disabled={isTraining}
									class="rounded p-1 text-zinc-400 transition hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50"
									aria-label="Remove dataset"
								>
									<svg class="h-4 w-4" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
									</svg>
								</button>
							</div>
						</div>
					{/if}
				</section>

				<!-- Section 2: Model -->
				<section class="space-y-3 border-t border-zinc-800/80 pt-5">
					<div class="flex items-center justify-between">
						<span class="text-xs font-semibold tracking-wider text-zinc-300 uppercase">
							Model
						</span>
					</div>

					<div class="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
						<!-- Option 1: Start fresh -->
						<label
							class={`flex cursor-pointer items-start gap-3 rounded-lg border p-3.5 transition-all ${
								modelMode === "fresh"
									? "border-amber-500/70 bg-amber-500/10 ring-1 ring-amber-500/40"
									: "border-zinc-800 bg-zinc-950/40 hover:border-zinc-700 hover:bg-zinc-900/50"
							}`}
						>
							<input
								type="radio"
								name="model-mode"
								value="fresh"
								disabled={isTraining}
								checked={modelMode === "fresh"}
								onchange={() => {
									modelMode = "fresh";
								}}
								class="mt-0.5 accent-amber-500"
							/>
							<div>
								<span class="block text-sm font-medium text-zinc-200">Start fresh</span>
								<span class="block text-xs text-zinc-400">Initialize a pretrained detection or segmentation model</span>
							</div>
						</label>

						<!-- Option 2: Continue from existing -->
						<label
							class={`flex cursor-pointer items-start gap-3 rounded-lg border p-3.5 transition-all ${
								modelMode === "continue"
									? "border-amber-500/70 bg-amber-500/10 ring-1 ring-amber-500/40"
									: "border-zinc-800 bg-zinc-950/40 hover:border-zinc-700 hover:bg-zinc-900/50"
							}`}
						>
							<input
								type="radio"
								name="model-mode"
								value="continue"
								disabled={isTraining}
								checked={modelMode === "continue"}
								onchange={() => {
									modelMode = "continue";
								}}
								class="mt-0.5 accent-amber-500"
							/>
							<div>
								<span class="block text-sm font-medium text-zinc-200">Continue from existing</span>
								<span class="block text-xs text-zinc-400">Resume from custom .pt checkpoint</span>
							</div>
						</label>
					</div>

					<div class="space-y-1.5">
						<span class="block text-xs font-medium text-zinc-400">Task</span>
						<div class="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
							<label
								class={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition-all ${
									trainTask === "detect"
										? "border-amber-500/70 bg-amber-500/10 ring-1 ring-amber-500/40"
										: "border-zinc-800 bg-zinc-950/40 hover:border-zinc-700 hover:bg-zinc-900/50"
								}`}
							>
								<input
									type="radio"
									name="train-task"
									value="detect"
									disabled={isTraining}
									checked={trainTask === "detect"}
									onchange={() => selectTrainTask("detect")}
									class="mt-0.5 accent-amber-500"
								/>
								<div>
									<span class="block text-sm font-medium text-zinc-200">Object Detection</span>
									<span class="block text-xs text-zinc-400">Boxes around each part</span>
								</div>
							</label>
							<label
								class={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 transition-all ${
									trainTask === "segment"
										? "border-amber-500/70 bg-amber-500/10 ring-1 ring-amber-500/40"
										: "border-zinc-800 bg-zinc-950/40 hover:border-zinc-700 hover:bg-zinc-900/50"
								}`}
							>
								<input
									type="radio"
									name="train-task"
									value="segment"
									disabled={isTraining}
									checked={trainTask === "segment"}
									onchange={() => selectTrainTask("segment")}
									class="mt-0.5 accent-amber-500"
								/>
								<div>
									<span class="block text-sm font-medium text-zinc-200">Instance Segmentation</span>
									<span class="block text-xs text-zinc-400">Polygon masks (task="segment")</span>
								</div>
							</label>
						</div>
					</div>

					{#if isSegmentationRun && isDatasetSelected}
						{#if labelCheckPending}
							<p class="text-xs text-zinc-400">Checking label files for polygon annotations…</p>
						{:else if labelWarning}
							<div class="rounded-lg border border-amber-500/50 bg-amber-500/10 px-3.5 py-2.5 text-xs leading-relaxed text-amber-100/90">
								{labelWarning}
							</div>
						{:else if labelCheckNote}
							<p class="text-xs text-emerald-400/90">{labelCheckNote}</p>
						{/if}
					{/if}

					<!-- Dynamic Model Inputs based on choice -->
					{#if modelMode === "fresh"}
						<div class="space-y-1.5 pt-1">
							<label for="fresh-model-select" class="block text-xs font-medium text-zinc-400">
								Select Base Architecture
							</label>
							<select
								id="fresh-model-select"
								bind:value={freshModel}
								disabled={isTraining}
								class="w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
							>
								{#if visibleModelOptions.some((o) => o.family === "YOLO11")}
									<optgroup label={trainTask === "segment" ? "YOLO11 Segmentation" : "YOLO11 Series"}>
										{#each visibleModelOptions.filter((o) => o.family === "YOLO11") as opt (opt.id)}
											<option value={opt.id}>
												{opt.label} ({opt.id}) — {opt.desc}
											</option>
										{/each}
									</optgroup>
								{/if}
								{#if visibleModelOptions.some((o) => o.family === "YOLO26")}
									<optgroup label={trainTask === "segment" ? "YOLO26 Segmentation" : "YOLO26 Series"}>
										{#each visibleModelOptions.filter((o) => o.family === "YOLO26") as opt (opt.id)}
											<option value={opt.id}>
												{opt.label} ({opt.id}) — {opt.desc}
											</option>
										{/each}
									</optgroup>
								{/if}
							</select>
						</div>
					{:else}
						<div class="space-y-1.5 pt-1">
							<label for="model-file-input" class="block text-xs font-medium text-zinc-400">
								Checkpoint File (.pt)
							</label>
							<input
								id="model-file-input"
								bind:this={modelInputRef}
								type="file"
								accept=".pt"
								onchange={handleExistingModelFileInputChange}
								class="hidden"
							/>

							{#if !existingModelFileName}
								<button
									type="button"
									onclick={handlePickExistingModel}
									disabled={isTraining}
									class="flex w-full items-center justify-center gap-2 rounded-lg border border-dashed border-zinc-700/80 bg-zinc-950/40 px-4 py-3 text-sm text-zinc-300 transition hover:border-amber-500/70 hover:bg-zinc-900/60 focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500/40 disabled:opacity-50"
								>
									<svg class="h-4 w-4 text-amber-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
									</svg>
									<span>Select existing checkpoint (.pt)</span>
								</button>
							{:else}
								<div class="flex items-center justify-between rounded-lg border border-amber-500/30 bg-amber-500/5 p-3 px-4">
									<div class="flex items-center gap-3 min-w-0">
										<svg
											class="h-5 w-5 shrink-0 text-amber-400"
											aria-hidden="true"
											fill="none"
											stroke="currentColor"
											viewBox="0 0 24 24"
										>
											<path
												stroke-linecap="round"
												stroke-linejoin="round"
												stroke-width="2"
												d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"
											/>
										</svg>
										<div class="min-w-0">
											<span class="block truncate text-sm font-medium text-zinc-100" title={normalizePath(existingModelPath) || existingModelFileName}>
												{existingModelFileName}
											</span>
											<span class="block truncate text-xs text-amber-400/90" title={normalizePath(existingModelPath) || existingModelFileName}>
												{normalizePath(existingModelPath) || "Checkpoint selected"}
											</span>
										</div>
									</div>
									<div class="flex items-center gap-2 pl-3 shrink-0">
										<button
											type="button"
											onclick={handlePickExistingModel}
											disabled={isTraining}
											class="rounded bg-zinc-800 px-2.5 py-1 text-xs font-medium text-zinc-300 transition hover:bg-zinc-700 hover:text-white disabled:opacity-50"
										>
											Change
										</button>
										<button
											type="button"
											onclick={clearExistingModel}
											disabled={isTraining}
											class="rounded p-1 text-zinc-400 transition hover:bg-zinc-800 hover:text-zinc-200 disabled:opacity-50"
											aria-label="Remove model file"
										>
											<svg class="h-4 w-4" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
												<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
											</svg>
										</button>
									</div>
								</div>
							{/if}
						</div>
					{/if}

					<!-- Freeze backbone checkbox -->
					<div class="relative flex items-center justify-between rounded-lg border border-zinc-800/90 bg-zinc-950/40 px-3.5 py-2.5">
						<div class="flex items-center gap-2.5">
							<input
								id="freeze-backbone-checkbox"
								type="checkbox"
								bind:checked={freezeBackbone}
								disabled={isTraining}
								class="h-4 w-4 rounded border-zinc-700 bg-zinc-900 accent-amber-500 focus:ring-amber-500 disabled:opacity-50"
							/>
							<label for="freeze-backbone-checkbox" class="cursor-pointer text-sm font-medium text-zinc-200 select-none">
								Freeze backbone (first {freezeLayersCount} layers)
							</label>
							<button
								type="button"
								onclick={(e) => {
									e.stopPropagation();
									activeTooltip = activeTooltip === "freeze" ? null : "freeze";
								}}
								class="inline-flex h-4 w-4 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-[10px] font-bold text-zinc-400 transition hover:border-amber-500/80 hover:text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-500"
								aria-label="Freeze backbone help"
								title="Freezes early layers so the model mainly learns the new head. Useful when you have limited labeled data."
							>
								?
							</button>
						</div>

						<span class={`font-mono text-xs ${freezeBackbone ? "text-amber-400 font-medium" : "text-zinc-500"}`}>
							{freezeBackbone ? `freeze=${freezeLayersCount}` : "freeze=0"}
						</span>

						{#if activeTooltip === "freeze"}
							<!-- svelte-ignore a11y_no_static_element_interactions -->
							<div
								onclick={(e) => e.stopPropagation()}
								onkeydown={(e) => e.stopPropagation()}
								class="absolute bottom-full left-4 z-30 mb-2 w-72 rounded-lg border border-amber-500/40 bg-zinc-900 p-2.5 text-xs leading-relaxed text-zinc-200 shadow-xl backdrop-blur-md"
							>
								<div class="flex items-start justify-between gap-2 border-b border-zinc-800 pb-1">
									<span class="font-semibold text-amber-400">Freeze backbone</span>
									<button
										type="button"
										onclick={(e) => {
											e.stopPropagation();
											activeTooltip = null;
										}}
										class="text-zinc-400 transition hover:text-zinc-100"
										aria-label="Close tooltip"
									>
										&times;
									</button>
								</div>
								<p class="mt-1 text-zinc-300">
									Freezes early layers so the model mainly learns the new head. Useful when you have limited labeled data.
								</p>
							</div>
						{/if}
					</div>
				</section>

				<!-- Section 3: Output -->
				<section class="space-y-2 border-t border-zinc-800/80 pt-5">
					<label for="run-name-input" class="block text-xs font-semibold tracking-wider text-zinc-300 uppercase">
						Output
					</label>
					<div class="relative">
						<input
							id="run-name-input"
							type="text"
							bind:value={runName}
							disabled={isTraining}
							placeholder="BB-Parts-v3"
							class="w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 placeholder-zinc-500 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
						/>
					</div>
					<p class="text-xs text-zinc-500">Run identifier for checkpoint directories and evaluation metrics</p>
				</section>

				<!-- Section 4: Parameters / Settings -->
				<section class="space-y-3 border-t border-zinc-800/80 pt-5">
					<div class="flex items-center justify-between gap-2">
						<span class="block text-xs font-semibold tracking-wider text-zinc-300 uppercase">
							Parameters &amp; Settings
						</span>
						<button
							type="button"
							onclick={handleRepairCuda}
							disabled={isTraining || isRepairingCuda}
							class="inline-flex items-center gap-1.5 rounded-md border border-zinc-700/80 bg-zinc-800/80 px-2.5 py-1 text-xs font-medium text-zinc-300 transition hover:border-amber-500/60 hover:bg-zinc-800 hover:text-amber-300 disabled:opacity-50"
							title="Re-detect GPU and driver version and install the matching PyTorch CUDA wheel"
						>
							{#if isRepairingCuda}
								<svg class="h-3.5 w-3.5 animate-spin text-amber-400" aria-hidden="true" fill="none" viewBox="0 0 24 24">
									<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
									<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
								</svg>
								<span>Repairing CUDA…</span>
							{:else}
								<svg class="h-3.5 w-3.5 text-amber-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
								</svg>
								<span>Repair CUDA</span>
							{/if}
						</button>
					</div>

					{#if isRepairingCuda}
						<div class="flex items-center gap-2.5 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3.5 py-2.5 text-xs font-medium text-amber-300">
							<svg class="h-4 w-4 shrink-0 animate-spin text-amber-400" aria-hidden="true" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
							</svg>
							<span>{statusMessage}</span>
						</div>
					{/if}

					<div class="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
						<!-- Epochs and Batch -->
						<div class="grid grid-cols-2 gap-3">
							<div class="space-y-1.5">
								<label for="epochs-input" class="flex h-5 items-center text-xs font-medium text-zinc-400">
									Epochs
								</label>
								<input
									id="epochs-input"
									type="number"
									min="1"
									step="1"
									bind:value={epochs}
									disabled={isTraining}
									class="w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
								/>
							</div>

							<div class="space-y-1.5">
								<label for="batch-input" class="flex h-5 items-center text-xs font-medium text-zinc-400">
									Batch
								</label>
								<input
									id="batch-input"
									type="number"
									min="1"
									step="1"
									bind:value={batch}
									disabled={isTraining}
									class="w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
								/>
							</div>
						</div>

						<!-- Image size -->
						<div class="space-y-1.5">
							<label for="image-size-select" class="flex h-5 items-center text-xs font-medium text-zinc-400">
								Image size
							</label>
							<select
								id="image-size-select"
								bind:value={imageSize}
								disabled={isTraining}
								class="w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
							>
								<option value={640}>640 px</option>
								<option value={960}>960 px</option>
								<option value={1280}>1280 px</option>
							</select>
						</div>

						<!-- Device -->
						<div class="space-y-1.5 min-w-0">
							<div class="flex h-5 items-center justify-between gap-2 whitespace-nowrap">
								<label for="device-select" class="block text-xs font-medium text-zinc-400 shrink-0">
									Device
								</label>
								{#if gpuInfo?.cuda_available && gpuInfo.device_name}
									<span class="flex items-center gap-1.5 text-[11px] font-medium text-emerald-400 whitespace-nowrap truncate min-w-0" title={gpuInfo.device_name}>
										<span class="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400"></span>
										<span class="truncate">{shortenGpuName(gpuInfo.device_name)}</span>
									</span>
								{:else if gpuInfo && !gpuInfo.cuda_available}
									<span class="text-[11px] text-zinc-500 whitespace-nowrap">
										CPU only (No CUDA GPU)
									</span>
								{/if}
							</div>
							<select
								id="device-select"
								value={isCustomDevice ? "custom" : device}
								onchange={(e) => {
									const val = e.currentTarget.value;
									if (val === "custom") {
										isCustomDevice = true;
										device = customDevice || "0";
									} else {
										isCustomDevice = false;
										device = val;
									}
								}}
								disabled={isTraining}
								class="w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
							>
								{#if gpuInfo?.cuda_available}
									<option value="0">
										{gpuInfo.device_name ? `GPU 0: ${ gpuInfo.device_name }` : "GPU (device=0)"}
									</option>
									<option value="cpu">CPU (device=cpu)</option>
									<option value="custom">Custom device...</option>
								{:else if gpuInfo !== null}
									<option value="cpu">CPU (device=cpu)</option>
									<option value="custom">Custom device...</option>
								{:else}
									<option value="0">GPU (device=0)</option>
									<option value="cpu">CPU (device=cpu)</option>
									<option value="custom">Custom device...</option>
								{/if}
							</select>

							{#if isCustomDevice}
								<input
									type="text"
									placeholder="e.g. 0,1 or mps"
									bind:value={customDevice}
									oninput={() => {
										device = customDevice.trim();
									}}
									disabled={isTraining}
									class="mt-1.5 w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2 text-xs text-zinc-100 placeholder-zinc-500 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
								/>
							{/if}
						</div>

						<!-- amp -->
						<div class="space-y-1.5">
							<div class="relative flex h-5 items-center justify-between">
								<div class="flex items-center gap-1.5">
									<label for="amp-toggle" class="text-xs font-medium text-zinc-400">
										amp
									</label>
									<button
										type="button"
										onclick={(e) => {
											e.stopPropagation();
											activeTooltip = activeTooltip === "amp" ? null : "amp";
										}}
										class="inline-flex h-4 w-4 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-[10px] font-bold text-zinc-400 transition hover:border-amber-500/80 hover:text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-500"
										aria-label="amp help"
										title="Automatic Mixed Precision. Uses FP16 where safe to reduce VRAM. Leave on unless you hit numerical problems."
									>
										?
									</button>
								</div>

								{#if activeTooltip === "amp"}
									<!-- svelte-ignore a11y_no_static_element_interactions -->
									<div
										onclick={(e) => e.stopPropagation()}
										onkeydown={(e) => e.stopPropagation()}
										class="absolute bottom-full right-0 z-30 mb-2 w-64 rounded-lg border border-amber-500/40 bg-zinc-900 p-2.5 text-xs leading-relaxed text-zinc-200 shadow-xl backdrop-blur-md"
									>
										<div class="flex items-start justify-between gap-2 border-b border-zinc-800 pb-1">
											<span class="font-semibold text-amber-400">amp</span>
											<button
												type="button"
												onclick={(e) => {
													e.stopPropagation();
													activeTooltip = null;
												}}
												class="text-zinc-400 transition hover:text-zinc-100"
												aria-label="Close tooltip"
											>
												&times;
											</button>
										</div>
										<p class="mt-1 text-zinc-300">
											Automatic Mixed Precision. Uses FP16 where safe to reduce VRAM. Leave on unless you hit numerical problems.
										</p>
									</div>
								{/if}
							</div>

							<button
								id="amp-toggle"
								type="button"
								role="switch"
								aria-checked={amp}
								onclick={() => {
									if (!isTraining) amp = !amp;
								}}
								disabled={isTraining}
								class={`flex h-[42px] w-full items-center justify-between rounded-lg border px-3.5 py-2 text-sm font-medium transition ${
									amp
										? "border-amber-500/50 bg-amber-500/10 text-amber-300"
										: "border-zinc-700/80 bg-zinc-900 text-zinc-400 hover:border-zinc-600"
								} disabled:opacity-50`}
							>
								<span>{amp ? "Enabled (FP16)" : "Disabled (FP32)"}</span>
								<span
									class={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out ${
										amp ? "bg-amber-500" : "bg-zinc-700"
									}`}
								>
									<span
										class={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition duration-200 ease-in-out ${
											amp ? "translate-x-4" : "translate-x-0"
										}`}
									></span>
								</span>
							</button>
						</div>
					</div>
				</section>

				<!-- Section 5: Augmentation Controls -->
				<section class="space-y-3.5 border-t border-zinc-800/80 pt-5">
					<div class="flex flex-wrap items-center justify-between gap-2">
						<div class="flex items-center gap-2">
							<span class="block text-xs font-semibold tracking-wider text-zinc-300 uppercase">
								Augmentation
							</span>
							<span class="rounded border border-zinc-800 bg-zinc-950/60 px-2 py-0.5 font-mono text-[11px] text-zinc-400">
								{activeAugSummary}
							</span>
						</div>

						<!-- Preset buttons: Bench / Off / Light / Medium -->
						<div class="flex items-center gap-1 rounded-lg border border-zinc-800 bg-zinc-950/60 p-0.5">
							<span class="px-2 text-[11px] text-zinc-500">Preset:</span>
							<button
								type="button"
								onclick={() => applyAugPreset("bench")}
								disabled={isTraining}
								class={`rounded-md px-2 py-0.5 text-xs font-medium transition ${
									activeAugPreset === "bench"
										? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
										: "text-zinc-400 hover:text-zinc-200"
								} disabled:opacity-50`}
							>
								Bench
							</button>
							<button
								type="button"
								onclick={() => applyAugPreset("off")}
								disabled={isTraining}
								class={`rounded-md px-2 py-0.5 text-xs font-medium transition ${
									activeAugPreset === "off"
										? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
										: "text-zinc-400 hover:text-zinc-200"
								} disabled:opacity-50`}
							>
								Off
							</button>
							<button
								type="button"
								onclick={() => applyAugPreset("light")}
								disabled={isTraining}
								class={`rounded-md px-2 py-0.5 text-xs font-medium transition ${
									activeAugPreset === "light"
										? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
										: "text-zinc-400 hover:text-zinc-200"
								} disabled:opacity-50`}
							>
								Light
							</button>
							<button
								type="button"
								onclick={() => applyAugPreset("medium")}
								disabled={isTraining}
								class={`rounded-md px-2 py-0.5 text-xs font-medium transition ${
									activeAugPreset === "medium"
										? "bg-amber-500/20 text-amber-300 border border-amber-500/40"
										: "text-zinc-400 hover:text-zinc-200"
								} disabled:opacity-50`}
							>
								Medium
							</button>
						</div>
					</div>

					<!-- Row 1: Mosaic (dropdown 0.0 / 0.5 / 1.0), MixUp (slider 0.0–0.5), Copy-paste (on/off) -->
					<div class="grid grid-cols-1 gap-4 sm:grid-cols-3">
						<!-- Mosaic dropdown -->
						<div class="space-y-1.5">
							<div class="relative flex h-5 items-center justify-between">
								<div class="flex items-center gap-1.5">
									<label for="mosaic-select" class="text-xs font-medium text-zinc-400">
										Mosaic
									</label>
									<button
										type="button"
										onclick={(e) => {
											e.stopPropagation();
											activeTooltip = activeTooltip === "mosaic" ? null : "mosaic";
										}}
										class="inline-flex h-4 w-4 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-[10px] font-bold text-zinc-400 transition hover:border-amber-500/80 hover:text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-500"
										aria-label="mosaic help"
										title="Probability of stitching four images into one. 0.0 disables it. Useful to turn off for controlled industrial scenes."
									>
										?
									</button>
								</div>

								{#if activeTooltip === "mosaic"}
									<!-- svelte-ignore a11y_no_static_element_interactions -->
									<div
										onclick={(e) => e.stopPropagation()}
										onkeydown={(e) => e.stopPropagation()}
										class="absolute bottom-full left-0 z-30 mb-2 w-64 rounded-lg border border-amber-500/40 bg-zinc-900 p-2.5 text-xs leading-relaxed text-zinc-200 shadow-xl backdrop-blur-md"
									>
										<div class="flex items-start justify-between gap-2 border-b border-zinc-800 pb-1">
											<span class="font-semibold text-amber-400">mosaic</span>
											<button
												type="button"
												onclick={(e) => {
													e.stopPropagation();
													activeTooltip = null;
												}}
												class="text-zinc-400 transition hover:text-zinc-100"
												aria-label="Close tooltip"
											>
												&times;
											</button>
										</div>
										<p class="mt-1 text-zinc-300">
											Probability of stitching four images into one. 0.0 disables it. Useful to turn off for controlled industrial scenes.
										</p>
									</div>
								{/if}
							</div>

							<select
								id="mosaic-select"
								bind:value={mosaic}
								disabled={isTraining}
								class="h-[42px] w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
							>
								<option value={0.0}>0.0 — Off (Bench default)</option>
								<option value={0.5}>0.5 — Moderate</option>
								<option value={1.0}>1.0 — Full (Ultralytics default)</option>
							</select>
						</div>

						<!-- MixUp slider 0.0–0.5 -->
						<div class="space-y-1.5">
							<div class="relative flex h-5 items-center justify-between">
								<div class="flex items-center gap-1.5">
									<label for="mixup-slider" class="text-xs font-medium text-zinc-400">
										MixUp
									</label>
									<button
										type="button"
										onclick={(e) => {
											e.stopPropagation();
											activeTooltip = activeTooltip === "mixup" ? null : "mixup";
										}}
										class="inline-flex h-4 w-4 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-[10px] font-bold text-zinc-400 transition hover:border-amber-500/80 hover:text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-500"
										aria-label="mixup help"
										title="Probability of blending two images and their labels. Usually left at 0.0 for clean bench photos."
									>
										?
									</button>
								</div>
								<span class="font-mono text-xs text-amber-400">{mixup.toFixed(2)}</span>

								{#if activeTooltip === "mixup"}
									<!-- svelte-ignore a11y_no_static_element_interactions -->
									<div
										onclick={(e) => e.stopPropagation()}
										onkeydown={(e) => e.stopPropagation()}
										class="absolute bottom-full left-0 z-30 mb-2 w-64 rounded-lg border border-amber-500/40 bg-zinc-900 p-2.5 text-xs leading-relaxed text-zinc-200 shadow-xl backdrop-blur-md"
									>
										<div class="flex items-start justify-between gap-2 border-b border-zinc-800 pb-1">
											<span class="font-semibold text-amber-400">mixup</span>
											<button
												type="button"
												onclick={(e) => {
													e.stopPropagation();
													activeTooltip = null;
												}}
												class="text-zinc-400 transition hover:text-zinc-100"
												aria-label="Close tooltip"
											>
												&times;
											</button>
										</div>
										<p class="mt-1 text-zinc-300">
											Probability of blending two images and their labels. Usually left at 0.0 for clean bench photos.
										</p>
									</div>
								{/if}
							</div>

							<div class="flex h-[42px] items-center gap-3 rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5">
								<input
									id="mixup-slider"
									type="range"
									min="0"
									max="0.5"
									step="0.05"
									bind:value={mixup}
									disabled={isTraining}
									class="w-full accent-amber-500 disabled:opacity-50"
								/>
							</div>
						</div>

						<!-- Copy-paste on/off -->
						<div class="space-y-1.5">
							<div class="relative flex h-5 items-center justify-between">
								<div class="flex items-center gap-1.5">
									<label for="copy-paste-toggle" class="text-xs font-medium text-zinc-400">
										Copy-paste
									</label>
									<button
										type="button"
										onclick={(e) => {
											e.stopPropagation();
											activeTooltip = activeTooltip === "copy_paste" ? null : "copy_paste";
										}}
										class="inline-flex h-4 w-4 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-[10px] font-bold text-zinc-400 transition hover:border-amber-500/80 hover:text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-500"
										aria-label="copy-paste help"
										title="Copy-paste object patches across training images. Default off for clean bench work."
									>
										?
									</button>
								</div>

								{#if activeTooltip === "copy_paste"}
									<!-- svelte-ignore a11y_no_static_element_interactions -->
									<div
										onclick={(e) => e.stopPropagation()}
										onkeydown={(e) => e.stopPropagation()}
										class="absolute bottom-full right-0 z-30 mb-2 w-64 rounded-lg border border-amber-500/40 bg-zinc-900 p-2.5 text-xs leading-relaxed text-zinc-200 shadow-xl backdrop-blur-md"
									>
										<div class="flex items-start justify-between gap-2 border-b border-zinc-800 pb-1">
											<span class="font-semibold text-amber-400">copy_paste</span>
											<button
												type="button"
												onclick={(e) => {
													e.stopPropagation();
													activeTooltip = null;
												}}
												class="text-zinc-400 transition hover:text-zinc-100"
												aria-label="Close tooltip"
											>
												&times;
											</button>
										</div>
										<p class="mt-1 text-zinc-300">
											Copy-paste object patches across training images. Default off (0.0) for controlled bench scenes.
										</p>
									</div>
								{/if}
							</div>

							<button
								id="copy-paste-toggle"
								type="button"
								role="switch"
								aria-checked={copyPaste}
								onclick={() => {
									if (!isTraining) copyPaste = !copyPaste;
								}}
								disabled={isTraining}
								class={`flex h-[42px] w-full items-center justify-between rounded-lg border px-3.5 py-2 text-sm font-medium transition ${
									copyPaste
										? "border-amber-500/50 bg-amber-500/10 text-amber-300"
										: "border-zinc-700/80 bg-zinc-900 text-zinc-400 hover:border-zinc-600"
								} disabled:opacity-50`}
							>
								<span>{copyPaste ? "On (0.3)" : "Off (0.0)"}</span>
								<span
									class={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out ${
										copyPaste ? "bg-amber-500" : "bg-zinc-700"
									}`}
								>
									<span
										class={`inline-block h-4 w-4 transform rounded-full bg-white shadow transition duration-200 ease-in-out ${
											copyPaste ? "translate-x-4" : "translate-x-0"
										}`}
									></span>
								</span>
							</button>
						</div>
					</div>

					<!-- Row 2: Degrees / Translate / Scale number inputs -->
					<div class="grid grid-cols-1 gap-4 sm:grid-cols-3">
						<!-- Degrees -->
						<div class="space-y-1.5">
							<div class="flex h-5 items-center justify-between">
								<label for="degrees-input" class="text-xs font-medium text-zinc-400" title="180° gives top-down rotational invariance on a bench.">
									Degrees (±rot)
								</label>
								<span class="font-mono text-[11px] text-zinc-500">0–180°</span>
							</div>
							<input
								id="degrees-input"
								type="number"
								min="0"
								max="180"
								step="1"
								bind:value={degrees}
								disabled={isTraining}
								class="h-[42px] w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
							/>
						</div>

						<!-- Translate -->
						<div class="space-y-1.5">
							<div class="flex h-5 items-center justify-between">
								<label for="translate-input" class="text-xs font-medium text-zinc-400">
									Translate (±shift)
								</label>
								<span class="font-mono text-[11px] text-zinc-500">0.0–1.0</span>
							</div>
							<input
								id="translate-input"
								type="number"
								min="0"
								max="1"
								step="0.05"
								bind:value={translate}
								disabled={isTraining}
								class="h-[42px] w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
							/>
						</div>

						<!-- Scale -->
						<div class="space-y-1.5">
							<div class="relative flex h-5 items-center justify-between">
								<div class="flex items-center gap-1.5">
									<label for="scale-input" class="text-xs font-medium text-zinc-400">
										Scale (±zoom)
									</label>
									<button
										type="button"
										onclick={(e) => {
											e.stopPropagation();
											activeTooltip = activeTooltip === "scale" ? null : "scale";
										}}
										class="inline-flex h-4 w-4 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-[10px] font-bold text-zinc-400 transition hover:border-amber-500/80 hover:text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-500"
										aria-label="scale help"
										title="Random zoom range. Bench default 0.05 is ±5% for a fixed working distance."
									>
										?
									</button>
								</div>
								<span class="font-mono text-[11px] text-zinc-500">0.0–1.0</span>

								{#if activeTooltip === "scale"}
									<!-- svelte-ignore a11y_no_static_element_interactions -->
									<div
										onclick={(e) => e.stopPropagation()}
										onkeydown={(e) => e.stopPropagation()}
										class="absolute bottom-full right-0 z-30 mb-2 w-64 rounded-lg border border-amber-500/40 bg-zinc-900 p-2.5 text-xs leading-relaxed text-zinc-200 shadow-xl backdrop-blur-md"
									>
										<div class="flex items-start justify-between gap-2 border-b border-zinc-800 pb-1">
											<span class="font-semibold text-amber-400">scale</span>
											<button
												type="button"
												onclick={(e) => {
													e.stopPropagation();
													activeTooltip = null;
												}}
												class="text-zinc-400 transition hover:text-zinc-100"
												aria-label="Close tooltip"
											>
												&times;
											</button>
										</div>
										<p class="mt-1 text-zinc-300">
											Random zoom range. 0.05 means ±5%. The bench default keeps part sizes close to a fixed working distance.
										</p>
									</div>
								{/if}
							</div>

							<input
								id="scale-input"
								type="number"
								min="0"
								max="1"
								step="0.01"
								bind:value={scale}
								disabled={isTraining}
								class="h-[42px] w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
							/>
						</div>
					</div>

					<div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
						<div class="space-y-1.5">
							<div class="relative flex h-5 items-center justify-between">
								<div class="flex items-center gap-1.5">
									<label for="flipud-input" class="text-xs font-medium text-zinc-400">
										Flip UD
									</label>
									<button
										type="button"
										onclick={(e) => {
											e.stopPropagation();
											activeTooltip = activeTooltip === "flipud" ? null : "flipud";
										}}
										class="inline-flex h-4 w-4 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-[10px] font-bold text-zinc-400 transition hover:border-amber-500/80 hover:text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-500"
										aria-label="flipud help"
										title="Probability of a vertical flip. 0.5 gives top-down parts the same invariance as a horizontal flip."
									>
										?
									</button>
								</div>
								<span class="font-mono text-[11px] text-zinc-500">0.0–1.0</span>

								{#if activeTooltip === "flipud"}
									<!-- svelte-ignore a11y_no_static_element_interactions -->
									<div
										onclick={(e) => e.stopPropagation()}
										onkeydown={(e) => e.stopPropagation()}
										class="absolute bottom-full left-0 z-30 mb-2 w-64 rounded-lg border border-amber-500/40 bg-zinc-900 p-2.5 text-xs leading-relaxed text-zinc-200 shadow-xl backdrop-blur-md"
									>
										<div class="flex items-start justify-between gap-2 border-b border-zinc-800 pb-1">
											<span class="font-semibold text-amber-400">flipud</span>
											<button
												type="button"
												onclick={(e) => {
													e.stopPropagation();
													activeTooltip = null;
												}}
												class="text-zinc-400 transition hover:text-zinc-100"
												aria-label="Close tooltip"
											>
												&times;
											</button>
										</div>
										<p class="mt-1 text-zinc-300">
											Probability of flipping the image upside down. Bench default is 0.5 so a top-down camera is invariant to which way the part is placed.
										</p>
									</div>
								{/if}
							</div>
							<input
								id="flipud-input"
								type="number"
								min="0"
								max="1"
								step="0.05"
								bind:value={flipud}
								disabled={isTraining}
								class="h-[42px] w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
							/>
						</div>

						<div class="space-y-1.5">
							<div class="relative flex h-5 items-center justify-between">
								<div class="flex items-center gap-1.5">
									<label for="close-mosaic-input" class="text-xs font-medium text-zinc-400">
										Close mosaic
									</label>
									<button
										type="button"
										onclick={(e) => {
											e.stopPropagation();
											activeTooltip = activeTooltip === "close_mosaic" ? null : "close_mosaic";
										}}
										class="inline-flex h-4 w-4 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-[10px] font-bold text-zinc-400 transition hover:border-amber-500/80 hover:text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-500"
										aria-label="close mosaic help"
										title="Turns mosaic off for the final N epochs. Bench default is 10."
									>
										?
									</button>
								</div>
								<span class="font-mono text-[11px] text-zinc-500">epochs</span>

								{#if activeTooltip === "close_mosaic"}
									<!-- svelte-ignore a11y_no_static_element_interactions -->
									<div
										onclick={(e) => e.stopPropagation()}
										onkeydown={(e) => e.stopPropagation()}
										class="absolute bottom-full right-0 z-30 mb-2 w-64 rounded-lg border border-amber-500/40 bg-zinc-900 p-2.5 text-xs leading-relaxed text-zinc-200 shadow-xl backdrop-blur-md"
									>
										<div class="flex items-start justify-between gap-2 border-b border-zinc-800 pb-1">
											<span class="font-semibold text-amber-400">close_mosaic</span>
											<button
												type="button"
												onclick={(e) => {
													e.stopPropagation();
													activeTooltip = null;
												}}
												class="text-zinc-400 transition hover:text-zinc-100"
												aria-label="Close tooltip"
											>
												&times;
											</button>
										</div>
										<p class="mt-1 text-zinc-300">
											Disables mosaic for the last N epochs so the model finishes on whole images. 10 is the bench default.
										</p>
									</div>
								{/if}
							</div>
							<input
								id="close-mosaic-input"
								type="number"
								min="0"
								step="1"
								bind:value={closeMosaic}
								disabled={isTraining}
								class="h-[42px] w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
							/>
						</div>
					</div>
				</section>

				<!-- Section 6: Go Button -->
				<div class="border-t border-zinc-800/80 pt-5 space-y-3">
					<label class="flex items-start gap-3 rounded-lg border border-zinc-800/90 bg-zinc-950/40 px-3.5 py-2.5">
						<input
							type="checkbox"
							bind:checked={autoExportOnnx}
							disabled={isTraining}
							class="mt-0.5 h-4 w-4 rounded border-zinc-700 bg-zinc-900 accent-amber-500 focus:ring-amber-500 disabled:opacity-50"
						/>
						<span>
							<span class="block text-sm font-medium text-zinc-200">Auto-export ONNX on completion</span>
							<span class="block text-xs text-zinc-400">
								After a successful run, export best.pt with format="onnx", the selected image size, dynamic=False, and simplify=True.
							</span>
						</span>
					</label>

					{#if epochCountdownText}
						<div class="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 space-y-2">
							<div class="flex flex-wrap items-center justify-between gap-2">
								<span class="font-mono text-sm font-semibold text-amber-300 whitespace-pre">
									{epochCountdownText}
								</span>
								<span class="font-mono text-xs font-medium text-amber-400/90">
									{epochProgressPercent}%
								</span>
							</div>
							<div class="font-mono text-[11px] text-zinc-300/90 truncate" title={activeRunConfigBanner}>
								{activeRunConfigBanner}
							</div>
							<div class="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
								<div
									class="h-full rounded-full bg-amber-500 transition-all duration-300"
									style={`width: ${epochProgressPercent}%`}
								></div>
							</div>
						</div>
					{/if}

					<button
						type="submit"
						disabled={!isFormValid || isTraining}
						class={`flex w-full items-center justify-center gap-2 rounded-lg py-3 text-sm font-semibold tracking-wide transition-all ${
							isFormValid && !isTraining
								? "cursor-pointer border border-amber-400/80 bg-amber-500 text-zinc-950 shadow-[0_0_20px_rgba(245,158,11,0.25)] hover:bg-amber-400 hover:shadow-[0_0_25px_rgba(245,158,11,0.4)] active:scale-[0.99]"
								: "cursor-not-allowed border border-zinc-800 bg-zinc-800/50 text-zinc-500 opacity-60"
						}`}
					>
						{#if isRepairingCuda || isCheckingGpu}
							<svg class="h-4 w-4 animate-spin text-zinc-400" aria-hidden="true" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
							</svg>
							<span>{statusMessage}</span>
						{:else if isExportingOnnx}
							<svg class="h-4 w-4 animate-spin text-zinc-950" aria-hidden="true" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
							</svg>
							<span class="text-zinc-950 font-medium">Exporting ONNX...</span>
						{:else if isTraining}
							<svg class="h-4 w-4 animate-spin text-zinc-950" aria-hidden="true" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
							</svg>
							<span class="text-zinc-950 font-medium">Training in progress...</span>
						{:else}
							<svg class="h-4 w-4" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
							</svg>
							<span>Train</span>
						{/if}
					</button>
				</div>

				<!-- Status Line -->
				<div class="flex items-center justify-between border-t border-zinc-800/60 pt-4 text-xs text-zinc-400">
					<div class="flex items-center gap-2">
						<span class={`h-2 w-2 rounded-full ${isTraining || isRepairingCuda ? "bg-amber-400 animate-pulse shadow-[0_0_8px_rgba(245,158,11,0.8)]" : isTorchReady ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]" : "bg-amber-500/80 shadow-[0_0_6px_rgba(245,158,11,0.6)]"}`}></span>
						<span>{statusMessage}</span>
					</div>
					{#if (logs.length > 0 || isTraining || isRepairingCuda) && !isLogPanelOpen}
						<button
							type="button"
							onclick={() => {
								isLogPanelOpen = true;
							}}
							class="text-amber-400 hover:text-amber-300 font-medium transition"
						>
							View Log &rarr;
						</button>
					{/if}
				</div>
			</form>
		</div>

		<!-- Right: Resizable Training Log Panel -->
		{#if isLogPanelOpen}
			<!-- Resizer Divider Handle -->
			<!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
			<div
				role="separator"
				aria-orientation="vertical"
				aria-valuenow={logPanelWidth}
				tabindex="-1"
				onmousedown={startResize}
				class={`group relative flex w-2 shrink-0 cursor-col-resize select-none items-center justify-center border-x border-zinc-800/80 bg-zinc-900/60 transition-colors hover:bg-amber-500/70 ${
					isDragging ? "bg-amber-500" : ""
				}`}
				title="Drag horizontally to resize"
			>
				<div class="h-8 w-0.5 rounded-full bg-zinc-600 transition-colors group-hover:bg-zinc-950"></div>
			</div>

			<!-- Right Log Aside -->
			<aside
				style="width: {logPanelWidth}px;"
				class="flex h-full shrink-0 flex-col overflow-hidden border-l border-zinc-800/80 bg-zinc-950"
			>
				<!-- Log Panel Header -->
				<div class="flex shrink-0 flex-col border-b border-zinc-800/80 bg-zinc-900/90 px-4 py-2.5 gap-1">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class={`h-2 w-2 rounded-full ${isTraining ? "bg-amber-400 animate-pulse shadow-[0_0_8px_rgba(245,158,11,0.9)]" : "bg-emerald-500"}`}></span>
							<span class="text-xs font-semibold tracking-wider text-zinc-300 uppercase">
								Training Log
							</span>
							{#if isTraining}
								<span class="rounded border border-amber-500/30 bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-400">
									Streaming
								</span>
							{/if}
						</div>

						<div class="flex items-center gap-1.5">
							<span class="font-mono text-[11px] text-zinc-500 mr-1">
								{logs.length} lines
							</span>
							<button
								type="button"
								onclick={() => {
									logs = [];
								}}
								disabled={isTraining || logs.length === 0}
								class="rounded bg-zinc-800/80 px-2 py-1 text-[11px] text-zinc-400 transition hover:bg-zinc-700 hover:text-zinc-200 disabled:opacity-40"
							>
								Clear
							</button>
							<button
								type="button"
								onclick={copyLogsToClipboard}
								disabled={logs.length === 0}
								class="rounded bg-zinc-800/80 p-1 text-zinc-400 transition hover:bg-zinc-700 hover:text-zinc-200 disabled:opacity-40"
								title={hasCopiedLogs ? "Copied!" : "Copy logs to clipboard"}
								aria-label="Copy logs to clipboard"
							>
								{#if hasCopiedLogs}
									<svg class="h-3.5 w-3.5 text-emerald-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7" />
									</svg>
								{:else}
									<svg class="h-3.5 w-3.5" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
									</svg>
								{/if}
							</button>
							<button
								type="button"
								onclick={() => {
									isLogPanelOpen = false;
								}}
								class="ml-1 rounded p-1 text-zinc-400 transition hover:bg-zinc-800 hover:text-zinc-200"
								title="Collapse panel"
								aria-label="Close log panel"
							>
								<svg class="h-4 w-4" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" />
								</svg>
							</button>
						</div>
					</div>
					<div class="font-mono text-[11px] text-amber-300/90 truncate" title={activeRunConfigBanner}>
						{activeRunConfigBanner}
					</div>
				</div>

				<!-- Results Folder Banner & ONNX Export (if present) -->
				{#if resultsDir}
					<div class="m-3 mb-0 space-y-2.5 shrink-0">
						<!-- Run directory card -->
						<div class="relative rounded-lg border border-amber-500/40 bg-amber-500/10 p-3.5 text-zinc-200 space-y-2.5">
							<!-- Absolute top-right action buttons: Open Folder & Copy Path -->
							<div class="absolute top-2.5 right-2.5 flex items-center gap-1.5 z-10">
								<!-- Open Folder -->
								<button
									type="button"
									onclick={() => {
										if (resultsDir) handleOpenFolder(resultsDir);
									}}
									class="inline-flex items-center gap-1.5 rounded bg-zinc-900/90 hover:bg-zinc-800 border border-zinc-700/80 px-2.5 py-1 text-xs font-medium text-zinc-200 transition hover:text-white"
									title="Open results folder in file explorer"
								>
									<svg class="h-3.5 w-3.5 text-amber-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
									</svg>
									<span>Open Folder</span>
								</button>

								<!-- Copy Path -->
								<button
									type="button"
									onclick={handleCopyResultsPath}
									class="inline-flex items-center gap-1.5 rounded bg-zinc-900/90 hover:bg-zinc-800 border border-zinc-700/80 px-2.5 py-1 text-xs font-medium text-zinc-200 transition hover:text-white"
									title="Copy directory path to clipboard"
								>
									{#if hasCopiedResultsPath}
										<svg class="h-3.5 w-3.5 text-emerald-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7" />
										</svg>
										<span class="text-emerald-400 font-medium">Copied!</span>
									{:else}
										<svg class="h-3.5 w-3.5 text-zinc-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
										</svg>
										<span>Copy Path</span>
									{/if}
								</button>
							</div>

							<div class="flex items-start justify-between gap-3 pr-48">
								<div class="flex items-start gap-2.5 min-w-0">
									<svg class="h-4 w-4 shrink-0 text-amber-400 mt-0.5" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
									</svg>
									<div class="min-w-0">
										<p class="text-[11px] font-semibold uppercase tracking-wider text-amber-300">
											Training Results Directory
										</p>
										<p class="mt-0.5 font-mono text-xs font-medium text-zinc-100 select-all break-all">
											{normalizePath(resultsDir)}
										</p>
									</div>
								</div>
							</div>

							<!-- Convert to ONNX -->
							<div class="pt-1 border-t border-amber-500/20">
								<button
									type="button"
									onclick={handleExportToOnnx}
									disabled={isExportingOnnx}
									class="inline-flex items-center gap-1.5 rounded bg-amber-500 hover:bg-amber-400 text-zinc-950 font-semibold px-2.5 py-1 text-xs transition shadow-sm hover:shadow active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed"
									title="Export trained best.pt model to ONNX format"
								>
									{#if isExportingOnnx}
										<svg class="h-3.5 w-3.5 animate-spin" aria-hidden="true" fill="none" viewBox="0 0 24 24">
											<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
											<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
										</svg>
										<span>Exporting to ONNX...</span>
									{:else}
										<svg class="h-3.5 w-3.5" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
										</svg>
										<span>{onnxPath ? "Re-export ONNX" : "Convert to ONNX"}</span>
									{/if}
								</button>
							</div>
						</div>

						<!-- ONNX Success Card -->
						{#if onnxPath}
							<div class="relative rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-3 text-zinc-200 space-y-2">
								<!-- Absolute top-right action buttons for ONNX -->
								<div class="absolute top-2.5 right-2.5 flex items-center gap-1.5 z-10">
									<!-- Open Folder -->
									<button
										type="button"
										onclick={() => {
											if (onnxPath) handleOpenFolder(onnxPath);
										}}
										class="inline-flex items-center gap-1.5 rounded bg-zinc-900/90 hover:bg-zinc-800 border border-zinc-700/80 px-2.5 py-1 text-xs font-medium text-zinc-200 transition hover:text-white"
										title="Open ONNX model folder in file explorer"
									>
										<svg class="h-3.5 w-3.5 text-emerald-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 19a2 2 0 01-2-2V7a2 2 0 012-2h4l2 2h4a2 2 0 012 2v1M5 19h14a2 2 0 002-2v-5a2 2 0 00-2-2H9a2 2 0 00-2 2v5a2 2 0 01-2 2z" />
										</svg>
										<span>Open Folder</span>
									</button>

									<!-- Copy ONNX Path -->
									<button
										type="button"
										onclick={handleCopyOnnxPath}
										class="inline-flex items-center gap-1.5 rounded bg-zinc-900/90 hover:bg-zinc-800 border border-zinc-700/80 px-2.5 py-1 text-xs font-medium text-zinc-200 transition hover:text-white"
										title="Copy ONNX path to clipboard"
									>
										{#if hasCopiedOnnxPath}
											<svg class="h-3.5 w-3.5 text-emerald-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
												<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7" />
											</svg>
											<span class="text-emerald-400 font-medium">Copied!</span>
										{:else}
											<svg class="h-3.5 w-3.5 text-zinc-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
												<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
											</svg>
											<span>Copy ONNX Path</span>
										{/if}
									</button>
								</div>

								<div class="flex items-start gap-2.5 min-w-0 pr-48">
									<div class="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-500/20 text-emerald-400 mt-0.5">
										<svg class="h-3.5 w-3.5" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7" />
										</svg>
									</div>
									<div class="min-w-0">
										<p class="text-[11px] font-semibold uppercase tracking-wider text-emerald-300">
											ONNX Model Ready
										</p>
										<p class="mt-0.5 font-mono text-xs font-medium text-zinc-100 select-all break-all">
											{normalizePath(onnxPath)}
										</p>
										<p class="mt-1 text-xs text-emerald-400/90 font-medium">
											This model is optimized and ready for inference in LS-Vision.
										</p>
									</div>
								</div>
							</div>
						{/if}

						<!-- ONNX Error Alert -->
						{#if onnxError}
							<div class="rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-red-300 text-xs flex items-start gap-2">
								<svg class="h-4 w-4 shrink-0 text-red-400 mt-0.5" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
									<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
								</svg>
								<div class="min-w-0">
									<p class="font-semibold">ONNX Export Failed</p>
									<p class="mt-0.5 text-red-200/90">{onnxError}</p>
								</div>
							</div>
						{/if}
					</div>
				{/if}

				<!-- Live Log Terminal Area -->
				<div
					bind:this={logContainerRef}
					class="flex-1 overflow-y-auto p-4 font-mono text-xs leading-relaxed text-zinc-300 space-y-1 select-text bg-zinc-950 overscroll-none"
				>
					{#if logs.length === 0}
						<div class="flex h-full flex-col items-center justify-center text-center text-zinc-600">
							<svg class="h-8 w-8 mb-2 opacity-50" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
							</svg>
							<p>Ready to stream training output.</p>
						</div>
					{:else}
						{#each logs as logLine, i (i)}
							<div class="flex gap-2">
								<span class="w-8 shrink-0 text-right text-zinc-600 select-none">{i + 1}</span>
								<span class={logLine.includes("Error") || logLine.includes("[ERROR]") ? "text-red-400" : logLine.includes("[LS-Trainer]") ? "text-amber-400" : "text-zinc-300"}>
									{logLine}
								</span>
							</div>
						{/each}
					{/if}
				</div>
			</aside>
		{/if}
	</div>
</div>
