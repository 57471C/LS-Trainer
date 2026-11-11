<script lang="ts">
import { Channel, invoke, isTauri } from "@tauri-apps/api/core";
import { onMount } from "svelte";
import AugmentationSection from "./components/AugmentationSection.svelte";
import DatasetSection from "./components/DatasetSection.svelte";
import Header from "./components/Header.svelte";
import LogPanel from "./components/LogPanel.svelte";
import ModelSection from "./components/ModelSection.svelte";
import OutputSection from "./components/OutputSection.svelte";
import ParametersSection from "./components/ParametersSection.svelte";
import TrainingAction from "./components/TrainingAction.svelte";
import SettingsModal from "./components/SettingsModal.svelte";
import {
	freshModelOptions,
	loadSavedFamily,
	loadSavedFreshModel,
	loadSavedShowSeg,
	modelFamilies,
	persistModelPreferences,
} from "./models";
import type {
	AugPreset,
	BaseModelId,
	ExportResponse,
	FinalMetrics,
	GpuInfo,
	ImageSize,
	LabelCheckResult,
	ModelFamily,
	ModelMode,
	TrainingResponse,
	TrainTask,
} from "./types";
import {
	formatDurationHm,
	formatEpochRate,
	formatTimeRemaining,
	normalizePath,
	shortenGpuName,
} from "./utils";

const initialFamily = loadSavedFamily();
const initialShowSeg = loadSavedShowSeg();
const initialFamilyMeta =
	modelFamilies.find((f) => f.id === initialFamily) ?? modelFamilies[0];
const initialSavedFreshModel = loadSavedFreshModel(initialFamily);

// Form state
let datasetPath = $state<string>("");
let datasetFileName = $state<string>("");
let modelMode = $state<ModelMode>("fresh");
let selectedFamily = $state<ModelFamily>(initialFamily);
let showSegmentationModels = $state<boolean>(initialShowSeg);
let existingModelPath = $state<string>("");
let existingModelFileName = $state<string>("");
let freshModel = $state<BaseModelId>(
	initialSavedFreshModel ?? initialFamilyMeta.defaultDetectModel,
);
let trainTask = $state<TrainTask>("detect");
let freezeBackbone = $state<boolean>(false);
const freezeLayersCount = 10;
let runName = $state<string>("");
let epochs = $state<number>(60);
let batch = $state<number>(4);
let imageSize = $state<ImageSize>(960);
let workersOverride = $state<string>("auto");
let device = $state<string>("0");
let isCustomDevice = $state<boolean>(false);
let customDevice = $state<string>("");

// Hardware & Environment
let gpuInfo = $state<GpuInfo | null>(null);
let isCheckingGpu = $state<boolean>(true);
let isRepairingCuda = $state<boolean>(false);

// Augmentation controls
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

// Label inspection
let labelWarning = $state<string | null>(null);
let labelCheckNote = $state<string | null>(null);
let labelCheckPending = $state<boolean>(false);

// Active tooltip
let activeTooltip = $state<string | null>(null);

// Training & Log state
let isTraining = $state<boolean>(false);
let isStoppingTraining = $state<boolean>(false);
let logs = $state<string[]>([]);
let resultsDir = $state<string | null>(null);
let statusMessage = $state<string>("Ready");
let isExportingOnnx = $state<boolean>(false);
let onnxPath = $state<string | null>(null);
let onnxError = $state<string | null>(null);
let isLogPanelOpen = $state<boolean>(false);
let settingsOpen = $state<boolean>(false);
let dataDirectory = $state<string>("");

// Progress & Metrics
let currentEpoch = $state<number>(0);
let totalEpochs = $state<number>(0);
let currentEpochStartTime = $state<number | null>(null);
let trainingStartTime = $state<number | null>(null);
let trainingEndTime = $state<number | null>(null);
let currentTimestamp = $state<number>(Date.now());
let recentEpochDurations = $state<number[]>([]);
let firstEpochEstimatedSec = $state<number | null>(null);
let finalMetrics = $state<FinalMetrics>({
	map50: null,
	map50_95: null,
});

// Augmentation presets
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

$effect(() => {
	persistModelPreferences(selectedFamily, showSegmentationModels, freshModel);
});

$effect(() => {
	if (!isTraining) return;
	const interval = setInterval(() => {
		currentTimestamp = Date.now();
	}, 1000);
	return () => clearInterval(interval);
});

const selectedFamilyInfo = $derived(
	modelFamilies.find((f) => f.id === selectedFamily) ?? modelFamilies[0],
);

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

const activeTaskLabel = $derived(isSegmentationRun ? "segment" : "detect");

const autoDetectedWorkers = $derived(gpuInfo?.recommended_workers ?? 4);
const effectiveWorkers = $derived(
	workersOverride === "auto" ? autoDetectedWorkers : Number(workersOverride),
);
const activeWorkersLabel = $derived(
	workersOverride === "auto"
		? `workers=${effectiveWorkers} (auto)`
		: `workers=${effectiveWorkers}`,
);

const activeRunConfigBanner = $derived(
	`${activeTaskLabel} · ${activeModelLabel} · ${activeWorkersLabel} · ${activeFreezeLabel} · ${activeAugSummary}`,
);

const remainingEpochs = $derived(Math.max(0, totalEpochs - currentEpoch));
const rollingAvgEpochSec = $derived(
	recentEpochDurations.length > 0
		? recentEpochDurations.reduce((sum, d) => sum + d, 0) /
				recentEpochDurations.length
		: firstEpochEstimatedSec,
);

const elapsedSeconds = $derived.by(() => {
	if (!trainingStartTime) return 0;
	const end = trainingEndTime ?? currentTimestamp;
	return Math.max(0, Math.floor((end - trainingStartTime) / 1000));
});

const elapsedLabel = $derived.by(() => {
	if (!trainingStartTime) return "";
	return `elapsed ${formatDurationHm(elapsedSeconds)}`;
});

const epochRateLabel = $derived.by(() => {
	if (rollingAvgEpochSec === null || rollingAvgEpochSec <= 0) return "";
	return formatEpochRate(rollingAvgEpochSec);
});

const epochEtaLabel = $derived.by(() => {
	if (currentEpoch <= 0 || totalEpochs <= 0) return "";
	if (!isTraining && remainingEpochs === 0) return "Complete";
	if (rollingAvgEpochSec === null || rollingAvgEpochSec <= 0) {
		return "estimating…";
	}
	const secLeft = remainingEpochs * rollingAvgEpochSec;
	return formatTimeRemaining(secLeft);
});

const epochCountdownText = $derived.by(() => {
	if (totalEpochs <= 0) return "";
	if (currentEpoch <= 0 && !isTraining) return "";
	const epDisplay = currentEpoch <= 0 ? 0 : currentEpoch;
	const parts = [
		`Epoch ${epDisplay} / ${totalEpochs}`,
		currentEpoch <= 0 ? "(initializing...)" : `(${remainingEpochs} remaining)`,
	];
	if (elapsedLabel) {
		parts.push(elapsedLabel);
	}
	if (epochRateLabel) {
		parts.push(epochRateLabel);
	}
	if (epochEtaLabel) {
		parts.push(epochEtaLabel);
	}
	return parts.join("   ");
});
const epochProgressPercent = $derived(
	totalEpochs > 0
		? Math.min(100, Math.round((currentEpoch / totalEpochs) * 100))
		: 0,
);

const isDatasetSelected = $derived(
	Boolean(datasetPath.trim() || datasetFileName.trim()),
);
const isModelSelected = $derived(
	modelMode === "continue"
		? Boolean(existingModelPath.trim() || existingModelFileName.trim())
		: Boolean(freshModel),
);
const selectedFreshModelOption = $derived(
	freshModelOptions.find((o) => o.id === freshModel),
);
const isSelectedModelSupported = $derived(
	modelMode === "continue" ||
		(selectedFamilyInfo.supported &&
			(selectedFreshModelOption?.supported ?? true)),
);
const selectedModelButtonLabel = $derived(
	modelMode === "fresh" &&
		selectedFreshModelOption &&
		selectedFreshModelOption.supported === false
		? selectedFreshModelOption.label
		: selectedFamilyInfo.label,
);
const isTorchReady = $derived(
	!isCheckingGpu &&
		!isRepairingCuda &&
		(gpuInfo === null
			? !isTauri()
			: Boolean(
					gpuInfo.xpu_available ||
						gpuInfo.cuda_available ||
						gpuInfo.cpu_fallback_confirmed,
				)),
);
const isFormValid = $derived(
	isDatasetSelected &&
		isModelSelected &&
		isSelectedModelSupported &&
		!isTraining &&
		isTorchReady,
);

function processIncomingLogLine(line: string) {
	if (line.startsWith("[LS-Trainer Status] ")) {
		statusMessage = line.replace("[LS-Trainer Status] ", "").trim();
		return;
	}

	// Parse live/final mAP metrics emitted by RF-DETR or YOLO
	const metricsMatch = line.match(
		/mAP50=([0-9]*\.?[0-9]+)\s*\|\s*mAP50-95=([0-9]*\.?[0-9]+)/i,
	);
	if (metricsMatch) {
		const m50 = Number.parseFloat(metricsMatch[1]);
		const m95 = Number.parseFloat(metricsMatch[2]);
		if (!Number.isNaN(m50) && !Number.isNaN(m95)) {
			finalMetrics = {
				map50: m50,
				map50_95: m95,
			};
		}
	}

	// Detect Ultralytics or RF-DETR epoch progress row e.g. "     12/80      2.41G ..." or "Epoch 12/80"
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

			// Replace in-place if previous line is an incomplete tqdm bar for the same epoch
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
	if (isTraining || isRepairingCuda) return;
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
			device = updated.xpu_available
				? "xpu"
				: updated.cuda_available
					? "0"
					: "cpu";
			statusMessage = updated.status_summary;
		} catch (err) {
			const errMsg = err instanceof Error ? err.message : String(err);
			logs = [...logs, `[LS-Trainer Error] ${errMsg}`];
			statusMessage = "GPU repair encountered an error";
		} finally {
			isRepairingCuda = false;
		}
	} else {
		logs = [
			...logs,
			"[LS-Trainer] Simulated GPU repair in Browser Mode",
			"[LS-Trainer] Verified PyTorch environment.",
		];
		statusMessage = "Ready";
		isRepairingCuda = false;
	}
}

onMount(async () => {
	if (isTauri()) {
		try {
			const s = await invoke<{ data_directory: string }>("get_settings");
			dataDirectory = s.data_directory ?? "";
		} catch (e) {
			console.warn("Failed to load settings:", e);
		}
		isCheckingGpu = true;
		statusMessage = "Checking GPU and PyTorch environment…";
		try {
			gpuInfo = await invoke<GpuInfo>("get_gpu_info");
			if (gpuInfo.xpu_available) {
				device = "xpu";
			} else if (gpuInfo.cuda_available) {
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
			"Polygon check runs in the desktop app when a dataset archive or folder is selected.";
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
					"No polygon annotations were found. Instance segmentation needs polygon masks.";
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

async function handleStopTraining() {
	if (!isTraining || isStoppingTraining) return;
	isStoppingTraining = true;
	statusMessage = "Stopping training...";
	logs = [...logs, "[LS-Trainer] Stop requested by user..."];

	if (isTauri()) {
		try {
			await invoke("stop_training");
		} catch (err) {
			console.error("Failed to stop training:", err);
			logs = [...logs, `[LS-Trainer Error] Stop training failed: ${err}`];
		}
	} else {
		// Browser development preview simulation
		setTimeout(() => {
			isTraining = false;
			isStoppingTraining = false;
			statusMessage = "Stopped by user";
			trainingEndTime = Date.now();
			logs = [...logs, "[LS-Trainer] Training stopped by user."];
		}, 300);
	}
}

async function handleStartTraining() {
	if (!isFormValid || isTraining) return;

	isTraining = true;
	isStoppingTraining = false;
	isLogPanelOpen = true;
	logs = [];
	resultsDir = null;
	onnxPath = null;
	onnxError = null;
	currentEpoch = 0;
	totalEpochs = Number(epochs);
	currentEpochStartTime = null;
	trainingStartTime = Date.now();
	trainingEndTime = null;
	currentTimestamp = Date.now();
	recentEpochDurations = [];
	firstEpochEstimatedSec = null;
	finalMetrics = { map50: null, map50_95: null };
	statusMessage = "Training in progress...";

	const effectiveRunName = runName.trim() || "BB-Parts-v3";
	const effectiveDataset = datasetPath.trim() || datasetFileName.trim();
	const effectiveModel =
		modelMode === "continue"
			? existingModelPath.trim() || existingModelFileName.trim()
			: freshModel;
	const effectiveFamily: ModelFamily =
		modelMode === "continue"
			? /rt-?detr/i.test(effectiveModel)
				? "RT-DETR"
				: /\.pth$/i.test(effectiveModel) || /^rf-?detr/i.test(effectiveModel)
					? "RF-DETR"
					: selectedFamily
			: selectedFamily;

	const effectiveCopyPaste = copyPaste ? 0.3 : 0.0;
	const taskName = isSegmentationRun ? "segment" : "detect";
	const requestPayload = {
		dataset_path: effectiveDataset,
		model_family: effectiveFamily,
		model_choice: effectiveModel,
		run_name: effectiveRunName,
		epochs: Number(epochs),
		batch: Number(batch),
		imgsz: Number(imageSize),
		workers: workersOverride === "auto" ? null : Number(workersOverride),
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
			if (
				(response.map50 !== undefined && response.map50 !== null) ||
				(response.map50_95 !== undefined && response.map50_95 !== null)
			) {
				finalMetrics = {
					map50: response.map50 ?? finalMetrics.map50,
					map50_95: response.map50_95 ?? finalMetrics.map50_95,
				};
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
			if (isStoppingTraining || /stopped by user/i.test(response.message)) {
				statusMessage = "Stopped by user";
			} else {
				statusMessage = response.success
					? onnxPath
						? `Finished — ${normalizePath(onnxPath)}`
						: `Finished — ${normalizePath(response.results_dir)}`
					: "Process finished with status warnings";
			}
		} catch (err) {
			const errMsg = err instanceof Error ? err.message : String(err);
			logs = [...logs, `[LS-Trainer Error] ${errMsg}`];
			if (isStoppingTraining || /stopped by user/i.test(errMsg)) {
				statusMessage = "Stopped by user";
			} else {
				statusMessage = "Training encountered an error";
			}
		} finally {
			trainingEndTime = Date.now();
			isTraining = false;
			isStoppingTraining = false;
		}
	} else {
		// Browser development preview simulation
		const freezeArgStr = freezeBackbone ? ` freeze=${freezeLayersCount}` : "";
		const isRfDetrSim = effectiveFamily === "RF-DETR";
		const isRtDetrSim = effectiveFamily === "RT-DETR";
		logs = [
			"[LS-Trainer] Running outside Tauri webview (Browser Mode)",
			`[LS-Trainer] Simulated training run '${effectiveRunName}' (${effectiveFamily})`,
			isRtDetrSim
				? `[LS-Trainer] Prepared LS-Marker COCO splits for RT-DETR (${effectiveModel})`
				: isRfDetrSim
					? `[LS-Trainer] Prepared LS-Marker COCO splits for RF-DETR (${effectiveModel})`
					: `[LS-Trainer] Ultralytics task="${taskName}"`,
			`[LS-Trainer] Active config: ${activeRunConfigBanner}`,
			isRtDetrSim
				? `[LS-Trainer] Command: python ls_trainer_rtdetr_run.py --model ${effectiveModel} --dataset "${effectiveDataset}" --epochs ${epochs} --batch ${batch} --resolution ${imageSize}`
				: isRfDetrSim
					? `[LS-Trainer] Command: python ls_trainer_rfdetr_run.py --model ${effectiveModel} --dataset "${effectiveDataset}" --epochs ${epochs} --batch ${batch} --resolution ${imageSize}`
					: `[LS-Trainer] Command: yolo ${taskName} train task=${taskName} data="${effectiveDataset}" model="${effectiveModel}" name="${effectiveRunName}" epochs=${epochs} batch=${batch} imgsz=${imageSize} device="${device}" amp=${amp}${freezeArgStr} mosaic=${mosaic} close_mosaic=${closeMosaic} mixup=${mixup} copy_paste=${effectiveCopyPaste} degrees=${degrees} flipud=${flipud} translate=${translate} scale=${scale}`,
			`[LS-Trainer] Starting ${activeModelLabel} epochs (1/${epochs})...`,
			"[LS-Trainer Metrics] Final | mAP50=0.9140 | mAP50-95=0.7280",
			"[LS-Trainer] Training finished successfully.",
		];
		currentEpoch = Number(epochs);
		totalEpochs = Number(epochs);
		finalMetrics = { map50: 0.914, map50_95: 0.728 };
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
		trainingEndTime = Date.now();
		isTraining = false;
	}
}

async function handleExportToOnnx() {
	if (!resultsDir || isExportingOnnx) return;
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
	<Header
		{isLogPanelOpen}
		{isTraining}
		logCount={logs.length}
		onToggleLogPanel={() => {
			isLogPanelOpen = !isLogPanelOpen;
		}}
		onOpenSettings={() => {
			settingsOpen = true;
		}}
	/>

	<!-- Main Workspace Split -->
	<div class="relative flex flex-1 overflow-hidden">
		<!-- Left: Form Area (Scrollable) -->
		<div class="flex flex-1 justify-center overflow-y-auto overscroll-none px-4 pt-4 pb-8 sm:px-8 sm:pt-6 sm:pb-10">
			<form
				onsubmit={(e) => {
					e.preventDefault();
					handleStartTraining();
				}}
				class="w-full max-w-4xl space-y-6 rounded-xl border border-zinc-800/90 bg-zinc-900/50 p-6 shadow-2xl backdrop-blur-sm sm:p-8"
			>
				<!-- Section 1: Dataset -->
				<DatasetSection
					bind:datasetPath
					bind:datasetFileName
					{selectedFamily}
					{isTraining}
				/>

				<!-- Section 2: Model & Task -->
				<ModelSection
					bind:modelMode
					bind:selectedFamily
					bind:showSegmentationModels
					bind:freshModel
					bind:trainTask
					bind:existingModelPath
					bind:existingModelFileName
					bind:freezeBackbone
					{freezeLayersCount}
					{isTraining}
					{isSegmentationRun}
					{isDatasetSelected}
					{labelCheckPending}
					{labelWarning}
					{labelCheckNote}
					{activeTooltip}
					onToggleTooltip={(name) => {
						activeTooltip = activeTooltip === name ? null : name;
					}}
				/>

				<!-- Settings Panel: Output, Parameters & Augmentation (Single Continuous Container) -->
				<div class="flex flex-col gap-6 w-full rounded-xl border border-zinc-800/90 bg-zinc-950/40 p-5 sm:p-6">
					<!-- Section 3: Output -->
					<OutputSection
						bind:runName
						{isTraining}
					/>

					<!-- Section 4: Parameters & Settings -->
					<ParametersSection
						bind:epochs
						bind:batch
						bind:imageSize
						bind:workersOverride
						bind:device
						bind:isCustomDevice
						bind:customDevice
						bind:amp
						{gpuInfo}
						{isTraining}
						{isRepairingCuda}
						{statusMessage}
						{autoDetectedWorkers}
						{effectiveWorkers}
						{activeTooltip}
						onRepairCuda={handleRepairCuda}
						onToggleTooltip={(name) => {
							activeTooltip = activeTooltip === name ? null : name;
						}}
					/>

					<!-- Section 5: Augmentation Controls -->
					<AugmentationSection
						bind:mosaic
						bind:mixup
						bind:copyPaste
						bind:degrees
						bind:flipud
						bind:translate
						bind:scale
						bind:closeMosaic
						{isTraining}
						{activeAugSummary}
						{activeAugPreset}
						{activeTooltip}
						onApplyPreset={applyAugPreset}
						onToggleTooltip={(name) => {
							activeTooltip = activeTooltip === name ? null : name;
						}}
					/>
				</div>

				<!-- Section 6: Action & Status -->
				<TrainingAction
					bind:autoExportOnnx
					{isFormValid}
					{isTraining}
					{isCheckingGpu}
					{isRepairingCuda}
					{isExportingOnnx}
					{isTorchReady}
					{isLogPanelOpen}
					{modelMode}
					selectedFamilySupported={isSelectedModelSupported}
					selectedFamilyLabel={selectedModelButtonLabel}
					{statusMessage}
					{epochCountdownText}
					{epochProgressPercent}
					{activeRunConfigBanner}
					{finalMetrics}
					hasLogs={logs.length > 0}
					onOpenLogPanel={() => {
						isLogPanelOpen = true;
					}}
					{isStoppingTraining}
					onStopTraining={handleStopTraining}
				/>
			</form>
		</div>

		<!-- Right: Resizable Training Log Panel -->
		<LogPanel
			{isLogPanelOpen}
			{isTraining}
			{isExportingOnnx}
			{logs}
			{resultsDir}
			{onnxPath}
			{onnxError}
			{activeRunConfigBanner}
			{finalMetrics}
			onClose={() => {
				isLogPanelOpen = false;
			}}
			onClearLogs={() => {
				logs = [];
			}}
			onExportOnnx={handleExportToOnnx}
		/>
	</div>

	<SettingsModal
		open={settingsOpen}
		onClose={() => { settingsOpen = false; }}
		{dataDirectory}
		onSaved={(path) => { dataDirectory = path; }}
	/>
</div>

