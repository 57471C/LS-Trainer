<script lang="ts">
import { Channel, invoke, isTauri } from "@tauri-apps/api/core";
import { onMount } from "svelte";

type ModelMode = "continue" | "fresh";
type TrainTask = "detect" | "segment";
type ImageSize = 640 | 960 | 1280;
type ModelFamily = "YOLO" | "RF-DETR" | "RT-DETR" | "YOLOX";
type BaseModelId =
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
	| "yolo26s-seg.pt"
	| "rf-detr-nano"
	| "rf-detr-small"
	| "rf-detr-medium"
	| "rf-detr-base"
	| "rf-detr-large"
	| "rf-detr-seg-nano"
	| "rf-detr-seg-small"
	| "rf-detr-seg-medium"
	| "rf-detr-seg-large"
	| "rf-detr-seg-preview"
	| "rtdetr-r18"
	| "rtdetr-r34"
	| "rtdetr-r50"
	| "rtdetr-l.pt"
	| "rtdetr-x.pt"
	| "yolox-nano"
	| "yolox-tiny"
	| "yolox-s"
	| "yolox-m"
	| "yolox-l"
	| "yolox-x";
type AugPreset = "off" | "light" | "medium" | "bench" | "custom";

const STORAGE_KEY_FAMILY = "ls_trainer.model_family";
const STORAGE_KEY_SHOW_SEG = "ls_trainer.show_segmentation_models";

interface TrainingResponse {
	success: boolean;
	results_dir: string;
	message: string;
	map50?: number | null;
	map50_95?: number | null;
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
	recommended_workers?: number;
	physical_cores?: number;
	logical_cores?: number;
}

interface LabelCheckResult {
	polygons_found: boolean;
	bboxes_found: boolean;
	files_scanned: number;
	lines_scanned: number;
	warning: string | null;
}

interface ModelFamilyInfo {
	id: ModelFamily;
	label: string;
	vendor: string;
	badge: string;
	supported: boolean;
	defaultDetectModel: BaseModelId;
	defaultSegModel: BaseModelId | null;
	description: string;
}

const modelFamilies: ModelFamilyInfo[] = [
	{
		id: "YOLO",
		label: "YOLO",
		vendor: "Ultralytics",
		badge: "Active",
		supported: true,
		defaultDetectModel: "yolo11m.pt",
		defaultSegModel: "yolo11s-seg.pt",
		description: "Ultralytics YOLO11 & YOLO26 detection and segmentation",
	},
	{
		id: "RF-DETR",
		label: "RF-DETR",
		vendor: "Roboflow",
		badge: "Active",
		supported: true,
		defaultDetectModel: "rf-detr-medium",
		defaultSegModel: "rf-detr-seg-small",
		description: "Roboflow DINOv2 real-time vision transformer (Apache 2.0)",
	},
	{
		id: "RT-DETR",
		label: "RT-DETR",
		vendor: "Baidu / Ultralytics",
		badge: "Coming soon",
		supported: false,
		defaultDetectModel: "rtdetr-l.pt",
		defaultSegModel: null,
		description: "Real-Time Detection Transformer end-to-end detector",
	},
	{
		id: "YOLOX",
		label: "YOLOX",
		vendor: "Megvii",
		badge: "Coming soon",
		supported: false,
		defaultDetectModel: "yolox-s",
		defaultSegModel: null,
		description: "Anchor-free decoupled-head detector (Apache 2.0)",
	},
];

const freshModelOptions: {
	id: BaseModelId;
	family: ModelFamily;
	group: string;
	task: TrainTask;
	label: string;
	desc: string;
}[] = [
	// YOLO11 Detection
	{
		id: "yolo11n.pt",
		family: "YOLO",
		group: "YOLO11 Series",
		task: "detect",
		label: "YOLO11n",
		desc: "Nano (fastest, lightweight)",
	},
	{
		id: "yolo11s.pt",
		family: "YOLO",
		group: "YOLO11 Series",
		task: "detect",
		label: "YOLO11s",
		desc: "Small (balanced fast)",
	},
	{
		id: "yolo11m.pt",
		family: "YOLO",
		group: "YOLO11 Series",
		task: "detect",
		label: "YOLO11m",
		desc: "Medium (default, high accuracy)",
	},
	{
		id: "yolo11l.pt",
		family: "YOLO",
		group: "YOLO11 Series",
		task: "detect",
		label: "YOLO11l",
		desc: "Large (extra accuracy)",
	},
	{
		id: "yolo11x.pt",
		family: "YOLO",
		group: "YOLO11 Series",
		task: "detect",
		label: "YOLO11x",
		desc: "XLarge (maximum capacity)",
	},
	// YOLO26 Detection
	{
		id: "yolo26n.pt",
		family: "YOLO",
		group: "YOLO26 Series",
		task: "detect",
		label: "YOLO26n",
		desc: "Nano (next-gen lightweight)",
	},
	{
		id: "yolo26s.pt",
		family: "YOLO",
		group: "YOLO26 Series",
		task: "detect",
		label: "YOLO26s",
		desc: "Small (next-gen balanced)",
	},
	{
		id: "yolo26m.pt",
		family: "YOLO",
		group: "YOLO26 Series",
		task: "detect",
		label: "YOLO26m",
		desc: "Medium (next-gen high accuracy)",
	},
	{
		id: "yolo26l.pt",
		family: "YOLO",
		group: "YOLO26 Series",
		task: "detect",
		label: "YOLO26l",
		desc: "Large (next-gen extra accuracy)",
	},
	{
		id: "yolo26x.pt",
		family: "YOLO",
		group: "YOLO26 Series",
		task: "detect",
		label: "YOLO26x",
		desc: "XLarge (next-gen maximum capacity)",
	},
	// YOLO Segmentation (-seg)
	{
		id: "yolo11s-seg.pt",
		family: "YOLO",
		group: "YOLO11 Segmentation (-seg)",
		task: "segment",
		label: "YOLO11s-seg",
		desc: "Small instance segmentation",
	},
	{
		id: "yolo11m-seg.pt",
		family: "YOLO",
		group: "YOLO11 Segmentation (-seg)",
		task: "segment",
		label: "YOLO11m-seg",
		desc: "Medium instance segmentation",
	},
	{
		id: "yolo26s-seg.pt",
		family: "YOLO",
		group: "YOLO26 Segmentation (-seg)",
		task: "segment",
		label: "YOLO26s-seg",
		desc: "Small next-gen segmentation",
	},
	// RF-DETR Detection (Nano / Small / Medium / Large / Base)
	{
		id: "rf-detr-nano",
		family: "RF-DETR",
		group: "RF-DETR Detection",
		task: "detect",
		label: "RF-DETR Nano",
		desc: "Nano (ultra-fast real-time transformer)",
	},
	{
		id: "rf-detr-small",
		family: "RF-DETR",
		group: "RF-DETR Detection",
		task: "detect",
		label: "RF-DETR Small",
		desc: "Small (balanced low-latency transformer)",
	},
	{
		id: "rf-detr-medium",
		family: "RF-DETR",
		group: "RF-DETR Detection",
		task: "detect",
		label: "RF-DETR Medium",
		desc: "Medium (default, high accuracy industrial detector)",
	},
	{
		id: "rf-detr-large",
		family: "RF-DETR",
		group: "RF-DETR Detection",
		task: "detect",
		label: "RF-DETR Large",
		desc: "Large (128M backbone, maximum accuracy)",
	},
	{
		id: "rf-detr-base",
		family: "RF-DETR",
		group: "RF-DETR Detection",
		task: "detect",
		label: "RF-DETR Base",
		desc: "Base (29M DINOv2 backbone)",
	},
	// RF-DETR Segmentation (-seg)
	{
		id: "rf-detr-seg-nano",
		family: "RF-DETR",
		group: "RF-DETR Segmentation (-seg)",
		task: "segment",
		label: "RF-DETR Seg Nano",
		desc: "Nano instance segmentation",
	},
	{
		id: "rf-detr-seg-small",
		family: "RF-DETR",
		group: "RF-DETR Segmentation (-seg)",
		task: "segment",
		label: "RF-DETR Seg Small",
		desc: "Small instance segmentation (default)",
	},
	{
		id: "rf-detr-seg-medium",
		family: "RF-DETR",
		group: "RF-DETR Segmentation (-seg)",
		task: "segment",
		label: "RF-DETR Seg Medium",
		desc: "Medium instance segmentation",
	},
	{
		id: "rf-detr-seg-large",
		family: "RF-DETR",
		group: "RF-DETR Segmentation (-seg)",
		task: "segment",
		label: "RF-DETR Seg Large",
		desc: "Large instance segmentation",
	},
	{
		id: "rf-detr-seg-preview",
		family: "RF-DETR",
		group: "RF-DETR Segmentation (-seg)",
		task: "segment",
		label: "RF-DETR-seg Preview",
		desc: "Instance segmentation head (Preview)",
	},
	// RT-DETR
	{
		id: "rtdetr-r18",
		family: "RT-DETR",
		group: "RT-DETR Series",
		task: "detect",
		label: "RT-DETR-R18",
		desc: "ResNet-18 (lightweight real-time DETR)",
	},
	{
		id: "rtdetr-r34",
		family: "RT-DETR",
		group: "RT-DETR Series",
		task: "detect",
		label: "RT-DETR-R34",
		desc: "ResNet-34 (balanced speed & accuracy)",
	},
	{
		id: "rtdetr-r50",
		family: "RT-DETR",
		group: "RT-DETR Series",
		task: "detect",
		label: "RT-DETR-R50",
		desc: "ResNet-50 (high accuracy transformer)",
	},
	{
		id: "rtdetr-l.pt",
		family: "RT-DETR",
		group: "RT-DETR Series",
		task: "detect",
		label: "RT-DETR-L",
		desc: "HGNetv2 Large (default, high accuracy)",
	},
	{
		id: "rtdetr-x.pt",
		family: "RT-DETR",
		group: "RT-DETR Series",
		task: "detect",
		label: "RT-DETR-X",
		desc: "HGNetv2 XLarge (maximum capacity)",
	},
	// YOLOX
	{
		id: "yolox-nano",
		family: "YOLOX",
		group: "YOLOX Series",
		task: "detect",
		label: "YOLOX-Nano",
		desc: "Nano (0.9M params, edge/embedded)",
	},
	{
		id: "yolox-tiny",
		family: "YOLOX",
		group: "YOLOX Series",
		task: "detect",
		label: "YOLOX-Tiny",
		desc: "Tiny (5.1M params, fast lightweight)",
	},
	{
		id: "yolox-s",
		family: "YOLOX",
		group: "YOLOX Series",
		task: "detect",
		label: "YOLOX-S",
		desc: "Small (9.0M params, default balanced)",
	},
	{
		id: "yolox-m",
		family: "YOLOX",
		group: "YOLOX Series",
		task: "detect",
		label: "YOLOX-M",
		desc: "Medium (25.3M params, high accuracy)",
	},
	{
		id: "yolox-l",
		family: "YOLOX",
		group: "YOLOX Series",
		task: "detect",
		label: "YOLOX-L",
		desc: "Large (54.2M params, extra accuracy)",
	},
	{
		id: "yolox-x",
		family: "YOLOX",
		group: "YOLOX Series",
		task: "detect",
		label: "YOLOX-X",
		desc: "XLarge (99.1M params, maximum capacity)",
	},
];

function loadSavedFamily(): ModelFamily {
	if (typeof window === "undefined") return "YOLO";
	try {
		const raw = window.localStorage.getItem(STORAGE_KEY_FAMILY);
		if (
			raw === "YOLO" ||
			raw === "RF-DETR" ||
			raw === "RT-DETR" ||
			raw === "YOLOX"
		) {
			return raw;
		}
	} catch {
		// ignore localStorage read errors
	}
	return "YOLO";
}

function loadSavedShowSeg(): boolean {
	if (typeof window === "undefined") return false;
	try {
		return window.localStorage.getItem(STORAGE_KEY_SHOW_SEG) === "true";
	} catch {
		return false;
	}
}

const initialFamily = loadSavedFamily();
const initialShowSeg = loadSavedShowSeg();
const initialFamilyMeta =
	modelFamilies.find((f) => f.id === initialFamily) ?? modelFamilies[0];

let datasetPath = $state<string>("");
let datasetFileName = $state<string>("");
let modelMode = $state<ModelMode>("fresh");
let selectedFamily = $state<ModelFamily>(initialFamily);
let showSegmentationModels = $state<boolean>(initialShowSeg);
let existingModelPath = $state<string>("");
let existingModelFileName = $state<string>("");
let freshModel = $state<BaseModelId>(initialFamilyMeta.defaultDetectModel);
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

function persistModelPreferences(family: ModelFamily, showSeg: boolean) {
	if (typeof window === "undefined") return;
	try {
		window.localStorage.setItem(STORAGE_KEY_FAMILY, family);
		window.localStorage.setItem(STORAGE_KEY_SHOW_SEG, String(showSeg));
	} catch {
		// ignore localStorage write errors
	}
}

$effect(() => {
	persistModelPreferences(selectedFamily, showSegmentationModels);
});

const selectedFamilyInfo = $derived(
	modelFamilies.find((f) => f.id === selectedFamily) ?? modelFamilies[0],
);

const familySegModelCount = $derived(
	freshModelOptions.filter(
		(o) =>
			o.family === selectedFamily &&
			(o.task === "segment" || /-seg/i.test(o.id)),
	).length,
);

const visibleModelOptions = $derived(
	freshModelOptions.filter(
		(o) =>
			o.family === selectedFamily &&
			(showSegmentationModels || (o.task !== "segment" && !/-seg/i.test(o.id))),
	),
);

const visibleModelGroups = $derived.by(() => {
	const groups: { name: string; options: typeof freshModelOptions }[] = [];
	for (const opt of visibleModelOptions) {
		const existing = groups.find((g) => g.name === opt.group);
		if (existing) {
			existing.options.push(opt);
		} else {
			groups.push({ name: opt.group, options: [opt] });
		}
	}
	return groups;
});

function selectModelFamily(family: ModelFamily) {
	if (isTraining) return;
	selectedFamily = family;
	const meta = modelFamilies.find((f) => f.id === family) ?? modelFamilies[0];
	const allowed = freshModelOptions.filter(
		(o) =>
			o.family === family &&
			(showSegmentationModels || (o.task !== "segment" && !/-seg/i.test(o.id))),
	);
	if (!allowed.some((o) => o.id === freshModel)) {
		if (
			trainTask === "segment" &&
			showSegmentationModels &&
			meta.defaultSegModel
		) {
			freshModel = meta.defaultSegModel;
		} else {
			freshModel = meta.defaultDetectModel;
			trainTask = "detect";
		}
	}
}

function toggleShowSegmentationModels(nextValue: boolean) {
	if (isTraining) return;
	showSegmentationModels = nextValue;
	if (!nextValue) {
		const current = freshModelOptions.find((o) => o.id === freshModel);
		if (current && (current.task === "segment" || /-seg/i.test(current.id))) {
			const meta =
				modelFamilies.find((f) => f.id === selectedFamily) ?? modelFamilies[0];
			freshModel = meta.defaultDetectModel;
			trainTask = "detect";
		}
	}
}

function selectTrainTask(task: TrainTask) {
	if (isTraining) return;
	trainTask = task;
	const meta =
		modelFamilies.find((f) => f.id === selectedFamily) ?? modelFamilies[0];
	if (task === "segment") {
		if (!showSegmentationModels) {
			showSegmentationModels = true;
		}
		const current = freshModelOptions.find((o) => o.id === freshModel);
		if (
			!current ||
			current.family !== selectedFamily ||
			current.task !== "segment"
		) {
			if (meta.defaultSegModel) {
				freshModel = meta.defaultSegModel;
			}
		}
	} else {
		const current = freshModelOptions.find((o) => o.id === freshModel);
		if (
			!current ||
			current.family !== selectedFamily ||
			current.task !== "detect"
		) {
			freshModel = meta.defaultDetectModel;
		}
	}
}

function handleFreshModelChange(nextModel: BaseModelId) {
	freshModel = nextModel;
	const found = freshModelOptions.find((o) => o.id === nextModel);
	if (found) {
		trainTask =
			found.task === "segment" || /-seg/i.test(found.id) ? "segment" : "detect";
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

// Epoch countdown, rolling ETA & validation metrics tracking
let currentEpoch = $state<number>(0);
let totalEpochs = $state<number>(0);
let currentEpochStartTime = $state<number | null>(null);
let recentEpochDurations = $state<number[]>([]);
let firstEpochEstimatedSec = $state<number | null>(null);
let finalMetrics = $state<{ map50: number | null; map50_95: number | null }>({
	map50: null,
	map50_95: null,
});

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

function formatMetricPercent(v: number | null | undefined): string {
	if (v === null || v === undefined || Number.isNaN(v)) return "—";
	return `${(v * 100).toFixed(1)}%`;
}

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
const isSelectedFamilySupported = $derived(
	modelMode === "continue" || selectedFamilyInfo.supported,
);
const isTorchReady = $derived(
	!isCheckingGpu &&
		!isRepairingCuda &&
		(gpuInfo === null
			? !isTauri()
			: Boolean(gpuInfo.cuda_available || gpuInfo.cpu_fallback_confirmed)),
);
const isFormValid = $derived(
	isDatasetSelected &&
		isModelSelected &&
		isSelectedFamilySupported &&
		!isTraining &&
		isTorchReady,
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

async function handlePickDatasetFolder() {
	if (isTauri()) {
		try {
			const picked = await invoke<string | null>("pick_dataset_folder");
			if (picked) {
				const norm = normalizePath(picked);
				datasetPath = norm;
				const cleanName = norm.replace(/\\/g, "/").split("/").pop();
				datasetFileName = cleanName || norm;
			}
			return;
		} catch (err) {
			console.warn("Native folder picker unavailable:", err);
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
			? /\.pth$/i.test(effectiveModel) || /^rf-?detr/i.test(effectiveModel)
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
		const isRfDetrSim = effectiveFamily === "RF-DETR";
		logs = [
			"[LS-Trainer] Running outside Tauri webview (Browser Mode)",
			`[LS-Trainer] Simulated training run '${effectiveRunName}' (${effectiveFamily})`,
			isRfDetrSim
				? `[LS-Trainer] Prepared LS-Marker COCO splits for RF-DETR (${effectiveModel})`
				: `[LS-Trainer] Ultralytics task="${taskName}"`,
			`[LS-Trainer] Active config: ${activeRunConfigBanner}`,
			isRfDetrSim
				? `[LS-Trainer] Command: python ls_trainer_rfdetr_run.py --model ${effectiveModel} --dataset "${effectiveDataset}" --epochs ${epochs} --batch ${batch} --resolution ${imageSize}`
				: `[LS-Trainer] Command: yolo ${taskName} train task=${taskName} data="${effectiveDataset}" model="${effectiveModel}" name="${effectiveRunName}" epochs=${epochs} batch=${batch} imgsz=${imageSize} device="${device}" amp=${amp}${freezeArgStr} mosaic=${mosaic} close_mosaic=${closeMosaic} mixup=${mixup} copy_paste=${effectiveCopyPaste} degrees=${degrees} flipud=${flipud} translate=${translate} scale=${scale}`,
			`[LS-Trainer] Starting ${activeModelLabel} epochs (1/${epochs})...`,
			`[LS-Trainer Metrics] Final | mAP50=0.9140 | mAP50-95=0.7280`,
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
						<span class="text-xs text-zinc-500">
							{selectedFamily === "RF-DETR"
								? "LS-Marker COCO (.tar, .tar.gz, or folder)"
								: "Supported: .tar, .tar.gz, or folder"}
						</span>
					</div>

					<input
						id="dataset-file-input"
						bind:this={datasetInputRef}
						type="file"
						accept=".tar,.tar.gz,.tgz,.json,.yaml,.yml,application/x-tar,application/gzip,application/x-gzip"
						onchange={handleDatasetFileInputChange}
						class="hidden"
					/>

					{#if !datasetFileName}
						<div class="grid grid-cols-1 gap-2.5 sm:grid-cols-2">
							<button
								type="button"
								onclick={handlePickDataset}
								disabled={isTraining}
								class="group flex w-full flex-col items-center justify-center rounded-lg border-2 border-dashed border-zinc-700/80 bg-zinc-950/40 p-4 text-center transition-all hover:border-amber-500/70 hover:bg-zinc-900/60 focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500/40 disabled:opacity-50"
							>
								<svg
									class="mb-1.5 h-6 w-6 text-zinc-400 transition-colors group-hover:text-amber-400"
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
									LS-Marker .tar / .tar.gz export or manifest
								</p>
							</button>

							<button
								type="button"
								onclick={handlePickDatasetFolder}
								disabled={isTraining}
								class="group flex w-full flex-col items-center justify-center rounded-lg border-2 border-dashed border-zinc-700/80 bg-zinc-950/40 p-4 text-center transition-all hover:border-amber-500/70 hover:bg-zinc-900/60 focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500/40 disabled:opacity-50"
							>
								<svg
									class="mb-1.5 h-6 w-6 text-zinc-400 transition-colors group-hover:text-amber-400"
									aria-hidden="true"
									fill="none"
									stroke="currentColor"
									viewBox="0 0 24 24"
								>
									<path
										stroke-linecap="round"
										stroke-linejoin="round"
										stroke-width="1.75"
										d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z"
									/>
								</svg>
								<p class="text-sm font-medium text-zinc-200">
									Choose dataset folder
								</p>
								<p class="mt-0.5 text-xs text-zinc-500">
									Unpacked LS-Marker COCO or YOLO directory
								</p>
							</button>
						</div>
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
										{normalizePath(datasetPath) || "Dataset selected"}
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
									Archive
								</button>
								<button
									type="button"
									onclick={handlePickDatasetFolder}
									disabled={isTraining}
									class="rounded bg-zinc-800 px-2.5 py-1 text-xs font-medium text-zinc-300 transition hover:bg-zinc-700 hover:text-white disabled:opacity-50"
								>
									Folder
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
								<span class="block text-xs text-zinc-400">Resume from custom .pt or .pth checkpoint</span>
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
						<div class="space-y-3 pt-1">
							<!-- Model Family Switcher -->
							<div class="space-y-1.5">
								<div class="flex items-center justify-between">
									<span class="block text-xs font-medium text-zinc-400">
										Model Family
									</span>
									<span class="font-mono text-[11px] text-zinc-500">
										{selectedFamilyInfo.vendor}
									</span>
								</div>
								<div
									role="radiogroup"
									aria-label="Model Family"
									class="grid grid-cols-2 gap-1.5 rounded-lg border border-zinc-800/90 bg-zinc-950/60 p-1 sm:grid-cols-4"
								>
									{#each modelFamilies as fam (fam.id)}
										{@const isSelected = selectedFamily === fam.id}
										<button
											type="button"
											role="radio"
											aria-checked={isSelected}
											disabled={isTraining}
											onclick={() => selectModelFamily(fam.id)}
											class={`group relative flex flex-col items-start justify-between rounded-md px-2.5 py-2 text-left transition-all ${
												isSelected
													? fam.supported
														? "border border-amber-500/60 bg-amber-500/15 text-zinc-100 shadow-[inset_0_1px_0_rgba(245,158,11,0.2)]"
														: "border border-zinc-600/80 bg-zinc-800/90 text-zinc-100"
													: "border border-transparent bg-transparent text-zinc-400 hover:bg-zinc-900/80 hover:text-zinc-200"
											} disabled:opacity-50`}
										>
											<div class="flex w-full items-center justify-between gap-1">
												<span class="font-mono text-xs font-semibold tracking-tight">
													{fam.label}
												</span>
												{#if fam.supported}
													<span
														class={`h-1.5 w-1.5 rounded-full ${
															isSelected
																? "bg-amber-400 shadow-[0_0_6px_rgba(245,158,11,0.8)]"
																: "bg-emerald-500/70"
														}`}
														title="Supported in training pipeline"
													></span>
												{:else}
													<span class="rounded border border-zinc-700/80 bg-zinc-900/90 px-1 py-0.5 font-mono text-[9px] uppercase tracking-wider text-zinc-400">
														Soon
													</span>
												{/if}
											</div>
											<span class="mt-0.5 text-[10px] text-zinc-500 group-hover:text-zinc-400">
												{fam.vendor}
											</span>
										</button>
									{/each}
								</div>
							</div>

							<!-- Architecture Size Selector + Show Segmentation Toggle -->
							<div class="space-y-1.5">
								<div class="flex flex-wrap items-center justify-between gap-2">
									<label for="fresh-model-select" class="block text-xs font-medium text-zinc-400">
										Select Base Architecture ({selectedFamilyInfo.label})
									</label>
									<label
										class={`inline-flex cursor-pointer items-center gap-2 rounded-md border px-2.5 py-1 text-xs transition select-none ${
											showSegmentationModels
												? "border-amber-500/40 bg-amber-500/10 text-amber-200"
												: "border-zinc-800 bg-zinc-950/50 text-zinc-400 hover:border-zinc-700 hover:text-zinc-300"
										} ${isTraining ? "opacity-50 pointer-events-none" : ""}`}
									>
										<input
											type="checkbox"
											checked={showSegmentationModels}
											disabled={isTraining}
											onchange={(e) => toggleShowSegmentationModels(e.currentTarget.checked)}
											class="h-3.5 w-3.5 rounded border-zinc-700 bg-zinc-900 text-amber-500 focus:ring-amber-500 focus:ring-offset-zinc-950"
										/>
										<span>Show segmentation models</span>
										<span class="rounded bg-zinc-800/90 px-1.5 py-0.2 font-mono text-[10px] text-zinc-400">
											{familySegModelCount}
										</span>
									</label>
								</div>

								<select
									id="fresh-model-select"
									value={freshModel}
									onchange={(e) => handleFreshModelChange(e.currentTarget.value as BaseModelId)}
									disabled={isTraining}
									class="w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
								>
									{#each visibleModelGroups as group (group.name)}
										<optgroup label={group.name}>
											{#each group.options as opt (opt.id)}
												<option value={opt.id}>
													{opt.label} ({opt.id}){ !selectedFamilyInfo.supported ? " [Coming soon]" : "" } — {opt.desc}
												</option>
											{/each}
										</optgroup>
									{/each}
								</select>

								{#if !showSegmentationModels && familySegModelCount > 0}
									<p class="text-[11px] text-zinc-500">
										{familySegModelCount} segmentation (`-seg`) variant{familySegModelCount === 1 ? "" : "s"} hidden. Enable “Show segmentation models” to include polygon mask weights.
									</p>
								{:else if showSegmentationModels && familySegModelCount === 0}
									<p class="text-[11px] text-zinc-500">
										No `-seg` / segmentation checkpoints are cataloged for {selectedFamilyInfo.label} yet.
									</p>
								{/if}
							</div>

							{#if !selectedFamilyInfo.supported}
								<div class="flex items-start gap-2.5 rounded-lg border border-amber-500/35 bg-amber-500/10 px-3.5 py-2.5 text-xs leading-relaxed text-amber-200/90">
									<svg class="mt-0.5 h-4 w-4 shrink-0 text-amber-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
									</svg>
									<div>
										<span class="font-semibold text-amber-300">{selectedFamilyInfo.label} backend coming soon.</span>
										<span class="block text-amber-100/80">
											Model catalog and size presets are ready to inspect, but training execution is currently wired to <button type="button" onclick={() => selectModelFamily("YOLO")} class="font-mono underline decoration-amber-400/70 hover:text-amber-200">YOLO</button> and <button type="button" onclick={() => selectModelFamily("RF-DETR")} class="font-mono underline decoration-amber-400/70 hover:text-amber-200">RF-DETR</button>.
										</span>
									</div>
								</div>
							{/if}
						</div>
					{:else}
						<div class="space-y-1.5 pt-1">
							<label for="model-file-input" class="block text-xs font-medium text-zinc-400">
								Checkpoint File (.pt / .pth)
							</label>
							<input
								id="model-file-input"
								bind:this={modelInputRef}
								type="file"
								accept=".pt,.pth"
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
									<span>Select existing checkpoint (.pt / .pth)</span>
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

						<!-- Image size & DataLoader Workers -->
						<div class="grid grid-cols-2 gap-3">
							<div class="space-y-1.5">
								<label for="image-size-select" class="flex h-5 items-center text-xs font-medium text-zinc-400">
									Image size
								</label>
								<select
									id="image-size-select"
									bind:value={imageSize}
									disabled={isTraining}
									class="w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3 py-2.5 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
								>
									<option value={640}>640 px</option>
									<option value={960}>960 px</option>
									<option value={1280}>1280 px</option>
								</select>
							</div>

							<div class="space-y-1.5">
								<label
									for="workers-select"
									class="flex h-5 items-center justify-between gap-1 text-xs font-medium text-zinc-400"
									title={gpuInfo?.physical_cores
										? `Detected ${gpuInfo.physical_cores} physical / ${gpuInfo.logical_cores ?? gpuInfo.physical_cores} logical CPU cores (capped at 8, 1 core reserved)`
										: "DataLoader worker processes (auto-detected from physical CPU cores, capped at 8)"}
								>
									<span>Workers</span>
									<span class="font-mono text-[10px] text-zinc-500">
										{workersOverride === "auto" ? `auto=${autoDetectedWorkers}` : `=${effectiveWorkers}`}
									</span>
								</label>
								<select
									id="workers-select"
									bind:value={workersOverride}
									disabled={isTraining}
									class="w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-2.5 py-2.5 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
								>
									<option value="auto">Auto ({autoDetectedWorkers})</option>
									<option value="0">0 (main)</option>
									<option value="1">1 worker</option>
									<option value="2">2 workers</option>
									<option value="4">4 workers</option>
									<option value="6">6 workers</option>
									<option value="7">7 workers</option>
									<option value="8">8 workers</option>
								</select>
							</div>
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
								After a successful run, export the best checkpoint to ONNX with the selected image size, dynamic=False, and simplify=True.
							</span>
						</span>
					</label>

					{#if epochCountdownText}
						<div class="rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3 space-y-2">
							<div class="flex flex-wrap items-center justify-between gap-2">
								<span class="font-mono text-sm font-semibold text-amber-300 whitespace-pre">
									{epochCountdownText}
								</span>
								<div class="flex items-center gap-3">
									{#if finalMetrics.map50 !== null || finalMetrics.map50_95 !== null}
										<span class="rounded border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 font-mono text-xs font-medium text-emerald-300">
											mAP50: {formatMetricPercent(finalMetrics.map50)} · mAP50-95: {formatMetricPercent(finalMetrics.map50_95)}
										</span>
									{/if}
									<span class="font-mono text-xs font-medium text-amber-400/90">
										{epochProgressPercent}%
									</span>
								</div>
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
						{:else if modelMode === "fresh" && !selectedFamilyInfo.supported}
							<svg class="h-4 w-4 text-zinc-500" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
							</svg>
							<span>{selectedFamilyInfo.label} Training — Coming Soon</span>
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

							{#if finalMetrics.map50 !== null || finalMetrics.map50_95 !== null}
								<div class="grid grid-cols-2 gap-2 rounded-md border border-zinc-800/80 bg-zinc-950/70 px-3 py-2">
									<div>
										<span class="block text-[10px] font-medium uppercase tracking-wider text-zinc-400">
											Final mAP50
										</span>
										<span class="font-mono text-sm font-semibold text-emerald-400">
											{formatMetricPercent(finalMetrics.map50)}
										</span>
									</div>
									<div>
										<span class="block text-[10px] font-medium uppercase tracking-wider text-zinc-400">
											Final mAP50-95
										</span>
										<span class="font-mono text-sm font-semibold text-amber-300">
											{formatMetricPercent(finalMetrics.map50_95)}
										</span>
									</div>
								</div>
							{/if}

							<!-- Convert to ONNX -->
							<div class="pt-1 border-t border-amber-500/20">
								<button
									type="button"
									onclick={handleExportToOnnx}
									disabled={isExportingOnnx}
									class="inline-flex items-center gap-1.5 rounded bg-amber-500 hover:bg-amber-400 text-zinc-950 font-semibold px-2.5 py-1 text-xs transition shadow-sm hover:shadow active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed"
									title="Export best checkpoint to ONNX format"
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
