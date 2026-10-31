import type {
	BaseModelId,
	ModelFamily,
	ModelFamilyInfo,
	TrainTask,
} from "./types";

export const STORAGE_KEY_FAMILY = "ls_trainer.model_family";
export const STORAGE_KEY_SHOW_SEG = "ls_trainer.show_segmentation_models";

export const modelFamilies: ModelFamilyInfo[] = [
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

export interface FreshModelOption {
	id: BaseModelId;
	family: ModelFamily;
	group: string;
	task: TrainTask;
	label: string;
	desc: string;
}

export const freshModelOptions: FreshModelOption[] = [
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

export function loadSavedFamily(): ModelFamily {
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

export function loadSavedShowSeg(): boolean {
	if (typeof window === "undefined") return false;
	try {
		return window.localStorage.getItem(STORAGE_KEY_SHOW_SEG) === "true";
	} catch {
		return false;
	}
}

export function persistModelPreferences(family: ModelFamily, showSeg: boolean) {
	if (typeof window === "undefined") return;
	try {
		window.localStorage.setItem(STORAGE_KEY_FAMILY, family);
		window.localStorage.setItem(STORAGE_KEY_SHOW_SEG, String(showSeg));
	} catch {
		// ignore localStorage write errors
	}
}
