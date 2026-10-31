export type ModelMode = "continue" | "fresh";
export type TrainTask = "detect" | "segment";
export type ImageSize = 640 | 960 | 1280;
export type ModelFamily = "YOLO" | "RF-DETR" | "RT-DETR" | "YOLOX";

export type BaseModelId =
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

export type AugPreset = "off" | "light" | "medium" | "bench" | "custom";

export interface TrainingResponse {
	success: boolean;
	results_dir: string;
	message: string;
	map50?: number | null;
	map50_95?: number | null;
}

export interface ExportResponse {
	success: boolean;
	onnx_path: string;
	message: string;
}

export interface GpuInfo {
	xpu_available?: boolean;
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

export interface LabelCheckResult {
	polygons_found: boolean;
	bboxes_found: boolean;
	files_scanned: number;
	lines_scanned: number;
	warning: string | null;
}

export interface ModelFamilyInfo {
	id: ModelFamily;
	label: string;
	vendor: string;
	badge: string;
	supported: boolean;
	defaultDetectModel: BaseModelId;
	defaultSegModel: BaseModelId | null;
	description: string;
}

export interface FinalMetrics {
	map50?: number | null;
	map50_95?: number | null;
}
