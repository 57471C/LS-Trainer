<script lang="ts">
import { Channel, invoke, isTauri } from "@tauri-apps/api/core";

type ModelMode = "continue" | "fresh";
type ImageSize = 640 | 960 | 1280;
type YoloModel = "yolo11n" | "yolo11s" | "yolo11m" | "yolo11l" | "yolo11x";

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

const freshModelOptions: { id: YoloModel; label: string; desc: string }[] = [
	{ id: "yolo11n", label: "yolo11n", desc: "Nano (fastest, lightweight)" },
	{ id: "yolo11s", label: "yolo11s", desc: "Small (balanced fast)" },
	{ id: "yolo11m", label: "yolo11m", desc: "Medium (high accuracy)" },
	{ id: "yolo11l", label: "yolo11l", desc: "Large (extra accuracy)" },
	{ id: "yolo11x", label: "yolo11x", desc: "XLarge (maximum capacity)" },
];

let datasetPath = $state<string>("");
let datasetFileName = $state<string>("");
let modelMode = $state<ModelMode>("fresh");
let existingModelPath = $state<string>("");
let existingModelFileName = $state<string>("");
let freshModel = $state<YoloModel>("yolo11n");
let runName = $state<string>("");
let epochs = $state<number>(60);
let imageSize = $state<ImageSize>(960);
let device = $state<string>("auto");

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
const isFormValid = $derived(
	isDatasetSelected && isModelSelected && !isTraining,
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
	statusMessage = "Training in progress...";

	const effectiveRunName = runName.trim() || "BB-Parts-v3";
	const effectiveDataset = datasetPath.trim() || datasetFileName.trim();
	const effectiveModel =
		modelMode === "continue"
			? existingModelPath.trim() || existingModelFileName.trim()
			: freshModel;

	const requestPayload = {
		dataset_path: effectiveDataset,
		model_choice: effectiveModel,
		run_name: effectiveRunName,
		epochs: Number(epochs),
		imgsz: Number(imageSize),
		device,
	};

	console.log("LS-Trainer Starting Training:", requestPayload);

	if (isTauri()) {
		try {
			const onLog = new Channel<string>();
			onLog.onmessage = (message) => {
				logs = [...logs, message];
			};

			const response = await invoke<TrainingResponse>("start_training", {
				onLog,
				request: requestPayload,
			});

			resultsDir = normalizePath(response.results_dir);
			statusMessage = response.success
				? `Finished — ${normalizePath(response.results_dir)}`
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
		logs = [
			"[LS-Trainer] Running outside Tauri webview (Browser Mode)",
			`[LS-Trainer] Simulated training run '${effectiveRunName}'`,
			`[LS-Trainer] Command: yolo detect train data="${effectiveDataset}" model="${effectiveModel}" name="${effectiveRunName}" epochs=${epochs} imgsz=${imageSize} device="${device}"`,
			"[LS-Trainer] Unpacking archive simulation complete.",
			`[LS-Trainer] Starting YOLO11 epochs (1/${epochs})...`,
			"[LS-Trainer] Training finished successfully.",
		];
		resultsDir = normalizePath(
			`C:/Scripts/LS-Trainer/runs/detect/${effectiveRunName}/`,
		);
		statusMessage = `Finished — ${normalizePath(resultsDir)}`;
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
			`[LS-Trainer] Export command: yolo export model=${normalizePath(resultsDir)}weights\\best.pt format=onnx imgsz=640`,
			`[LS-Trainer] Export completed successfully: ${normalizePath(resultsDir)}weights\\best.onnx`,
			"[LS-Trainer] Note: This model is optimized and ready for inference in LS-Vision.",
		];
		onnxPath = normalizePath(`${resultsDir}weights/best.onnx`);
		isExportingOnnx = false;
	}
}
</script>

<div class="flex h-screen w-screen flex-col overflow-hidden bg-zinc-950 font-sans text-zinc-100 antialiased selection:bg-amber-500/30 selection:text-amber-200">
	<!-- Top Bar -->
	<header class="flex h-14 shrink-0 items-center justify-between border-b border-zinc-800/80 bg-zinc-900/90 px-6 backdrop-blur-md z-10">
		<div class="flex items-center gap-3">
			<div class="flex h-7 w-7 items-center justify-center rounded-md border border-amber-500/40 bg-amber-500/10">
				<span class="h-2 w-2 rounded-sm bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.8)]"></span>
			</div>
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
				v0.1.0
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
				class="w-full max-w-2xl space-y-6 rounded-xl border border-zinc-800/90 bg-zinc-900/50 p-6 shadow-2xl backdrop-blur-sm sm:p-8"
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
								<span class="block text-xs text-zinc-400">Initialize a pretrained YOLO11 base</span>
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
								{#each freshModelOptions as opt (opt.id)}
									<option value={opt.id}>
										{opt.label} — {opt.desc}
									</option>
								{/each}
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

				<!-- Section 4: Parameters -->
				<section class="space-y-3 border-t border-zinc-800/80 pt-5">
					<span class="block text-xs font-semibold tracking-wider text-zinc-300 uppercase">
						Parameters
					</span>

					<div class="grid grid-cols-1 gap-4 sm:grid-cols-3">
						<!-- Epochs -->
						<div class="space-y-1.5">
							<label for="epochs-input" class="block text-xs font-medium text-zinc-400">
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

						<!-- Image size -->
						<div class="space-y-1.5">
							<label for="image-size-select" class="block text-xs font-medium text-zinc-400">
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
						<div class="space-y-1.5">
							<label for="device-input" class="block text-xs font-medium text-zinc-400">
								Device
							</label>
							<div class="relative">
								<input
									id="device-input"
									type="text"
									list="device-options"
									bind:value={device}
									disabled={isTraining}
									placeholder="auto"
									class="w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 placeholder-zinc-500 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
								/>
								<datalist id="device-options">
									<option value="auto">auto (recommended)</option>
									<option value="cpu">cpu</option>
									<option value="0">0 (GPU cuda:0)</option>
									<option value="mps">mps (Apple Silicon)</option>
								</datalist>
							</div>
						</div>
					</div>
				</section>

				<!-- Section 5: Go Button -->
				<div class="border-t border-zinc-800/80 pt-5">
					<button
						type="submit"
						disabled={!isFormValid || isTraining}
						class={`flex w-full items-center justify-center gap-2 rounded-lg py-3 text-sm font-semibold tracking-wide transition-all ${
							isFormValid && !isTraining
								? "cursor-pointer border border-amber-400/80 bg-amber-500 text-zinc-950 shadow-[0_0_20px_rgba(245,158,11,0.25)] hover:bg-amber-400 hover:shadow-[0_0_25px_rgba(245,158,11,0.4)] active:scale-[0.99]"
								: "cursor-not-allowed border border-zinc-800 bg-zinc-800/50 text-zinc-500 opacity-60"
						}`}
					>
						{#if isTraining}
							<svg class="h-4 w-4 animate-spin text-zinc-950" aria-hidden="true" fill="none" viewBox="0 0 24 24">
								<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
								<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
							</svg>
							<span class="text-zinc-950 font-medium">Training in progress...</span>
						{:else}
							<svg class="h-4 w-4" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.2" d="M14 5l7 7m0 0l-7 7m7-7H3" />
							</svg>
							<span>Go</span>
						{/if}
					</button>
				</div>

				<!-- Status Line -->
				<div class="flex items-center justify-between border-t border-zinc-800/60 pt-4 text-xs text-zinc-400">
					<div class="flex items-center gap-2">
						<span class={`h-2 w-2 rounded-full ${isTraining ? "bg-amber-400 animate-pulse shadow-[0_0_8px_rgba(245,158,11,0.8)]" : "bg-amber-500/80 shadow-[0_0_6px_rgba(245,158,11,0.6)]"}`}></span>
						<span>{statusMessage}</span>
					</div>
					{#if (logs.length > 0 || isTraining) && !isLogPanelOpen}
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
				<div class="flex h-12 shrink-0 items-center justify-between border-b border-zinc-800/80 bg-zinc-900/90 px-4">
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
