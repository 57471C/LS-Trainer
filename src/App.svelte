<script lang="ts">
import { Channel, invoke, isTauri } from "@tauri-apps/api/core";

type ModelMode = "continue" | "fresh";
type ImageSize = 640 | 960 | 1280;
type DeviceOption = "auto" | "cpu" | "0" | "mps";
type YoloModel = "yolo11n" | "yolo11s" | "yolo11m" | "yolo11l" | "yolo11x";

interface TrainingResponse {
	success: boolean;
	results_dir: string;
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
let device = $state<DeviceOption>("auto");

let isTraining = $state<boolean>(false);
let logs = $state<string[]>([]);
let resultsDir = $state<string | null>(null);
let statusMessage = $state<string>("Ready");

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

async function handlePickDataset() {
	if (isTauri()) {
		try {
			const picked = await invoke<string | null>("pick_dataset_file");
			if (picked) {
				datasetPath = picked;
				const cleanName = picked.replace(/\\/g, "/").split("/").pop();
				datasetFileName = cleanName || picked;
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
		datasetPath = (file as unknown as { path?: string }).path || file.name;
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
				existingModelPath = picked;
				const cleanName = picked.replace(/\\/g, "/").split("/").pop();
				existingModelFileName = cleanName || picked;
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
		existingModelPath =
			(file as unknown as { path?: string }).path || file.name;
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
	logs = [];
	resultsDir = null;
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

			resultsDir = response.results_dir;
			statusMessage = response.success
				? `Finished — ${response.results_dir}`
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
			`[LS-Trainer] Running outside Tauri webview (Browser Mode)`,
			`[LS-Trainer] Simulated training run '${effectiveRunName}'`,
			`[LS-Trainer] Command: yolo detect train data="${effectiveDataset}" model="${effectiveModel}" name="${effectiveRunName}" epochs=${epochs} imgsz=${imageSize} device="${device}"`,
			`[LS-Trainer] Unpacking archive simulation complete.`,
			`[LS-Trainer] Starting YOLO11 epochs (1/${epochs})...`,
			`[LS-Trainer] Training finished successfully.`,
		];
		resultsDir = `runs/detect/${effectiveRunName}/`;
		statusMessage = `Finished — ${resultsDir}`;
		isTraining = false;
	}
}
</script>

<div class="min-h-screen bg-zinc-950 font-sans text-zinc-100 antialiased selection:bg-amber-500/30 selection:text-amber-200">
	<!-- Top Bar -->
	<header class="sticky top-0 z-20 flex h-14 items-center justify-between border-b border-zinc-800/80 bg-zinc-900/90 px-6 backdrop-blur-md">
		<div class="flex items-center gap-3">
			<div class="flex h-7 w-7 items-center justify-center rounded-md border border-amber-500/40 bg-amber-500/10">
				<span class="h-2 w-2 rounded-sm bg-amber-500 shadow-[0_0_8px_rgba(245,158,11,0.8)]"></span>
			</div>
			<div>
				<h1 class="text-sm font-bold tracking-wider text-zinc-100 uppercase">LS-Trainer</h1>
			</div>
		</div>
		<div class="flex items-center gap-2">
			<span class="rounded border border-zinc-700/60 bg-zinc-800/80 px-2.5 py-0.5 font-mono text-xs text-zinc-400">
				v0.1.0
			</span>
		</div>
	</header>

	<!-- Main Form Content -->
	<main class="mx-auto max-w-2xl px-4 py-8 sm:px-6">
		<form
			onsubmit={(e) => {
				e.preventDefault();
				handleStartTraining();
			}}
			class="space-y-6 rounded-xl border border-zinc-800/90 bg-zinc-900/50 p-6 shadow-2xl backdrop-blur-sm sm:p-8"
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
								<p class="truncate text-sm font-medium text-zinc-100" title={datasetPath || datasetFileName}>
									{datasetFileName}
								</p>
								<p class="truncate text-xs text-amber-400/90" title={datasetPath || datasetFileName}>
									{datasetPath || "Archive selected"}
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
										<span class="block truncate text-sm font-medium text-zinc-100" title={existingModelPath || existingModelFileName}>
											{existingModelFileName}
										</span>
										<span class="block truncate text-xs text-amber-400/90" title={existingModelPath || existingModelFileName}>
											{existingModelPath || "Checkpoint selected"}
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
						<label for="device-select" class="block text-xs font-medium text-zinc-400">
							Device
						</label>
						<select
							id="device-select"
							bind:value={device}
							disabled={isTraining}
							class="w-full rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
						>
							<option value="auto">auto (recommended)</option>
							<option value="0">0 (GPU cuda:0)</option>
							<option value="cpu">cpu</option>
							<option value="mps">mps (Apple Silicon)</option>
						</select>
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

			<!-- Live Log Panel (Shown when training is active or logs exist) -->
			{#if logs.length > 0 || isTraining}
				<div class="space-y-2 border-t border-zinc-800/80 pt-5">
					<div class="flex items-center justify-between">
						<div class="flex items-center gap-2">
							<span class={`h-2 w-2 rounded-full ${isTraining ? "bg-amber-400 animate-pulse shadow-[0_0_8px_rgba(245,158,11,0.9)]" : "bg-emerald-500"}`}></span>
							<span class="text-xs font-semibold tracking-wider text-zinc-300 uppercase">
								Training Log
							</span>
							{#if isTraining}
								<span class="rounded bg-amber-500/10 px-2 py-0.5 text-[10px] font-medium text-amber-400 border border-amber-500/30">
									Streaming
								</span>
							{/if}
						</div>
						<div class="flex items-center gap-2">
							<span class="font-mono text-[11px] text-zinc-500">
								{logs.length} lines
							</span>
							<button
								type="button"
								onclick={() => { logs = []; }}
								disabled={isTraining}
								class="rounded bg-zinc-800/80 px-2 py-0.5 text-[11px] text-zinc-400 transition hover:bg-zinc-700 hover:text-zinc-200 disabled:opacity-40"
							>
								Clear
							</button>
						</div>
					</div>

					<div
						bind:this={logContainerRef}
						class="max-h-64 overflow-y-auto rounded-lg border border-zinc-800 bg-zinc-950 p-3.5 font-mono text-xs leading-relaxed text-zinc-300 shadow-inner select-text"
					>
						{#each logs as logLine, i (i)}
							<div class="flex gap-2">
								<span class="w-8 shrink-0 text-right text-zinc-600 select-none">{i + 1}</span>
								<span class={logLine.includes("Error") || logLine.includes("[ERROR]") ? "text-red-400" : logLine.includes("[LS-Trainer]") ? "text-amber-400" : "text-zinc-300"}>
									{logLine}
								</span>
							</div>
						{/each}
					</div>
				</div>
			{/if}

			<!-- Results Folder Banner (Shown when finished) -->
			{#if resultsDir}
				<div class="flex items-start gap-3 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3.5 text-zinc-200">
					<svg class="h-5 w-5 shrink-0 text-amber-400 mt-0.5" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
					</svg>
					<div class="min-w-0">
						<p class="text-xs font-semibold uppercase tracking-wider text-amber-300">
							Training Results Directory
						</p>
						<p class="mt-0.5 font-mono text-xs font-medium text-zinc-100 select-all break-all">
							{resultsDir}
						</p>
					</div>
				</div>
			{/if}

			<!-- Status Line -->
			<div class="flex items-center justify-center gap-2 border-t border-zinc-800/60 pt-4 text-xs text-zinc-400">
				<span class={`h-2 w-2 rounded-full ${isTraining ? "bg-amber-400 animate-pulse shadow-[0_0_8px_rgba(245,158,11,0.8)]" : "bg-amber-500/80 shadow-[0_0_6px_rgba(245,158,11,0.6)]"}`}></span>
				<span>{statusMessage}</span>
			</div>
		</form>
	</main>
</div>
