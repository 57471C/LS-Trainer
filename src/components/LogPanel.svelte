<script lang="ts">
import { invoke, isTauri } from "@tauri-apps/api/core";
import type { FinalMetrics } from "../types";
import { formatMetricPercent, normalizePath } from "../utils";

interface LogPanelProps {
	isLogPanelOpen: boolean;
	isTraining: boolean;
	isExportingOnnx: boolean;
	logs: string[];
	resultsDir: string | null;
	onnxPath: string | null;
	onnxError: string | null;
	activeRunConfigBanner: string;
	finalMetrics: FinalMetrics;
	onClose: () => void;
	onClearLogs: () => void;
	onExportOnnx: () => void;
}

const {
	isLogPanelOpen,
	isTraining,
	isExportingOnnx,
	logs,
	resultsDir,
	onnxPath,
	onnxError,
	activeRunConfigBanner,
	finalMetrics,
	onClose,
	onClearLogs,
	onExportOnnx,
}: LogPanelProps = $props();

let logPanelWidth = $state<number>(540);
let isDragging = $state<boolean>(false);
let hasCopiedLogs = $state<boolean>(false);
let hasCopiedResultsPath = $state<boolean>(false);
let hasCopiedOnnxPath = $state<boolean>(false);
let logContainerRef: HTMLDivElement | undefined = $state();

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

async function copyLogsToClipboard() {
	if (logs.length === 0) return;
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

function handleWindowKeydown(e: KeyboardEvent) {
	if (e.key === "F12" && !e.ctrlKey && !e.metaKey && !e.altKey && !e.shiftKey) {
		e.preventDefault();
		void copyLogsToClipboard();
	}
}

async function handleOpenFolder(folderPath: string) {
	if (!folderPath) return;
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
	if (!resultsDir) return;
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
	if (!onnxPath) return;
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
</script>

<svelte:window onkeydown={handleWindowKeydown} />

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
		<div class="flex shrink-0 flex-col gap-1 border-b border-zinc-800/80 bg-zinc-900/90 px-4 py-2.5">
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
					<span class="mr-1 font-mono text-[11px] text-zinc-500">
						{logs.length} lines
					</span>
					<button
						type="button"
						onclick={onClearLogs}
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
						title={hasCopiedLogs ? "Copied!" : "Copy logs to clipboard (F12)"}
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
						onclick={onClose}
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
			<div class="truncate font-mono text-[11px] text-amber-300/90" title={activeRunConfigBanner}>
				{activeRunConfigBanner}
			</div>
		</div>

		<!-- Results Folder Banner & ONNX Export (if present) -->
		{#if resultsDir}
			<div class="m-3 mb-0 shrink-0 space-y-2.5">
				<!-- Run directory card -->
				<div class="space-y-2.5 rounded-lg border border-amber-500/40 bg-amber-500/10 p-3.5 text-zinc-200">
					<div class="flex flex-wrap items-center justify-between gap-2">
						<div class="flex items-center gap-2 min-w-0">
							<svg class="h-4 w-4 shrink-0 text-amber-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
								<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-6l-2-2H5a2 2 0 00-2 2z" />
							</svg>
							<p class="truncate text-[11px] font-semibold uppercase tracking-wider text-amber-300">
								Training Results Directory
							</p>
						</div>
						<div class="flex items-center gap-1.5 shrink-0">
							<!-- Open Folder -->
							<button
								type="button"
								onclick={() => {
									if (resultsDir) handleOpenFolder(resultsDir);
								}}
								class="inline-flex items-center gap-1.5 rounded border border-zinc-700/80 bg-zinc-900/90 px-2.5 py-1 text-xs font-medium text-zinc-200 transition hover:bg-zinc-800 hover:text-white"
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
								class="inline-flex items-center gap-1.5 rounded border border-zinc-700/80 bg-zinc-900/90 px-2.5 py-1 text-xs font-medium text-zinc-200 transition hover:bg-zinc-800 hover:text-white"
								title="Copy directory path to clipboard"
							>
								{#if hasCopiedResultsPath}
									<svg class="h-3.5 w-3.5 text-emerald-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7" />
									</svg>
									<span class="font-medium text-emerald-400">Copied!</span>
								{:else}
									<svg class="h-3.5 w-3.5 text-zinc-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
									</svg>
									<span>Copy Path</span>
								{/if}
							</button>
						</div>
					</div>

					<p class="select-all font-mono text-xs font-medium break-all text-zinc-100 min-w-0">
						{normalizePath(resultsDir)}
					</p>

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
					<div class="border-t border-amber-500/20 pt-1">
						<button
							type="button"
							onclick={onExportOnnx}
							disabled={isExportingOnnx}
							class="inline-flex items-center gap-1.5 rounded bg-amber-500 px-2.5 py-1 text-xs font-semibold text-zinc-950 shadow-sm transition hover:bg-amber-400 hover:shadow active:scale-[0.98] disabled:cursor-not-allowed disabled:opacity-50"
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
					<div class="space-y-2 rounded-lg border border-emerald-500/40 bg-emerald-500/10 p-3 text-zinc-200">
						<div class="flex flex-wrap items-center justify-between gap-2">
							<div class="flex items-center gap-2 min-w-0">
								<div class="flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-emerald-500/20 text-emerald-400">
									<svg class="h-3.5 w-3.5" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
										<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7" />
									</svg>
								</div>
								<p class="truncate text-[11px] font-semibold uppercase tracking-wider text-emerald-300">
									ONNX Model Ready
								</p>
							</div>
							<div class="flex items-center gap-1.5 shrink-0">
								<!-- Open Folder -->
								<button
									type="button"
									onclick={() => {
										if (onnxPath) handleOpenFolder(onnxPath);
									}}
									class="inline-flex items-center gap-1.5 rounded border border-zinc-700/80 bg-zinc-900/90 px-2.5 py-1 text-xs font-medium text-zinc-200 transition hover:bg-zinc-800 hover:text-white"
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
									class="inline-flex items-center gap-1.5 rounded border border-zinc-700/80 bg-zinc-900/90 px-2.5 py-1 text-xs font-medium text-zinc-200 transition hover:bg-zinc-800 hover:text-white"
									title="Copy ONNX path to clipboard"
								>
									{#if hasCopiedOnnxPath}
										<svg class="h-3.5 w-3.5 text-emerald-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2.5" d="M5 13l4 4L19 7" />
										</svg>
										<span class="font-medium text-emerald-400">Copied!</span>
									{:else}
										<svg class="h-3.5 w-3.5 text-zinc-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
										</svg>
										<span>Copy ONNX Path</span>
									{/if}
								</button>
							</div>
						</div>

						<div class="space-y-1">
							<p class="select-all font-mono text-xs font-medium break-all text-zinc-100 min-w-0">
								{normalizePath(onnxPath)}
							</p>
							<p class="text-xs font-medium text-emerald-400/90">
								This model is optimized and ready for inference in LS-Vision.
							</p>
						</div>
					</div>
				{/if}

				<!-- ONNX Error Alert -->
				{#if onnxError}
					<div class="flex items-start gap-2 rounded-lg border border-red-500/40 bg-red-500/10 p-3 text-xs text-red-300">
						<svg class="mt-0.5 h-4 w-4 shrink-0 text-red-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
			class="flex-1 space-y-1 overflow-y-auto overscroll-none bg-zinc-950 p-4 font-mono text-xs leading-relaxed text-zinc-300 select-text"
		>
			{#if logs.length === 0}
				<div class="flex h-full flex-col items-center justify-center text-center text-zinc-600">
					<svg class="mb-2 h-8 w-8 opacity-50" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
						<path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M8 9l3 3-3 3m5 0h3M5 20h14a2 2 0 002-2V6a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
					</svg>
					<p>Ready to stream training output.</p>
				</div>
			{:else}
				{#each logs as logLine, i (i)}
					<div class="flex gap-2">
						<span class="w-8 shrink-0 select-none text-right text-zinc-600">{i + 1}</span>
						<span class={logLine.includes("Error") || logLine.includes("[ERROR]") ? "text-red-400" : logLine.includes("[LS-Trainer]") ? "text-amber-400" : "text-zinc-300"}>
							{logLine}
						</span>
					</div>
				{/each}
			{/if}
		</div>
	</aside>
{/if}
