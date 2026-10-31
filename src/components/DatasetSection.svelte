<script lang="ts">
import { invoke, isTauri } from "@tauri-apps/api/core";
import type { ModelFamily } from "../types";
import { normalizePath } from "../utils";

interface DatasetSectionProps {
	datasetPath: string;
	datasetFileName: string;
	selectedFamily: ModelFamily;
	isTraining: boolean;
}

let {
	datasetPath = $bindable(""),
	datasetFileName = $bindable(""),
	selectedFamily,
	isTraining,
}: DatasetSectionProps = $props();

let datasetInputRef: HTMLInputElement | undefined = $state();

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
</script>

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

