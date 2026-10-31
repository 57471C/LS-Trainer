<script lang="ts">
import type { GpuInfo, ImageSize } from "../types";
import { shortenGpuName } from "../utils";

interface ParametersSectionProps {
	epochs: number;
	batch: number;
	imageSize: ImageSize;
	workersOverride: string;
	device: string;
	isCustomDevice: boolean;
	customDevice: string;
	amp: boolean;
	gpuInfo: GpuInfo | null;
	isTraining: boolean;
	isRepairingCuda: boolean;
	statusMessage: string;
	autoDetectedWorkers: number;
	effectiveWorkers: number;
	activeTooltip: string | null;
	onRepairCuda: () => void;
	onToggleTooltip: (name: string) => void;
}

let {
	epochs = $bindable(60),
	batch = $bindable(4),
	imageSize = $bindable(960),
	workersOverride = $bindable("auto"),
	device = $bindable("0"),
	isCustomDevice = $bindable(false),
	customDevice = $bindable(""),
	amp = $bindable(true),
	gpuInfo,
	isTraining,
	isRepairingCuda,
	statusMessage,
	autoDetectedWorkers,
	effectiveWorkers,
	activeTooltip,
	onRepairCuda,
	onToggleTooltip,
}: ParametersSectionProps = $props();
</script>

<section class="w-full space-y-3">
	<div class="flex items-center justify-between gap-2">
		<span class="block text-xs font-semibold tracking-wider text-zinc-300 uppercase">
			Parameters &amp; Settings
		</span>
		<button
			type="button"
			onclick={onRepairCuda}
			disabled={isTraining || isRepairingCuda}
			class="inline-flex items-center gap-1.5 rounded-md border border-zinc-700/80 bg-zinc-800/80 px-2.5 py-1 text-xs font-medium text-zinc-300 transition hover:border-amber-500/60 hover:bg-zinc-800 hover:text-amber-300 disabled:opacity-50"
			title="Re-detect GPU (Intel Arc XPU or NVIDIA CUDA) and install the matching PyTorch wheel"
		>
			{#if isRepairingCuda}
				<svg class="h-3.5 w-3.5 animate-spin text-amber-400" aria-hidden="true" fill="none" viewBox="0 0 24 24">
					<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
					<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
				</svg>
				<span>{gpuInfo?.recommended_wheel === "xpu" ? "Repairing XPU…" : "Repairing GPU…"}</span>
			{:else}
				<svg class="h-3.5 w-3.5 text-amber-400" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
				</svg>
				<span>{gpuInfo?.recommended_wheel === "xpu" ? "Repair XPU" : "Repair CUDA"}</span>
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

	<div class="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-3">
		<!-- Epochs -->
		<div class="space-y-1.5 min-w-0">
			<label for="epochs-input" class="flex h-5 items-center text-xs font-medium text-zinc-400 truncate">
				Epochs
			</label>
			<input
				id="epochs-input"
				type="number"
				min="1"
				step="1"
				bind:value={epochs}
				disabled={isTraining}
				class="h-[42px] w-full min-w-0 rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
			/>
		</div>

		<!-- Batch -->
		<div class="space-y-1.5 min-w-0">
			<label for="batch-input" class="flex h-5 items-center text-xs font-medium text-zinc-400 truncate">
				Batch
			</label>
			<input
				id="batch-input"
				type="number"
				min="1"
				step="1"
				bind:value={batch}
				disabled={isTraining}
				class="h-[42px] w-full min-w-0 rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
			/>
		</div>

		<!-- Image size -->
		<div class="space-y-1.5 min-w-0">
			<label for="image-size-select" class="flex h-5 items-center text-xs font-medium text-zinc-400 truncate">
				Image size
			</label>
			<select
				id="image-size-select"
				bind:value={imageSize}
				disabled={isTraining}
				class="h-[42px] w-full min-w-0 max-w-full truncate rounded-lg border border-zinc-700/80 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
			>
				<option value={640}>640 px</option>
				<option value={960}>960 px</option>
				<option value={1280}>1280 px</option>
			</select>
		</div>

		<!-- DataLoader Workers -->
		<div class="space-y-1.5 min-w-0">
			<div
				class="flex h-5 items-center justify-between gap-1 text-xs font-medium text-zinc-400 min-w-0"
				title={gpuInfo?.physical_cores
					? `Detected ${gpuInfo.physical_cores} physical / ${gpuInfo.logical_cores ?? gpuInfo.physical_cores} logical CPU cores (capped at 8, 1 core reserved)`
					: "DataLoader worker processes (auto-detected from physical CPU cores, capped at 8)"}
			>
				<label for="workers-select" class="shrink-0 cursor-pointer">Workers</label>
				<span class="font-mono text-[10px] text-zinc-500 truncate">
					{workersOverride === "auto" ? `auto=${autoDetectedWorkers}` : `=${effectiveWorkers}`}
				</span>
			</div>
			<select
				id="workers-select"
				bind:value={workersOverride}
				disabled={isTraining}
				class="h-[42px] w-full min-w-0 max-w-full truncate rounded-lg border border-zinc-700/80 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
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

		<!-- Device -->
		<div class="space-y-1.5 min-w-0">
			<div class="flex h-5 items-center justify-between gap-1.5 min-w-0">
				<label for="device-select" class="block text-xs font-medium text-zinc-400 shrink-0 cursor-pointer">
					Device
				</label>
				{#if (gpuInfo?.xpu_available || gpuInfo?.cuda_available) && gpuInfo.device_name}
					<span class="flex items-center gap-1 text-[11px] font-medium text-emerald-400 truncate min-w-0" title={gpuInfo.device_name}>
						<span class="h-1.5 w-1.5 shrink-0 rounded-full bg-emerald-400"></span>
						<span class="truncate">{shortenGpuName(gpuInfo.device_name)}</span>
					</span>
				{:else if gpuInfo && !gpuInfo.xpu_available && !gpuInfo.cuda_available}
					<span class="text-[11px] text-zinc-500 truncate min-w-0" title="CPU only (No XPU/CUDA GPU)">
						CPU only
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
				class="h-[42px] w-full min-w-0 max-w-full truncate rounded-lg border border-zinc-700/80 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
			>
				{#if gpuInfo?.xpu_available}
					<option value="xpu">
						{gpuInfo.device_name ? `Intel Arc XPU: ${shortenGpuName(gpuInfo.device_name)}` : "Intel Arc XPU (device=xpu)"}
					</option>
					<option value="cpu">CPU (device=cpu)</option>
					<option value="custom">Custom device...</option>
				{:else if gpuInfo?.cuda_available}
					<option value="0">
						{gpuInfo.device_name ? `GPU 0: ${gpuInfo.device_name}` : "GPU (device=0)"}
					</option>
					<option value="cpu">CPU (device=cpu)</option>
					<option value="custom">Custom device...</option>
				{:else if gpuInfo !== null}
					<option value="cpu">CPU (device=cpu)</option>
					<option value="custom">Custom device...</option>
				{:else}
					<option value="0">GPU (device=0)</option>
					<option value="xpu">Intel Arc XPU (device=xpu)</option>
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
					class="mt-1.5 h-[36px] w-full min-w-0 rounded-lg border border-zinc-700/80 bg-zinc-900 px-3 py-1.5 text-xs text-zinc-100 placeholder-zinc-500 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
				/>
			{/if}
		</div>

		<!-- amp -->
		<div class="space-y-1.5 min-w-0">
			<div class="relative flex h-5 items-center justify-between">
				<div class="flex items-center gap-1.5">
					<label for="amp-toggle" class="text-xs font-medium text-zinc-400 cursor-pointer">
						amp
					</label>
					<button
						type="button"
						onclick={(e) => {
							e.stopPropagation();
							onToggleTooltip("amp");
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
									onToggleTooltip("amp");
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
				class={`flex h-[42px] w-full min-w-0 items-center justify-between rounded-lg border px-3 py-2 text-sm font-medium transition ${
					amp
						? "border-amber-500/50 bg-amber-500/10 text-amber-300"
						: "border-zinc-700/80 bg-zinc-900 text-zinc-400 hover:border-zinc-600"
				} disabled:opacity-50`}
			>
				<span class="truncate">{amp ? "Enabled (FP16)" : "Disabled (FP32)"}</span>
				<span
					class={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out ml-2 ${
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
