<script lang="ts">
import type { FinalMetrics, ModelMode } from "../types";
import { formatMetricPercent } from "../utils";

interface TrainingActionProps {
	autoExportOnnx: boolean;
	isFormValid: boolean;
	isTraining: boolean;
	isCheckingGpu: boolean;
	isRepairingCuda: boolean;
	isExportingOnnx: boolean;
	isTorchReady: boolean;
	isLogPanelOpen: boolean;
	modelMode: ModelMode;
	selectedFamilySupported: boolean;
	selectedFamilyLabel: string;
	statusMessage: string;
	epochCountdownText: string;
	epochProgressPercent: number;
	activeRunConfigBanner: string;
	finalMetrics: FinalMetrics;
	hasLogs: boolean;
	onOpenLogPanel: () => void;
	isStoppingTraining?: boolean;
	onStopTraining?: () => void;
}

let {
	autoExportOnnx = $bindable(true),
	isFormValid,
	isTraining,
	isCheckingGpu,
	isRepairingCuda,
	isExportingOnnx,
	isTorchReady,
	isLogPanelOpen,
	modelMode,
	selectedFamilySupported,
	selectedFamilyLabel,
	statusMessage,
	epochCountdownText,
	epochProgressPercent,
	activeRunConfigBanner,
	finalMetrics,
	hasLogs,
	onOpenLogPanel,
	isStoppingTraining = false,
	onStopTraining,
}: TrainingActionProps = $props();
</script>

<!-- Section 6: Go Button & Status -->
<div class="w-full space-y-4">
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
		<div class="space-y-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-4 py-3">
			<div class="flex flex-wrap items-center justify-between gap-2">
				<span class="font-mono text-sm font-semibold whitespace-pre text-amber-300">
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
					{#if isTraining}
						<button
							type="button"
							onclick={onStopTraining}
							disabled={isStoppingTraining}
							class="inline-flex items-center gap-1.5 rounded border border-rose-500/50 bg-rose-500/20 px-2.5 py-1 text-xs font-semibold text-rose-300 shadow-sm transition hover:bg-rose-500/30 hover:border-rose-500/70 hover:text-rose-200 disabled:cursor-not-allowed disabled:opacity-50 active:scale-[0.98]"
							title="Stop training run gracefully"
						>
							{#if isStoppingTraining}
								<svg class="h-3.5 w-3.5 animate-spin text-rose-300" viewBox="0 0 24 24" fill="none">
									<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
									<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
								</svg>
								<span>Stopping...</span>
							{:else}
								<svg class="h-3.5 w-3.5 text-rose-400" viewBox="0 0 20 20" fill="currentColor">
									<rect x="5" y="5" width="10" height="10" rx="1.5" />
								</svg>
								<span>Stop training</span>
							{/if}
						</button>
					{/if}
				</div>
			</div>
			<div class="truncate font-mono text-[11px] text-zinc-300/90" title={activeRunConfigBanner}>
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
			<span class="font-medium text-zinc-950">Exporting ONNX...</span>
		{:else if isStoppingTraining}
			<svg class="h-4 w-4 animate-spin text-zinc-950" aria-hidden="true" fill="none" viewBox="0 0 24 24">
				<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
				<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
			</svg>
			<span class="font-medium text-zinc-950">Stopping training...</span>
		{:else if isTraining}
			<svg class="h-4 w-4 animate-spin text-zinc-950" aria-hidden="true" fill="none" viewBox="0 0 24 24">
				<circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
				<path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z"></path>
			</svg>
			<span class="font-medium text-zinc-950">Training in progress...</span>
		{:else if modelMode === "fresh" && !selectedFamilySupported}
			<svg class="h-4 w-4 text-zinc-500" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24">
				<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
			</svg>
			<span>{selectedFamilyLabel} Training — Coming Soon</span>
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
		<span class={`h-2 w-2 rounded-full ${isStoppingTraining ? "bg-rose-400 animate-pulse shadow-[0_0_8px_rgba(244,63,94,0.8)]" : isTraining || isRepairingCuda ? "bg-amber-400 animate-pulse shadow-[0_0_8px_rgba(245,158,11,0.8)]" : isTorchReady ? "bg-emerald-400 shadow-[0_0_6px_rgba(52,211,153,0.6)]" : "bg-amber-500/80 shadow-[0_0_6px_rgba(245,158,11,0.6)]"}`}></span>
		<span>{statusMessage}</span>
	</div>
	{#if (hasLogs || isTraining || isRepairingCuda) && !isLogPanelOpen}
		<button
			type="button"
			onclick={onOpenLogPanel}
			class="font-medium text-amber-400 transition hover:text-amber-300"
		>
			View Log &rarr;
		</button>
	{/if}
</div>
