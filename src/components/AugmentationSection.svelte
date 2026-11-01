<script lang="ts">
import type { AugPreset } from "../types";

interface AugmentationSectionProps {
	mosaic: number;
	mixup: number;
	copyPaste: boolean;
	degrees: number;
	flipud: number;
	translate: number;
	scale: number;
	closeMosaic: number;
	isTraining: boolean;
	activeAugSummary: string;
	activeAugPreset: AugPreset;
	activeTooltip: string | null;
	onApplyPreset: (preset: "off" | "light" | "medium" | "bench") => void;
	onToggleTooltip: (name: string) => void;
}

let {
	mosaic = $bindable(0.0),
	mixup = $bindable(0.0),
	copyPaste = $bindable(false),
	degrees = $bindable(180),
	flipud = $bindable(0.5),
	translate = $bindable(0.1),
	scale = $bindable(0.05),
	closeMosaic = $bindable(10),
	isTraining,
	activeAugSummary,
	activeAugPreset,
	activeTooltip,
	onApplyPreset,
	onToggleTooltip,
}: AugmentationSectionProps = $props();
</script>

<section class="w-full space-y-3.5">
	<div class="flex flex-wrap items-center justify-between gap-2">
		<div class="flex items-center gap-2 min-w-0 max-w-full">
			<span class="block text-xs font-semibold tracking-wider text-zinc-300 uppercase shrink-0">
				Augmentation
			</span>
			<span
				class="truncate rounded border border-zinc-800 bg-zinc-950/60 px-2 py-0.5 font-mono text-[11px] text-zinc-400 max-w-xs sm:max-w-md lg:max-w-lg"
				title={activeAugSummary}
			>
				{activeAugSummary}
			</span>
		</div>

		<!-- Preset buttons: Bench / Off / Light / Medium -->
		<div class="flex items-center gap-1 rounded-lg border border-zinc-800 bg-zinc-950/60 p-0.5 shrink-0">
			<span class="px-2 text-[11px] text-zinc-500">Preset:</span>
			<button
				type="button"
				onclick={() => onApplyPreset("bench")}
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
				onclick={() => onApplyPreset("off")}
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
				onclick={() => onApplyPreset("light")}
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
				onclick={() => onApplyPreset("medium")}
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
		<div class="space-y-1.5 min-w-0">
			<div class="relative flex h-5 items-center justify-between">
				<div class="flex items-center gap-1.5 min-w-0">
					<label for="mosaic-select" class="text-xs font-medium text-zinc-400 truncate cursor-pointer">
						Mosaic
					</label>
					<button
						type="button"
						onclick={(e) => {
							e.stopPropagation();
							onToggleTooltip("mosaic");
						}}
						class="inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-[10px] font-bold text-zinc-400 transition hover:border-amber-500/80 hover:text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-500"
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
									onToggleTooltip("mosaic");
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
				class="h-[42px] w-full min-w-0 max-w-full truncate rounded-lg border border-zinc-700/80 bg-zinc-900 px-3 py-2 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
			>
				<option value={0.0}>0.0 — Off (Bench default)</option>
				<option value={0.5}>0.5 — Moderate</option>
				<option value={1.0}>1.0 — Full (Ultralytics default)</option>
			</select>
		</div>

		<!-- MixUp slider 0.0–0.5 -->
		<div class="space-y-1.5 min-w-0">
			<div class="relative flex h-5 items-center justify-between">
				<div class="flex items-center gap-1.5 min-w-0">
					<label for="mixup-slider" class="text-xs font-medium text-zinc-400 truncate cursor-pointer">
						MixUp
					</label>
					<button
						type="button"
						onclick={(e) => {
							e.stopPropagation();
							onToggleTooltip("mixup");
						}}
						class="inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-[10px] font-bold text-zinc-400 transition hover:border-amber-500/80 hover:text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-500"
						aria-label="mixup help"
						title="Probability of blending two images and their labels. Usually left at 0.0 for clean bench photos."
					>
						?
					</button>
				</div>
				<span class="font-mono text-xs text-amber-400 shrink-0">{mixup.toFixed(2)}</span>

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
									onToggleTooltip("mixup");
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
					class="w-full min-w-0 accent-amber-500 disabled:opacity-50 cursor-pointer"
				/>
			</div>
		</div>

		<!-- Copy-paste on/off -->
		<div class="space-y-1.5 min-w-0">
			<div class="relative flex h-5 items-center justify-between">
				<div class="flex items-center gap-1.5 min-w-0">
					<label for="copy-paste-toggle" class="text-xs font-medium text-zinc-400 cursor-pointer">
						Copy-paste
					</label>
					<button
						type="button"
						onclick={(e) => {
							e.stopPropagation();
							onToggleTooltip("copy_paste");
						}}
						class="inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-[10px] font-bold text-zinc-400 transition hover:border-amber-500/80 hover:text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-500"
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
									onToggleTooltip("copy_paste");
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
				class={`flex h-[42px] w-full min-w-0 items-center justify-between rounded-lg border px-3 py-2 text-sm font-medium transition ${
					copyPaste
						? "border-amber-500/50 bg-amber-500/10 text-amber-300"
						: "border-zinc-700/80 bg-zinc-900 text-zinc-400 hover:border-zinc-600"
				} disabled:opacity-50`}
			>
				<span class="truncate">{copyPaste ? "On (0.3)" : "Off (0.0)"}</span>
				<span
					class={`relative inline-flex h-5 w-9 shrink-0 items-center rounded-full border-2 border-transparent transition-colors duration-200 ease-in-out ml-2 ${
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
		<div class="space-y-1.5 min-w-0">
			<div class="flex h-5 items-center justify-between gap-1">
				<label for="degrees-input" class="text-xs font-medium text-zinc-400 truncate cursor-pointer" title="180° gives top-down rotational invariance on a bench.">
					Degrees (±rot)
				</label>
				<span class="font-mono text-[11px] text-zinc-500 shrink-0">0–180°</span>
			</div>
			<input
				id="degrees-input"
				type="number"
				min="0"
				max="180"
				step="1"
				bind:value={degrees}
				disabled={isTraining}
				class="h-[42px] w-full min-w-0 rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
			/>
		</div>

		<!-- Translate -->
		<div class="space-y-1.5 min-w-0">
			<div class="flex h-5 items-center justify-between gap-1">
				<label for="translate-input" class="text-xs font-medium text-zinc-400 truncate cursor-pointer">
					Translate (±shift)
				</label>
				<span class="font-mono text-[11px] text-zinc-500 shrink-0">0.0–1.0</span>
			</div>
			<input
				id="translate-input"
				type="number"
				min="0"
				max="1"
				step="0.05"
				bind:value={translate}
				disabled={isTraining}
				class="h-[42px] w-full min-w-0 rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
			/>
		</div>

		<!-- Scale -->
		<div class="space-y-1.5 min-w-0">
			<div class="relative flex h-5 items-center justify-between gap-1">
				<div class="flex items-center gap-1.5 min-w-0">
					<label for="scale-input" class="text-xs font-medium text-zinc-400 truncate cursor-pointer">
						Scale (±zoom)
					</label>
					<button
						type="button"
						onclick={(e) => {
							e.stopPropagation();
							onToggleTooltip("scale");
						}}
						class="inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-[10px] font-bold text-zinc-400 transition hover:border-amber-500/80 hover:text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-500"
						aria-label="scale help"
						title="Random zoom range. Bench default 0.05 is ±5% for a fixed working distance."
					>
						?
					</button>
				</div>
				<span class="font-mono text-[11px] text-zinc-500 shrink-0">0.0–1.0</span>

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
									onToggleTooltip("scale");
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
				class="h-[42px] w-full min-w-0 rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
			/>
		</div>
	</div>

	<!-- Row 3: Flip UD and Close mosaic -->
	<div class="grid grid-cols-1 gap-4 sm:grid-cols-2">
		<div class="space-y-1.5 min-w-0">
			<div class="relative flex h-5 items-center justify-between gap-1">
				<div class="flex items-center gap-1.5 min-w-0">
					<label for="flipud-input" class="text-xs font-medium text-zinc-400 truncate cursor-pointer">
						Flip UD
					</label>
					<button
						type="button"
						onclick={(e) => {
							e.stopPropagation();
							onToggleTooltip("flipud");
						}}
						class="inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-[10px] font-bold text-zinc-400 transition hover:border-amber-500/80 hover:text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-500"
						aria-label="flipud help"
						title="Probability of a vertical flip. 0.5 gives top-down parts the same invariance as a horizontal flip."
					>
						?
					</button>
				</div>
				<span class="font-mono text-[11px] text-zinc-500 shrink-0">0.0–1.0</span>

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
									onToggleTooltip("flipud");
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
				class="h-[42px] w-full min-w-0 rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
			/>
		</div>

		<div class="space-y-1.5 min-w-0">
			<div class="relative flex h-5 items-center justify-between gap-1">
				<div class="flex items-center gap-1.5 min-w-0">
					<label for="close-mosaic-input" class="text-xs font-medium text-zinc-400 truncate cursor-pointer">
						Close mosaic
					</label>
					<button
						type="button"
						onclick={(e) => {
							e.stopPropagation();
							onToggleTooltip("close_mosaic");
						}}
						class="inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full border border-zinc-700 bg-zinc-800 text-[10px] font-bold text-zinc-400 transition hover:border-amber-500/80 hover:text-amber-300 focus:outline-none focus:ring-1 focus:ring-amber-500"
						aria-label="close mosaic help"
						title="Turns mosaic off for the final N epochs. Bench default is 10."
					>
						?
					</button>
				</div>
				<span class="font-mono text-[11px] text-zinc-500 shrink-0">epochs</span>

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
									onToggleTooltip("close_mosaic");
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
				class="h-[42px] w-full min-w-0 rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2 text-sm text-zinc-100 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
			/>
		</div>
	</div>
</section>

