<script lang="ts">
import type { UnpackedDatasetInfo } from "../types";
import { formatByteSize, normalizePath } from "../utils";

interface Props {
	info: UnpackedDatasetInfo | null;
	deleting?: boolean;
	error?: string | null;
	onDelete: () => void;
	onKeep: () => void;
}

const {
	info,
	deleting = false,
	error = null,
	onDelete,
	onKeep,
}: Props = $props();

let panel: HTMLDivElement | undefined = $state();
let acceptBackdropClose = $state(false);

$effect(() => {
	if (!info) {
		acceptBackdropClose = false;
		return;
	}
	panel?.focus();
	const timer = setTimeout(() => {
		acceptBackdropClose = true;
	}, 0);
	return () => clearTimeout(timer);
});
</script>

<svelte:window
	onkeydown={(e) => {
		if (!info || deleting || e.key !== "Escape") return;
		e.preventDefault();
		onKeep();
	}}
/>

{#if info}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
		role="presentation"
		onclick={(e) => {
			if (!acceptBackdropClose || deleting || e.target !== e.currentTarget) return;
			onKeep();
		}}
	>
		<div
			bind:this={panel}
			class="w-full max-w-lg rounded-xl border border-zinc-700/80 bg-zinc-900 p-5 shadow-2xl"
			role="dialog"
			aria-modal="true"
			aria-labelledby="unpacked-dataset-title"
			tabindex="-1"
		>
			<div class="mb-3 flex items-center justify-between">
				<h2
					id="unpacked-dataset-title"
					class="text-sm font-semibold tracking-wide text-zinc-100"
				>
					Delete unpacked dataset?
				</h2>
				<button
					type="button"
					onclick={onKeep}
					disabled={deleting}
					class="rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-white disabled:opacity-50"
					aria-label="Keep unpacked dataset"
				>
					✕
				</button>
			</div>

			<p class="mb-3 text-xs leading-relaxed text-zinc-300">
				This training run unpacked an archive into the dataset store. Delete this folder to free disk space, or keep it for inspection?
			</p>

			<div class="mb-4 space-y-2 rounded-lg border border-zinc-800 bg-zinc-950/80 p-3">
				<div>
					<span class="block text-[11px] font-medium text-zinc-500 uppercase">Path</span>
					<code class="mt-0.5 block font-mono text-xs break-all text-zinc-200">
						{normalizePath(info.path)}
					</code>
				</div>
				<div class="flex flex-wrap gap-4 pt-1 text-xs text-zinc-300">
					<div>
						<span class="text-zinc-500">Images:</span>
						<span class="ml-1 font-mono font-medium text-zinc-100">
							{info.image_count.toLocaleString()}
						</span>
					</div>
					<div>
						<span class="text-zinc-500">Size:</span>
						<span class="ml-1 font-mono font-medium text-zinc-100">
							{formatByteSize(info.size_bytes)}
						</span>
					</div>
				</div>
			</div>

			{#if error}
				<p class="mb-3 text-xs text-red-400">{error}</p>
			{/if}

			<div class="flex justify-end gap-2">
				<button
					type="button"
					onclick={onKeep}
					disabled={deleting}
					class="rounded-md border border-zinc-700 px-3.5 py-1.5 text-xs font-medium text-zinc-300 hover:bg-zinc-800 disabled:opacity-50"
				>
					Keep
				</button>
				<button
					type="button"
					onclick={onDelete}
					disabled={deleting}
					class="rounded-md bg-rose-600 px-3.5 py-1.5 text-xs font-medium text-white hover:bg-rose-500 disabled:opacity-50"
				>
					{deleting ? "Deleting…" : "Delete"}
				</button>
			</div>
		</div>
	</div>
{/if}
