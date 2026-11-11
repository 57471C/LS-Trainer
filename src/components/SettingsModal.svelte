<script lang="ts">
	import { invoke, isTauri } from "@tauri-apps/api/core";

	interface Props {
		open: boolean;
		onClose: () => void;
		dataDirectory: string;
		onSaved: (path: string) => void;
	}

	const { open, onClose, dataDirectory, onSaved }: Props = $props();

	let draft = $state("");
	let saving = $state(false);
	let error = $state<string | null>(null);

	$effect(() => {
		if (open) {
			draft = dataDirectory;
			error = null;
		}
	});

	async function pickFolder() {
		if (!isTauri()) return;
		try {
			const picked = await invoke<string | null>("pick_dataset_folder");
			if (picked) draft = picked;
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		}
	}

	function useDefaultMount() {
		draft = "/mnt/datasets";
	}

	async function save() {
		saving = true;
		error = null;
		try {
			if (isTauri()) {
				const result = await invoke<{ data_directory: string }>("set_data_directory", {
					path: draft.trim(),
				});
				onSaved(result.data_directory);
			} else {
				onSaved(draft.trim());
			}
			onClose();
		} catch (e) {
			error = e instanceof Error ? e.message : String(e);
		} finally {
			saving = false;
		}
	}
</script>

{#if open}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-sm"
		onclick={onClose}
		onkeydown={(e) => e.key === "Escape" && onClose()}
		role="dialog"
		aria-modal="true"
	>
		<div
			class="w-full max-w-lg rounded-xl border border-zinc-700/80 bg-zinc-900 p-5 shadow-2xl"
			onclick|stopPropagation
			onkeydown|stopPropagation
		>
			<div class="mb-4 flex items-center justify-between">
				<h2 class="text-sm font-semibold tracking-wide text-zinc-100 uppercase">Settings</h2>
				<button
					type="button"
					onclick={onClose}
					class="rounded p-1 text-zinc-400 hover:bg-zinc-800 hover:text-white"
				>
					✕
				</button>
			</div>

			<label class="mb-1 block text-xs font-medium text-zinc-400">
				Data directory (runs & extraction)
			</label>
			<p class="mb-2 text-[11px] leading-relaxed text-zinc-500">
				Leave empty to use the project folder. When set, training outputs go to
				<code class="text-zinc-400"><data>/runs/<task>/<run></code>.
			</p>

			<div class="mb-3 flex gap-2">
				<input
					type="text"
					bind:value={draft}
					placeholder="/mnt/datasets"
					class="flex-1 rounded-md border border-zinc-700 bg-zinc-950 px-3 py-1.5 text-sm text-zinc-100 outline-none focus:border-amber-500/60"
				/>
				<button
					type="button"
					onclick={pickFolder}
					class="rounded-md border border-zinc-700 bg-zinc-800 px-3 py-1.5 text-xs text-zinc-300 hover:bg-zinc-700"
				>
					Browse
				</button>
			</div>

			<button
				type="button"
				onclick={useDefaultMount}
				class="mb-4 text-xs text-amber-400/80 hover:text-amber-300"
			>
				Use /mnt/datasets
			</button>

			{#if error}
				<p class="mb-3 text-xs text-red-400">{error}</p>
			{/if}

			<div class="flex justify-end gap-2">
				<button
					type="button"
					onclick={onClose}
					class="rounded-md border border-zinc-700 px-3 py-1.5 text-xs text-zinc-400 hover:bg-zinc-800"
				>
					Cancel
				</button>
				<button
					type="button"
				onclick={save}
					disabled={saving}
					class="rounded-md bg-amber-600 px-3 py-1.5 text-xs font-medium text-white hover:bg-amber-500 disabled:opacity-50"
				>
					{saving ? "Saving…" : "Save"}
				</button>
			</div>
		</div>
	</div>
{/if}
