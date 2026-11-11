<script lang="ts">
interface HeaderProps {
	isLogPanelOpen: boolean;
	isTraining: boolean;
	logCount: number;
	onToggleLogPanel: () => void;
	onOpenSettings: () => void;
}

const { isLogPanelOpen, isTraining, logCount, onToggleLogPanel, onOpenSettings }: HeaderProps =
	$props();
</script>

<header class="z-10 flex h-14 shrink-0 items-center justify-between border-b border-zinc-800/80 bg-zinc-900/90 px-6 backdrop-blur-md">
	<div class="flex items-center gap-3">
		<img
			src="/LS-Trainer.svg"
			alt="LS-Trainer Logo"
			class="h-7 w-7 rounded-md select-none"
		/>
		<div>
			<h1 class="text-sm font-bold tracking-wider text-zinc-100 uppercase">LS-Trainer</h1>
		</div>
	</div>
	<div class="flex items-center gap-2.5">
		{#if logCount > 0 || isTraining}
			<button
				type="button"
				onclick={onToggleLogPanel}
				class={`flex items-center gap-1.5 rounded-md border px-2.5 py-1 text-xs font-medium transition ${
					isLogPanelOpen
						? "border-amber-500/50 bg-amber-500/10 text-amber-300"
						: "border-zinc-700 bg-zinc-800/80 text-zinc-300 hover:bg-zinc-700 hover:text-white"
				}`}
			>
				<span class={`h-1.5 w-1.5 rounded-full ${isTraining ? "bg-amber-400 animate-pulse" : "bg-emerald-500"}`}></span>
				<span>{isLogPanelOpen ? "Hide Log" : "Show Log"}</span>
				<span class="font-mono text-[10px] text-zinc-400">({logCount})</span>
			</button>
		{/if}
		<button
			type="button"
			onclick={onOpenSettings}
			title="Settings"
			class="flex h-7 w-7 items-center justify-center rounded-md border border-zinc-700/60 bg-zinc-800/80 text-zinc-400 transition hover:bg-zinc-700 hover:text-zinc-100"
		>
			<svg class="h-3.5 w-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
				<circle cx="12" cy="12" r="3" />
				<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
			</svg>
		</button>
		<span class="rounded border border-zinc-700/60 bg-zinc-800/80 px-2.5 py-0.5 font-mono text-xs text-zinc-400">
			v0.1.6
		</span>
	</div>
</header>
