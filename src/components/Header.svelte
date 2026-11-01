<script lang="ts">
interface HeaderProps {
	isLogPanelOpen: boolean;
	isTraining: boolean;
	logCount: number;
	onToggleLogPanel: () => void;
}

const { isLogPanelOpen, isTraining, logCount, onToggleLogPanel }: HeaderProps =
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
		<span class="rounded border border-zinc-700/60 bg-zinc-800/80 px-2.5 py-0.5 font-mono text-xs text-zinc-400">
			v0.1.5
		</span>
	</div>
</header>

