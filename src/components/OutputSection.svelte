<script lang="ts">
interface OutputSectionProps {
	runName: string;
	isTraining: boolean;
}

let { runName = $bindable(""), isTraining }: OutputSectionProps = $props();
</script>

<section class="w-full space-y-2">
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
			class="w-full min-w-0 rounded-lg border border-zinc-700/80 bg-zinc-900 px-3.5 py-2.5 text-sm text-zinc-100 placeholder-zinc-500 transition focus:border-amber-500 focus:outline-none focus:ring-1 focus:ring-amber-500 disabled:opacity-50"
		/>
	</div>
	<p class="text-xs text-zinc-500">Run identifier for checkpoint directories and evaluation metrics</p>
</section>

