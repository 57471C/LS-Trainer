export function formatMetricPercent(v: number | null | undefined): string {
	if (v === null || v === undefined || Number.isNaN(v)) return "—";
	return `${(v * 100).toFixed(1)}%`;
}

/**
 * Formats a duration in seconds into "{hours}h {minutes}m" (e.g. "0h 23m", "14h 18m").
 */
export function formatDurationHm(totalSeconds: number): string {
	const sec = Math.max(0, Math.round(totalSeconds));
	const totalMins = Math.floor(sec / 60);
	const hours = Math.floor(totalMins / 60);
	const mins = totalMins % 60;
	return `${hours}h ${mins.toString().padStart(2, "0")}m`;
}

/**
 * Formats seconds per epoch into a compact average (e.g. "~11m/epoch", "~45s/epoch", "~1h 10m/epoch").
 */
export function formatEpochRate(secPerEpoch: number): string {
	if (secPerEpoch <= 0 || Number.isNaN(secPerEpoch)) return "";
	if (secPerEpoch < 60) {
		return `~${Math.max(1, Math.round(secPerEpoch))}s/epoch`;
	}
	const totalMins = Math.round(secPerEpoch / 60);
	const hours = Math.floor(totalMins / 60);
	const mins = totalMins % 60;
	if (hours > 0) {
		return `~${hours}h ${mins.toString().padStart(2, "0")}m/epoch`;
	}
	return `~${mins}m/epoch`;
}

/**
 * Formats estimated remaining seconds into human-friendly time (e.g. "~14h 18m left", "~23m left", "~30s left").
 */
export function formatTimeRemaining(secLeft: number): string {
	if (secLeft <= 0 || Number.isNaN(secLeft)) return "";
	if (secLeft < 45) {
		return `~${Math.max(1, Math.round(secLeft))}s left`;
	}
	const totalMins = Math.round(secLeft / 60);
	const hours = Math.floor(totalMins / 60);
	const mins = totalMins % 60;
	if (hours > 0) {
		return `~${hours}h ${mins.toString().padStart(2, "0")}m left`;
	}
	return `~${mins}m left`;
}

export function shortenGpuName(name: string): string {
	return name
		.replace(/^NVIDIA GeForce\s+/i, "")
		.replace(/^NVIDIA\s+/i, "")
		.replace(/\(R\)/gi, "")
		.replace(/\(TM\)/gi, "")
		.trim();
}

export function normalizePath(path: string | null | undefined): string {
	if (!path) return "";
	const trimmed = path.trim();
	const isWindows =
		(typeof navigator !== "undefined" &&
			(/windows|win32|win64/i.test(navigator.userAgent) ||
				navigator.platform?.startsWith("Win"))) ||
		/^[a-zA-Z]:[\\/]/.test(trimmed);

	if (isWindows) {
		let win = trimmed.replace(/\//g, "\\");
		const isUnc = win.startsWith("\\\\");
		if (isUnc) {
			win = `\\\\${win.slice(2).replace(/\\+/g, "\\")}`;
		} else {
			win = win.replace(/\\+/g, "\\");
		}
		return win;
	}

	return trimmed.replace(/\\/g, "/").replace(/\/+/g, "/");
}
