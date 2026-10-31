export function formatMetricPercent(v: number | null | undefined): string {
	if (v === null || v === undefined || Number.isNaN(v)) return "—";
	return `${(v * 100).toFixed(1)}%`;
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
