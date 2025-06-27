export interface StatusInfo {
	color: string;
	bgColor: string;
	icon: string;
	emoji: string;
	label: string;
}

export type ExperimentStatus =
	| "running"
	| "completed"
	| "failed"
	| "starting"
	| "stopping"
	| "unknown"ddd;

export const STATUS_PALETTE: Record<ExperimentStatus, StatusInfo> = {
	running: {
		color: "text-green-400",
		bgColor: "bg-green-900/20",
		icon: "▶",
		emoji: "🟢",
		label: "Running",
	},
	completed: {
		color: "text-blue-400",
		bgColor: "bg-blue-900/20",
		icon: "✓",
		emoji: "✅",
		label: "Completed",
	},
	failed: {
		color: "text-red-400",
		bgColor: "bg-red-900/20",
		icon: "✗",
		emoji: "❌",
		label: "Failed",
	},
	starting: {
		color: "text-yellow-400",
		bgColor: "bg-yellow-900/20",
		icon: "⧖",
		emoji: "🟡",
		label: "Starting",
	},
	stopping: {
		color: "text-orange-400",
		bgColor: "bg-orange-900/20",
		icon: "⏸",
		emoji: "🟠",
		label: "Stopping",
	},
	unknown: {
		color: "text-slate-400",
		bgColor: "bg-slate-900/20",
		icon: "?",
		emoji: "❓",
		label: "Unknown",
	},
};

export function getStatusInfo(status: string): StatusInfo {
	const normalizedStatus = status.toLowerCase() as ExperimentStatus;
	return STATUS_PALETTE[normalizedStatus] || STATUS_PALETTE.unknown;
}

export function getStatusBadgeClasses(status: string): string {
	const info = getStatusInfo(status);
	return `inline-flex items-center px-2 py-1 rounded-full text-xs font-medium ${info.color} ${info.bgColor}`;
}
