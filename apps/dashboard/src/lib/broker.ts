import type { ZodTypeAny } from "zod";

/** Raw frame structure coming from BrokerRelay via WebSocket. */
export interface BrokerFrame {
	topic: string;
	data: unknown;
}

// ---------------------------------------------------------------------------
// Parsed, type-safe event objects consumed by the Zustand store / UI.
// ---------------------------------------------------------------------------

export interface StatusEvent {
	type: "status";
	experiment_id: string;
	status: string;
	message?: string;
	timestamp?: number;
}

export interface ProgressEvent {
	type: "progress";
	experiment_id: string;
	current_step: number;
	total_steps: number;
	progress: number; // 0..1
	epoch?: number;
	timestamp?: number;
}

export interface MetricsEvent {
	type: "metrics";
	experiment_id: string;
	metrics: Record<string, number>;
	timestamp?: number;
}

export type BrokerEvent = StatusEvent | ProgressEvent | MetricsEvent;

const TOPIC_MAP: Record<
	string,
	StatusEvent["type"] | ProgressEvent["type"] | MetricsEvent["type"]
> = {
	"training.status": "status",
	"training.progress": "progress",
	"training.metrics": "metrics",
};

/**
 * Convert a low-level broker frame into a typed `BrokerEvent`.
 *
 * Returns `null` for unknown or invalid frames so callers can safely ignore.
 */
export function parseBrokerFrame(raw: BrokerFrame): BrokerEvent | null {
	const logicalType = TOPIC_MAP[raw.topic];
	if (!logicalType || typeof raw.data !== "object" || raw.data === null) {
		if (import.meta.env.DEV) {
			/* eslint-disable no-console */
			console.warn("[broker] Unhandled frame", raw);
		}
		return null;
	}

	// Narrow data – `as` cast after runtime guards for performance / simplicity.
	const d: any = raw.data;

	switch (logicalType) {
		case "status": {
			if (
				typeof d.experiment_id !== "string" ||
				typeof d.status !== "string"
			) {
				return null;
			}
			return {
				type: "status",
				experiment_id: d.experiment_id,
				status: d.status,
				message: typeof d.message === "string" ? d.message : undefined,
				timestamp:
					typeof d.timestamp === "number" ? d.timestamp : undefined,
			};
		}
		case "progress": {
			if (
				typeof d.experiment_id !== "string" ||
				typeof d.current_step !== "number" ||
				typeof d.total_steps !== "number"
			) {
				return null;
			}
			return {
				type: "progress",
				experiment_id: d.experiment_id,
				current_step: d.current_step,
				total_steps: d.total_steps,
				progress:
					typeof d.progress === "number"
						? d.progress
						: d.current_step / (d.total_steps || 1),
				epoch: typeof d.epoch === "number" ? d.epoch : undefined,
				timestamp:
					typeof d.timestamp === "number" ? d.timestamp : undefined,
			};
		}
		case "metrics": {
			if (
				typeof d.experiment_id !== "string" ||
				typeof d.metrics !== "object" ||
				d.metrics === null
			) {
				return null;
			}
			return {
				type: "metrics",
				experiment_id: d.experiment_id,
				metrics: d.metrics as Record<string, number>,
				timestamp:
					typeof d.timestamp === "number" ? d.timestamp : undefined,
			};
		}
		default:
			return null;
	}
}
