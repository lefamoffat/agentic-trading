import { signal, computed } from "@preact/signals";
import { openWs } from "./api";
import { parseBrokerFrame, BrokerEvent } from "./broker";

// Types
export type Experiment = Record<string, unknown>;
export type EquityPoint = { ts: number; value: number };
export type ConnectionState = "connecting" | "open" | "closed";

// Core signals
export const experimentsById = signal<Record<string, Partial<Experiment>>>({});
export const logsById = signal<Record<string, string[]>>({});
export const equityById = signal<Record<string, EquityPoint[]>>({});
export const connection = signal<ConnectionState>("closed");

// Computed signals for convenience
export const experiments = computed(() => experimentsById.value);
export const experimentCount = computed(
	() => Object.keys(experimentsById.value).length
);
export const runningExperiments = computed(
	() =>
		Object.values(experimentsById.value).filter(
			(exp: any) => exp.status === "running"
		).length
);
export const completedExperiments = computed(
	() =>
		Object.values(experimentsById.value).filter(
			(exp: any) => exp.status === "completed"
		).length
);

// Helper functions to access specific data
export const getExperiment = (id: string) =>
	computed(() => experimentsById.value[id]);
export const getLogs = (id: string) => computed(() => logsById.value[id] ?? []);
export const getEquity = (id: string) =>
	computed(() => equityById.value[id] ?? []);

// Broker subscription management
let wsConnection: WebSocket | null = null;
let buffer: BrokerEvent[] = [];
let flushTimer: number | null = null;

const flush = () => {
	if (buffer.length === 0) return;
	const events = buffer;
	buffer = [];
	applyEvents(events);
};

const applyEvents = (events: BrokerEvent[]) => {
	if (events.length === 0) return;

	const currentExperiments = { ...experimentsById.value };
	const currentLogs = { ...logsById.value };
	const currentEquity = { ...equityById.value };

	for (const event of events) {
		switch (event.type) {
			case "status": {
				currentExperiments[event.experiment_id] = {
					...currentExperiments[event.experiment_id],
					status: event.status,
					id: event.experiment_id,
				};
				break;
			}
			case "progress": {
				currentExperiments[event.experiment_id] = {
					...currentExperiments[event.experiment_id],
					progress: event.progress,
					current_step: event.current_step,
					total_steps: event.total_steps,
					id: event.experiment_id,
				};
				break;
			}
			case "metrics": {
				// Handle equity curve updates
				if (event.metrics.equity) {
					const arr = currentEquity[event.experiment_id] ?? [];
					arr.push({
						ts: event.timestamp ?? Date.now(),
						value: event.metrics.equity,
					});
					// Trim to last 2000 points
					if (arr.length > 2000) {
						arr.splice(0, arr.length - 2000);
					}
					currentEquity[event.experiment_id] = arr;
				}
				break;
			}
		}
	}

	// Update signals atomically
	experimentsById.value = currentExperiments;
	logsById.value = currentLogs;
	equityById.value = currentEquity;
};

export const subscribeBroker = () => {
	if (
		wsConnection &&
		(connection.value === "open" || connection.value === "connecting")
	) {
		return; // Already connected
	}

	connection.value = "connecting";

	wsConnection = openWs("/ws/experiments");

	wsConnection.onopen = () => {
		connection.value = "open";
	};

	wsConnection.onclose = () => {
		connection.value = "closed";
		wsConnection = null;
	};

	wsConnection.onerror = () => {
		connection.value = "closed";
		wsConnection = null;
	};

	wsConnection.onmessage = (evt) => {
		const parsed = parseBrokerFrame(JSON.parse(evt.data));
		if (!parsed) return;

		if (document.hidden) {
			buffer.push(parsed);
			// Flush once per 2s max while hidden to keep buffer bounded
			if (buffer.length > 1000) {
				buffer.splice(0, buffer.length - 1000);
			}
			if (!flushTimer) {
				flushTimer = window.setTimeout(() => {
					flush();
					flushTimer = null;
				}, 2000);
			}
		} else {
			applyEvents([parsed]);
		}
	};
};

// Cleanup function (for testing or manual cleanup)
export const disconnectBroker = () => {
	if (wsConnection) {
		wsConnection.close();
		wsConnection = null;
	}
	connection.value = "closed";
	if (flushTimer) {
		clearTimeout(flushTimer);
		flushTimer = null;
	}
	buffer = [];
};
