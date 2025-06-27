/* @jsxImportSource preact */
import { useEffect } from "preact/hooks";
import { get } from "../lib/api";
import LiveChart from "./live-chart";
import { useAutoScroll } from "../lib/hooks";
import {
	getExperiment,
	getEquity,
	getLogs,
	subscribeBroker,
	experimentsById,
} from "../lib/store";
import { getStatusInfo, getStatusBadgeClasses } from "../lib/status-palette";
import { toChartData, downsampleSeries } from "../lib/timeseries";

type Props = { id: string };

export default function ExperimentDetail({ id }: Props) {
	const experiment = getExperiment(id);
	const equity = getEquity(id);
	const logs = getLogs(id);
	const logRef = useAutoScroll(logs.value);

	// fetch initial data and start broker subscription
	useEffect(() => {
		// Start broker subscription for real-time updates
		subscribeBroker();

		// Fetch initial experiment data
		get<Record<string, unknown>>(`/experiments/${id}`).then((data) => {
			experimentsById.value = {
				...experimentsById.value,
				[id]: data,
			};
		});
	}, [id]);

	// Convert and downsample equity data for chart component
	const downsampledEquity = downsampleSeries(equity.value, {
		maxPoints: 1000,
		method: "last",
	});
	const chartData = toChartData(downsampledEquity);

	if (!experiment.value) return <p>Loading…</p>;

	const statusInfo = getStatusInfo(
		(experiment.value.status as string) || "unknown"
	);
	const progress =
		experiment.value.current_step && experiment.value.total_steps
			? ((experiment.value.current_step as number) /
					(experiment.value.total_steps as number)) *
			  100
			: 0;

	return (
		<div class="space-y-6">
			<section class="bg-surface p-4 rounded">
				<h2 class="font-semibold text-lg mb-4">Experiment Info</h2>
				<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
					<div>
						<label class="text-sm text-slate-400">Status</label>
						<div class="mt-1">
							<span
								class={getStatusBadgeClasses(
									(experiment.value.status as string) ||
										"unknown"
								)}
							>
								<span class="mr-1">{statusInfo.icon}</span>
								{statusInfo.label}
							</span>
						</div>
					</div>

					{experiment.value.agent_type && (
						<div>
							<label class="text-sm text-slate-400">
								Agent Type
							</label>
							<p class="mt-1 font-mono text-sm">
								{experiment.value.agent_type}
							</p>
						</div>
					)}

					{experiment.value.symbol && (
						<div>
							<label class="text-sm text-slate-400">Symbol</label>
							<p class="mt-1 font-mono text-sm">
								{experiment.value.symbol}
							</p>
						</div>
					)}

					{experiment.value.start_time && (
						<div>
							<label class="text-sm text-slate-400">
								Started
							</label>
							<p class="mt-1 text-sm">
								{new Date(
									experiment.value.start_time as number
								).toLocaleString()}
							</p>
						</div>
					)}

					{experiment.value.current_step &&
						experiment.value.total_steps && (
							<div class="md:col-span-2">
								<label class="text-sm text-slate-400">
									Progress
								</label>
								<div class="mt-2 flex items-center space-x-3">
									<div class="flex-1 bg-slate-700 rounded-full h-3">
										<div
											class="bg-primary h-3 rounded-full transition-all duration-300"
											style={{
												width: `${Math.min(
													progress,
													100
												)}%`,
											}}
										/>
									</div>
									<span class="text-sm text-slate-300 min-w-0">
										{(
											experiment.value
												.current_step as number
										).toLocaleString()}{" "}
										/{" "}
										{(
											experiment.value
												.total_steps as number
										).toLocaleString()}
									</span>
									<span class="text-sm text-slate-400">
										({progress.toFixed(1)}%)
									</span>
								</div>
							</div>
						)}
				</div>
			</section>

			<section class="bg-surface p-4 rounded">
				<h2 class="font-semibold text-lg mb-2">Equity Curve</h2>
				<LiveChart data={chartData as any} />
			</section>

			<section class="bg-surface p-4 rounded">
				<h2 class="font-semibold text-lg mb-2">Logs</h2>
				<div
					ref={logRef}
					class="h-60 overflow-y-auto text-xs font-mono space-y-1"
				>
					{logs.value.map((logEntry: any, index: number) => (
						<div
							key={index}
							class={
								logEntry.includes("ERROR") ? "text-red-400" : ""
							}
						>
							{logEntry}
						</div>
					))}
				</div>
			</section>
		</div>
	);
}
