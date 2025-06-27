/* @jsxImportSource preact */
import { useEffect } from "preact/hooks";
import type { LineData } from "lightweight-charts";
import LiveChart from "./live-chart";
import {
	runningExperiments,
	completedExperiments,
	subscribeBroker,
	experiments,
} from "../lib/store";

export default function SummaryDashboard() {
	useEffect(() => {
		// Start broker subscription for real-time updates
		subscribeBroker();
	}, []);

	// Calculate summary statistics from signals
	const allExperiments = Object.values(experiments.value);
	const successfulExperiments = allExperiments.filter(
		(exp: any) =>
			exp.status === "completed" &&
			exp.metrics &&
			(exp.metrics.profit_pct > 0 || exp.metrics.sharpe_ratio > 0)
	);

	const stats = {
		running: runningExperiments.value,
		completed: completedExperiments.value,
		total: allExperiments.length,
		win_rate:
			allExperiments.length > 0
				? successfulExperiments.length / allExperiments.length
				: 0,
	};

	// Aggregate equity data across all experiments
	const equity: LineData[] = []; // TODO: Implement aggregation when equity data structure is finalized

	return (
		<section class="flex flex-col gap-6 text-slate-200">
			<div class="grid grid-cols-1 sm:grid-cols-4 gap-4">
				<StatCard
					label="Total"
					value={stats.total}
					color="text-slate-300"
				/>
				<StatCard
					label="Running"
					value={stats.running}
					color="text-green-400"
				/>
				<StatCard
					label="Completed"
					value={stats.completed}
					color="text-blue-400"
				/>
				<StatCard
					label="Success Rate"
					value={`${(stats.win_rate * 100).toFixed(1)}%`}
					color={
						stats.win_rate > 0.5 ? "text-green-400" : "text-red-400"
					}
				/>
			</div>
			<div class="bg-surface rounded-lg p-4 shadow-inner">
				<h2 class="text-lg font-semibold mb-2">Equity Curve</h2>
				<LiveChart data={equity} />
			</div>
		</section>
	);
}

function StatCard({
	label,
	value,
	color = "text-primary",
}: {
	label: string;
	value: string | number;
	color?: string;
}) {
	return (
		<div class="bg-surface p-4 rounded-lg shadow flex flex-col items-start">
			<span class="text-sm text-slate-400 uppercase tracking-wide">
				{label}
			</span>
			<span class={`text-2xl font-bold mt-1 ${color}`}>{value}</span>
		</div>
	);
}
