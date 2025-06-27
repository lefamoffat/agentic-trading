/* @jsxImportSource preact */
import { useEffect, useState } from "preact/hooks";
import type { LineData } from "lightweight-charts";
import LiveChart from "./LiveChart";
import { openWs } from "../lib/api";

interface SummaryMsg {
	running: number;
	completed: number;
	win_rate: number; // 0-1
	equity_series: [number, number][]; // [timestamp, value]
}

export default function SummaryDashboard() {
	const [stats, setStats] = useState<SummaryMsg | null>(null);
	const [equity, setEquity] = useState<LineData[]>([]);

	useEffect(() => {
		const ws = openWs("/ws/summary");

		ws.onmessage = (evt) => {
			try {
				const msg: SummaryMsg = JSON.parse(evt.data);
				setStats(msg);
				const series: LineData[] = msg.equity_series.map(([ts, v]) => ({
					time: ts as number,
					value: v as number,
				}));
				setEquity(series);
			} catch (err) {
				// eslint-disable-next-line no-console
				console.error("Bad WS message", err);
			}
		};
		ws.onerror = (err) => console.error("WS summary error", err);
		return () => ws.close();
	}, []);

	return (
		<section class="flex flex-col gap-6 text-slate-200">
			<div class="grid grid-cols-1 sm:grid-cols-3 gap-4">
				<StatCard label="Running" value={stats?.running ?? "–"} />
				<StatCard label="Completed" value={stats?.completed ?? "–"} />
				<StatCard
					label="Win-Rate"
					value={
						stats ? `${(stats.win_rate * 100).toFixed(1)}%` : "–"
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

function StatCard({ label, value }: { label: string; value: string | number }) {
	return (
		<div class="bg-surface p-4 rounded-lg shadow flex flex-col items-start">
			<span class="text-sm text-slate-400 uppercase tracking-wide">
				{label}
			</span>
			<span class="text-2xl font-bold text-primary mt-1">{value}</span>
		</div>
	);
}
