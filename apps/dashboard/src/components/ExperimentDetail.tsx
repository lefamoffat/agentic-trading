/* @jsxImportSource preact */
import { useEffect, useState } from "preact/hooks";
import { get, openWs } from "../lib/api";
import LiveChart from "./LiveChart";
import { useAutoScroll } from "../lib/hooks";

type Props = { id: string };

type ExpData = {
	id: string;
	status: string;
	metrics: Record<string, number>;
	equity_series: [number, number][];
};

type LogMsg = { ts: number; level: string; message: string };

export default function ExperimentDetail({ id }: Props) {
	const [data, setData] = useState<ExpData | null>(null);
	const [equity, setEquity] = useState<{ time: number; value: number }[]>([]);
	const [logs, setLogs] = useState<LogMsg[]>([]);
	const logRef = useAutoScroll(logs);

	// fetch initial
	useEffect(() => {
		get<ExpData>(`/experiments/${id}`).then((d) => {
			setData(d);
			setEquity(d.equity_series.map(([t, v]) => ({ time: t, value: v })));
		});
	}, [id]);

	// live
	useEffect(() => {
		const ws = openWs(`/ws/experiments/${id}`);
		ws.onmessage = (evt) => {
			try {
				const msg = JSON.parse(evt.data);
				if (msg.type === "equity") {
					setEquity((arr) => [
						...arr,
						{ time: msg.data[0], value: msg.data[1] },
					]);
				} else if (msg.type === "log") {
					setLogs((ls) => [...ls, msg.data]);
				} else if (msg.type === "meta") {
					setData((old) => ({ ...old, ...msg.data } as ExpData));
				}
			} catch {}
		};
		return () => ws.close();
	}, [id]);

	if (!data) return <p>Loading…</p>;

	return (
		<div class="space-y-6">
			<section class="bg-surface p-4 rounded">
				<h2 class="font-semibold text-lg mb-2">Info</h2>
				<p>
					Status: <span class="capitalize">{data.status}</span>
				</p>
			</section>

			<section class="bg-surface p-4 rounded">
				<h2 class="font-semibold text-lg mb-2">Equity Curve</h2>
				<LiveChart data={equity as any} />
			</section>

			<section class="bg-surface p-4 rounded">
				<h2 class="font-semibold text-lg mb-2">Logs</h2>
				<div
					ref={logRef}
					class="h-60 overflow-y-auto text-xs font-mono space-y-1"
				>
					{logs.map((l) => (
						<div
							key={l.ts}
							class={l.level === "error" ? "text-red-400" : ""}
						>
							[{new Date(l.ts).toLocaleTimeString()}]{" "}
							{l.level.toUpperCase()}: {l.message}
						</div>
					))}
				</div>
			</section>
		</div>
	);
}
