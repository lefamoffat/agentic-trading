/* @jsxImportSource preact */
import { useEffect, useState } from "preact/hooks";
import { get, post, openWs } from "../lib/api";
import { showToast } from "./Toast";

type ExperimentRow = {
	experiment_id?: string;
	id?: string;
	status: string;
	start_time?: number;
	started_at?: string;
};

export default function ExperimentsTable() {
	const [experiments, setExperiments] = useState<
		Record<string, ExperimentRow>
	>({});

	// initial fetch
	useEffect(() => {
		get<ExperimentRow[]>("/experiments?limit=100").then((rows) => {
			setExperiments(
				Object.fromEntries(
					rows.map((r) => [r.experiment_id ?? r.id, r])
				)
			);
		});
	}, []);

	// live updates
	useEffect(() => {
		const ws = openWs("/ws/experiments");
		ws.onmessage = (evt) => {
			try {
				const msg = JSON.parse(evt.data);
				if (msg && msg.experiment_id) {
					setExperiments((map) => ({
						...map,
						[msg.experiment_id]: {
							...(map[msg.experiment_id] ?? {}),
							...msg,
						},
					}));
				}
			} catch {}
		};
		return () => ws.close();
	}, []);

	const stopExp = async (id: string) => {
		try {
			await post(`/experiments/${id}/stop`);
			showToast(`Stop requested for ${id}`);
		} catch (err) {
			showToast(`Failed to stop ${id}`);
		}
	};

	const rows = Object.values(experiments).sort((a, b) =>
		(a.start_time ?? 0) < (b.start_time ?? 0) ? 1 : -1
	);

	if (rows.length === 0) {
		return <p class="text-slate-400">No experiments found.</p>;
	}

	return (
		<table class="min-w-full bg-surface rounded-lg overflow-hidden">
			<thead class="bg-slate-700 text-slate-300 text-left text-sm">
				<tr>
					<th class="px-4 py-2">ID</th>
					<th class="px-4 py-2">Status</th>
					<th class="px-4 py-2">Started</th>
					<th class="px-4 py-2">Action</th>
				</tr>
			</thead>
			<tbody>
				{rows.map((row) => (
					<tr key={row.id} class="odd:bg-slate-800 even:bg-slate-900">
						<td class="px-4 py-2 font-mono text-primary underline">
							<a
								href={`/experiments/${
									row.experiment_id ?? row.id
								}`}
							>
								{row.experiment_id ?? row.id}
							</a>
						</td>
						<td class="px-4 py-2 capitalize">{row.status}</td>
						<td class="px-4 py-2">
							{row.start_time
								? new Date(row.start_time).toLocaleString()
								: ""}
						</td>
						<td class="px-4 py-2">
							{row.status === "running" && (
								<button
									class="text-red-400 hover:text-red-300"
									onClick={() => stopExp(row.id ?? "")}
									aria-label={`Stop experiment ${row.id}`}
								>
									Stop
								</button>
							)}
						</td>
					</tr>
				))}
			</tbody>
		</table>
	);
}
