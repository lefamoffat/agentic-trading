/* @jsxImportSource preact */
import { useEffect } from "preact/hooks";
import { get, post } from "../lib/api";
import { showToast } from "./toast";
import { experiments, subscribeBroker, experimentsById } from "../lib/store";
import { getStatusInfo, getStatusBadgeClasses } from "../lib/status-palette";

export default function ExperimentsTable() {
	// initial fetch and broker subscription
	useEffect(() => {
		// Start broker subscription for real-time updates
		subscribeBroker();

		// Fetch initial experiments data and populate store
		get<Record<string, unknown>[]>("/experiments?limit=100")
			.then((rows) => {
				const experimentMap: Record<
					string,
					Record<string, unknown>
				> = {};
				for (const experiment of rows) {
					const experimentId = experiment.experiment_id as string;
					if (experimentId) {
						experimentMap[experimentId] = experiment;
					}
				}
				// Merge with existing store data
				experimentsById.value = {
					...experimentsById.value,
					...experimentMap,
				};
			})
			.catch((error) => {
				console.error("Failed to fetch initial experiments:", error);
				showToast("Failed to load experiments");
			});
	}, []);

	const stopExp = async (id: string) => {
		try {
			await post(`/experiments/${id}/stop`);
			showToast(`Stop requested for ${id}`);
		} catch (err) {
			showToast(`Failed to stop ${id}`);
		}
	};

	const rows = Object.entries(experiments.value)
		.map(([id, exp]) => ({ id, ...(exp as any) }))
		.sort((a: any, b: any) =>
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
					<th class="px-4 py-2">Progress</th>
					<th class="px-4 py-2">Started</th>
					<th class="px-4 py-2">Action</th>
				</tr>
			</thead>
			<tbody>
				{rows.map((row) => {
					const statusInfo = getStatusInfo(row.status || "unknown");
					const progress =
						row.current_step && row.total_steps
							? (row.current_step / row.total_steps) * 100
							: 0;

					return (
						<tr
							key={row.id}
							class="odd:bg-slate-800 even:bg-slate-900"
						>
							<td class="px-4 py-2 font-mono text-primary underline">
								<a href={`/experiments/${row.id}`}>{row.id}</a>
							</td>
							<td class="px-4 py-2">
								<span
									class={getStatusBadgeClasses(
										row.status || "unknown"
									)}
								>
									<span class="mr-1">{statusInfo.icon}</span>
									{statusInfo.label}
								</span>
							</td>
							<td class="px-4 py-2">
								{row.current_step && row.total_steps ? (
									<div class="flex items-center space-x-2">
										<div class="flex-1 bg-slate-700 rounded-full h-2">
											<div
												class="bg-primary h-2 rounded-full transition-all duration-300"
												style={{
													width: `${Math.min(
														progress,
														100
													)}%`,
												}}
											/>
										</div>
										<span class="text-xs text-slate-400 min-w-0">
											{row.current_step.toLocaleString()}/
											{row.total_steps.toLocaleString()}
										</span>
									</div>
								) : (
									<span class="text-slate-500 text-sm">
										—
									</span>
								)}
							</td>
							<td class="px-4 py-2">
								{row.start_time
									? new Date(
											row.start_time * 1000
									  ).toLocaleString()
									: ""}
							</td>
							<td class="px-4 py-2">
								{row.status === "running" && (
									<button
										class="text-red-400 hover:text-red-300 text-sm px-2 py-1 rounded border border-red-400/30 hover:bg-red-400/10 transition-colors"
										onClick={() => stopExp(row.id)}
										aria-label={`Stop experiment ${row.id}`}
									>
										Stop
									</button>
								)}
							</td>
						</tr>
					);
				})}
			</tbody>
		</table>
	);
}
