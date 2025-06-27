/* @jsxImportSource preact */
import { useState } from "preact/hooks";
import { post } from "../lib/api";
import { showToast } from "./Toast";

export default function LaunchExperimentForm() {
	const [loading, setLoading] = useState(false);
	const [form, setForm] = useState({
		agent_type: "ppo",
		symbol: "EUR/USD",
		timeframe: "1h",
		timesteps: 50000,
		learning_rate: 0.0003,
		initial_balance: 10000,
	});

	const update = (key: string, value: any) =>
		setForm((f) => ({ ...f, [key]: value }));

	const submit = async (e: Event) => {
		e.preventDefault();
		setLoading(true);
		try {
			const res: any = await post("/experiments", form);
			showToast(`Experiment ${res.experiment_id} starting…`);
			setTimeout(() => {
				window.location.href = "/experiments";
			}, 800);
		} catch {
			showToast("Failed to launch experiment");
		} finally {
			setLoading(false);
		}
	};

	return (
		<form class="space-y-4 max-w-md" onSubmit={submit}>
			{(["agent_type", "symbol", "timeframe"] as const).map((k) => (
				<div key={k} class="flex flex-col">
					<label class="text-sm mb-1 capitalize" for={k}>
						{k.replace("_", " ")}
					</label>
					<input
						required
						id={k}
						value={(form as any)[k]}
						onInput={(e: any) => update(k, e.target.value)}
						class="bg-slate-800 border border-slate-600 rounded px-3 py-2"
					/>
				</div>
			))}
			{(["timesteps", "learning_rate", "initial_balance"] as const).map(
				(k) => (
					<div key={k} class="flex flex-col">
						<label class="text-sm mb-1 capitalize" for={k}>
							{k.replace("_", " ")}
						</label>
						<input
							required
							type="number"
							step="any"
							id={k}
							value={(form as any)[k]}
							onInput={(e: any) =>
								update(k, parseFloat(e.target.value))
							}
							class="bg-slate-800 border border-slate-600 rounded px-3 py-2"
						/>
					</div>
				)
			)}
			<button
				type="submit"
				class="mt-4 bg-primary px-4 py-2 rounded disabled:opacity-50"
				disabled={loading}
			>
				{loading ? "Launching…" : "Launch Experiment"}
			</button>
		</form>
	);
}
