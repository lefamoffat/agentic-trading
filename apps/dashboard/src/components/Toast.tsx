/* @jsxImportSource preact */
import { useEffect, useState } from "preact/hooks";

type ToastMsg = { id: number; text: string };

let counter = 0;

export function showToast(text: string) {
	window.dispatchEvent(
		new CustomEvent<ToastMsg>("toast", { detail: { id: ++counter, text } })
	);
}

export default function ToastHost() {
	const [messages, setMessages] = useState<ToastMsg[]>([]);

	useEffect(() => {
		const handler = (e: Event) => {
			const detail = (e as CustomEvent<ToastMsg>).detail;
			setMessages((msgs) => [...msgs, detail]);
			setTimeout(() => {
				setMessages((msgs) => msgs.filter((m) => m.id !== detail.id));
			}, 4000);
		};
		window.addEventListener("toast", handler);
		return () => window.removeEventListener("toast", handler);
	}, []);

	return (
		<div class="fixed bottom-4 right-4 flex flex-col gap-2 z-50">
			{messages.map((m) => (
				<div
					key={m.id}
					class="bg-surface text-slate-100 px-4 py-2 rounded shadow-lg border border-primary/30"
				>
					{m.text}
				</div>
			))}
		</div>
	);
}
