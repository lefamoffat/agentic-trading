/*
Shared Preact hooks for real-time dashboard.
*/
import { useEffect, useRef } from "preact/hooks";

export function useAutoScroll(dep: any) {
	const container = useRef<HTMLDivElement>(null);
	useEffect(() => {
		const el = container.current;
		if (!el) return;
		el.scrollTop = el.scrollHeight;
	}, [dep]);
	return container;
}
