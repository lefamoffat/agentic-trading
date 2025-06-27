/* @jsxImportSource preact */
import { useEffect, useRef } from "preact/hooks";
import {
	createChart,
	IChartApi,
	LineData,
	LineSeries,
} from "lightweight-charts";

type LiveChartProps = {
	data: Array<LineData>; // initial seed data
	dark?: boolean;
};

// Simple responsive line chart suitable for equity curves & price series.
export default function LiveChart({ data, dark = true }: LiveChartProps) {
	const containerRef = useRef<HTMLDivElement>(null);
	const chartRef = useRef<IChartApi | null>(null);
	const seriesRef = useRef<ReturnType<IChartApi["addSeries"]> | null>(null);

	useEffect(() => {
		if (!containerRef.current) return;

		// Destroy previous chart (hot-module-reload)
		chartRef.current?.remove();

		const chart = createChart(containerRef.current, {
			layout: {
				background: { color: dark ? "#0f172a" : "#ffffff" },
				textColor: dark ? "#e2e8f0" : "#334155",
			},
			width: containerRef.current.clientWidth,
			height: 300,
			timeScale: { timeVisible: true, secondsVisible: false },
			grid: {
				vertLines: { visible: false },
				horzLines: { visible: false },
			},
		});

		const series = chart.addSeries(LineSeries, {
			color: "#14b8a6",
			lineWidth: 2,
		});
		series.setData(data);

		chartRef.current = chart;
		seriesRef.current = series;

		const handleResize = () =>
			chart.applyOptions({ width: containerRef.current!.clientWidth });
		window.addEventListener("resize", handleResize);

		return () => {
			window.removeEventListener("resize", handleResize);
			chart.remove();
		};
	}, []);

	// Imperative api to push new point
	useEffect(() => {
		if (seriesRef.current && data.length) {
			seriesRef.current.setData(data);
		}
	}, [data]);

	return <div ref={containerRef} class="w-full" />;
}
