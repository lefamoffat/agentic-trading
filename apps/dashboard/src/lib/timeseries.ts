export interface TimePoint {
	ts: number;
	value: number;
}

export interface DownsampleOptions {
	maxPoints: number;
	method: "average" | "last" | "max" | "min";
}

/**
 * Append a new point to a timeseries, maintaining chronological order
 */
export function appendPoint(
	series: TimePoint[],
	newPoint: TimePoint
): TimePoint[] {
	const result = [...series];

	// Find insertion point to maintain chronological order
	let insertIndex = result.length;
	for (let i = result.length - 1; i >= 0; i--) {
		if (result[i].ts <= newPoint.ts) {
			insertIndex = i + 1;
			break;
		}
	}

	// Insert at correct position
	result.splice(insertIndex, 0, newPoint);
	return result;
}

/**
 * Trim timeseries to maximum number of points, keeping most recent
 */
export function trimSeries(
	series: TimePoint[],
	maxPoints: number
): TimePoint[] {
	if (series.length <= maxPoints) {
		return series;
	}

	// Sort by timestamp to ensure we keep the most recent points
	const sorted = [...series].sort((a, b) => a.ts - b.ts);
	return sorted.slice(-maxPoints);
}

/**
 * Downsample timeseries to reduce number of points while preserving shape
 */
export function downsampleSeries(
	series: TimePoint[],
	options: DownsampleOptions
): TimePoint[] {
	if (series.length <= options.maxPoints) {
		return series;
	}

	const sorted = [...series].sort((a, b) => a.ts - b.ts);
	const bucketSize = Math.ceil(sorted.length / options.maxPoints);
	const result: TimePoint[] = [];

	for (let i = 0; i < sorted.length; i += bucketSize) {
		const bucket = sorted.slice(i, i + bucketSize);

		if (bucket.length === 0) continue;

		let aggregatedValue: number;
		const lastTimestamp = bucket[bucket.length - 1].ts;

		switch (options.method) {
			case "average":
				aggregatedValue =
					bucket.reduce((sum, p) => sum + p.value, 0) / bucket.length;
				break;
			case "last":
				aggregatedValue = bucket[bucket.length - 1].value;
				break;
			case "max":
				aggregatedValue = Math.max(...bucket.map((p) => p.value));
				break;
			case "min":
				aggregatedValue = Math.min(...bucket.map((p) => p.value));
				break;
		}

		result.push({
			ts: lastTimestamp,
			value: aggregatedValue,
		});
	}

	return result;
}

/**
 * Get time range (min/max timestamps) from a timeseries
 */
export function getTimeRange(
	series: TimePoint[]
): { min: number; max: number } | null {
	if (series.length === 0) {
		return null;
	}

	const timestamps = series.map((p) => p.ts);
	return {
		min: Math.min(...timestamps),
		max: Math.max(...timestamps),
	};
}

/**
 * Get value range (min/max values) from a timeseries
 */
export function getValueRange(
	series: TimePoint[]
): { min: number; max: number } | null {
	if (series.length === 0) {
		return null;
	}

	const values = series.map((p) => p.value);
	return {
		min: Math.min(...values),
		max: Math.max(...values),
	};
}

/**
 * Filter timeseries to a specific time range
 */
export function filterTimeRange(
	series: TimePoint[],
	startTime: number,
	endTime: number
): TimePoint[] {
	return series.filter(
		(point) => point.ts >= startTime && point.ts <= endTime
	);
}

/**
 * Convert equity points to lightweight-charts format
 */
export function toChartData(
	series: TimePoint[]
): Array<{ time: number; value: number }> {
	return series.map((point) => ({
		time: Math.floor(point.ts / 1000), // Convert ms to seconds for lightweight-charts
		value: point.value,
	}));
}
