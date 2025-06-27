/*
Minimal browser-side API helpers mirroring the CLI's behaviour. All
requests are issued against the PUBLIC_API_URL env variable configured in
`apps/dashboard/.env` (or system env at runtime).
*/

import type { ZodSchema } from "zod";

// `import.meta.env` type is provided by Astro during build; cast to any to avoid TS error in isolated utils
// eslint-disable-next-line @typescript-eslint/no-explicit-any
const API_URL =
	((import.meta as any).env.PUBLIC_API_URL as string | undefined)?.replace(
		/\/$/,
		""
	) ?? "";

if (!API_URL) {
	// eslint-disable-next-line no-console
	console.warn(
		"PUBLIC_API_URL is not defined – dashboard will not be able to talk to the backend"
	);
}

// Ensure backend broker is production-ready; fetch once at module load.
(async () => {
	try {
		const res = await fetch(`${API_URL}/health`);
		const data = await res.json();
		if (data.messaging_backend?.broker_type === "memory") {
			// Display blocking error page
			document.body.innerHTML =
				'<div style="color:#f87171;font-family:monospace;padding:2rem">' +
				"⚠️ API is running with in-memory message broker, which is NOT supported! ⚠️" +
				"</div>";
			throw new Error(
				"Backend broker = memory (unsupported for dashboard)"
			);
		}
	} catch (err) {
		// eslint-disable-next-line no-console
		console.error(err);
	}
})();

export async function get<T = unknown>(
	path: string,
	schema?: ZodSchema<T>
): Promise<T> {
	const res = await fetch(`${API_URL}${path}`, { credentials: "omit" });
	if (!res.ok) throw new Error(`GET ${path} failed: ${res.status}`);
	const json = await res.json();
	if (schema && import.meta.env.MODE !== "production") {
		return schema.parse(json);
	}
	return json as T;
}

export async function post<T = unknown>(
	path: string,
	body?: unknown,
	schema?: ZodSchema<T>
): Promise<T> {
	const res = await fetch(`${API_URL}${path}`, {
		method: "POST",
		headers: { "Content-Type": "application/json" },
		body: body ? JSON.stringify(body) : undefined,
	});
	if (!res.ok) throw new Error(`POST ${path} failed: ${res.status}`);
	const json = await res.json();
	if (schema && import.meta.env.MODE !== "production") {
		return schema.parse(json);
	}
	return json as T;
}

// WebSocket helper -----------------------------------------------------------
export function openWs(path: string): WebSocket {
	const wsBase = API_URL.replace(/^http/, "ws");
	return new WebSocket(`${wsBase}${path}`);
}
