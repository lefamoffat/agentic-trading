import { defineConfig } from "astro/config";
import tailwind from "@astrojs/tailwind";
import preact from "@astrojs/preact";

export default defineConfig({
	integrations: [tailwind(), preact()],
	devServer: {
		proxy: {
			// HTTP endpoints
			"/experiments": "http://localhost:8000",
			"/experiments/summary": "http://localhost:8000",
			// WebSocket endpoints – need ws:true
			"/ws": {
				target: "ws://localhost:8000",
				ws: true,
			},
		},
	},
	outDir: "./dist",
});
