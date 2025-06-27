/** @type {import('tailwindcss').Config} */
export default {
	darkMode: "class",
	content: ["./src/**/*.{astro,html,js,ts,jsx,tsx}"],
	theme: {
		extend: {
			colors: {
				background: "#0f172a", // slate-900
				surface: "#1e293b", // slate-800
				primary: {
					DEFAULT: "#14b8a6", // teal-500
					light: "#2dd4bf",
					dark: "#0d9488",
				},
			},
		},
	},
	plugins: [],
};
