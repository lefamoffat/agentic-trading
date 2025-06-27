# Agentic Trading – Dashboard

This Astro (+Preact & TailwindCSS) web UI provides real-time monitoring of training experiments and, in the future, live trading sessions driven by the Agentic-Trading back-end.

---

## Quick start

```bash
# 1) Install JS deps
cd apps/dashboard
npm install

# 2) Export the API base-url for the dashboard to talk to
# (The FastAPI server must be running on this port)
export PUBLIC_API_URL="http://127.0.0.1:8000"

# 3) Start the dev server (auto-reload + API proxy)
npm run dev
# open http://localhost:4321 in your browser
```

> The `dev` script automatically proxies `/experiments*` and all `/ws/*` WebSocket
> endpoints to the FastAPI service running at `localhost:8000`, so CORS is not an
> issue during development.

## Production build & serving

```bash
cd apps/dashboard
npm run build  # static assets generated into apps/dashboard/dist
```

The API application mounts these assets automatically:

```python
# apps/api/__init__.py (excerpt)
from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory="apps/dashboard/dist", html=True), name="dashboard")
```

After building, the dashboard is available at the root path of the API service.

## Technology choices

-   **Astro 5** – server-first island architecture
-   **Preact 10** – tiny (~4 KB) runtime for interactive islands
-   **TailwindCSS** – utility-first styling; dark theme by default
-   **TradingView Lightweight-Charts** – performant `<canvas>` chart used for equity curves & price
-   **WebSocket** streaming\*\* – live KPI updates via the existing API (`/ws/…`)

All components are built from scratch (no shadcn/ui) but follow the same modern,
a11y-focused design language.

---

### Scripts

| command           | purpose                                     |
| ----------------- | ------------------------------------------- |
| `npm run dev`     | dev server + HMR at <http://localhost:4321> |
| `npm run build`   | generate production assets in `dist/`       |
| `npm run preview` | preview built site (should mirror prod)     |

### Environment variables

| Var              | Example                 | Description                     |
| ---------------- | ----------------------- | ------------------------------- |
| `PUBLIC_API_URL` | `http://127.0.0.1:8000` | Base URL of the FastAPI backend |

Only `PUBLIC_API_URL` is required; Astro exposes any `PUBLIC_*` vars to the
browser bundle for runtime use.
