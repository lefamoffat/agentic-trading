# Interactive Simulation Mode

The **Interactive Simulation Mode** lets you visually back-test any registered model against historical data directly from your browser.

## Key Features

-   **Model Picker** – Browse the MLflow Model Registry and select any version.
-   **Custom Date Range** – Choose the exact period you'd like to simulate.
-   **Rich Visuals** – Plotly charts for price with trade markers and equity curve.
-   **Performance Metrics** – Total profit, Sharpe ratio, drawdown and trade log summarised.

## Launching the App

1. Ensure you have generated features (`scripts/features/build_features.py`) and trained at least one model.
2. Start the MLflow tracking server (if not already running):

    ```bash
    uv run scripts/setup/launch_mlflow.sh
    ```

3. Run the Streamlit application:

    ```bash
    uv run streamlit run scripts/analysis/run_simulation.py
    ```

4. Navigate to <http://localhost:8501> in your browser.

## Under the Hood

The UI is backed by:

| Layer                 | Location                             | Responsibility                          |
| --------------------- | ------------------------------------ | --------------------------------------- |
| **Streamlit**         | `scripts/analysis/run_simulation.py` | UI & user interaction                   |
| **Backend Helper**    | `scripts/analysis/backend.py`        | Loads model, orchestrates simulation    |
| **Simulation Engine** | `src/simulation/backtester.py`       | Step-by-step back-test using Qlib & SB3 |
| **Visuals**           | `src/simulation/visualisation.py`    | Generates Plotly figures                |

All components are unit-tested and follow the project's coding standards.

## Extending

-   **Multiple Assets** – Parameterise `symbol` & `timeframe` in the UI.
-   **Additional Metrics** – Compute win-rate, expectancy, etc., in the back-tester.
-   **Export** – Add buttons to download trade logs and equity curves.

Feel free to iterate and improve—just remember to add tests! 🎨🚀
