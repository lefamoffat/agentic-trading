from __future__ import annotations

"""Data pipeline commands (download, feature engineering, preparation)."""

import sys
from pathlib import Path
from typing import Optional

import typer

from apps.cli import app, _run, SCRIPTS_DIR
from src.types import Timeframe  # imported late here previously, acceptable

# ---------------------------------------------------------------------------
# Download raw data
# ---------------------------------------------------------------------------


@app.command("download-data")
def download_data(
    symbol: str = typer.Option("EUR/USD", help="Trading symbol e.g. EUR/USD"),
    timeframe: str = typer.Option("1h", help="Timeframe e.g. 1h, 1d"),
    bars: int = typer.Option(365, help="Number of bars to fetch"),
) -> None:
    """Download raw historical data for a symbol & timeframe."""

    script = SCRIPTS_DIR / "data" / "download_historical.py"
    _run(
        [
            sys.executable,
            str(script),
            "--symbol",
            symbol,
            "--timeframe",
            timeframe,
            "--bars",
            str(bars),
        ]
    )


# ---------------------------------------------------------------------------
# Build engineered features
# ---------------------------------------------------------------------------


@app.command("build-features")
def build_features(
    symbol: str = typer.Option("EUR/USD", help="Trading symbol e.g. EUR/USD"),
    timeframe: str = typer.Option("1h", help="Timeframe e.g. 1h, 1d"),
) -> None:
    """Generate Qlib feature CSVs."""

    script = SCRIPTS_DIR / "features" / "build_features.py"
    _run([sys.executable, str(script), "--symbol", symbol, "--timeframe", timeframe])


# ---------------------------------------------------------------------------
# End-to-end preparation pipeline
# ---------------------------------------------------------------------------


@app.command("prepare-data")
def prepare_data(
    symbol: Optional[str] = typer.Option(None, help="Trading symbol e.g. EUR/USD"),
    timeframe: Optional[str] = typer.Option(None, help="Timeframe e.g. 1h, 1d"),
    bars: int = typer.Option(365, help="Number of bars to fetch"),
) -> None:
    """Run download ➜ Qlib dump ➜ build-features in one go."""

    # Interactive prompts when values are omitted
    if symbol is None:
        symbol = typer.prompt("Enter trading symbol", default="EUR/USD")
    if timeframe is None:
        timeframe = typer.prompt("Enter timeframe (e.g. 1h, 1d)", default="1h")

    # 1) download raw data
    download_data(symbol=symbol, timeframe=timeframe, bars=bars)

    # 2) dump to Qlib binary format
    try:
        freq = Timeframe.from_standard(timeframe).qlib_name
    except ValueError as exc:  # noqa: BLE001
        typer.secho(str(exc), fg=typer.colors.RED)
        raise typer.Exit(code=1) from exc

    csv_path = f"data/qlib_source/{timeframe}"
    dump_script = SCRIPTS_DIR / "data" / "dump_bin.py"
    _run(
        [
            sys.executable,
            str(dump_script),
            "dump_all",
            "--csv-path",
            csv_path,
            "--qlib-dir",
            "data/qlib_data",
            "--freq",
            freq,
            "--include_fields",
            "open,high,low,close,volume,factor",
            "--date_field_name",
            "date",
        ]
    )

    # 3) feature engineering
    build_features(symbol=symbol, timeframe=timeframe) 