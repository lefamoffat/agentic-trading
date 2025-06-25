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

    _run(
        [
            sys.executable,
            "-m",
            "src.market_data.download",
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

    _run([sys.executable, "-m", "src.market_data.features", "--symbol", symbol, "--timeframe", timeframe])

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
    # Use python -c to bypass Fire help system issues
    dump_command = f"""
import sys
sys.argv = ['dump_bin.py', 'dump_all', '--csv-path', '{csv_path}', '--qlib-dir', 'data/qlib_data', '--freq', '{freq}', '--include_fields', 'open,high,low,close,volume,factor', '--date_field_name', 'date']
from src.market_data.qlib.dump_bin import DumpDataAll
import fire
fire.Fire({{'dump_all': DumpDataAll}})
"""
    _run([sys.executable, "-c", dump_command])

    # 3) feature engineering
    build_features(symbol=symbol, timeframe=timeframe) 