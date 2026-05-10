"""Render simple candlestick charts from the local canonical market store.

The script is intended for quick visual data-quality checks against a broker or
market-data terminal. It reads canonical minute-bar parquet files directly and
writes one interactive HTML chart per symbol.

Example:
    conda run -n quant python examples/render_kline_quality_check.py
    conda run -n quant python examples/render_kline_quality_check.py \
        --symbols 000001.SZ 600519.SH --start 2025-01-01 --end 2025-03-31
    conda run -n quant python examples/render_kline_quality_check.py \
        --symbols 600519.SH --chart-frequency 1m --start 2025-12-01 --end 2025-12-05
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
import re
import sys
from typing import Iterable

import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = WORKSPACE_ROOT / "quant_dataset"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "research_store" / "kline_quality_check"
DEFAULT_SYMBOLS = ("000001.SZ", "000002.SZ", "600000.SH", "600519.SH")
DEFAULT_START = "2025-10-01"
DEFAULT_END = "2025-12-31"

FREQUENCY_LABELS = {
    "1m": "1分钟",
    "5m": "5分钟",
    "15m": "15分钟",
    "30m": "30分钟",
    "60m": "60分钟",
}

MARKET_ASSET_FILE_PREFIXES = {
    ("CN", "equity"): "market_cn_equity_full",
    ("CN", "index"): "market_cn_index_full",
    ("CN", "fund"): "market_cn_fund_full",
    ("HK", "equity"): "market_hk_full",
}


@dataclass(frozen=True, slots=True)
class Instrument:
    canonical_code: str
    display_name: str
    instrument_id: str


def main() -> int:
    args = parse_args()
    try:
        import duckdb  # noqa: F401
    except ImportError:
        print(
            "Missing dependency: duckdb. Install it in the quant environment with "
            "`conda run -n quant python -m pip install duckdb`.",
            file=sys.stderr,
        )
        return 2
    try:
        import plotly.graph_objects as go  # noqa: F401
        from plotly.subplots import make_subplots  # noqa: F401
    except ImportError:
        print(
            "Missing dependency: plotly. Install it in the quant environment with "
            "`conda run -n quant python -m pip install plotly`.",
            file=sys.stderr,
        )
        return 2

    start = parse_date(args.start, "--start")
    end = parse_date(args.end, "--end")
    if end < start:
        raise SystemExit("--end must be on or after --start")

    canonical_root = args.quant_dataset_root.resolve() / "canonical_store"
    ensure_store_exists(canonical_root)

    effective_source_frequency = (
        args.source_frequency
        if args.chart_frequency == "1d"
        else args.chart_frequency
    )
    data_files = find_market_bar_files(
        canonical_root=canonical_root,
        market=args.market,
        asset_type=args.asset_type,
        frequency=effective_source_frequency,
        start=start,
        end=end,
    )
    if not data_files:
        raise SystemExit(
            "No minute-bar parquet files found for "
            f"market={args.market}, asset_type={args.asset_type}, "
            f"frequency={effective_source_frequency}, years={start.year}-{end.year}"
        )

    requested_symbols = normalize_requested_symbols(
        args.symbols, market=args.market, asset_type=args.asset_type
    )
    instruments = resolve_instruments(
        canonical_root=canonical_root,
        symbols=requested_symbols,
        market=args.market,
        asset_type=args.asset_type,
    )
    if not instruments:
        raise SystemExit(
            "None of the requested symbols were resolved. Use canonical codes such "
            "as 000001.SZ or 600519.SH."
        )

    bars = load_bars(
        files=data_files,
        instruments=instruments,
        market=args.market,
        asset_type=args.asset_type,
        frequency=effective_source_frequency,
        start=start,
        end=end,
    )
    if bars.empty:
        raise SystemExit(
            "No bars matched the requested symbols/date range. Try a wider date "
            "range or verify the symbols are present in the canonical store."
        )

    if args.chart_frequency == "1d":
        plot_frame = aggregate_daily(bars)
        title_frequency = f"1d from {effective_source_frequency}"
    else:
        plot_frame = bars.copy()
        title_frequency = args.chart_frequency

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = write_symbol_charts(
        frame=plot_frame,
        instruments=instruments,
        output_dir=args.output_dir,
        title_frequency=title_frequency,
        adjustment=args.adjustment,
        max_bars_per_chart=args.max_bars_per_chart,
    )
    write_index(
        output_dir=args.output_dir,
        chart_paths=written,
        args=args,
        instruments=instruments,
        source_files=data_files,
        effective_source_frequency=effective_source_frequency,
    )

    print(f"wrote {len(written)} chart(s) to {args.output_dir.resolve()}")
    for path in written:
        print(path.resolve())
    print(f"index: {(args.output_dir / 'index.html').resolve()}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render candlestick charts for quick local market-data checks."
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=list(DEFAULT_SYMBOLS),
        help="Canonical symbols to plot. Bare CN equity codes are expanded by exchange.",
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end", default=DEFAULT_END, help="End date, YYYY-MM-DD.")
    parser.add_argument(
        "--chart-frequency",
        choices=("1d", "1m", "5m", "15m", "30m", "60m"),
        default="1d",
        help="Rendered K-line frequency. 1d is aggregated from source minute bars.",
    )
    parser.add_argument(
        "--source-frequency",
        choices=tuple(FREQUENCY_LABELS),
        default="1m",
        help="Minute frequency used when --chart-frequency=1d.",
    )
    parser.add_argument("--market", default="CN", help="Market filter, e.g. CN or HK.")
    parser.add_argument("--asset-type", default="equity", help="Asset type filter.")
    parser.add_argument(
        "--adjustment",
        default="raw",
        choices=("raw",),
        help="Price adjustment label for the chart title. Only raw bars are read.",
    )
    parser.add_argument(
        "--quant-dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Path to the sibling quant_dataset repository.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated HTML charts.",
    )
    parser.add_argument(
        "--max-bars-per-chart",
        type=int,
        default=5000,
        help="Skip charts with more bars than this limit unless you raise it.",
    )
    return parser.parse_args()


def parse_date(value: str, option_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"{option_name} must be YYYY-MM-DD, got {value!r}") from exc


def ensure_store_exists(canonical_root: Path) -> None:
    required = [
        canonical_root / "v1/reference/records=instrument_master",
        canonical_root / "v1/reference/records=instrument_alias",
        canonical_root / "v1/market/records=minute_bar",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise SystemExit(f"Canonical store is missing required paths:\n{joined}")


def normalize_requested_symbols(
    symbols: Iterable[str], *, market: str, asset_type: str
) -> list[str]:
    normalized: list[str] = []
    for symbol in symbols:
        value = symbol.strip().upper()
        if not value:
            continue
        normalized.append(value)
        if market.upper() == "CN" and asset_type.lower() == "equity":
            inferred = infer_cn_equity_canonical_code(value)
            if inferred is not None:
                normalized.append(inferred)
    return list(dict.fromkeys(normalized))


def infer_cn_equity_canonical_code(symbol: str) -> str | None:
    if "." in symbol or not re.fullmatch(r"\d{6}", symbol):
        return None
    if symbol.startswith("6"):
        return f"{symbol}.SH"
    return f"{symbol}.SZ"


def resolve_instruments(
    *,
    canonical_root: Path,
    symbols: list[str],
    market: str,
    asset_type: str,
) -> list[Instrument]:
    master_path = sql_string(
        str((canonical_root / "v1/reference/records=instrument_master/*.parquet").resolve())
    )
    alias_path = sql_string(
        str((canonical_root / "v1/reference/records=instrument_alias/*.parquet").resolve())
    )
    placeholders = ", ".join("?" for _ in symbols)
    query = f"""
        WITH master AS (
            SELECT DISTINCT instrument_id, canonical_code, display_name, market, asset_type
            FROM read_parquet({master_path}, union_by_name=true)
            WHERE upper(market) = ? AND lower(asset_type) = ?
        ),
        alias AS (
            SELECT DISTINCT instrument_id, upper(alias_code) AS alias_code
            FROM read_parquet({alias_path}, union_by_name=true)
            WHERE upper(market) = ? AND lower(asset_type) = ?
        )
        SELECT DISTINCT
            master.instrument_id,
            master.canonical_code,
            master.display_name
        FROM master
        LEFT JOIN alias USING (instrument_id)
        WHERE upper(master.canonical_code) IN ({placeholders})
           OR alias.alias_code IN ({placeholders})
        ORDER BY master.canonical_code
    """
    params = [
        market.upper(),
        asset_type.lower(),
        market.upper(),
        asset_type.lower(),
        *symbols,
        *symbols,
    ]
    import duckdb

    with duckdb.connect() as connection:
        rows = connection.execute(query, params).fetchall()
    return [
        Instrument(
            instrument_id=str(row[0]),
            canonical_code=str(row[1]),
            display_name=str(row[2]),
        )
        for row in rows
    ]


def find_market_bar_files(
    *,
    canonical_root: Path,
    market: str,
    asset_type: str,
    frequency: str,
    start: date,
    end: date,
) -> list[Path]:
    label = FREQUENCY_LABELS[frequency]
    prefix = MARKET_ASSET_FILE_PREFIXES.get((market.upper(), asset_type.lower()))
    if prefix is None:
        prefix = "*"
    market_root = canonical_root / "v1/market/records=minute_bar"
    files: list[Path] = []
    for year in range(start.year, end.year + 1):
        pattern = f"{prefix}*__{label}_*__{year}*.parquet"
        files.extend(sorted(market_root.glob(pattern)))
    return list(dict.fromkeys(files))


def load_bars(
    *,
    files: list[Path],
    instruments: list[Instrument],
    market: str,
    asset_type: str,
    frequency: str,
    start: date,
    end: date,
) -> pd.DataFrame:
    file_sql = sql_path_list(files)
    symbol_values = [item.canonical_code for item in instruments]
    placeholders = ", ".join("?" for _ in symbol_values)
    start_text = datetime.combine(start, time.min).isoformat()
    end_text = datetime.combine(end, time.max).isoformat()
    query = f"""
        SELECT
            canonical_code,
            raw_name,
            bar_end_time,
            open_price,
            high_price,
            low_price,
            close_price,
            volume,
            turnover,
            raw_file_path,
            source_row_number
        FROM read_parquet({file_sql}, union_by_name=true)
        WHERE upper(market) = ?
          AND lower(asset_type) = ?
          AND lower(frequency) = ?
          AND canonical_code IN ({placeholders})
          AND bar_end_time >= ?
          AND bar_end_time <= ?
        ORDER BY canonical_code, bar_end_time
    """
    params = [
        market.upper(),
        asset_type.lower(),
        frequency,
        *symbol_values,
        start_text,
        end_text,
    ]
    import duckdb

    with duckdb.connect() as connection:
        frame = connection.execute(query, params).fetchdf()
    if frame.empty:
        return frame
    frame["bar_end_time"] = pd.to_datetime(frame["bar_end_time"])
    for column in ("open_price", "high_price", "low_price", "close_price", "volume", "turnover"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def aggregate_daily(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values(["canonical_code", "bar_end_time"]).copy()
    ordered["trade_date"] = ordered["bar_end_time"].dt.date
    grouped = ordered.groupby(["canonical_code", "raw_name", "trade_date"], sort=True)
    daily = grouped.agg(
        bar_end_time=("bar_end_time", "max"),
        open_price=("open_price", "first"),
        high_price=("high_price", "max"),
        low_price=("low_price", "min"),
        close_price=("close_price", "last"),
        volume=("volume", "sum"),
        turnover=("turnover", "sum"),
        source_rows=("source_row_number", "count"),
    )
    return daily.reset_index()


def write_symbol_charts(
    *,
    frame: pd.DataFrame,
    instruments: list[Instrument],
    output_dir: Path,
    title_frequency: str,
    adjustment: str,
    max_bars_per_chart: int,
) -> list[Path]:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    written: list[Path] = []
    display_names = {item.canonical_code: item.display_name for item in instruments}
    for symbol in [item.canonical_code for item in instruments]:
        symbol_frame = frame.loc[frame["canonical_code"] == symbol].copy()
        if symbol_frame.empty:
            print(f"skip {symbol}: no bars in selected range", file=sys.stderr)
            continue
        if len(symbol_frame) > max_bars_per_chart:
            print(
                f"skip {symbol}: {len(symbol_frame)} bars exceeds --max-bars-per-chart="
                f"{max_bars_per_chart}",
                file=sys.stderr,
            )
            continue

        symbol_frame = symbol_frame.sort_values("bar_end_time")
        x_values = format_x_values(symbol_frame["bar_end_time"], title_frequency)
        up = symbol_frame["close_price"] >= symbol_frame["open_price"]
        volume_colors = up.map({True: "#d62728", False: "#2ca02c"}).tolist()
        display_name = display_names.get(symbol, "")
        first_time = symbol_frame["bar_end_time"].iloc[0]
        last_time = symbol_frame["bar_end_time"].iloc[-1]
        title = (
            f"{symbol} {display_name} | {title_frequency} | {adjustment} | "
            f"{first_time:%Y-%m-%d} to {last_time:%Y-%m-%d}"
        )

        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.72, 0.28],
            vertical_spacing=0.03,
        )
        figure.add_trace(
            go.Candlestick(
                x=x_values,
                open=symbol_frame["open_price"],
                high=symbol_frame["high_price"],
                low=symbol_frame["low_price"],
                close=symbol_frame["close_price"],
                increasing_line_color="#d62728",
                decreasing_line_color="#2ca02c",
                name="price",
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Bar(
                x=x_values,
                y=symbol_frame["volume"],
                marker_color=volume_colors,
                name="volume",
            ),
            row=2,
            col=1,
        )
        figure.update_layout(
            title=title,
            template="plotly_white",
            height=760,
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
            margin={"l": 64, "r": 28, "t": 72, "b": 48},
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        )
        figure.update_xaxes(type="category", row=1, col=1)
        figure.update_xaxes(type="category", row=2, col=1)
        figure.update_yaxes(title_text="price", row=1, col=1)
        figure.update_yaxes(title_text="volume", row=2, col=1)

        output_path = output_dir / f"{safe_filename(symbol)}_{safe_filename(title_frequency)}.html"
        figure.write_html(output_path, include_plotlyjs="cdn", full_html=True)
        written.append(output_path)
    return written


def format_x_values(values: pd.Series, frequency: str) -> list[str]:
    if frequency.startswith("1d"):
        return [item.strftime("%Y-%m-%d") for item in values]
    return [item.strftime("%Y-%m-%d %H:%M") for item in values]


def write_index(
    *,
    output_dir: Path,
    chart_paths: list[Path],
    args: argparse.Namespace,
    instruments: list[Instrument],
    source_files: list[Path],
    effective_source_frequency: str,
) -> None:
    links = "\n".join(
        f'<li><a href="{path.name}">{path.stem}</a></li>' for path in chart_paths
    )
    instrument_rows = "\n".join(
        "<tr>"
        f"<td>{item.canonical_code}</td>"
        f"<td>{item.display_name}</td>"
        f"<td>{item.instrument_id}</td>"
        "</tr>"
        for item in instruments
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>K-line quality check</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; line-height: 1.5; }}
    code {{ background: #f4f4f5; padding: 2px 4px; border-radius: 4px; }}
    table {{ border-collapse: collapse; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 10px; text-align: left; }}
  </style>
</head>
<body>
  <h1>K-line quality check</h1>
  <p>
    market=<code>{args.market}</code>,
    asset_type=<code>{args.asset_type}</code>,
    chart_frequency=<code>{args.chart_frequency}</code>,
    source_frequency=<code>{effective_source_frequency}</code>,
    range=<code>{args.start}</code> to <code>{args.end}</code>
  </p>
  <h2>Charts</h2>
  <ul>{links}</ul>
  <h2>Resolved instruments</h2>
  <table>
    <thead><tr><th>canonical_code</th><th>display_name</th><th>instrument_id</th></tr></thead>
    <tbody>{instrument_rows}</tbody>
  </table>
  <h2>Source files</h2>
  <p>{len(source_files)} parquet file(s) scanned.</p>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def sql_path_list(paths: list[Path]) -> str:
    if len(paths) == 1:
        return sql_string(str(paths[0].resolve()))
    return "[" + ", ".join(sql_string(str(path.resolve())) for path in paths) + "]"


def sql_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


if __name__ == "__main__":
    raise SystemExit(main())
