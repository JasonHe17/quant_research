"""Static HTML reports for reviewing backtest trade decisions."""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


DECISION_TRACE_COLUMNS = (
    "timestamp",
    "instrument_id",
    "action",
    "current_weight",
    "aim_weight",
    "target_weight",
    "delta_weight",
    "rank",
    "score",
    "expected_edge_bps",
    "estimated_cost_bps",
    "priority",
    "decision_reason",
    "constraint_flags",
)


@dataclass(frozen=True, slots=True)
class DecisionReportConfig:
    """Display limits for a static decision-review report."""

    max_instruments: int = 60
    max_timestamps: int = 80
    max_decisions: int = 500
    max_kline_charts_per_timestamp: int = 50
    title: str = "Backtest Decision Review"

    def __post_init__(self) -> None:
        if self.max_instruments <= 0:
            raise ValueError("max_instruments must be positive")
        if self.max_timestamps <= 0:
            raise ValueError("max_timestamps must be positive")
        if self.max_decisions <= 0:
            raise ValueError("max_decisions must be positive")
        if self.max_kline_charts_per_timestamp <= 0:
            raise ValueError("max_kline_charts_per_timestamp must be positive")
        if not self.title:
            raise ValueError("title must be non-empty")


def render_decision_report(
    *,
    summary: dict[str, Any],
    decision_trace: pd.DataFrame | None,
    policy_diagnostics: pd.DataFrame | None,
    trades: pd.DataFrame | None,
    equity_curve: pd.DataFrame | None,
    output_path: str | Path,
    market_context: pd.DataFrame | None = None,
    kline_windows: pd.DataFrame | None = None,
    config: DecisionReportConfig | None = None,
) -> Path:
    """Render a self-contained HTML report for manual decision review."""

    report_config = config or DecisionReportConfig()
    decisions = _prepare_decisions(decision_trace)
    context = _prepare_market_context(market_context)
    decisions = _merge_market_context(decisions, context)
    diagnostics = _prepare_frame(policy_diagnostics)
    trades_frame = _prepare_frame(trades)
    equity = _prepare_frame(equity_curve)
    kline = _prepare_kline_windows(kline_windows)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    html_text = "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            "<head>",
            '<meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            f"<title>{_escape(report_config.title)}</title>",
            f"<style>{_stylesheet()}</style>",
            "</head>",
            "<body>",
            '<main class="page">',
            _header_html(summary, report_config),
            _metric_grid_html(summary, decisions, diagnostics, trades_frame),
            _overview_grid_html(equity, diagnostics, decisions),
            _heatmap_html(decisions, report_config),
            _kline_explorer_html(kline, report_config),
            _timestamp_review_html(decisions, diagnostics, report_config),
            _top_decisions_html(decisions, report_config),
            _trades_html(trades_frame, report_config),
            "</main>",
            "</body>",
            "</html>",
            "",
        ]
    )
    output.write_text(html_text, encoding="utf-8")
    return output


def _prepare_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    return frame.copy()


def _prepare_decisions(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=[*DECISION_TRACE_COLUMNS, "abs_delta_weight"])
    decisions = frame.copy()
    for column in DECISION_TRACE_COLUMNS:
        if column not in decisions.columns:
            decisions[column] = pd.NA
    for column in (
        "current_weight",
        "aim_weight",
        "target_weight",
        "delta_weight",
        "rank",
        "score",
        "expected_edge_bps",
        "estimated_cost_bps",
        "priority",
    ):
        decisions[column] = pd.to_numeric(decisions[column], errors="coerce")
    missing_delta = decisions["delta_weight"].isna()
    decisions.loc[missing_delta, "delta_weight"] = (
        decisions.loc[missing_delta, "target_weight"].fillna(0.0)
        - decisions.loc[missing_delta, "current_weight"].fillna(0.0)
    )
    missing_action = decisions["action"].isna() | (decisions["action"].astype(str) == "")
    decisions.loc[missing_action, "action"] = decisions.loc[missing_action].apply(
        _action_from_row,
        axis=1,
    )
    decisions["abs_delta_weight"] = decisions["delta_weight"].abs().fillna(0.0)
    return decisions.loc[:, [*DECISION_TRACE_COLUMNS, "abs_delta_weight"]]


def _prepare_market_context(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    context = frame.copy()
    if "timestamp" not in context.columns or "instrument_id" not in context.columns:
        return pd.DataFrame()
    context["instrument_id"] = context["instrument_id"].astype(str)
    for column in (
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
        "turnover",
        "bar_return",
        "execution_target_weight",
        "executed_shares",
        "executed_notional",
        "avg_trade_price",
        "total_cost",
        "trade_count",
    ):
        if column in context.columns:
            context[column] = pd.to_numeric(context[column], errors="coerce")
    return context.drop_duplicates(
        subset=["timestamp", "instrument_id"],
        keep="first",
    )


def _prepare_kline_windows(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    windows = frame.copy()
    if "decision_timestamp" not in windows.columns and "timestamp" in windows.columns:
        windows = windows.rename(columns={"timestamp": "decision_timestamp"})
    required = {
        "decision_timestamp",
        "instrument_id",
        "bar_time",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
    }
    if not required.issubset(windows.columns):
        return pd.DataFrame()
    windows["instrument_id"] = windows["instrument_id"].astype(str)
    for column in (
        "current_weight",
        "target_weight",
        "delta_weight",
        "rank",
        "score",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "volume",
        "turnover",
        "bar_offset",
        "marker_price",
        "executed_shares",
        "executed_notional",
        "avg_trade_price",
    ):
        if column in windows.columns:
            windows[column] = pd.to_numeric(windows[column], errors="coerce")
    if "is_execution_bar" in windows.columns:
        windows["is_execution_bar"] = windows["is_execution_bar"].fillna(False).astype(bool)
    else:
        windows["is_execution_bar"] = False
    return windows.sort_values(
        ["decision_timestamp", "instrument_id", "bar_offset", "bar_time"],
        kind="stable",
    )


def _merge_market_context(
    decisions: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    if decisions.empty or context.empty:
        return decisions
    merge_columns = [
        column
        for column in (
            "timestamp",
            "instrument_id",
            "canonical_code",
            "raw_name",
            "exec_time",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "volume",
            "turnover",
            "bar_return",
            "tradable_bar",
            "limit_up_open",
            "limit_down_open",
            "execution_target_weight",
            "executed_side",
            "executed_shares",
            "executed_notional",
            "avg_trade_price",
            "total_cost",
            "trade_count",
        )
        if column in context.columns
    ]
    return decisions.merge(
        context.loc[:, merge_columns],
        on=["timestamp", "instrument_id"],
        how="left",
    )


def _action_from_row(row: pd.Series) -> str:
    current = _finite_float(row.get("current_weight"), default=0.0)
    target = _finite_float(row.get("target_weight"), default=0.0)
    reason = str(row.get("decision_reason") or "")
    delta = target - current
    if abs(delta) <= 1e-12:
        return "no_trade" if reason in {"below_edge", "below_weight_band"} else "hold"
    if current <= 0 and target > 0:
        return "entry"
    if current > 0 and target <= 0:
        return "exit"
    if delta > 0:
        return "resize_up"
    return "resize_down"


def _header_html(summary: dict[str, Any], config: DecisionReportConfig) -> str:
    params = summary.get("params", {}) if isinstance(summary, dict) else {}
    if not isinstance(params, dict):
        params = {}
    period = f"{params.get('start', 'n/a')} to {params.get('end', 'n/a')}"
    policy = params.get("trade_policy", "n/a")
    return f"""
<section class="header">
  <div>
    <h1>{_escape(config.title)}</h1>
    <p>{_escape(period)} - policy {_escape(policy)}</p>
  </div>
  <div class="header-meta">
    <span>top_n {_escape(params.get("top_n", "n/a"))}</span>
    <span>{_escape(params.get("data_access_mode", "n/a"))}</span>
  </div>
</section>
""".strip()


def _metric_grid_html(
    summary: dict[str, Any],
    decisions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    trades: pd.DataFrame,
) -> str:
    metrics = summary.get("metrics", {}) if isinstance(summary, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    policy = summary.get("policy_diagnostics", {}) if isinstance(summary, dict) else {}
    if not isinstance(policy, dict):
        policy = {}
    decision_timestamps = (
        int(decisions["timestamp"].nunique())
        if not decisions.empty and "timestamp" in decisions.columns
        else int(policy.get("decision_timestamp_count") or len(diagnostics))
    )
    cards = [
        ("Total return", _fmt_percent(metrics.get("total_return"))),
        ("Max drawdown", _fmt_percent(metrics.get("max_drawdown"))),
        ("Final equity", _fmt_number(metrics.get("final_equity"))),
        ("Trades", _fmt_count(metrics.get("trade_count", len(trades)))),
        ("Decision times", _fmt_count(decision_timestamps)),
        ("Planned turnover", _fmt_percent(policy.get("planned_gross_turnover"))),
    ]
    return '<section class="metric-grid">' + "".join(
        f'<article class="metric"><span>{_escape(label)}</span><strong>{_escape(value)}</strong></article>'
        for label, value in cards
    ) + "</section>"


def _overview_grid_html(
    equity: pd.DataFrame,
    diagnostics: pd.DataFrame,
    decisions: pd.DataFrame,
) -> str:
    action_counts = _count_values(decisions, "action")
    reason_counts = _count_values(decisions, "decision_reason")
    return f"""
<section class="overview-grid">
  <article class="panel wide">
    <h2>Equity Curve</h2>
    {_line_svg(equity, x_column="timestamp", y_columns=("equity",), colors=("#2563eb",))}
  </article>
  <article class="panel">
    <h2>Actions</h2>
    {_bar_counts_html(action_counts, ACTION_COLORS)}
  </article>
  <article class="panel">
    <h2>Decision Reasons</h2>
    {_bar_counts_html(reason_counts, REASON_COLORS)}
  </article>
  <article class="panel wide">
    <h2>Policy Diagnostics</h2>
    {_line_svg(
        diagnostics,
        x_column="timestamp",
        y_columns=("planned_gross_turnover", "target_gross_exposure"),
        colors=("#0891b2", "#d97706"),
    )}
  </article>
</section>
""".strip()


ACTION_COLORS = {
    "entry": "#15803d",
    "exit": "#be123c",
    "resize_up": "#2563eb",
    "resize_down": "#d97706",
    "hold": "#64748b",
    "no_trade": "#94a3b8",
}
REASON_COLORS = {
    "entry_rank": "#15803d",
    "exit_rank": "#be123c",
    "hold_buffer": "#64748b",
    "below_edge": "#d97706",
    "below_weight_band": "#ca8a04",
    "risk_reduction": "#7c3aed",
    "turnover_budget_limited": "#b45309",
}


def _line_svg(
    frame: pd.DataFrame,
    *,
    x_column: str,
    y_columns: tuple[str, ...],
    colors: tuple[str, ...],
    width: int = 760,
    height: int = 220,
) -> str:
    if frame.empty or x_column not in frame.columns:
        return _empty_state("No series available.")
    columns = [column for column in y_columns if column in frame.columns]
    if not columns:
        return _empty_state("No series available.")
    plot = frame.loc[:, [x_column, *columns]].copy()
    for column in columns:
        plot[column] = pd.to_numeric(plot[column], errors="coerce")
    plot = plot.dropna(subset=columns, how="all").reset_index(drop=True)
    if plot.empty:
        return _empty_state("No numeric values available.")

    values = pd.concat([plot[column] for column in columns], ignore_index=True).dropna()
    y_min = float(values.min())
    y_max = float(values.max())
    if math.isclose(y_min, y_max):
        padding = abs(y_min) * 0.05 or 1.0
        y_min -= padding
        y_max += padding
    left = 48
    right = 16
    top = 18
    bottom = 34
    inner_width = width - left - right
    inner_height = height - top - bottom
    denominator = max(len(plot) - 1, 1)

    lines = []
    for index, column in enumerate(columns):
        points = []
        for row_index, value in enumerate(plot[column]):
            if pd.isna(value):
                continue
            x = left + inner_width * row_index / denominator
            y = top + inner_height * (1 - (float(value) - y_min) / (y_max - y_min))
            points.append(f"{x:.1f},{y:.1f}")
        if points:
            color = colors[index % len(colors)]
            lines.append(
                f'<polyline points="{" ".join(points)}" fill="none" '
                f'stroke="{color}" stroke-width="2.5" stroke-linejoin="round" '
                f'stroke-linecap="round" />'
            )
    if not lines:
        return _empty_state("No numeric values available.")

    legend = "".join(
        f'<span><i style="background:{colors[index % len(colors)]}"></i>{_escape(column)}</span>'
        for index, column in enumerate(columns)
    )
    start_label = _short_timestamp(plot.loc[0, x_column])
    end_label = _short_timestamp(plot.loc[len(plot) - 1, x_column])
    svg = f"""
<div class="chart-wrap">
  <svg viewBox="0 0 {width} {height}" role="img" aria-label="Line chart">
    <line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" class="axis" />
    <line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" class="axis" />
    <text x="4" y="{top + 6}" class="axis-label">{_escape(_fmt_compact(y_max))}</text>
    <text x="4" y="{height - bottom}" class="axis-label">{_escape(_fmt_compact(y_min))}</text>
    <text x="{left}" y="{height - 8}" class="axis-label">{_escape(start_label)}</text>
    <text x="{width - right}" y="{height - 8}" text-anchor="end" class="axis-label">{_escape(end_label)}</text>
    {"".join(lines)}
  </svg>
  <div class="legend">{legend}</div>
</div>
""".strip()
    return svg


def _bar_counts_html(counts: dict[str, int], colors: dict[str, str]) -> str:
    if not counts:
        return _empty_state("No decisions available.")
    maximum = max(counts.values()) or 1
    rows = []
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:10]:
        width = max(4, int(count / maximum * 100))
        color = colors.get(label, "#475569")
        rows.append(
            f"""
<div class="bar-row">
  <span>{_escape(label)}</span>
  <div class="bar-track"><i style="width:{width}%;background:{color}"></i></div>
  <strong>{_escape(count)}</strong>
</div>
""".strip()
        )
    return '<div class="bars">' + "".join(rows) + "</div>"


def _heatmap_html(decisions: pd.DataFrame, config: DecisionReportConfig) -> str:
    if decisions.empty:
        return _section_empty("Decision Heatmap", "No decision trace was written.")

    active = decisions.loc[
        (decisions["abs_delta_weight"] > 1e-12)
        | (decisions["target_weight"].fillna(0.0) > 1e-12)
        | decisions["action"].isin(["entry", "exit", "resize_up", "resize_down"])
    ].copy()
    if active.empty:
        active = decisions.copy()
    scores = (
        active.groupby("instrument_id", dropna=False)
        .agg(
            decision_count=("instrument_id", "size"),
            abs_delta=("abs_delta_weight", "sum"),
            max_target=("target_weight", "max"),
        )
        .reset_index()
        .sort_values(
            ["abs_delta", "decision_count", "max_target", "instrument_id"],
            ascending=[False, False, False, True],
        )
    )
    instruments = [
        str(value)
        for value in scores["instrument_id"].head(config.max_instruments).tolist()
    ]
    timestamps = _sample_values(
        sorted(active["timestamp"].dropna().unique().tolist(), key=str),
        config.max_timestamps,
    )
    subset = active.loc[
        active["instrument_id"].astype(str).isin(instruments)
        & active["timestamp"].isin(timestamps)
    ].copy()
    if subset.empty:
        return _section_empty("Decision Heatmap", "No active decisions after filtering.")
    lookup: dict[tuple[str, object], pd.Series] = {}
    for row in subset.sort_values("abs_delta_weight", ascending=False).itertuples():
        key = (str(row.instrument_id), row.timestamp)
        lookup.setdefault(key, pd.Series(row._asdict()))
    label_by_id = _instrument_labels(subset)

    header_cells = "".join(
        f'<th title="{_escape(value)}">{_escape(_short_timestamp(value))}</th>'
        for value in timestamps
    )
    body_rows = []
    for instrument_id in instruments:
        cells = []
        for timestamp in timestamps:
            row = lookup.get((instrument_id, timestamp))
            cells.append(_heatmap_cell(row))
        label = label_by_id.get(instrument_id, instrument_id)
        body_rows.append(
            f'<tr><th class="instrument">{_escape(label)}</th>{"".join(cells)}</tr>'
        )

    return f"""
<section class="panel full">
  <div class="section-heading">
    <h2>Decision Heatmap</h2>
    <p>Rows are the most active instruments; columns are sampled rebalance timestamps.</p>
  </div>
  <div class="heatmap-scroll">
    <table class="heatmap">
      <thead><tr><th class="instrument">instrument</th>{header_cells}</tr></thead>
      <tbody>{"".join(body_rows)}</tbody>
    </table>
  </div>
</section>
""".strip()


def _heatmap_cell(row: pd.Series | None) -> str:
    if row is None:
        return '<td class="heat-cell empty"></td>'
    action = str(row.get("action") or "")
    target = _finite_float(row.get("target_weight"), default=0.0)
    delta = _finite_float(row.get("delta_weight"), default=0.0)
    text = ""
    if target > 0:
        text = _fmt_weight(target)
    elif action == "entry":
        text = "E"
    elif action == "exit":
        text = "X"
    elif action == "resize_up":
        text = "+"
    elif action == "resize_down":
        text = "-"
    elif action == "no_trade":
        text = "."
    title = " | ".join(
        [
            str(row.get("canonical_code") or row.get("instrument_id") or ""),
            str(row.get("timestamp") or ""),
            f"exec_time={row.get('exec_time') or ''}",
            f"action={action}",
            f"target={_fmt_percent(target)}",
            f"delta={_fmt_percent(delta)}",
            f"open={_fmt_number(row.get('open_price'))}",
            f"close={_fmt_number(row.get('close_price'))}",
            f"reason={row.get('decision_reason') or ''}",
            f"flags={row.get('constraint_flags') or ''}",
        ]
    )
    magnitude = min(abs(delta) * 6 + target * 2, 1.0)
    alpha = 0.22 + magnitude * 0.65
    return (
        f'<td class="heat-cell action-{_escape(action)}" '
        f'style="--alpha:{alpha:.2f}" title="{_escape(title)}">{_escape(text)}</td>'
    )


def _kline_explorer_html(
    windows: pd.DataFrame,
    config: DecisionReportConfig,
) -> str:
    if windows.empty:
        return _section_empty(
            "K-Line Decision Explorer",
            "No decision K-line windows were written.",
        )
    payload = _kline_payload(windows, config)
    if not payload["times"]:
        return _section_empty(
            "K-Line Decision Explorer",
            "No K-line windows available after report limits.",
        )
    json_payload = _json_script_payload(payload)
    return f"""
<section class="panel full kline-explorer">
  <div class="section-heading">
    <h2>K-Line Decision Explorer</h2>
    <p>Drag the time axis to inspect holdings and trade markers at one rebalance slice.</p>
  </div>
  <div class="kline-controls" aria-label="K-line time controls">
    <button type="button" id="kline-prev" aria-label="Previous timestamp">Prev</button>
    <input id="kline-slider" type="range" min="0" max="{len(payload["times"]) - 1}" value="0" step="1" aria-label="Decision timestamp">
    <button type="button" id="kline-next" aria-label="Next timestamp">Next</button>
    <output id="kline-current"></output>
  </div>
  <div class="kline-time-strip" id="kline-time-strip"></div>
  <div class="kline-slice-summary" id="kline-slice-summary"></div>
  <div class="kline-stack" id="kline-stack"></div>
  <script type="application/json" id="decision-kline-payload">{json_payload}</script>
  <script>{_kline_script()}</script>
</section>
""".strip()


def _kline_payload(
    windows: pd.DataFrame,
    config: DecisionReportConfig,
) -> dict[str, object]:
    timestamps = _sample_values(
        sorted(windows["decision_timestamp"].dropna().unique().tolist(), key=_timestamp_key),
        config.max_timestamps,
    )
    slices: dict[str, list[dict[str, object]]] = {}
    times: list[dict[str, object]] = []
    for timestamp in timestamps:
        frame = windows.loc[windows["decision_timestamp"] == timestamp].copy()
        if frame.empty:
            continue
        decision_rows = _kline_decision_rows(frame)
        if decision_rows.empty:
            continue
        decision_rows = decision_rows.head(config.max_kline_charts_per_timestamp)
        charts = []
        for row in decision_rows.itertuples(index=False):
            instrument_id = str(getattr(row, "instrument_id"))
            chart_frame = frame.loc[frame["instrument_id"].astype(str) == instrument_id].copy()
            chart_frame = chart_frame.sort_values(["bar_offset", "bar_time"], kind="stable")
            bars = [_kline_bar_payload(bar) for bar in chart_frame.itertuples(index=False)]
            bars = [bar for bar in bars if bar is not None]
            if not bars:
                continue
            charts.append(_kline_chart_payload(row, bars))
        if not charts:
            continue
        key = _timestamp_key(timestamp)
        slices[key] = charts
        buy_count = sum(1 for chart in charts if chart.get("markerSide") == "buy")
        sell_count = sum(1 for chart in charts if chart.get("markerSide") == "sell")
        times.append(
            {
                "key": key,
                "label": _short_timestamp(timestamp),
                "chartCount": len(charts),
                "buyCount": buy_count,
                "sellCount": sell_count,
            }
        )
    return {"times": times, "slices": slices}


def _kline_decision_rows(frame: pd.DataFrame) -> pd.DataFrame:
    sort_frame = frame.copy()
    sort_frame["_is_exec"] = sort_frame["is_execution_bar"].astype(int)
    sort_frame = sort_frame.sort_values(
        ["instrument_id", "_is_exec", "bar_offset"],
        ascending=[True, False, True],
        kind="stable",
    ).drop_duplicates("instrument_id", keep="first")
    sort_frame["_abs_notional"] = (
        pd.to_numeric(sort_frame.get("executed_notional"), errors="coerce")
        .abs()
        .fillna(0.0)
    )
    sort_frame["_abs_delta"] = (
        pd.to_numeric(sort_frame.get("delta_weight"), errors="coerce")
        .abs()
        .fillna(0.0)
    )
    sort_frame["_target"] = (
        pd.to_numeric(sort_frame.get("target_weight"), errors="coerce").fillna(0.0)
    )
    sort_frame["_rank"] = pd.to_numeric(sort_frame.get("rank"), errors="coerce").fillna(1e12)
    return sort_frame.sort_values(
        ["_abs_notional", "_abs_delta", "_target", "_rank", "instrument_id"],
        ascending=[False, False, False, True, True],
        kind="stable",
    ).drop(columns=["_is_exec", "_abs_notional", "_abs_delta", "_target", "_rank"])


def _kline_chart_payload(row: object, bars: list[dict[str, object]]) -> dict[str, object]:
    marker_side = _json_string(getattr(row, "marker_side", ""))
    marker_price = _json_float(getattr(row, "marker_price", None))
    executed_notional = _json_float(getattr(row, "executed_notional", None))
    executed_side = _json_string(getattr(row, "executed_side", ""))
    return {
        "instrumentId": _json_string(getattr(row, "instrument_id", "")),
        "label": _instrument_label_from_values(
            getattr(row, "instrument_id", ""),
            getattr(row, "canonical_code", None),
            getattr(row, "raw_name", None),
        ),
        "action": _json_string(getattr(row, "action", "")),
        "reason": _json_string(getattr(row, "decision_reason", "")),
        "execTime": _json_string(getattr(row, "exec_time", "")),
        "currentWeight": _json_float(getattr(row, "current_weight", None)),
        "targetWeight": _json_float(getattr(row, "target_weight", None)),
        "deltaWeight": _json_float(getattr(row, "delta_weight", None)),
        "rank": _json_float(getattr(row, "rank", None)),
        "score": _json_float(getattr(row, "score", None)),
        "markerSide": marker_side,
        "markerPrice": marker_price,
        "markerSource": "trade" if executed_side and executed_notional else "decision",
        "executedSide": executed_side,
        "executedNotional": executed_notional,
        "avgTradePrice": _json_float(getattr(row, "avg_trade_price", None)),
        "bars": bars,
    }


def _kline_bar_payload(row: object) -> dict[str, object] | None:
    open_price = _json_float(getattr(row, "open_price", None))
    high_price = _json_float(getattr(row, "high_price", None))
    low_price = _json_float(getattr(row, "low_price", None))
    close_price = _json_float(getattr(row, "close_price", None))
    if None in {open_price, high_price, low_price, close_price}:
        return None
    return {
        "time": _short_timestamp(getattr(row, "bar_time", "")),
        "offset": _json_float(getattr(row, "bar_offset", None)),
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": _json_float(getattr(row, "volume", None)),
        "turnover": _json_float(getattr(row, "turnover", None)),
        "isExec": bool(getattr(row, "is_execution_bar", False)),
    }


def _instrument_label_from_values(
    instrument_id: object,
    canonical_code: object,
    raw_name: object,
) -> str:
    code = "" if canonical_code is None or pd.isna(canonical_code) else str(canonical_code)
    name = "" if raw_name is None or pd.isna(raw_name) else str(raw_name)
    label = code or str(instrument_id)
    if name:
        label = f"{label} {name}"
    return label


def _json_script_payload(payload: dict[str, object]) -> str:
    text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def _json_float(value: object) -> float | None:
    number = _finite_float(value)
    if number is None:
        return None
    return float(number)


def _json_string(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _timestamp_key(value: object) -> str:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(timestamp):
        return str(value)
    return timestamp.isoformat()


def _kline_script() -> str:
    return r"""
(function () {
  const payloadNode = document.getElementById("decision-kline-payload");
  const slider = document.getElementById("kline-slider");
  const current = document.getElementById("kline-current");
  const summary = document.getElementById("kline-slice-summary");
  const stack = document.getElementById("kline-stack");
  const strip = document.getElementById("kline-time-strip");
  const prev = document.getElementById("kline-prev");
  const next = document.getElementById("kline-next");
  if (!payloadNode || !slider || !current || !summary || !stack || !strip) return;

  const payload = JSON.parse(payloadNode.textContent || "{\"times\":[],\"slices\":{}}");
  const times = payload.times || [];
  const slices = payload.slices || {};
  if (!times.length) return;

  function pct(value) {
    return Number.isFinite(value) ? (value * 100).toFixed(2) + "%" : "n/a";
  }
  function num(value, digits) {
    return Number.isFinite(value) ? value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits }) : "n/a";
  }
  function compact(value) {
    if (!Number.isFinite(value)) return "n/a";
    const abs = Math.abs(value);
    if (abs >= 1000000) return (value / 1000000).toFixed(2) + "m";
    if (abs >= 1000) return (value / 1000).toFixed(2) + "k";
    return value.toFixed(2);
  }
  function clampIndex(value) {
    return Math.max(0, Math.min(times.length - 1, Number(value) || 0));
  }
  function setIndex(index) {
    slider.value = String(clampIndex(index));
    render();
  }
  function renderTimeStrip(index) {
    strip.innerHTML = "";
    times.forEach((time, idx) => {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "kline-time-dot" + (idx === index ? " active" : "");
      button.title = time.label + " | charts=" + time.chartCount + " buy=" + time.buyCount + " sell=" + time.sellCount;
      button.setAttribute("aria-label", time.label);
      button.addEventListener("click", () => setIndex(idx));
      strip.appendChild(button);
    });
  }
  function render() {
    const index = clampIndex(slider.value);
    const time = times[index];
    const charts = slices[time.key] || [];
    current.value = time.label + " (" + (index + 1) + "/" + times.length + ")";
    summary.textContent = charts.length + " charts | buy " + time.buyCount + " | sell " + time.sellCount;
    renderTimeStrip(index);
    stack.innerHTML = "";
    charts.forEach((chart) => stack.appendChild(renderChart(chart)));
  }
  function renderChart(chart) {
    const row = document.createElement("article");
    row.className = "kline-row";

    const header = document.createElement("div");
    header.className = "kline-row-head";
    const title = document.createElement("div");
    title.className = "kline-title";
    title.textContent = chart.label || chart.instrumentId || "";
    const meta = document.createElement("div");
    meta.className = "kline-meta";
    meta.textContent = [
      chart.action || "n/a",
      "target " + pct(chart.targetWeight),
      "delta " + pct(chart.deltaWeight),
      "rank " + num(chart.rank, 0),
      chart.execTime || ""
    ].filter(Boolean).join(" | ");
    const trade = document.createElement("div");
    trade.className = "kline-trade";
    trade.textContent = [
      chart.executedSide || chart.markerSide || "no fill",
      "notional " + compact(chart.executedNotional),
      "avg " + num(chart.avgTradePrice, 2)
    ].join(" | ");
    header.append(title, meta, trade);

    const svg = drawKlineSvg(chart);
    row.append(header, svg);
    return row;
  }
  function drawKlineSvg(chart) {
    const ns = "http://www.w3.org/2000/svg";
    const width = 1120;
    const height = 276;
    const left = 54;
    const right = 24;
    const top = 18;
    const bottom = 42;
    const volumeTop = 218;
    const priceBottom = 204;
    const innerWidth = width - left - right;
    const bars = chart.bars || [];
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);
    svg.setAttribute("role", "img");
    svg.setAttribute("aria-label", "K-line chart for " + (chart.label || chart.instrumentId || ""));
    svg.classList.add("kline-svg");
    if (!bars.length) return svg;

    const highs = bars.map((bar) => bar.high).filter(Number.isFinite);
    const lows = bars.map((bar) => bar.low).filter(Number.isFinite);
    const volumes = bars.map((bar) => bar.volume || 0);
    let yMin = Math.min.apply(null, lows);
    let yMax = Math.max.apply(null, highs);
    if (!Number.isFinite(yMin) || !Number.isFinite(yMax)) return svg;
    if (yMin === yMax) {
      yMin -= Math.max(0.01, Math.abs(yMin) * 0.01);
      yMax += Math.max(0.01, Math.abs(yMax) * 0.01);
    } else {
      const pad = (yMax - yMin) * 0.08;
      yMin -= pad;
      yMax += pad;
    }
    const maxVolume = Math.max.apply(null, volumes) || 1;
    const xStep = innerWidth / Math.max(bars.length, 1);
    const candleWidth = Math.max(4, Math.min(14, xStep * 0.58));
    const xAt = (index) => left + xStep * (index + 0.5);
    const yAt = (value) => top + (priceBottom - top) * (1 - (value - yMin) / (yMax - yMin));
    const vHeight = (value) => Math.max(1, (height - bottom - volumeTop) * (value || 0) / maxVolume);

    addLine(svg, left, top, left, height - bottom, "axis");
    addLine(svg, left, priceBottom, width - right, priceBottom, "axis");
    addLine(svg, left, height - bottom, width - right, height - bottom, "axis faint");
    addText(svg, 6, top + 6, num(yMax, 2), "axis-label");
    addText(svg, 6, priceBottom, num(yMin, 2), "axis-label");
    addText(svg, left, height - 8, bars[0].time || "", "axis-label");
    addText(svg, width - right, height - 8, bars[bars.length - 1].time || "", "axis-label end");

    bars.forEach((bar, index) => {
      const x = xAt(index);
      const up = bar.close >= bar.open;
      const cls = up ? "candle up" : "candle down";
      addLine(svg, x, yAt(bar.high), x, yAt(bar.low), cls + " wick");
      const yOpen = yAt(bar.open);
      const yClose = yAt(bar.close);
      const bodyY = Math.min(yOpen, yClose);
      const bodyH = Math.max(2, Math.abs(yOpen - yClose));
      addRect(svg, x - candleWidth / 2, bodyY, candleWidth, bodyH, cls + " body");
      const vh = vHeight(bar.volume || 0);
      addRect(svg, x - candleWidth / 2, height - bottom - vh, candleWidth, vh, "volume-bar");
      if (bar.isExec) {
        addLine(svg, x, top, x, height - bottom, "exec-line");
      }
    });

    const execIndex = bars.findIndex((bar) => bar.isExec);
    if (execIndex >= 0 && chart.markerSide && Number.isFinite(chart.markerPrice)) {
      const x = xAt(execIndex);
      const y = yAt(chart.markerPrice);
      const side = chart.markerSide === "sell" ? "sell" : "buy";
      const points = side === "buy"
        ? [[x, y - 15], [x - 8, y - 2], [x + 8, y - 2]]
        : [[x, y + 15], [x - 8, y + 2], [x + 8, y + 2]];
      const polygon = document.createElementNS(ns, "polygon");
      polygon.setAttribute("points", points.map((point) => point.join(",")).join(" "));
      polygon.setAttribute("class", "trade-marker " + side);
      svg.appendChild(polygon);
      addText(svg, x + 10, y + (side === "buy" ? -8 : 14), side.toUpperCase(), "trade-label " + side);
    }
    return svg;
  }
  function addLine(svg, x1, y1, x2, y2, cls) {
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", x1);
    line.setAttribute("y1", y1);
    line.setAttribute("x2", x2);
    line.setAttribute("y2", y2);
    line.setAttribute("class", cls);
    svg.appendChild(line);
  }
  function addRect(svg, x, y, width, height, cls) {
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", x);
    rect.setAttribute("y", y);
    rect.setAttribute("width", width);
    rect.setAttribute("height", height);
    rect.setAttribute("class", cls);
    svg.appendChild(rect);
  }
  function addText(svg, x, y, text, cls) {
    const node = document.createElementNS("http://www.w3.org/2000/svg", "text");
    node.setAttribute("x", x);
    node.setAttribute("y", y);
    node.setAttribute("class", cls);
    node.textContent = text;
    svg.appendChild(node);
  }
  slider.addEventListener("input", render);
  if (prev) prev.addEventListener("click", () => setIndex(clampIndex(slider.value) - 1));
  if (next) next.addEventListener("click", () => setIndex(clampIndex(slider.value) + 1));
  render();
})();
""".strip()


def _timestamp_review_html(
    decisions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    config: DecisionReportConfig,
) -> str:
    if decisions.empty:
        return _section_empty("Timestamp Review", "No decision trace was written.")
    timestamps = _review_timestamps(decisions, diagnostics)[:8]
    panels = []
    for timestamp in timestamps:
        frame = decisions.loc[decisions["timestamp"] == timestamp].copy()
        if frame.empty:
            continue
        frame = frame.sort_values(
            ["abs_delta_weight", "target_weight", "priority"],
            ascending=[False, False, True],
        ).head(14)
        action_counts = _count_values(frame, "action")
        summary = ", ".join(f"{key}: {value}" for key, value in action_counts.items())
        table = _html_table(
            frame,
            columns=(
                "instrument_id",
                "canonical_code",
                "exec_time",
                "action",
                "rank",
                "score",
                "current_weight",
                "target_weight",
                "delta_weight",
                "open_price",
                "close_price",
                "turnover",
                "executed_side",
                "executed_notional",
                "decision_reason",
                "constraint_flags",
            ),
            limit=14,
        )
        panels.append(
            f"""
<details class="timestamp-panel">
  <summary><strong>{_escape(_short_timestamp(timestamp))}</strong><span>{_escape(summary)}</span></summary>
  {table}
</details>
""".strip()
        )
    if not panels:
        return _section_empty("Timestamp Review", "No timestamp panels available.")
    return f"""
<section class="panel full">
  <div class="section-heading">
    <h2>Timestamp Review</h2>
    <p>High-turnover or high-activity rebalances for focused manual checking.</p>
  </div>
  {"".join(panels)}
</section>
""".strip()


def _review_timestamps(decisions: pd.DataFrame, diagnostics: pd.DataFrame) -> list[object]:
    if not diagnostics.empty and {
        "timestamp",
        "planned_gross_turnover",
    }.issubset(diagnostics.columns):
        ranked = diagnostics.copy()
        ranked["planned_gross_turnover"] = pd.to_numeric(
            ranked["planned_gross_turnover"],
            errors="coerce",
        ).fillna(0.0)
        return ranked.sort_values(
            ["planned_gross_turnover", "timestamp"],
            ascending=[False, True],
        )["timestamp"].dropna().tolist()
    grouped = (
        decisions.groupby("timestamp", dropna=False)["abs_delta_weight"]
        .sum()
        .reset_index()
        .sort_values(["abs_delta_weight", "timestamp"], ascending=[False, True])
    )
    return grouped["timestamp"].dropna().tolist()


def _top_decisions_html(decisions: pd.DataFrame, config: DecisionReportConfig) -> str:
    if decisions.empty:
        return _section_empty("Largest Decisions", "No decision trace was written.")
    ranked = decisions.sort_values(
        ["abs_delta_weight", "target_weight", "timestamp"],
        ascending=[False, False, True],
    )
    table = _html_table(
        ranked,
        columns=(
            "timestamp",
            "instrument_id",
            "canonical_code",
            "exec_time",
            "action",
            "rank",
            "score",
            "current_weight",
            "target_weight",
            "delta_weight",
            "open_price",
            "high_price",
            "low_price",
            "close_price",
            "bar_return",
            "turnover",
            "tradable_bar",
            "limit_up_open",
            "limit_down_open",
            "executed_side",
            "executed_shares",
            "executed_notional",
            "avg_trade_price",
            "expected_edge_bps",
            "estimated_cost_bps",
            "decision_reason",
            "constraint_flags",
        ),
        limit=config.max_decisions,
    )
    return f"""
<section class="panel full">
  <div class="section-heading">
    <h2>Largest Decisions</h2>
    <p>Sorted by absolute target-weight change.</p>
  </div>
  {table}
</section>
""".strip()


def _trades_html(trades: pd.DataFrame, config: DecisionReportConfig) -> str:
    if trades.empty:
        return _section_empty("Executed Trades", "No trades were written.")
    frame = trades.copy()
    if "notional" not in frame.columns and {"shares", "price"}.issubset(frame.columns):
        frame["notional"] = pd.to_numeric(frame["shares"], errors="coerce") * pd.to_numeric(
            frame["price"],
            errors="coerce",
        )
    if "notional" in frame.columns:
        frame["_abs_notional"] = pd.to_numeric(frame["notional"], errors="coerce").abs()
        frame = frame.sort_values("_abs_notional", ascending=False).drop(
            columns=["_abs_notional"]
        )
    columns = tuple(
        column
        for column in (
            "timestamp",
            "instrument_id",
            "side",
            "shares",
            "price",
            "notional",
            "commission",
            "stamp_tax",
            "slippage_cost",
            "total_cost",
        )
        if column in frame.columns
    )
    return f"""
<section class="panel full">
  <div class="section-heading">
    <h2>Executed Trades</h2>
    <p>Largest executed trades by absolute notional.</p>
  </div>
  {_html_table(frame, columns=columns, limit=min(config.max_decisions, 300))}
</section>
""".strip()


def _html_table(
    frame: pd.DataFrame,
    *,
    columns: tuple[str, ...],
    limit: int,
) -> str:
    if frame.empty or not columns:
        return _empty_state("No rows available.")
    visible = frame.loc[:, [column for column in columns if column in frame.columns]].head(limit)
    if visible.empty:
        return _empty_state("No rows available.")
    headers = "".join(f"<th>{_escape(column)}</th>" for column in visible.columns)
    body = []
    for row in visible.itertuples(index=False):
        cells = []
        for column, value in zip(visible.columns, row, strict=True):
            cells.append(f"<td>{_escape(_format_cell(column, value))}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    caption = ""
    if len(frame) > limit:
        caption = f'<p class="table-note">Showing {limit} of {len(frame)} rows.</p>'
    return (
        '<div class="table-scroll"><table class="data-table">'
        f"<thead><tr>{headers}</tr></thead><tbody>{''.join(body)}</tbody></table></div>{caption}"
    )


def _format_cell(column: str, value: object) -> str:
    if pd.isna(value):
        return ""
    if column in {"timestamp", "exec_time"}:
        return _short_timestamp(value)
    if column in {"current_weight", "aim_weight", "target_weight", "delta_weight"}:
        return _fmt_percent(value)
    if column in {"bar_return"}:
        return _fmt_percent(value)
    if column in {
        "expected_edge_bps",
        "estimated_cost_bps",
        "price",
        "score",
        "open_price",
        "high_price",
        "low_price",
        "close_price",
        "avg_trade_price",
    }:
        return _fmt_number(value, digits=4 if column == "score" else 2)
    if column in {
        "notional",
        "commission",
        "stamp_tax",
        "slippage_cost",
        "total_cost",
        "turnover",
        "executed_notional",
    }:
        return _fmt_number(value, digits=2)
    if column in {"rank", "priority", "shares", "volume", "executed_shares", "trade_count"}:
        return _fmt_count(value)
    return str(value)


def _instrument_labels(frame: pd.DataFrame) -> dict[str, str]:
    if frame.empty:
        return {}
    labels: dict[str, str] = {}
    for row in frame.itertuples(index=False):
        instrument_id = str(getattr(row, "instrument_id"))
        code = getattr(row, "canonical_code", None)
        name = getattr(row, "raw_name", None)
        label = str(code) if code is not None and not pd.isna(code) else instrument_id
        if name is not None and not pd.isna(name) and str(name):
            label = f"{label} {name}"
        labels.setdefault(instrument_id, label)
    return labels


def _count_values(frame: pd.DataFrame, column: str) -> dict[str, int]:
    if frame.empty or column not in frame.columns:
        return {}
    values = frame[column].fillna("").astype(str)
    values = values.loc[values != ""]
    return {str(key): int(value) for key, value in values.value_counts().items()}


def _sample_values(values: list[object], limit: int) -> list[object]:
    if len(values) <= limit:
        return values
    if limit == 1:
        return [values[0]]
    step = (len(values) - 1) / (limit - 1)
    indices = sorted({round(index * step) for index in range(limit)})
    return [values[index] for index in indices]


def _section_empty(title: str, message: str) -> str:
    return f"""
<section class="panel full">
  <h2>{_escape(title)}</h2>
  {_empty_state(message)}
</section>
""".strip()


def _empty_state(message: str) -> str:
    return f'<div class="empty">{_escape(message)}</div>'


def _short_timestamp(value: object) -> str:
    if pd.isna(value):
        return ""
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)
    if pd.isna(timestamp):
        return str(value)
    text = timestamp.isoformat()
    return text.replace("+08:00", "").replace("T", " ")


def _fmt_count(value: object) -> str:
    number = _finite_float(value)
    if number is None:
        return "0"
    return f"{int(round(number)):,}"


def _fmt_percent(value: object) -> str:
    number = _finite_float(value)
    if number is None:
        return "n/a"
    return f"{number * 100:.2f}%"


def _fmt_weight(value: object) -> str:
    number = _finite_float(value, default=0.0)
    return f"{number * 100:.0f}%"


def _fmt_number(value: object, *, digits: int = 2) -> str:
    number = _finite_float(value)
    if number is None:
        return "n/a"
    return f"{number:,.{digits}f}"


def _fmt_compact(value: object) -> str:
    number = _finite_float(value)
    if number is None:
        return "n/a"
    absolute = abs(number)
    if absolute >= 1_000_000:
        return f"{number / 1_000_000:.2f}m"
    if absolute >= 1_000:
        return f"{number / 1_000:.2f}k"
    if absolute >= 1:
        return f"{number:.2f}"
    return f"{number:.4f}"


def _finite_float(value: object, default: float | None = None) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _stylesheet() -> str:
    return """
:root {
  color-scheme: light;
  --bg: #f8fafc;
  --panel: #ffffff;
  --panel-soft: #f1f5f9;
  --text: #0f172a;
  --muted: #64748b;
  --border: #dbe3ee;
  --axis: #cbd5e1;
  --shadow: 0 18px 45px rgba(15, 23, 42, 0.08);
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.page {
  width: min(1440px, calc(100vw - 48px));
  margin: 0 auto;
  padding: 28px 0 48px;
}
.header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 24px;
  margin-bottom: 18px;
}
h1, h2 { margin: 0; letter-spacing: 0; }
h1 { font-size: 30px; line-height: 1.15; font-weight: 720; }
h2 { font-size: 17px; line-height: 1.25; font-weight: 680; }
p { color: var(--muted); margin: 6px 0 0; }
.header-meta {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  justify-content: flex-end;
}
.header-meta span {
  border: 1px solid var(--border);
  background: var(--panel);
  border-radius: 6px;
  padding: 7px 10px;
  color: #334155;
  font-size: 12px;
}
.metric-grid {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 12px;
  margin: 18px 0;
}
.metric, .panel {
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: var(--shadow);
}
.metric {
  border-radius: 8px;
  padding: 14px 15px;
  min-height: 78px;
}
.metric span {
  display: block;
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 8px;
}
.metric strong {
  display: block;
  font-size: 21px;
  line-height: 1.1;
  font-weight: 720;
}
.overview-grid {
  display: grid;
  grid-template-columns: minmax(0, 2fr) minmax(260px, 1fr) minmax(260px, 1fr);
  gap: 14px;
  margin-bottom: 14px;
}
.panel {
  border-radius: 8px;
  padding: 16px;
  min-width: 0;
}
.panel.full {
  margin-top: 14px;
}
.panel.wide {
  min-height: 300px;
}
.section-heading {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 20px;
  margin-bottom: 12px;
}
.section-heading p { max-width: 560px; text-align: right; }
.chart-wrap { margin-top: 10px; }
svg { display: block; width: 100%; height: auto; }
.axis { stroke: var(--axis); stroke-width: 1; }
.axis-label { fill: var(--muted); font-size: 11px; }
.legend {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  margin-top: 8px;
  color: var(--muted);
  font-size: 12px;
}
.legend i {
  display: inline-block;
  width: 18px;
  height: 3px;
  border-radius: 99px;
  margin-right: 6px;
  vertical-align: middle;
}
.bars {
  display: grid;
  gap: 10px;
  margin-top: 14px;
}
.bar-row {
  display: grid;
  grid-template-columns: minmax(90px, 1fr) minmax(80px, 2fr) 44px;
  align-items: center;
  gap: 10px;
  font-size: 12px;
}
.bar-row span {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.bar-row strong {
  text-align: right;
  font-size: 12px;
}
.bar-track {
  height: 8px;
  background: var(--panel-soft);
  border-radius: 999px;
  overflow: hidden;
}
.bar-track i {
  display: block;
  height: 100%;
  border-radius: inherit;
}
.empty {
  display: grid;
  place-items: center;
  min-height: 128px;
  color: var(--muted);
  background: var(--panel-soft);
  border: 1px dashed var(--border);
  border-radius: 8px;
  margin-top: 12px;
}
.heatmap-scroll, .table-scroll {
  overflow: auto;
  border: 1px solid var(--border);
  border-radius: 8px;
}
table {
  border-collapse: collapse;
  width: 100%;
}
.heatmap {
  width: max-content;
  min-width: 100%;
  font-size: 11px;
}
th, td {
  border-bottom: 1px solid var(--border);
  text-align: left;
}
.heatmap th {
  position: sticky;
  top: 0;
  z-index: 1;
  background: #eef2f7;
  color: #475569;
  font-weight: 650;
  padding: 7px 6px;
  white-space: nowrap;
}
.heatmap .instrument {
  position: sticky;
  left: 0;
  z-index: 2;
  min-width: 140px;
  max-width: 180px;
  background: #f8fafc;
  color: #334155;
}
.heat-cell {
  width: 46px;
  height: 30px;
  padding: 0;
  text-align: center;
  font-size: 10px;
  font-weight: 660;
  color: #0f172a;
  border-left: 1px solid #e7edf5;
  background: rgba(148, 163, 184, var(--alpha));
}
.heat-cell.empty { background: #ffffff; }
.action-entry { background: rgba(22, 163, 74, var(--alpha)); }
.action-exit { background: rgba(225, 29, 72, var(--alpha)); }
.action-resize_up { background: rgba(37, 99, 235, var(--alpha)); }
.action-resize_down { background: rgba(217, 119, 6, var(--alpha)); }
.action-hold { background: rgba(100, 116, 139, var(--alpha)); }
.action-no_trade { background: rgba(203, 213, 225, var(--alpha)); color: #64748b; }
.kline-controls {
  display: grid;
  grid-template-columns: auto minmax(220px, 1fr) auto minmax(210px, auto);
  align-items: center;
  gap: 10px;
  margin: 12px 0 10px;
}
.kline-controls button {
  border: 1px solid var(--border);
  background: #ffffff;
  color: #334155;
  border-radius: 6px;
  padding: 7px 11px;
  font: 12px/1.2 inherit;
  cursor: pointer;
}
.kline-controls button:hover { background: #f8fafc; }
.kline-controls input[type="range"] {
  width: 100%;
  accent-color: #2563eb;
}
.kline-controls output {
  color: #334155;
  font-size: 12px;
  text-align: right;
  white-space: nowrap;
}
.kline-time-strip {
  display: flex;
  gap: 4px;
  align-items: center;
  overflow-x: auto;
  padding: 4px 0 8px;
}
.kline-time-dot {
  flex: 0 0 auto;
  width: 14px;
  height: 14px;
  border: 1px solid #cbd5e1;
  border-radius: 999px;
  background: #ffffff;
  cursor: pointer;
}
.kline-time-dot.active {
  background: #2563eb;
  border-color: #2563eb;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.14);
}
.kline-slice-summary {
  color: var(--muted);
  font-size: 12px;
  margin-bottom: 8px;
}
.kline-stack {
  display: grid;
  gap: 0;
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  background: #ffffff;
}
.kline-row {
  padding: 12px 14px 14px;
  border-top: 1px solid var(--border);
}
.kline-row:first-child { border-top: 0; }
.kline-row-head {
  display: grid;
  grid-template-columns: minmax(190px, 1.15fr) minmax(280px, 2fr) minmax(230px, 1fr);
  gap: 14px;
  align-items: baseline;
  margin-bottom: 8px;
}
.kline-title {
  font-weight: 700;
  color: #0f172a;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.kline-meta, .kline-trade {
  color: var(--muted);
  font-size: 12px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.kline-trade { text-align: right; }
.kline-svg {
  display: block;
  width: 100%;
  min-height: 238px;
  background: #fbfdff;
  border: 1px solid #e7edf5;
  border-radius: 6px;
}
.kline-svg .axis {
  stroke: var(--axis);
  stroke-width: 1;
}
.kline-svg .axis.faint {
  stroke: #e7edf5;
}
.kline-svg .axis-label {
  fill: var(--muted);
  font-size: 11px;
}
.kline-svg .axis-label.end {
  text-anchor: end;
}
.kline-svg .candle {
  stroke-width: 1.2;
}
.kline-svg .candle.up {
  stroke: #dc2626;
  fill: #dc2626;
}
.kline-svg .candle.down {
  stroke: #16a34a;
  fill: #16a34a;
}
.kline-svg .volume-bar {
  fill: #cbd5e1;
  opacity: 0.78;
}
.kline-svg .exec-line {
  stroke: #2563eb;
  stroke-width: 1.4;
  stroke-dasharray: 5 4;
}
.kline-svg .trade-marker.buy {
  fill: #15803d;
}
.kline-svg .trade-marker.sell {
  fill: #be123c;
}
.kline-svg .trade-label {
  font-size: 12px;
  font-weight: 760;
}
.kline-svg .trade-label.buy {
  fill: #15803d;
}
.kline-svg .trade-label.sell {
  fill: #be123c;
}
.timestamp-panel {
  border: 1px solid var(--border);
  border-radius: 8px;
  margin-top: 10px;
  overflow: hidden;
}
.timestamp-panel summary {
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 12px 14px;
  background: #f8fafc;
}
.timestamp-panel summary span {
  color: var(--muted);
  font-size: 12px;
}
.data-table {
  font-size: 12px;
  min-width: 960px;
}
.data-table th {
  background: #f8fafc;
  color: #475569;
  font-weight: 680;
  padding: 9px 10px;
  white-space: nowrap;
}
.data-table td {
  padding: 8px 10px;
  vertical-align: top;
  white-space: nowrap;
}
.data-table tr:nth-child(even) td {
  background: #fbfdff;
}
.table-note {
  font-size: 12px;
  margin-top: 8px;
}
@media (max-width: 960px) {
  .page { width: min(100vw - 24px, 1440px); padding-top: 18px; }
  .header { align-items: flex-start; flex-direction: column; }
  .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .overview-grid { grid-template-columns: 1fr; }
  .section-heading { display: block; }
  .section-heading p { text-align: left; }
  .kline-controls { grid-template-columns: auto 1fr auto; }
  .kline-controls output { grid-column: 1 / -1; text-align: left; }
  .kline-row-head { grid-template-columns: 1fr; gap: 4px; }
  .kline-trade { text-align: left; }
}
"""
