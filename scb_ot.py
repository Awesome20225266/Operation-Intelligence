from __future__ import annotations

"""
SCB OT (SCB Operation Theatre)
-----------------------------
Read-only diagnostic page for SCB health across a site.

This module implements the STRICT, Excel-style SCB deviation logic required by the user.

Non-negotiables:
- Read-only: no writes to DuckDB.
- Computation runs ONLY after clicking "Plot Now".
- Time window is enforced at SQL level: 06:00 <= timestamp <= 18:00.
- Outlier removal is per SCB (NOT per timestamp):
  For each (inv_stn_name, inv_name, scb_name) independently, compute the median of that SCB's values
  in the selected range/window, and nullify only values where value > 3 * median.

Pipeline (exact):
1) Fetch raw data from the plant table (named like the site), filtered by date + time window.
2) Per-SCB outlier nulling: value > 3 * SCB_median => nullify only that cell.
3) Aggregate per SCB: SCB_sum = sum(valid values) per (inv_stn_name, inv_name, scb_name).
4) Capacity normalization: normalized_value = SCB_sum / load_kwp (from array_details).
   Missing/NULL/0 load_kwp => drop that SCB.
5) Median benchmark across SCBs (median of normalized_value):
   - If median_value == 0 => abort (warn).
   - If only one SCB remains => deviation is defined as 0%.
6) Deviation formula: ((normalized_value / median_value) - 1) * 100
7) Threshold filter: keep only deviation_pct <= T (if T provided).
8) Plot bar chart sorted ascending (worst first) with X label: inv_stn_name-inv_name-SCBx.
"""

from datetime import date
import re
from typing import Callable, Iterable, Optional

import numpy as np
import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from aws_duckdb import get_duckdb_connection


TIME_START = "06:00"
# Updated requirement: time filter is 06:00 onward (end of day). This remains enforced at SQL level.
TIME_END = "23:59:59"

# Absolute threshold used for:
# - Step 4: cell-level outlier nullification
# - Step 5: peak rejection
ABS_SCB_MAX = 1000.0


# -----------------------------------------------------------------------------
# REUSABLE PLOTLY HELPER: Inverter–SCB Diagnostic Map (VISUALIZATION ONLY)
# -----------------------------------------------------------------------------
def _build_inverter_scb_diagnostic_map(
    *,
    pivot_dev: pd.DataFrame,
    pivot_disc: pd.DataFrame,
    inverter_order: list,
    scb_order: list,
    title: str = "Inverter–SCB Performance & Fault Diagnostic Map",
    avg_da_by_inverter: Optional[dict[str, float]] = None,
    valid_scb_count_by_inverter: Optional[dict[str, int]] = None,
):
    """
    Builds a responsive 2-panel Plotly diagnostic map:
    - Left: Heatmap (Deviation %)
    - Right: Horizontal bar chart (Total Disconnected Strings)

    Assumes all computation is already done.
    Returns a Plotly Figure ready for st.plotly_chart().
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # -----------------------------
    # Normal heatmap diagnostic map (previous behavior)
    # - Left: heatmap of deviation (%) by inverter block vs SCB position
    # - Right: total disconnected strings per inverter block (barh)
    # -----------------------------
    n_scb = len(scb_order)
    n_inv = len(inverter_order)

    # Dynamic width ratios (simple + stable)
    heatmap_width_ratio = min(0.85, max(0.65, n_scb / (n_scb + 6))) if n_scb else 0.75
    bar_width_ratio = 1 - heatmap_width_ratio

    # Disconnected strings summary (hide zeros, sort desc)
    inv_disc_sum = pivot_disc.fillna(0).sum(axis=1)
    inv_disc_sum_all = inv_disc_sum.sort_values(ascending=False)
    inv_disc_sum = inv_disc_sum[inv_disc_sum > 0].sort_values(ascending=False)
    max_ds = float(inv_disc_sum.max()) if not inv_disc_sum.empty else 0.0

    # Severity-based coloring for disconnected strings (UI only)
    def _severity_color(v: int) -> str:
        if v >= 9:
            return "#7f1d1d"  # Critical (dark red)
        elif v >= 6:
            return "#dc2626"  # High (red)
        elif v >= 3:
            return "#f97316"  # Medium (orange)
        else:
            return "#fca5a5"  # Low (light red)

    # Dynamic sizing (works across ASPL/GSPL)
    row_height = 42 if n_inv >= 10 else 40
    calculated_height = max(500, n_inv * row_height)

    if n_inv >= 20:
        colorbar_y = -0.35
    elif n_inv >= 10:
        colorbar_y = -0.32
    else:
        colorbar_y = -0.28

    bottom_margin = max(120, 100 + int(n_inv / 3))

    # Heatmap text: show disconnected strings ONLY if > 0
    # (Visualization-only; keep DS text blank when value is 0.)
    disc_df = pd.DataFrame(pivot_disc).reindex(index=inverter_order, columns=scb_order).fillna(0)
    disc_vals = disc_df.to_numpy()
    disc_int = disc_vals.astype(int, copy=False)
    text_vals = np.where(disc_int > 0, disc_int.astype(str), "")

    # Force categorical axes for the heatmap so Plotly always renders discrete cells
    # (especially important when the DS bar chart is hidden and only the DA line is present).
    x_labels = [str(x) for x in scb_order]
    y_labels = [str(y) for y in inverter_order]

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[heatmap_width_ratio, bar_width_ratio],
        horizontal_spacing=0.12,  # Increased gap to prevent text overlap
    )

    # Heatmap Z is visualization-only.
    # IMPORTANT: rendering must be decoupled from disconnected-strings presence.
    # - Align pivot_dev to the provided axis ordering (does NOT change computations).
    # - Coerce to numeric; if *all* values become NaN (common when DS==0 induces upstream NaNs/misalignment),
    #   render a zero-filled matrix so Plotly still draws the heatmap.
    dev_df = pd.DataFrame(pivot_dev).reindex(index=inverter_order, columns=scb_order)
    # pd.to_numeric() accepts 1D only; coerce by raveling then reshaping back to 2D.
    z_vals = pd.to_numeric(dev_df.to_numpy().ravel(), errors="coerce").reshape(dev_df.shape)
    if dev_df is not None and not dev_df.empty and np.isnan(z_vals).all():
        z_vals = np.zeros_like(z_vals, dtype=float)

    # Plotly can appear blank when the heatmap's effective color range collapses (e.g., all zeros).
    # Force a stable, symmetric range around 0 for rendering ONLY.
    finite = np.isfinite(z_vals)
    if not finite.any():
        z_vals = np.zeros_like(z_vals, dtype=float)
        finite = np.isfinite(z_vals)
    max_abs = float(np.nanmax(np.abs(z_vals[finite]))) if finite.any() else 0.0
    if max_abs == 0.0:
        zmin, zmax = -1.0, 1.0
    else:
        zmin, zmax = -max_abs, max_abs

    # Plotly/Streamlit rendering safety:
    # Represent gaps as None (JSON null) rather than NaN to avoid silent "blank heatmap" failures
    # in some front-end render paths.
    z_plot = z_vals.astype(object)
    z_plot[~np.isfinite(z_vals)] = None

    fig.add_trace(
        go.Heatmap(
            z=z_plot,
            x=x_labels,
            y=y_labels,
            colorscale="RdYlGn",
            zmid=0,
            zmin=zmin,
            zmax=zmax,
            xgap=1,
            ygap=1,
            text=text_vals,
            texttemplate="%{text}",
            textfont=dict(color="black", size=9),
            hovertemplate=(
                "<span style='font-size:15px'><b>Inverter:</b> %{y}</span><br>"
                "<span style='font-size:15px'><b>SCB:</b> %{x}</span><br><br>"
                "<span style='font-size:14px'><b>Deviation:</b> %{z:.2f}%</span><br>"
                "<span style='font-size:14px'><b>Disconnected Strings:</b> %{text}</span>"
                "<extra></extra>"
            ),
            hoverlabel=dict(font_size=15),
            hoverongaps=True,
            showscale=False,  # Remove colorbar completely
        ),
        row=1,
        col=1,
    )

    # -----------------------------
    # RIGHT: BAR CHART
    # -----------------------------
    if not inv_disc_sum.empty:
        # Case 1: Has disconnected strings > 0, show bar chart
        fig.add_trace(
            go.Bar(
                x=inv_disc_sum.values,
                y=inv_disc_sum.index,
                orientation="h",
                name="DS",
                showlegend=True,
                marker=dict(color=[_severity_color(int(v)) for v in inv_disc_sum.values]),
                text=[int(v) for v in inv_disc_sum.values],
                textposition="outside",
                hovertemplate=(
                    "<span style='font-size:18px'><b>Inverter:</b> <b>%{y}</b></span><br>"
                    "<span style='font-size:18px'><b>Total Disconnected Strings:</b> <b>%{x}</b></span>"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )
    else:
        # Case 2: All disconnected strings are 0, add placeholder
        # This prevents the heatmap from failing to render
        fig.add_trace(
            go.Bar(
                x=[0],
                y=[inverter_order[0] if inverter_order else "N/A"],
                orientation="h",
                name="DS",
                showlegend=False,
                marker=dict(color="rgba(0,0,0,0)"),  # Transparent
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
        # Add text annotation explaining no disconnected strings
        fig.add_annotation(
            text="<b>No Disconnected Strings</b><br><span style='font-size:12px'>All strings are connected</span>",
            xref="x2",
            yref="paper",
            x=0.5,
            y=0.5,
            xanchor="center",
            yanchor="middle",
            showarrow=False,
            font=dict(size=14, color="#6b7280"),
            align="center",
        )

    # -----------------------------
    # RIGHT: LINE (Avg DA% per inverter) — NEW (visualization only)
    # Source MUST be the final pipeline output (eligible SCBs only) aggregated in render_scb_ot.
    # -----------------------------
    if avg_da_by_inverter:
        da_x: list[float] = []
        da_y: list[str] = []
        da_n: list[int] = []
        for inv in inverter_order:
            v = avg_da_by_inverter.get(str(inv))
            if v is None or pd.isna(v):
                continue
            da_y.append(str(inv))
            da_x.append(float(v))
            if valid_scb_count_by_inverter:
                da_n.append(int(valid_scb_count_by_inverter.get(str(inv), 0)))
            else:
                da_n.append(0)

        if da_x and da_y:
            # IMPORTANT:
            # Do NOT pass row/col here. make_subplots() will override xaxis/yaxis to x2/y2,
            # which would clip DA% values (0–100) against the DS bar x-axis range (typically 0–20).
            da_text = [f"{float(v):.0f}%" for v in da_x]
            fig.add_trace(
                go.Scatter(
                    x=da_x,
                    y=da_y,
                    mode="lines+markers+text",
                    name="Avg DA%",
                    xaxis="x3",
                    yaxis="y2",
                    line=dict(color="#0ea5a4", width=2),
                    marker=dict(size=8, color="#0ea5a4"),
                    text=da_text,
                    textposition="top center",
                    textfont=dict(size=11, color="rgba(2,6,23,0.70)"),
                    customdata=da_n,
                    hovertemplate=(
                        "<span style='font-size:18px'><b>Inverter:</b> <b>%{y}</b></span><br>"
                        "<span style='font-size:18px'><b>Average DA%:</b> <b>%{x:.2f}%</b></span><br>"
                        "<span style='font-size:18px'><b>Valid SCBs:</b> <b>%{customdata}</b></span>"
                        "<extra></extra>"
                    ),
                )
            )

    # Axes configuration
    fig.update_xaxes(
        title=dict(text="<b>SCB Number (Position)</b>", standoff=12, font=dict(size=16)),
        automargin=True,
        type="category",
        categoryorder="array",
        categoryarray=x_labels,
        row=1,
        col=1,
    )

    # Left heatmap: Y-axis
    fig.update_yaxes(
        title=dict(text="<b>Inverter (IS–INV)</b>", standoff=30),
        autorange="reversed",
        automargin=True,
        type="category",
        categoryorder="array",
        categoryarray=y_labels,
        row=1,
        col=1,
    )

    # Right bar chart: Y-axis (NO title — shared axis, labels already visible)
    fig.update_yaxes(
        title=None,
        automargin=True,
        showticklabels=True,
        row=1,
        col=2,
    )

    # Bold inverter labels on BOTH panels (tick text + consistent font)
    bold_ticks = [f"<b>{y}</b>" for y in y_labels]
    fig.update_yaxes(tickfont=dict(size=13, color="black"), row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=13, color="black"), row=1, col=2)
    fig.update_layout(
        yaxis=dict(ticktext=bold_ticks, tickvals=y_labels),
        yaxis2=dict(ticktext=bold_ticks, tickvals=y_labels),
    )

    # Right bar chart: Bottom X-axis
    fig.update_xaxes(
        title=dict(text="<b>Total Disconnected Strings</b>", standoff=12, font=dict(size=16)),
        automargin=True,
        range=[0, max_ds * 1.25] if max_ds > 0 else None,
        row=1,
        col=2,
    )

    # Right line chart: Top X-axis (secondary)
    # IMPORTANT: Uses a secondary axis overlaying the right panel's x-axis.
    fig.update_layout(
        xaxis3=dict(
            title=dict(text="<b>Avg DA% (0–100)</b>", standoff=12, font=dict(size=16)),
            range=[0, 100],
            overlaying="x2",
            side="top",
            ticksuffix="%",
            showgrid=False,
            anchor="y2",
        )
    )

    # Top X-axis labels removed per user request

    # Layout safety net
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.01, font=dict(size=18)),
        template="plotly_white",
        height=calculated_height,
        margin=dict(l=80, r=50, t=110, b=bottom_margin),
        hoverlabel=dict(font_size=18),
    )

    return fig


def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    # Centralized read-only connection (same approach used by dashboard.py)
    return get_duckdb_connection(db_local=db_path)


# -----------------------------------------------------------------------------
# REUSABLE PLOTLY HELPER: "Hierarchical Pulse" (VISUALIZATION ONLY)
# -----------------------------------------------------------------------------
def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def _norm_col(c: str) -> str:
    return str(c).strip().lower()


def _sanitize_table_guess(site_name: str) -> str:
    # Best-effort: some tables are lowercased/sanitized (e.g. GSPL-GAP -> gspl_gap)
    s = str(site_name).strip().lower()
    out = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    s2 = "".join(out).strip("_")
    return s2 or s.lower()


@st.cache_data(show_spinner=False)
def list_sites_from_array_details(db_path: str) -> list[str]:
    """
    STRICT requirement:
    - Site list must come from master.duckdb -> array_details.site_name
    - No fallback to plant_details (or any other table)
    """
    con = _connect(db_path)
    try:
        # Validate column presence explicitly (no fallback).
        info = con.execute("pragma table_info('array_details')").fetchdf()
        cols = [str(x) for x in info.get("name", pd.Series(dtype=str)).tolist()]
        if "site_name" not in cols:
            raise RuntimeError("`array_details.site_name` column not found (required for SCB OT).")
        # Trim to avoid duplicates like "TSPL" vs "TSPL " and to improve table resolution.
        rows = con.execute(
            """
            select distinct trim(site_name) as site_name
            from array_details
            where site_name is not null
              and trim(site_name) != ''
            order by 1
            """
        ).fetchall()
        return [str(r[0]) for r in rows if r and r[0] is not None and str(r[0]).strip() != ""]
    finally:
        con.close()


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    try:
        con.execute(f"select 1 from {_quote_ident(table)} limit 1")
        return True
    except Exception:
        return False


@st.cache_data(show_spinner=False)
def resolve_site_table_name(db_path: str, site_name: str) -> str:
    """
    Requirement says: "Resolve plant data table name dynamically (same as site name)".
    In practice, some pipelines store site tables sanitized/lowercase.

    We try:
    - exact site_name
    - sanitized lowercase variant
    """
    con = _connect(db_path)
    try:
        exact = str(site_name).strip()
        if exact and _table_exists(con, exact):
            return exact
        guess = _sanitize_table_guess(exact)
        if guess and _table_exists(con, guess):
            return guess
        # Return exact by default for error messaging downstream.
        return exact
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def get_table_columns(db_path: str, table: str) -> list[str]:
    con = _connect(db_path)
    try:
        info = con.execute(f"pragma table_info({_quote_ident(table)})").fetchdf()
        return [str(x) for x in info["name"].tolist()]
    finally:
        con.close()


def _norm_key(v: object) -> str:
    """
    Case-insensitive, trimmed join keys (required).
    We keep them lower() to avoid surprising locale issues with upper().
    """
    if v is None:
        return ""
    return str(v).strip().lower()


@st.cache_data(show_spinner=False)
def get_array_capacities(db_path: str, site_name: str) -> pd.DataFrame:
    """
    Capacity source (CRITICAL): master.duckdb -> array_details

    Required columns:
      site_name, inv_stn_name, inv_name, scb_name, load_kwp

    Join keys are case-insensitive, trimmed (handled in pandas via *_key columns).
    """
    con = _connect(db_path)
    try:
        # Validate required columns explicitly (no fallback).
        info = con.execute("pragma table_info('array_details')").fetchdf()
        cols = [str(x) for x in info.get("name", pd.Series(dtype=str)).tolist()]
        required = {"site_name", "inv_stn_name", "inv_name", "scb_name", "load_kwp"}
        missing = sorted(required - set(cols))
        if missing:
            raise RuntimeError(f"`array_details` is missing required columns: {missing}")

        # Filter site_name case-insensitively and trimmed (requirement).
        df = con.execute(
            """
            select
              site_name,
              inv_stn_name,
              inv_name,
              scb_name,
              load_kwp
            from array_details
            where lower(trim(site_name)) = lower(trim(?))
            """,
            [site_name],
        ).fetchdf()
    finally:
        con.close()

    # Normalize join keys (case-insensitive, trimmed) for robust matching.
    for c in ["site_name", "inv_stn_name", "inv_name", "scb_name"]:
        df[f"{c}_key"] = df[c].map(_norm_key)

    df["load_kwp"] = pd.to_numeric(df["load_kwp"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def get_array_string_nums(db_path: str, site_name: str) -> pd.DataFrame:
    """
    String-based normalization source (STRICT):
      array_details.string_num

    Required columns:
      site_name, inv_stn_name, inv_name, scb_name, string_num

    GSPL-only median enhancement:
      We also fetch inv_unit_name from array_details via the SAME query (no new query).
      If the column does not exist in the DB schema, we safely return NULL for inv_unit_name
      so non-GSPL behavior remains unchanged.
    """
    con = _connect(db_path)
    try:
        info = con.execute("pragma table_info('array_details')").fetchdf()
        cols = [str(x) for x in info.get("name", pd.Series(dtype=str)).tolist()]
        required = {"site_name", "inv_stn_name", "inv_name", "scb_name", "string_num"}
        missing = sorted(required - set(cols))
        if missing:
            raise RuntimeError(f"`array_details` is missing required columns: {missing}")

        unit_select = "inv_unit_name" if "inv_unit_name" in cols else "cast(null as varchar) as inv_unit_name"

        df = con.execute(
            """
            select
              site_name,
              inv_stn_name,
              inv_name,
              scb_name,
              string_num,
              """
            + unit_select
            + """
            from array_details
            where lower(trim(site_name)) = lower(trim(?))
            """,
            [site_name],
        ).fetchdf()
    finally:
        con.close()

    for c in ["site_name", "inv_stn_name", "inv_name", "scb_name"]:
        df[f"{c}_key"] = df[c].map(_norm_key)
    df["string_num"] = pd.to_numeric(df["string_num"], errors="coerce")
    return df


def _sql_date_expr(col: str) -> str:
    """
    Support both DD-MM-YYYY (Apollo tables) and YYYY-MM-DD (some other tables).
    We don't write back; this is only for filtering.
    """
    c = _quote_ident(col)
    return (
        f"coalesce("
        f"try_strptime(cast({c} as varchar), '%d-%m-%Y'), "
        f"try_strptime(cast({c} as varchar), '%Y-%m-%d')"
        f")::date"
    )


def _sql_time_expr(col: str) -> str:
    c = _quote_ident(col)
    return (
        f"coalesce("
        f"try_strptime(cast({c} as varchar), '%H:%M'), "
        f"try_strptime(cast({c} as varchar), '%H:%M:%S')"
        f")::time"
    )


@st.cache_data(show_spinner=False)
def _date_bounds_for_site_table(db_path: str, table: str) -> tuple[Optional[date], Optional[date]]:
    """
    Compute overall date bounds for a site table. This is used ONLY to set
    min/max for the date pickers (informational UX), not for deviation math.
    """
    date_expr = _sql_date_expr("date")
    con = _connect(db_path)
    try:
        row = con.execute(
            f"select min({date_expr}) as dmin, max({date_expr}) as dmax from {_quote_ident(table)}"
        ).fetchone()
    finally:
        con.close()
    if not row:
        return None, None
    return row[0], row[1]


@st.cache_data(show_spinner=False)
def _available_dates_for_site_table(db_path: str, table: str, scb_cols: tuple[str, ...]) -> set[date]:
    """
    Calendar availability logic (STRICT, separate from deviation computation):
    A date D is considered "available" if there exists ANY timestamp between
    06:00–18:00 on that date where ANY SCB column has a value > 0.

    This is ONLY for date-picker guidance (captions/warnings). It must NOT
    influence the SCB deviation math.
    """
    if not scb_cols:
        return set()

    date_expr = _sql_date_expr("date")
    time_expr = _sql_time_expr("timestamp")

    # Build dynamic OR condition: (SCB1 > 0 OR SCB2 > 0 OR ...)
    # Use TRY_CAST to handle string-typed columns safely.
    parts: list[str] = []
    for c in scb_cols:
        qc = _quote_ident(c)
        parts.append(f"coalesce(try_cast({qc} as double), 0) > 0")
    or_expr = "(" + " OR ".join(parts) + ")"

    sql = f"""
    select distinct {date_expr} as d
    from {_quote_ident(table)}
    where
      {time_expr} between time '{TIME_START}' and time '{TIME_END}'
      and {or_expr}
    order by 1
    """

    con = _connect(db_path)
    try:
        rows = con.execute(sql).fetchall()
    finally:
        con.close()

    out: set[date] = set()
    for r in rows:
        if r and isinstance(r[0], date):
            out.add(r[0])
    return out


def _fetch_raw_scb_data(
    *,
    db_path: str,
    table: str,
    from_date: date,
    to_date: date,
    scb_cols: Iterable[str],
) -> pd.DataFrame:
    cols = ["date", "timestamp", "inv_stn_name", "inv_name"] + list(scb_cols)
    select_list = ", ".join([_quote_ident(c) for c in cols])
    date_expr = _sql_date_expr("date")
    time_expr = _sql_time_expr("timestamp")

    sql = f"""
    select {select_list}
    from {_quote_ident(table)}
    where
      {date_expr} between ? and ?
      and {time_expr} between time '{TIME_START}' and time '{TIME_END}'
    """

    con = _connect(db_path)
    try:
        df = con.execute(sql, [from_date, to_date]).fetchdf()
    finally:
        con.close()

    return df


def _fetch_raw_scb_data_in_time_window(
    *,
    db_path: str,
    table: str,
    from_date: date,
    to_date: date,
    scb_cols: Iterable[str],
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """
    Same as _fetch_raw_scb_data, but with an explicit SQL time window.
    This is used for RULE 3.3 (06:00–18:00) while keeping time filtering at SQL level.
    """
    cols = ["date", "timestamp", "inv_stn_name", "inv_name"] + list(scb_cols)
    select_list = ", ".join([_quote_ident(c) for c in cols])
    date_expr = _sql_date_expr("date")
    time_expr = _sql_time_expr("timestamp")

    sql = f"""
    select {select_list}
    from {_quote_ident(table)}
    where
      {date_expr} between ? and ?
      and {time_expr} between time '{start_time}' and time '{end_time}'
    """

    con = _connect(db_path)
    try:
        return con.execute(sql, [from_date, to_date]).fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def fetch_scb_operational_data(
    *,
    db_path: str,
    table: str,
    from_date: date,
    to_date: date,
    scb_cols: tuple[str, ...],
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """
    PERFORMANCE ONLY (no logic change):
    Cached raw SCB fetch for a given site table + date range + time window.

    Cache key includes:
    - table (site)
    - from_date / to_date
    - time window (start_time / end_time)
    - scb_cols (schema-dependent)
    """
    return _fetch_raw_scb_data_in_time_window(
        db_path=db_path,
        table=table,
        from_date=from_date,
        to_date=to_date,
        scb_cols=list(scb_cols),
        start_time=start_time,
        end_time=end_time,
    )


@st.cache_data(show_spinner=False)
def _infer_sampling_interval_seconds_from_table(
    db_path: str, table: str, from_date: date, to_date: date
) -> Optional[int]:
    """
    Infer the site's native sampling interval (in seconds) from the operational window (06:00–18:00),
    using SQL-level time filtering.

    Approach:
    - Pull distinct (date, time) points in 06:00–18:00
    - For each date, compute successive deltas in seconds
    - Use the most common positive delta as the sampling interval
    """
    date_expr = _sql_date_expr("date")
    time_expr = _sql_time_expr("timestamp")

    con = _connect(db_path)
    try:
        slots = con.execute(
            f"""
            select distinct
              {date_expr} as d,
              {time_expr} as t
            from {_quote_ident(table)}
            where
              {date_expr} between ? and ?
              and {time_expr} between time '06:00' and time '18:00'
            order by 1,2
            """,
            [from_date, to_date],
        ).fetchdf()
    finally:
        con.close()

    if slots is None or slots.empty or "d" not in slots.columns or "t" not in slots.columns:
        return None

    # Compute deltas per-day (DuckDB returns Python date/time objects when casted).
    deltas: list[int] = []
    for _, g in slots.groupby("d", dropna=False):
        ts = g["t"].tolist()
        # ensure order
        ts = sorted([x for x in ts if x is not None])
        if len(ts) < 2:
            continue
        for a, b in zip(ts, ts[1:]):
            try:
                da = int(getattr(a, "hour")) * 3600 + int(getattr(a, "minute")) * 60 + int(getattr(a, "second", 0))
                db = int(getattr(b, "hour")) * 3600 + int(getattr(b, "minute")) * 60 + int(getattr(b, "second", 0))
                dsec = db - da
                if dsec > 0:
                    deltas.append(dsec)
            except Exception:
                continue

    if not deltas:
        return None

    vc = pd.Series(deltas).value_counts()
    if vc.empty:
        return None
    # Most common positive delta
    return int(vc.index[0])


def _expected_slots_per_day(interval_seconds: int) -> int:
    """
    Expected number of time slots between 06:00 and 18:00 inclusive, given a sampling interval.
    """
    window_seconds = 12 * 60 * 60  # 12 hours
    if interval_seconds <= 0:
        return 0
    return int(window_seconds // interval_seconds) + 1


def _median_s(series: pd.Series) -> Optional[float]:
    """
    Median helper that returns None if it can't compute a numeric median.
    """
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    try:
        m = float(s.median())
    except Exception:
        return None
    return m if pd.notna(m) else None


def _compute_deviation_median_based(
    *,
    df_raw: pd.DataFrame,
    site_name: str,
    scb_cols: list[str],
    db_path: str,
) -> pd.DataFrame:
    """
    Implements the required median-based SCB deviation logic, line-by-line auditable.

    Returns one row per SCB (per inverter context):
      inv_stn_name, inv_name, scb_name,
      scb_sum, load_kwp, normalized_value,
      median_value, deviation_pct,
      scb_label
    """
    base_cols = ["inv_stn_name", "inv_name"]
    if df_raw is None or df_raw.empty:
        return pd.DataFrame(columns=[*base_cols, "scb_name", "scb_sum", "load_kwp", "normalized_value", "median_value", "deviation_pct", "scb_label"])
    if not scb_cols:
        return pd.DataFrame(columns=[*base_cols, "scb_name", "scb_sum", "load_kwp", "normalized_value", "median_value", "deviation_pct", "scb_label"])

    # --- Step 1: Long-form values per SCB (keeps inverter context) ---
    # NOTE: We intentionally do NOT drop timestamps due to spikes.
    df = df_raw.copy()
    for c in base_cols:
        df[c] = df[c].astype(str)

    long_df = df.melt(
        id_vars=base_cols,
        value_vars=scb_cols,
        var_name="scb_name",
        value_name="scb_value",
    )
    long_df["scb_name"] = long_df["scb_name"].astype(str)
    long_df["scb_value"] = pd.to_numeric(long_df["scb_value"], errors="coerce")

    # Compute SCB-wise median per (inv_stn_name, inv_name, scb_name).
    # Requirement: outlier removal is per SCB, NOT per timestamp.
    scb_median = long_df.groupby([*base_cols, "scb_name"], dropna=False)["scb_value"].transform(_median_s)
    long_df["scb_median"] = scb_median

    # Edge case: SCB median == 0 -> drop SCB entirely (requirement).
    # We do it by nullifying all values so the SCB is removed downstream.
    median_bad = long_df["scb_median"].isna() | (pd.to_numeric(long_df["scb_median"], errors="coerce") <= 0)
    long_df.loc[median_bad, "scb_value"] = pd.NA

    # Outlier rule (strict): value > 3 * SCB_median -> nullify only that SCB cell.
    thr = pd.to_numeric(long_df["scb_median"], errors="coerce") * 3.0
    is_outlier = pd.to_numeric(long_df["scb_value"], errors="coerce") > thr
    long_df.loc[is_outlier, "scb_value"] = pd.NA

    # --- Step 2: Aggregate per SCB (after outlier nulling) ---
    sums = (
        long_df.dropna(subset=["scb_value"])
        .groupby([*base_cols, "scb_name"], as_index=False)["scb_value"]
        .sum()
        .rename(columns={"scb_value": "scb_sum"})
    )
    if sums.empty:
        return pd.DataFrame(columns=[*base_cols, "scb_name", "scb_sum", "load_kwp", "normalized_value", "median_value", "deviation_pct", "scb_label"])

    # --- Step 3: Capacity normalization (join to array_details) ---
    cap = get_array_capacities(db_path, site_name)
    if cap.empty:
        return pd.DataFrame(columns=[*base_cols, "scb_name", "scb_sum", "load_kwp", "normalized_value", "median_value", "deviation_pct", "scb_label"])

    # Build join keys (case-insensitive, trimmed) per requirement.
    sums["site_name"] = site_name
    for c in ["site_name", "inv_stn_name", "inv_name", "scb_name"]:
        sums[f"{c}_key"] = sums[c].map(_norm_key)

    joined = sums.merge(
        cap[["site_name_key", "inv_stn_name_key", "inv_name_key", "scb_name_key", "load_kwp"]],
        on=["site_name_key", "inv_stn_name_key", "inv_name_key", "scb_name_key"],
        how="left",
    )

    # Missing/NULL/0 load_kwp => drop that SCB (requirement).
    joined["load_kwp"] = pd.to_numeric(joined["load_kwp"], errors="coerce")
    joined = joined.dropna(subset=["load_kwp"])
    joined = joined[joined["load_kwp"] > 0].copy()
    if joined.empty:
        return pd.DataFrame(columns=[*base_cols, "scb_name", "scb_sum", "load_kwp", "normalized_value", "median_value", "deviation_pct", "scb_label"])

    joined["normalized_value"] = joined["scb_sum"] / joined["load_kwp"]

    # --- Step 4: Median benchmark across SCBs ---
    norm = pd.to_numeric(joined["normalized_value"], errors="coerce").dropna()
    if norm.empty:
        return pd.DataFrame(columns=[*base_cols, "scb_name", "scb_sum", "load_kwp", "normalized_value", "median_value", "deviation_pct", "scb_label"])

    if len(norm) == 1:
        # Edge case: Only one SCB remains -> deviation = 0% (mandatory).
        median_value = float(norm.iloc[0])
        joined["median_value"] = median_value
        joined["deviation_pct"] = 0.0
    else:
        median_value = float(norm.median())
        if pd.isna(median_value) or median_value == 0:
            # Mandatory abort condition.
            return pd.DataFrame(columns=[*base_cols, "scb_name", "scb_sum", "load_kwp", "normalized_value", "median_value", "deviation_pct", "scb_label"])
        joined["median_value"] = median_value
        # --- Step 5: deviation formula (strict) ---
        joined["deviation_pct"] = ((joined["normalized_value"] / median_value) - 1.0) * 100.0

    # --- X-axis label (mandatory) ---
    # inv_stn_name - inv_name - SCB_name (no spaces)
    # X-axis label format (MANDATORY): ISx-INVy-SCBz
    # We construct it from the table's inv_stn_name / inv_name / scb_name by extracting digits
    # and forcing the required prefixes. This keeps plotting deterministic even if casing varies.
    def _force_triplet_label(inv_stn: object, inv: object, scb: object) -> str:
        def _digits(x: object) -> Optional[str]:
            m = re.search(r"(\d+)", str(x).strip(), flags=re.IGNORECASE)
            return m.group(1) if m else None

        is_n = _digits(inv_stn)
        inv_n = _digits(inv)
        scb_n = _digits(scb)

        # If any component is missing digits, fall back to the raw token (still sortable as 999s).
        is_part = f"IS{is_n}" if is_n is not None else str(inv_stn).strip()
        inv_part = f"INV{inv_n}" if inv_n is not None else str(inv).strip()
        scb_part = f"SCB{scb_n}" if scb_n is not None else str(scb).strip()
        return f"{is_part}-{inv_part}-{scb_part}"

    joined["scb_label"] = joined.apply(lambda r: _force_triplet_label(r["inv_stn_name"], r["inv_name"], r["scb_name"]), axis=1)

    return joined[[*base_cols, "scb_name", "scb_sum", "load_kwp", "normalized_value", "median_value", "deviation_pct", "scb_label"]].copy()


def _parse_scb_label(label: str) -> tuple[int, int, int]:
    """
    STRICT hierarchical numeric parse for sorting.
    IS1-INV2-SCB9 -> (1, 2, 9)
    """
    m = re.search(r"IS(\d+)-INV(\d+)-SCB(\d+)", str(label).strip(), flags=re.IGNORECASE)
    if not m:
        return (999, 999, 999)
    try:
        return tuple(map(int, m.groups()))  # type: ignore[return-value]
    except Exception:
        return (999, 999, 999)


def _style_remarks(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """
    Color-coded remarks table (MANDATORY).
    - elimination -> red
    - missing/skip -> amber
    - insight -> blue
    - info -> neutral
    """
    def _row_style(row: pd.Series) -> list[str]:
        sev = str(row.get("severity", "")).lower()
        if sev == "eliminated":
            bg = "#fee2e2"  # red-100
        elif sev == "skipped":
            bg = "#ffedd5"  # amber-100
        elif sev == "insight":
            bg = "#dbeafe"  # blue-100
        else:
            bg = ""
        return [f"background-color: {bg}"] * len(row)

    return df.style.apply(_row_style, axis=1)


def build_scb_insight_summary(remarks_df: pd.DataFrame) -> pd.DataFrame:
    """
    UI-only "big hoarder view" summary derived from remarks_df.
    Must not change any computation outputs (purely re-grouping existing remarks).
    """
    if remarks_df is None or remarks_df.empty:
        return pd.DataFrame(columns=["Explanation", "Count", "Details"])

    def classify(msg: str) -> str:
        # NOTE: This is UI-only grouping derived from remarks.
        # It must remain threshold-independent and must not affect computation.
        if "night time bad value" in msg:
            return "Night Time Bad Value"
        if "very low energy contribution" in msg:
            return "SCB eliminated: very low energy contribution"
        if "value constant throughout" in msg:
            return "SCB eliminated: value constant throughout selected time window"
        if "Possible disconnected strings detected" in msg:
            return "Possible disconnected strings detected"
        return "Other"

    tmp = remarks_df.copy()
    tmp["message"] = tmp.get("message", "").astype(str)
    tmp["Explanation"] = tmp["message"].map(classify)

    grouped = (
        tmp.groupby("Explanation")
        .agg(
            Count=("scb_label", "count"),
            # Show impacted SCB labels (analyst-friendly), not the raw message text
            Details=("scb_label", lambda x: ", ".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )

    return grouped


def _detect_night_time_bad_scbs(
    *,
    db_path: str,
    table: str,
    from_date: date,
    to_date: date,
    scb_cols: list[str],
) -> set[str]:
    """
    Detect SCBs with abnormal night-time values in a raw time window.

    IMPORTANT (by design):
    - This runs independently of the main 06:00–18:00 analysis fetch.
    - We do NOT change any existing 06:00–18:00 SQL filters.
    - Instead, we query the site table directly with ONLY the 02:00–04:00 filter,
      so elimination decisions are not influenced by operational-window filtering.

    Rule (STRICT):
    - Window: 02:00 ≤ timestamp ≤ 04:00 (SQL enforced)
    - Metric: SUM(ABS(scb_value)) per SCB label
    - Eliminate if sum > 30

    Returns:
      set[str] of scb_label values to eliminate.
    """
    if not scb_cols:
        return set()

    # Fetch raw SCB values in the night-time window (SQL time filter only).
    night_df = _fetch_raw_scb_data_in_time_window(
        db_path=db_path,
        table=table,
        from_date=from_date,
        to_date=to_date,
        scb_cols=scb_cols,
        start_time="02:00",
        end_time="04:00",
    )
    if night_df is None or night_df.empty:
        return set()

    base_cols = ["inv_stn_name", "inv_name"]
    night_w = night_df.copy()
    for c in base_cols:
        night_w[c] = night_w[c].astype(str)

    night_long = night_w.melt(
        id_vars=base_cols,
        value_vars=scb_cols,
        var_name="scb_name",
        value_name="scb_value",
    )
    night_long["scb_name"] = night_long["scb_name"].astype(str)
    night_long["scb_value"] = pd.to_numeric(night_long["scb_value"], errors="coerce")

    def _force_triplet_label(inv_stn: object, inv: object, scb: object) -> str:
        def _digits(x: object) -> Optional[str]:
            m = re.search(r"(\d+)", str(x).strip(), flags=re.IGNORECASE)
            return m.group(1) if m else None

        is_n = _digits(inv_stn)
        inv_n = _digits(inv)
        scb_n = _digits(scb)
        is_part = f"IS{is_n}" if is_n is not None else str(inv_stn).strip()
        inv_part = f"INV{inv_n}" if inv_n is not None else str(inv).strip()
        scb_part = f"SCB{scb_n}" if scb_n is not None else str(scb).strip()
        return f"{is_part}-{inv_part}-{scb_part}"

    night_long["scb_label"] = night_long.apply(
        lambda r: _force_triplet_label(r["inv_stn_name"], r["inv_name"], r["scb_name"]),
        axis=1,
    )

    gcols = ["inv_stn_name", "inv_name", "scb_name", "scb_label"]
    night_sums = (
        night_long.groupby(gcols, dropna=False)["scb_value"]
        .apply(lambda s: float(pd.to_numeric(s, errors="coerce").dropna().abs().sum()))
        .reset_index(name="night_abs_sum")
    )

    bad = pd.to_numeric(night_sums["night_abs_sum"], errors="coerce") > 30.0
    return set(night_sums.loc[bad, "scb_label"].astype(str).tolist())


@st.cache_data(show_spinner=False)
def _kpi_total_scb_count(*, db_path: str, site_name: str) -> int:
    """
    KPI-ONLY query (threshold-independent).

    KPI 1 — Total SCB
    Source: array_details
    Logic (threshold-independent):
    COUNT of SCB instances in the site based on array_details mapping.

    NOTE:
    Elimination rules operate on SCB instances (inv_stn_name, inv_name, scb_name) and produce scb_label.
    To keep KPI 7 (Communicating SCB) consistent and non-negative, KPI 1 must count the same granularity.
    """
    con = _connect(db_path)
    try:
        info = con.execute("pragma table_info('array_details')").fetchdf()
        cols = [str(x) for x in info.get("name", pd.Series(dtype=str)).tolist()]
        required = {"site_name", "inv_stn_name", "inv_name", "scb_name"}
        if not required.issubset(set(cols)):
            return 0

        row = con.execute(
            """
            select count(distinct (
                lower(trim(cast(inv_stn_name as varchar))) || '|' ||
                lower(trim(cast(inv_name as varchar))) || '|' ||
                lower(trim(cast(scb_name as varchar)))
            )) as n
            from array_details
            where lower(trim(site_name)) = lower(trim(?))
              and scb_name is not null
              and trim(cast(scb_name as varchar)) != ''
            """,
            [site_name],
        ).fetchone()
        return int(row[0]) if row and row[0] is not None else 0
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def _kpi_total_scb_label_set(*, db_path: str, site_name: str) -> set[str]:
    """
    KPI-only base population (MANDATORY per requirements):

    TOTAL_SCB_SET = distinct (inv_stn_name, inv_name, scb_name) from array_details for the selected site.

    We return these as deterministic scb_label strings (ISx-INVy-SCBz) so we can perform set arithmetic
    against remarks_df scb_label values without touching deviation/plot outputs.
    """
    con = _connect(db_path)
    try:
        info = con.execute("pragma table_info('array_details')").fetchdf()
        cols = [str(x) for x in info.get("name", pd.Series(dtype=str)).tolist()]
        required = {"site_name", "inv_stn_name", "inv_name", "scb_name"}
        if not required.issubset(set(cols)):
            return set()

        df = con.execute(
            """
            select distinct
              trim(cast(inv_stn_name as varchar)) as inv_stn_name,
              trim(cast(inv_name as varchar)) as inv_name,
              trim(cast(scb_name as varchar)) as scb_name
            from array_details
            where lower(trim(site_name)) = lower(trim(?))
              and scb_name is not null
              and trim(cast(scb_name as varchar)) != ''
            """,
            [site_name],
        ).fetchdf()
    finally:
        con.close()

    if df is None or df.empty:
        return set()

    def _force_triplet_label(inv_stn: object, inv: object, scb: object) -> str:
        def _digits(x: object) -> Optional[str]:
            m = re.search(r"(\d+)", str(x).strip(), flags=re.IGNORECASE)
            return m.group(1) if m else None

        is_n = _digits(inv_stn)
        inv_n = _digits(inv)
        scb_n = _digits(scb)
        is_part = f"IS{is_n}" if is_n is not None else str(inv_stn).strip()
        inv_part = f"INV{inv_n}" if inv_n is not None else str(inv).strip()
        scb_part = f"SCB{scb_n}" if scb_n is not None else str(scb).strip()
        return f"{is_part}-{inv_part}-{scb_part}"

    labels = set()
    for r in df.itertuples(index=False):
        labels.add(_force_triplet_label(getattr(r, "inv_stn_name"), getattr(r, "inv_name"), getattr(r, "scb_name")))
    return labels


def _render_kpi_cards(kpis: dict[str, object]) -> None:
    """
    Presentational-only KPI cards (must not affect computation or plots).
    """
    st.markdown(
        """
<style>
  .scbot-kpi-wrap { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; }
  @media (max-width: 900px) { .scbot-kpi-wrap { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
  .scbot-kpi-card {
    border: 1px solid rgba(148,163,184,0.35);
    border-radius: 14px;
    padding: 12px 14px;
    background: linear-gradient(180deg, rgba(15,23,42,0.04), rgba(15,23,42,0.00));
    box-shadow: 0 6px 18px rgba(2,6,23,0.06);
  }
  .scbot-kpi-row { display: flex; align-items: center; justify-content: space-between; gap: 10px; }
  .scbot-kpi-title { font-size: 12px; font-weight: 600; color: rgba(30,41,59,0.85); }
  .scbot-kpi-value { font-size: 22px; font-weight: 800; color: rgba(2,6,23,0.92); letter-spacing: 0.2px; }
  .scbot-kpi-sub { margin-top: 4px; font-size: 11px; color: rgba(51,65,85,0.75); }
  .scbot-kpi-ic { width: 30px; height: 30px; border-radius: 10px; display:flex; align-items:center; justify-content:center;
                  background: rgba(59,130,246,0.10); border: 1px solid rgba(59,130,246,0.18); font-size: 16px; }
</style>
        """,
        unsafe_allow_html=True,
    )

    def _n(key: str) -> str:
        v = kpis.get(key)
        try:
            return f"{int(v)}"
        except Exception:
            return "—"

    cards = [
        ("Total SCB", "🧱", _n("total_scb"), "From array_details (distinct inv_stn + inv + scb)"),
        ("Communicating SCB", "📡", _n("communicating_scb"), "Total − all eliminated SCBs"),
        ("Disconnected Strings", "🔌", _n("disconnected_strings"), "From insights classification"),
        ("Night Time Bad Value", "🌙", _n("night_time_bad_value"), "Eliminated via 02:00–04:00 sum(|v|) > 30"),
        ("Constant Value", "🧊", _n("constant_value"), "Eliminated (nunique == 1)"),
        ("Low Energy", "⚡", _n("low_energy"), "KPI-only: excludes NULL-only SCBs"),
        ("Bad Availability", "🕳️", _n("bad_availability"), "Eliminated (blank_ratio > 65%)"),
    ]

    html = ['<div class="scbot-kpi-wrap">']
    for title, ic, value, sub in cards:
        # IMPORTANT: Do not indent HTML here.
        # In Markdown, leading spaces make Streamlit treat this as a code block and show raw tags.
        html.append(
            f"<div class='scbot-kpi-card'>"
            f"<div class='scbot-kpi-row'>"
            f"<div>"
            f"<div class='scbot-kpi-title'>{title}</div>"
            f"<div class='scbot-kpi-value'>{value}</div>"
            f"</div>"
            f"<div class='scbot-kpi-ic'>{ic}</div>"
            f"</div>"
            f"<div class='scbot-kpi-sub'>{sub}</div>"
            f"</div>"
        )
    html.append("</div>")  # .scbot-kpi-wrap
    st.markdown("\n".join(html), unsafe_allow_html=True)


def _compute_scb_ot_peak_pipeline(
    *,
    df_raw: pd.DataFrame,
    site_name: str,
    table: str,
    from_date: date,
    to_date: date,
    scb_cols: list[str],
    db_path: str,
    progress_hook: Optional[Callable[[int, str], None]] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Optional[str]]:
    """
    Implements the UPDATED computation pipeline (strict, in order):
    - Long format conversion
    - SCB elimination rules (constant / invalid median)
    - Cell-level outlier nullification: value > 5 * median
    - Peak selection (max valid value; reject peaks > 5*median)
    - Normalize by array_details.string_num
    - Median benchmark across normalized values
    - Deviation %
    - Disconnected strings insight

    Returns: (result_df, remarks_df, abort_reason)
    """
    remarks: list[dict[str, object]] = []

    # UI-only hook (must not affect computation). If provided, it allows the caller to
    # animate a smooth, phase-based progress bar without changing outputs.
    def _ph(pct: int, msg: str) -> None:
        if progress_hook is not None:
            try:
                progress_hook(int(pct), str(msg))
            except Exception:
                pass

    base_cols = ["inv_stn_name", "inv_name"]
    if df_raw is None or df_raw.empty or not scb_cols:
        return (
            pd.DataFrame(columns=[*base_cols, "scb_name", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings", "scb_label"]),
            pd.DataFrame(columns=["severity", "scb_label", "message"]),
            "no_data",
        )

    # STEP 1 — NIGHT-TIME ELIMINATION (NEW, STRICT, ALL SITES)
    # This MUST run before any stats/median/peak/normalization/deviation/disconnected-strings logic.
    # NOTE: The main df_raw fetch is still 06:00–18:00 (unchanged). This step uses its own SQL query
    # (02:00–04:00) so it is not affected by operational-window filtering.
    night_bad_labels = _detect_night_time_bad_scbs(
        db_path=db_path,
        table=table,
        from_date=from_date,
        to_date=to_date,
        scb_cols=list(scb_cols),
    )
    if night_bad_labels:
        for lbl in sorted(night_bad_labels):
            remarks.append(
                {
                    "severity": "eliminated",
                    "scb_label": lbl,
                    "message": "SCB eliminated: night time bad value",
                }
            )
    _ph(20, "Applying elimination rules…")

    # STEP 2 — LONG FORMAT CONVERSION (unchanged)
    wide = df_raw.copy()
    for c in base_cols:
        wide[c] = wide[c].astype(str)

    long_df = wide.melt(
        id_vars=base_cols,
        value_vars=scb_cols,
        var_name="scb_name",
        value_name="scb_value",
    )
    long_df["scb_name"] = long_df["scb_name"].astype(str)
    long_df["scb_value"] = pd.to_numeric(long_df["scb_value"], errors="coerce")

    # Build the mandatory label early so eliminations/skips can be shown to the user.
    def _force_triplet_label(inv_stn: object, inv: object, scb: object) -> str:
        def _digits(x: object) -> Optional[str]:
            m = re.search(r"(\d+)", str(x).strip(), flags=re.IGNORECASE)
            return m.group(1) if m else None

        is_n = _digits(inv_stn)
        inv_n = _digits(inv)
        scb_n = _digits(scb)
        is_part = f"IS{is_n}" if is_n is not None else str(inv_stn).strip()
        inv_part = f"INV{inv_n}" if inv_n is not None else str(inv).strip()
        scb_part = f"SCB{scb_n}" if scb_n is not None else str(scb).strip()
        return f"{is_part}-{inv_part}-{scb_part}"

    long_df["scb_label"] = long_df.apply(lambda r: _force_triplet_label(r["inv_stn_name"], r["inv_name"], r["scb_name"]), axis=1)

    # Remove night-eliminated SCBs BEFORE any downstream computations (stats, elimination rules, peaks, etc).
    if night_bad_labels:
        long_df = long_df[~long_df["scb_label"].isin(list(night_bad_labels))].copy()

    # Group stats per SCB
    gcols = ["inv_stn_name", "inv_name", "scb_name", "scb_label"]
    stats = (
        long_df.groupby(gcols, dropna=False)["scb_value"]
        .agg(
            n_nonnull=lambda s: int(pd.to_numeric(s, errors="coerce").dropna().shape[0]),
            nunique=lambda s: int(pd.to_numeric(s, errors="coerce").dropna().nunique()),
            # RULE 3.2 uses SCB SUM computed from RAW values (before outlier removal / peaks).
            scb_sum=lambda s: float(pd.to_numeric(s, errors="coerce").dropna().sum()),
            scb_median=lambda s: _median_s(pd.to_numeric(s, errors="coerce")),
        )
        .reset_index()
    )
    _ph(28, "Evaluating SCB quality checks…")

    # STEP 3 — SCB ELIMINATION RULES (NEW)
    # RULE 3.1 Constant/flat
    const_mask = (stats["n_nonnull"] > 0) & (stats["nunique"] == 1)
    for lbl in stats.loc[const_mask, "scb_label"].tolist():
        remarks.append({"severity": "eliminated", "scb_label": lbl, "message": "SCB eliminated: value constant throughout selected time window"})

    # RULE 3.2 (REPLACED) — LOW ENERGY / INVALID SCB
    # Compute scb_sum from RAW values (SQL time filtering already applied). If scb_sum <= 10, eliminate the SCB entirely.
    scb_sum_num = pd.to_numeric(stats["scb_sum"], errors="coerce").fillna(0.0)
    low_energy_mask = scb_sum_num <= 10.0
    for lbl in stats.loc[low_energy_mask, "scb_label"].tolist():
        if lbl not in set(stats.loc[const_mask, "scb_label"].tolist()):
            remarks.append(
                {
                    "severity": "eliminated",
                    "scb_label": lbl,
                    "message": "SCB eliminated: very low energy contribution (sum ≤ 10), treated as invalid SCB",
                }
            )

    # RULE 3.3 — INSUFFICIENT DATA AVAILABILITY (BAD DATA SCB)
    # Evaluate AFTER 3.1 and 3.2, and BEFORE any outlier nullification / peak selection / normalization.
    #
    # Operational window is STRICT: 06:00–18:00.
    # BLANK means NULL/missing only. Zero is a valid value and must NOT be treated as blank.
    #
    # expected slots are derived from the inferred native sampling interval.
    # blank_ratio = blank_count / total_expected_slots
    # eliminate if blank_ratio > 0.65
    eligible_for_33 = stats.loc[~(const_mask | low_energy_mask), gcols].copy()
    bad_avail_labels: set[str] = set()
    # PERFORMANCE: reuse 06:00–18:00 operational data + its long-form melt for both RULE 3.3 and DA%.
    op_df_0618: Optional[pd.DataFrame] = None
    op_long_0618: Optional[pd.DataFrame] = None
    if not eligible_for_33.empty:
        # Pull operational-window data via SQL (keeps time filtering at SQL level).
        op_df = fetch_scb_operational_data(
            db_path=db_path,
            table=table,
            from_date=from_date,
            to_date=to_date,
            scb_cols=tuple(scb_cols),
            start_time="06:00",
            end_time="18:00",
        )
        op_df_0618 = op_df
        # If no operational-window rows exist, we cannot compute availability reliably.
        # In that case we do not eliminate by 3.3 (avoid false positives).
        if op_df is not None and not op_df.empty:
            interval_sec = _infer_sampling_interval_seconds_from_table(db_path, table, from_date, to_date)
            if interval_sec is not None and interval_sec > 0:
                slots_per_day = _expected_slots_per_day(int(interval_sec))
                # Distinct operational dates observed in data
                try:
                    op_dates = pd.to_datetime(op_df["date"], errors="coerce", dayfirst=True).dt.date.dropna().unique().tolist()
                except Exception:
                    op_dates = []
                n_days = int(len(op_dates))
                total_expected_slots = int(slots_per_day * n_days) if n_days > 0 else 0

                if total_expected_slots > 0:
                    # Long form in 06:00–18:00; count non-blank per SCB.
                    # PERFORMANCE: melt once and reuse later for DA%.
                    op_w = op_df.copy()
                    for c in ["date", "timestamp", "inv_stn_name", "inv_name"]:
                        if c in op_w.columns:
                            op_w[c] = op_w[c].astype(str)
                    op_long = op_w.melt(
                        id_vars=["date", "timestamp", "inv_stn_name", "inv_name"],
                        value_vars=scb_cols,
                        var_name="scb_name",
                        value_name="scb_value",
                    )
                    op_long["scb_name"] = op_long["scb_name"].astype(str)
                    op_long["scb_value"] = pd.to_numeric(op_long["scb_value"], errors="coerce")
                    op_long["scb_label"] = op_long.apply(lambda r: _force_triplet_label(r["inv_stn_name"], r["inv_name"], r["scb_name"]), axis=1)
                    op_long_0618 = op_long

                    nonblank = (
                        op_long.groupby(["inv_stn_name", "inv_name", "scb_name", "scb_label"], dropna=False)["scb_value"]
                        .apply(lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum()))
                        .reset_index(name="nonblank_count")
                    )

                    # Only evaluate 3.3 for SCBs that survived 3.1 & 3.2
                    eval_df = eligible_for_33.merge(nonblank, on=gcols, how="left")
                    eval_df["nonblank_count"] = pd.to_numeric(eval_df["nonblank_count"], errors="coerce").fillna(0).astype(int)
                    eval_df["blank_count"] = total_expected_slots - eval_df["nonblank_count"]
                    eval_df["blank_ratio"] = eval_df["blank_count"].astype(float) / float(total_expected_slots)
                    bad = eval_df[eval_df["blank_ratio"] > 0.65]
                    bad_avail_labels = set(bad["scb_label"].tolist())
                    for lbl in sorted(bad_avail_labels):
                        remarks.append(
                            {
                                "severity": "eliminated",
                                "scb_label": lbl,
                                "message": "SCB eliminated: data missing for more than 65% of the operational window (06:00–18:00)",
                            }
                        )

    # Keep only eligible SCBs
    eligible = stats.loc[~(const_mask | low_energy_mask)].copy()
    if bad_avail_labels:
        eligible = eligible[~eligible["scb_label"].isin(list(bad_avail_labels))].copy()
    if eligible.empty:
        return (
            pd.DataFrame(columns=[*base_cols, "scb_name", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings", "scb_label"]),
            pd.DataFrame(remarks, columns=["severity", "scb_label", "message"]),
            "all_eliminated",
        )

    _ph(35, "Eliminations complete. Computing peaks & deviations…")

    # Join median back onto long_df for outlier nullification / peak selection
    long_ok = long_df.merge(
        eligible[gcols + ["scb_median"]],
        on=gcols,
        how="inner",
    )
    long_ok["scb_median"] = pd.to_numeric(long_ok["scb_median"], errors="coerce")

    # STEP 4 — CELL-LEVEL OUTLIER REMOVAL (UPDATED)
    # IMPORTANT CHANGE (strict):
    # Absolute cap only (no median multipliers here).
    # If scb_value > 1000 => nullify only that cell.
    outlier_mask = pd.to_numeric(long_ok["scb_value"], errors="coerce") > ABS_SCB_MAX
    long_ok.loc[outlier_mask, "scb_value"] = pd.NA

    # STEP 5 — PEAK SELECTION LOGIC (CORE CHANGE)
    peaks_rows: list[dict[str, object]] = []
    n_peak = int(getattr(eligible, "shape", [0])[0]) if eligible is not None else 0
    # More frequent progress updates (50 checkpoints) for smooth, continuous progress display
    step = max(1, n_peak // 50) if n_peak > 0 else 999999
    i_peak = 0
    for rr in eligible.itertuples(index=False):
        i_peak += 1
        # Keep progress moving during heavy peak selection (GSPL can be slow here).
        if (i_peak % step) == 0:
            pct = 35 + int((float(i_peak) / float(max(n_peak, 1))) * 25)  # 35 → 60
            _ph(pct, "Selecting SCB peaks…")
        inv_stn = getattr(rr, "inv_stn_name")
        inv = getattr(rr, "inv_name")
        scb = getattr(rr, "scb_name")
        lbl = getattr(rr, "scb_label")
        scb_median = float(getattr(rr, "scb_median"))

        s = long_ok[(long_ok["inv_stn_name"] == inv_stn) & (long_ok["inv_name"] == inv) & (long_ok["scb_name"] == scb)]["scb_value"]
        vals = pd.to_numeric(s, errors="coerce").dropna().sort_values(ascending=False).tolist()

        chosen: Optional[float] = None
        for v in vals:
            # Revised peak rejection rule (strict):
            # Reject peaks > 1000, otherwise accept.
            if float(v) <= ABS_SCB_MAX:
                chosen = float(v)
                break

        if chosen is None:
            remarks.append({"severity": "eliminated", "scb_label": lbl, "message": "SCB eliminated: all peak values exceed 1000"})
            continue

        peaks_rows.append(
            {
                "inv_stn_name": inv_stn,
                "inv_name": inv,
                "scb_name": scb,
                "scb_label": lbl,
                "scb_median": scb_median,
                "scb_peak": chosen,
            }
        )

    peaks = pd.DataFrame(peaks_rows)
    if peaks.empty:
        return (
            pd.DataFrame(columns=[*base_cols, "scb_name", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings", "scb_label"]),
            pd.DataFrame(remarks, columns=["severity", "scb_label", "message"]),
            "no_peaks",
        )

    # STEP 6 — STRING-BASED NORMALIZATION (UPDATED)
    _ph(62, "Normalizing & computing median benchmark…")
    # Join string_num from array_details
    string_df = get_array_string_nums(db_path, site_name)
    if string_df.empty:
        # If mapping is missing entirely, all SCBs will be skipped as invalid string_num.
        for lbl in peaks["scb_label"].tolist():
            remarks.append({"severity": "skipped", "scb_label": lbl, "message": "SCB skipped: string_num missing or invalid"})
        return (
            pd.DataFrame(columns=[*base_cols, "scb_name", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings", "scb_label"]),
            pd.DataFrame(remarks, columns=["severity", "scb_label", "message"]),
            "no_string_num",
        )

    # Build join keys (case-insensitive, trimmed)
    peaks["site_name"] = site_name
    for c in ["site_name", "inv_stn_name", "inv_name", "scb_name"]:
        peaks[f"{c}_key"] = peaks[c].map(_norm_key)

    joined = peaks.merge(
        string_df[["site_name_key", "inv_stn_name_key", "inv_name_key", "scb_name_key", "string_num", "inv_unit_name"]],
        on=["site_name_key", "inv_stn_name_key", "inv_name_key", "scb_name_key"],
        how="left",
    )

    joined["string_num"] = pd.to_numeric(joined["string_num"], errors="coerce")
    invalid_sn = joined["string_num"].isna() | (joined["string_num"] <= 0)
    for lbl in joined.loc[invalid_sn, "scb_label"].tolist():
        remarks.append({"severity": "skipped", "scb_label": lbl, "message": "SCB skipped: string_num missing or invalid"})
    joined = joined.loc[~invalid_sn].copy()
    if joined.empty:
        return (
            pd.DataFrame(columns=[*base_cols, "scb_name", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings", "scb_label"]),
            pd.DataFrame(remarks, columns=["severity", "scb_label", "message"]),
            "all_skipped_string_num",
        )

    joined["normalized_value"] = pd.to_numeric(joined["scb_peak"], errors="coerce") / pd.to_numeric(joined["string_num"], errors="coerce")
    _ph(63, "Computing median benchmark…")

    # STEP 7 — MEDIAN BENCHMARK
    # Default (non-GSPL): global median across all surviving SCBs (unchanged behavior).
    # GSPL-only: unit-wise median grouped by (inv_stn_name, inv_name, inv_unit_name).
    is_gspl = str(site_name).strip().upper() == "GSPL"
    norm = pd.to_numeric(joined["normalized_value"], errors="coerce").dropna()
    if norm.empty:
        return (
            pd.DataFrame(columns=[*base_cols, "scb_name", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings", "scb_label"]),
            pd.DataFrame(remarks, columns=["severity", "scb_label", "message"]),
            "no_normalized",
        )

    if not is_gspl:
        if len(norm) == 1:
            median_value = float(norm.iloc[0])
            joined["median_value"] = median_value
            joined["deviation_pct"] = 0.0
        else:
            median_value = float(norm.median())
            if pd.isna(median_value) or median_value == 0:
                return (
                    pd.DataFrame(columns=[*base_cols, "scb_name", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings", "scb_label"]),
                    pd.DataFrame(remarks, columns=["severity", "scb_label", "message"]),
                    "median_zero",
                )
            joined["median_value"] = median_value
            # STEP 8 — DEVIATION CALCULATION (UNCHANGED FORMULA)
            joined["deviation_pct"] = ((joined["normalized_value"] / median_value) - 1.0) * 100.0
    else:
        # GSPL-specific unit-wise median benchmark (AFTER elimination + normalization).
        # Group keys: inv_stn_name + inv_name + inv_unit_name
        grp_keys = ["inv_stn_name", "inv_name", "inv_unit_name"]
        tmp = joined.copy()
        tmp["normalized_value"] = pd.to_numeric(tmp["normalized_value"], errors="coerce")

        group_n = (
            tmp.groupby(grp_keys, dropna=False)["normalized_value"]
            .size()
            .reset_index(name="_group_n")
        )
        group_median = (
            tmp.groupby(grp_keys, dropna=False)["normalized_value"]
            .median()
            .reset_index()
            .rename(columns={"normalized_value": "median_value"})
        )

        tmp = tmp.merge(group_n, on=grp_keys, how="left").merge(group_median, on=grp_keys, how="left")

        # Per-group edge cases (must match existing intent, but applied unit-wise):
        # - group size == 1 => deviation = 0.0 and median_value = normalized_value
        one_mask = pd.to_numeric(tmp["_group_n"], errors="coerce").fillna(0).astype(int) <= 1
        tmp.loc[one_mask, "median_value"] = tmp.loc[one_mask, "normalized_value"]
        tmp.loc[one_mask, "deviation_pct"] = 0.0

        # - median_value == 0 or NaN => abort computation for that group (drop its SCBs)
        med = pd.to_numeric(tmp["median_value"], errors="coerce")
        bad_med = med.isna() | (med == 0)
        tmp = tmp.loc[~bad_med].copy()
        if tmp.empty:
            return (
                pd.DataFrame(columns=[*base_cols, "scb_name", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings", "scb_label"]),
                pd.DataFrame(remarks, columns=["severity", "scb_label", "message"]),
                "no_normalized",
            )

        many_mask = pd.to_numeric(tmp["_group_n"], errors="coerce").fillna(0).astype(int) > 1
        tmp.loc[many_mask, "deviation_pct"] = ((tmp.loc[many_mask, "normalized_value"] / tmp.loc[many_mask, "median_value"]) - 1.0) * 100.0

        # Clean up helper column
        joined = tmp.drop(columns=["_group_n"], errors="ignore")

    # STEP 9 — DISCONNECTED STRING CHECK (NEW INSIGHT)
    # disconnected_check = ((median_value * string_num) - SCB_PEAK) / median_value
    # IMPORTANT:
    # - Non-GSPL keeps the historical scalar median behavior.
    # - GSPL must use the SAME unit-wise median_value already attached per row (no new median computation).
    is_gspl = str(site_name).strip().upper() == "GSPL"
    _ph(65, "Computing deviations & fault signals…")
    if is_gspl:
        mv_series = pd.to_numeric(joined["median_value"], errors="coerce")
        disconnected_check = ((mv_series * pd.to_numeric(joined["string_num"], errors="coerce")) - pd.to_numeric(joined["scb_peak"], errors="coerce")) / mv_series
        disconnected_check = pd.to_numeric(disconnected_check, errors="coerce")
        joined["_disconnected_check"] = disconnected_check
        joined["disconnected_strings"] = joined["_disconnected_check"].apply(lambda x: int((x // 1)) if pd.notna(x) and float(x) > 1.0 else 0)
    else:
        mv = float(pd.to_numeric(joined["median_value"], errors="coerce").iloc[0]) if not joined.empty else 0.0
        if mv != 0:
            disconnected_check = ((mv * pd.to_numeric(joined["string_num"], errors="coerce")) - pd.to_numeric(joined["scb_peak"], errors="coerce")) / mv
            disconnected_check = pd.to_numeric(disconnected_check, errors="coerce")
            joined["_disconnected_check"] = disconnected_check
            joined["disconnected_strings"] = joined["_disconnected_check"].apply(lambda x: int((x // 1)) if pd.notna(x) and float(x) > 1.0 else 0)
        else:
            joined["disconnected_strings"] = 0

    for r in joined.itertuples(index=False):
        lbl = getattr(r, "scb_label")
        ds = int(getattr(r, "disconnected_strings"))
        if ds > 0:
            remarks.append({"severity": "insight", "scb_label": lbl, "message": f"Possible disconnected strings detected: {ds} (sum = {int(ds)})"})

    # STEP 10 — DATA AVAILABILITY % (DA%) (NEW, OBSERVATIONAL)
    # IMPORTANT:
    # - DA% is computed ONLY for eligible SCBs (those in the final output).
    # - Uses SQL-level 06:00–18:00 filtering as required.
    # - Must not change elimination/deviation logic (purely an added column).
    _ph(75, "Calculating Data Availability (DA%)…")
    da_map: dict[str, str] = {}
    try:
        # PERFORMANCE: reuse 06:00–18:00 df + melt if already available from RULE 3.3.
        op_df = op_df_0618
        if op_df is None:
            op_df = fetch_scb_operational_data(
                db_path=db_path,
                table=table,
                from_date=from_date,
                to_date=to_date,
                scb_cols=tuple(scb_cols),
                start_time="06:00",
                end_time="18:00",
            )
        total_ts = 0
        if op_df is not None and not op_df.empty and "date" in op_df.columns and "timestamp" in op_df.columns:
            total_ts = (
                op_df[["date", "timestamp"]]
                .astype(str)
                .dropna()
                .drop_duplicates()
                .shape[0]
            )

        eligible_labels = set(joined["scb_label"].astype(str).tolist())
        _ph(78, "Computing availability for each SCB…")
        if total_ts > 0 and eligible_labels:
            op_long = op_long_0618
            if op_long is None:
                op_w = op_df.copy()
                for c in ["date", "timestamp", "inv_stn_name", "inv_name"]:
                    if c in op_w.columns:
                        op_w[c] = op_w[c].astype(str)
                op_long = op_w.melt(
                    id_vars=["date", "timestamp", "inv_stn_name", "inv_name"],
                    value_vars=list(scb_cols),
                    var_name="scb_name",
                    value_name="scb_value",
                )
                op_long["scb_name"] = op_long["scb_name"].astype(str)
                op_long["scb_value"] = pd.to_numeric(op_long["scb_value"], errors="coerce")
                op_long["scb_label"] = op_long.apply(lambda r: _force_triplet_label(r["inv_stn_name"], r["inv_name"], r["scb_name"]), axis=1)

            # available = scb_value > 0 (strict)
            ok = op_long[(op_long["scb_label"].isin(eligible_labels)) & (pd.to_numeric(op_long["scb_value"], errors="coerce") > 0)].copy()
            ok["_dt"] = ok["date"].astype(str) + "|" + ok["timestamp"].astype(str)
            avail = ok.groupby("scb_label", dropna=False)["_dt"].nunique()

            for lbl in eligible_labels:
                a = int(avail.get(lbl, 0))
                da = (float(a) / float(total_ts)) * 100.0 if total_ts > 0 else 0.0
                da_map[str(lbl)] = f"{da:.2f}%"
        else:
            for lbl in eligible_labels:
                da_map[str(lbl)] = "0.00%"
    except Exception:
        # Fail-safe: never crash computation if DA% cannot be computed.
        da_map = {}

    joined["DA%"] = joined["scb_label"].astype(str).map(da_map).fillna("0.00%")
    _ph(85, "Insights ready. Rendering results…")

    out = joined[[*base_cols, "scb_name", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings", "scb_label", "DA%"]].copy()
    remarks_df = pd.DataFrame(remarks, columns=["severity", "scb_label", "message"])
    return out, remarks_df, None


def render_scb_ot(db_path: str) -> None:
    st.markdown("## SCB OT (SCB Operation Theatre)")
    st.caption("Read-only SCB deviation analysis (06:00–18:00 window). No DB writes.")

    # Username-based access control (UI/data-filter driven)
    from access_control import allowed_sites_for_user, is_admin, is_restricted_user

    username = st.session_state.get("user_info", {}).get("username")

    try:
        sites = list_sites_from_array_details(db_path)
    except Exception as e:
        st.error(f"Failed to read sites from array_details.site_name. {e}")
        return

    if not sites:
        st.info("No sites found in array_details.site_name.")
        return

    # Normalize site labels to prevent selection dropping on reruns (case/whitespace mismatches)
    sites = sorted({str(s).strip() for s in sites if str(s).strip()})

    allowed_sites = allowed_sites_for_user(username)
    if allowed_sites:
        allowed_l = {str(x).strip().lower() for x in allowed_sites}
        sites = [s for s in sites if str(s).strip().lower() in allowed_l]

    # Clamp previously selected sites (if any) onto current options so Streamlit doesn't drop the selection.
    prev_sel = st.session_state.get("scb_ot_site_multiselect", [])
    if isinstance(prev_sel, list) and prev_sel and sites:
        opt_by_lower = {s.lower(): s for s in sites}
        mapped = []
        for v in prev_sel:
            vv = str(v).strip()
            if vv in sites:
                mapped.append(vv)
            else:
                hit = opt_by_lower.get(vv.lower())
                if hit:
                    mapped.append(hit)
        st.session_state["scb_ot_site_multiselect"] = mapped

    def _reset_scb_ot_visual_state() -> None:
        """
        UI-only hard reset to prevent 'ghost' plots when switching sites.
        Does NOT touch computation logic; it only clears cached render outputs.
        """
        for k in [
            "scb_ot_last_fig",
            "scb_ot_last_table",
            "scb_ot_last_comments",
            "scb_ot_last_insights",
            "scb_ot_last_kpis",
            "scb_ot_last_meta",
        ]:
            try:
                st.session_state.pop(k, None)
            except Exception:
                pass
        st.session_state["scb_ot_plot_requested"] = False

    # UI REQUIREMENTS (STRICT):
    # - Site Name multiselect (MUST default empty)
    # - Threshold % (numeric, optional)
    # - Continuous Days (UI only, not used yet)
    # - From Date, To Date (with availability hints)
    # - Compute deviation ONLY on Plot Now
    # We render in two rows so date availability can be shown as soon as a site is selected.
    c_site, c_thr, c_n = st.columns([3.8, 1.8, 1.8], vertical_alignment="bottom")

    with c_site:
        # For restricted users: auto-select their single allowed site so Plot Now is enabled.
        if is_restricted_user(username) and sites:
            selected_sites = st.multiselect(
                "Site Name",
                options=sites,
                default=[sites[0]],
                disabled=True,
                key="scb_ot_site_multiselect",
            )
        else:
            # Admin behavior remains unchanged (empty by default).
            selected_sites = st.multiselect(
                "Site Name",
                options=sites,
                default=[],
                key="scb_ot_site_multiselect",
                on_change=_reset_scb_ot_visual_state,
            )

    # Defensive guard (safety net)
    if is_restricted_user(username):
        if len(selected_sites) > 0 and str(selected_sites[0]).strip().lower() != str(username).strip().lower():
            st.error("Unauthorized site access")
            st.stop()

    with c_thr:
        threshold_val = st.number_input(
            "Threshold (%)",
            value=-3.0,
            step=0.5,
            help="Deviation threshold. SCBs with deviation <= threshold will be shown.",
            key="scb_ot_threshold_number",
        )

    with c_n:
        # Kept for UI consistency with Operation Theatre (currently not used by SCB OT logic).
        default_n = st.session_state.get("scb_ot_continuous_days", 7)
        _n = st.number_input(
            "Continuous Days (N)",
            min_value=1,
            max_value=60,
            value=int(default_n),
            step=1,
            key="scb_ot_continuous_days_input",
        )
        st.session_state["scb_ot_continuous_days"] = int(_n)

    # Row 2: Date pickers + Plot button, with availability guidance once a site is selected.
    c_from, c_to, c_btn = st.columns([1.2, 1.2, 1.0], vertical_alignment="bottom")

    # Resolve site context early so the date picker can show availability hints BEFORE Plot Now.
    site_name: Optional[str] = None
    table: Optional[str] = None
    scb_cols: list[str] = []
    dmin: Optional[date] = None
    dmax: Optional[date] = None
    available_dates: set[date] = set()

    if len(selected_sites) == 1:
        site_name = selected_sites[0]
        table = resolve_site_table_name(db_path, site_name)
        try:
            cols = get_table_columns(db_path, table) if table else []
        except Exception as e:
            st.error(f"Failed to inspect site table '{table}'. {e}")
            cols = []

        if cols:
            required_base = {"date", "timestamp", "inv_stn_name", "inv_name"}
            missing_base = sorted(required_base - {str(c) for c in cols})
            if missing_base:
                st.error(f"Site table '{table}' is missing required columns: {missing_base}")
            else:
                scb_cols = [c for c in cols if str(c).upper().startswith("SCB")]
                if not scb_cols:
                    st.error("No SCB columns found in the selected site table.")
                else:
                    try:
                        dmin, dmax = _date_bounds_for_site_table(db_path, table)
                    except Exception:
                        dmin, dmax = None, None
                    try:
                        available_dates = _available_dates_for_site_table(db_path, table, tuple(scb_cols))
                    except Exception:
                        available_dates = set()

    with c_from:
        default_d1 = st.session_state.get("scb_ot_from_date", None)
        d1 = st.date_input(
            "From",
            value=default_d1,
            min_value=dmin if dmin else None,
            max_value=dmax if dmax else None,
            key="scb_ot_from",
            format="YYYY-MM-DD",
        )
        st.session_state["scb_ot_from_date"] = d1

    with c_to:
        default_d2 = st.session_state.get("scb_ot_to_date", None)
        d2 = st.date_input(
            "To",
            value=default_d2,
            min_value=dmin if dmin else None,
            max_value=dmax if dmax else None,
            key="scb_ot_to",
            format="YYYY-MM-DD",
        )
        st.session_state["scb_ot_to_date"] = d2

    # Plot Now (queued) — ensures progress bar mounts BEFORE heavy compute
    if "scb_ot_plot_requested" not in st.session_state:
        st.session_state["scb_ot_plot_requested"] = False

    def _on_plot_now() -> None:
        st.session_state["scb_ot_plot_requested"] = True

    with c_btn:
        # Mandatory: disabled until at least one site is selected.
        st.button(
            "Plot Now",
            type="primary",
            disabled=(len(selected_sites) == 0),
            on_click=_on_plot_now,
            width="stretch",
            key="scb_ot_plot_now_btn",
        )
    # Availability guidance (informational, not restrictive).
    if len(selected_sites) == 0:
        st.caption("Select a Site Name to see which dates have SCB data (06:00–18:00).")
    elif len(selected_sites) != 1:
        st.warning("Select exactly one Site Name to enable date availability hints and plotting.")
    else:
        if available_dates:
            ad_min = min(available_dates)
            ad_max = max(available_dates)
            st.caption(f"Available SCB data (06:00–18:00): **{ad_min.isoformat()} → {ad_max.isoformat()}**")
            if d1 not in available_dates:
                st.warning("Selected From date has no SCB data between 06:00–18:00")
            if d2 not in available_dates:
                st.warning("Selected To date has no SCB data between 06:00–18:00")
        elif dmin and dmax:
            st.caption(f"Table date bounds: {dmin.isoformat()} → {dmax.isoformat()}")
            st.caption("No available dates detected for SCB data (06:00–18:00). You can still pick dates, but results may be empty.")
        else:
            st.caption("No date information could be derived for the selected site table.")

    def _render_cached_results() -> bool:
        """
        Render last computed SCB OT outputs without recomputation.
        Returns True if something was rendered.
        """
        fig_last = st.session_state.get("scb_ot_last_fig")
        dev_table_last = st.session_state.get("scb_ot_last_table")
        comments_last = st.session_state.get("scb_ot_last_comments")
        insight_last = st.session_state.get("scb_ot_last_insights")
        kpis_last = st.session_state.get("scb_ot_last_kpis") or {}
        meta_last = st.session_state.get("scb_ot_last_meta") or {}

        if fig_last is None:
            return False

        # Hard-reset behavior: if the currently selected site differs from the cached result's site,
        # do not show any old plots/tables (prevents ghost/overlap UI). User must click Plot Now again.
        try:
            cached_site = str(meta_last.get("site_name") or "").strip()
            current_site = str(selected_sites[0]).strip() if isinstance(selected_sites, list) and len(selected_sites) == 1 else ""
            if cached_site and current_site and cached_site.lower() != current_site.lower():
                _reset_scb_ot_visual_state()
                return False
        except Exception:
            pass

        site_lbl = meta_last.get("site_name") or ""
        d1_lbl = meta_last.get("from_date") or ""
        d2_lbl = meta_last.get("to_date") or ""
        thr_lbl = meta_last.get("threshold")
        thr_txt = f", Threshold={thr_lbl}%" if thr_lbl is not None else ""
        if site_lbl and d1_lbl and d2_lbl:
            st.caption(f"Showing last computed results: **{site_lbl}** ({d1_lbl} → {d2_lbl}{thr_txt}). Change filters and click **Plot Now** to refresh.")
        else:
            st.caption("Showing last computed results. Change filters and click **Plot Now** to refresh.")

        # KPI cards: presentational only, threshold-independent.
        if isinstance(kpis_last, dict) and kpis_last:
            st.markdown("### 📌 KPIs")
            _render_kpi_cards(kpis_last)

        if isinstance(insight_last, pd.DataFrame) and not insight_last.empty:
            st.markdown("### 🧠 SCB Health Insights")
            st.dataframe(insight_last, hide_index=True, width="stretch")

        # Keep heatmap BELOW insights (matches original SCB OT layout)
        st.plotly_chart(fig_last, width="stretch", key="scb_ot_last_fig")

        if isinstance(dev_table_last, pd.DataFrame) and not dev_table_last.empty:
            with st.expander("Show table", expanded=False):
                st.dataframe(dev_table_last, width="stretch", hide_index=True)

        st.markdown("### SCB Comments")
        # Cached KPI bar (so it doesn't disappear on tab switch)
        try:
            below_cnt = kpis_last.get("below_threshold")
            with_comments_cnt = kpis_last.get("with_comments")
            if below_cnt is not None and with_comments_cnt is not None:
                k1, k2 = st.columns(2)
                with k1:
                    st.metric("Total SCBs below threshold", f"{int(below_cnt)}")
                with k2:
                    st.metric("Total SCBs with comments", f"{int(with_comments_cnt)}")
        except Exception:
            pass

        if isinstance(comments_last, pd.DataFrame) and not comments_last.empty:
            try:
                import add_comments

                add_comments._render_aggrid_table(comments_last, key="scb_ot_scb_comments_cached", height=360)  # type: ignore[attr-defined]
            except Exception:
                st.dataframe(comments_last, width="stretch", hide_index=True)
        else:
            st.caption("No cached SCB comments. Click **Plot Now** to fetch and compute comments for the selected window.")

        return True

    if not st.session_state.get("scb_ot_plot_requested"):
        if _render_cached_results():
            return
        st.caption("Click Plot Now to run the computation.")
        return

    # Consume flag (prevents double-run on reruns)
    st.session_state["scb_ot_plot_requested"] = False

    # Plot Now validation (strict)
    if len(selected_sites) != 1:
        st.warning("Select exactly one Site Name for SCB OT.")
        return
    if site_name is None or table is None or not scb_cols:
        st.warning("Selected site does not have a usable data table / SCB columns.")
        return

    from_date = d1
    to_date = d2
    if from_date > to_date:
        st.warning("From date must be <= To date.")
        return

    st.caption(f"Time window is enforced internally (SQL): **{TIME_START} onward** (data before is ignored).")

    import time

    # ENHANCED PROGRESS BAR with real-time percentage and time estimation
    # ====================================================================
    # Features:
    # - Real-time percentage display (1%, 2%, 3%... up to 100%)
    # - Time remaining in MM:SS format
    # - Smart capping for early-stage estimates
    # - Performance optimized (time updates every 0.5s)
    # - Clean single-line display (no shadow/overlay)
    prog_slot = st.empty()
    progress_bar = prog_slot.progress(0)

    current_pct = 0
    start_time = time.time()
    last_update_time = start_time

    def _delay(p: int) -> float:
        # Fast, responsive incremental progress (1%, 2%, 3%...) to show continuous activity.
        # Slight slowdown at the end for visual polish.
        if p >= 95:
            return 0.04  # Final stretch - slightly slower for intentional finish
        if p >= 90:
            return 0.03
        if p >= 80:
            return 0.025
        return 0.02  # Fast, responsive updates throughout most of the progress

    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS (e.g., 2:30, 1:15, 0:45)"""
        if seconds < 0:
            return "0:00"
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def _estimate_remaining_time(current_pct: int) -> str:
        """
        Estimate time remaining with smart capping.
        - Early stages (< 5%): Cap at 5 minutes to avoid wild estimates
        - Mid stages (5-10%): Cap at 3 minutes for stability
        - Later stages: No cap (estimates become accurate)
        """
        if current_pct <= 0:
            return "calculating..."
        
        elapsed = time.time() - start_time
        if elapsed < 1:  # Avoid division by very small numbers
            return "calculating..."
        
        # Estimate total time based on current progress
        estimated_total = (elapsed / current_pct) * 100
        remaining = estimated_total - elapsed
        
        # Smart capping for early estimates
        if current_pct < 5:
            remaining = min(remaining, 300)  # Cap at 5 minutes
        elif current_pct < 10:
            remaining = min(remaining, 180)  # Cap at 3 minutes
        
        return _format_time(remaining)

    def _animate_to(target: int, msg: str) -> None:
        nonlocal current_pct, last_update_time
        target = int(max(min(target, 100), 0))
        if target < current_pct:
            # Never move backwards.
            return
        
        # Enhanced progress display with percentage and time estimation
        for i in range(current_pct + 1, target + 1):
            # Update time estimate every 0.5 seconds for performance optimization
            current_time = time.time()
            if current_time - last_update_time >= 0.5 or i == target:
                time_remaining = _estimate_remaining_time(i)
                display_text = f"{msg} — {i}% • ⏱️ {time_remaining} remaining"
                last_update_time = current_time
            else:
                # Use previous time estimate to avoid recalculating every 1%
                time_remaining = _estimate_remaining_time(i)
                display_text = f"{msg} — {i}% • ⏱️ {time_remaining} remaining"
            
            progress_bar.progress(i, text=display_text)
            time.sleep(_delay(i))
        
        current_pct = target

    # Phase 1 — Data fetch (0 → 15)
    _animate_to(5, "Preparing SCB OT…")
    _animate_to(10, "Fetching SCB data (06:00–18:00)…")
    # PERFORMANCE: cached raw operational fetch (same SQL semantics; faster on reruns).
    df_raw = fetch_scb_operational_data(
        db_path=db_path,
        table=table,
        from_date=from_date,
        to_date=to_date,
        scb_cols=tuple(scb_cols),
        start_time=TIME_START,
        end_time=TIME_END,
    )
    if df_raw.empty:
        _animate_to(100, "No data found for the selected window.")
        st.info("No data found for the selected filters/time window.")
        prog_slot.empty()
        # status_text removed
        return

    _animate_to(15, "Data fetched. Applying elimination rules…")

    # Phase 2–4 are driven from inside the pipeline via progress_hook checkpoints.
    dev, remarks_df, abort_reason = _compute_scb_ot_peak_pipeline(
        df_raw=df_raw,
        site_name=site_name,
        table=table,
        from_date=from_date,
        to_date=to_date,
        scb_cols=list(scb_cols),
        db_path=db_path,
        progress_hook=_animate_to,
    )
    if abort_reason == "median_zero":
        _animate_to(100, "Aborted: median(normalized_value) is zero.")
        st.warning("Abort: median(normalized_value) is zero. Cannot compute deviations.")
        # Still show remarks if present (mandatory).
        if remarks_df is not None and not remarks_df.empty:
            st.markdown("### Remarks")
            st.dataframe(_style_remarks(remarks_df), width="stretch", hide_index=True)
        prog_slot.empty()
        # status_text removed
        return
    if dev is None or dev.empty:
        _animate_to(100, "No results (all SCBs eliminated/skipped).")
        st.warning("Unable to compute deviations (no SCBs remained after elimination/skip rules).")
        if remarks_df is not None and not remarks_df.empty:
            st.markdown("### Remarks")
            st.dataframe(_style_remarks(remarks_df), width="stretch", hide_index=True)
        prog_slot.empty()
        # status_text removed
        return

    # Phase 5 — Rendering (85 → 100)
    _animate_to(90, "Rendering results…")
    # Strict hierarchical sorting (IS -> INV -> SCB) before plotting.
    dev = dev.copy()
    dev["sort_key"] = dev["scb_label"].apply(_parse_scb_label)
    dev = dev.sort_values(["sort_key"], ascending=True).reset_index(drop=True)
    _animate_to(100, "SCB OT analysis completed.")
    prog_slot.empty()
    # FIXED: status_text removed, no need to empty it

    st.markdown("### Results")
    st.caption("Outliers are removed per SCB (cells nullified), peaks are selected per SCB, and normalization uses string_num (no DB writes).")

    # KPI computation layer (NEW, NON-INTRUSIVE)
    # IMPORTANT:
    # - KPIs are threshold-independent (threshold is only a plot filter).
    # - KPIs must not affect deviation/plot logic; they read from array_details + remarks only.
    def _compute_kpis_for_run() -> dict[str, int]:
        total_scb = _kpi_total_scb_count(db_path=db_path, site_name=str(site_name))

        if remarks_df is None or remarks_df.empty:
            return {
                "total_scb": int(total_scb),
                "communicating_scb": int(total_scb),
                "disconnected_strings": 0,
                "night_time_bad_value": 0,
                "constant_value": 0,
                "low_energy": 0,
                "bad_availability": 0,
            }

        tmp = remarks_df.copy()
        tmp["severity"] = tmp.get("severity", "").astype(str).str.lower()
        tmp["message"] = tmp.get("message", "").astype(str)
        msg_l = tmp["message"].astype(str).str.lower()

        eliminated_labels = set(tmp.loc[tmp["severity"] == "eliminated", "scb_label"].astype(str).tolist())

        def _labels(sev: str, substr: str) -> set[str]:
            m = (tmp["severity"] == sev.lower()) & msg_l.str.contains(substr.lower(), na=False)
            return set(tmp.loc[m, "scb_label"].astype(str).tolist())

        night_bad = _labels("eliminated", "night time bad value")
        const_bad = _labels("eliminated", "value constant throughout")
        low_energy = _labels("eliminated", "very low energy contribution")
        bad_avail = _labels("eliminated", "data missing for more than 65%")
        disconnected = _labels("insight", "possible disconnected strings detected")

        # Communicating SCB (AUTHORITATIVE):
        # TOTAL_SCB_SET comes ONLY from array_details (design level).
        # ELIMINATED_SCB_SET is the UNION of the four elimination categories below (set semantics, no double-count).
        total_set = _kpi_total_scb_label_set(db_path=db_path, site_name=str(site_name))
        eliminated_set = set(night_bad) | set(const_bad) | set(low_energy) | set(bad_avail)
        communicating_cnt = int(len(total_set - eliminated_set))

        # KPI 5 correction (KPI-only): exclude NULL-only SCBs from low-energy count.
        null_only_labels: set[str] = set()
        try:
            base_cols = ["inv_stn_name", "inv_name"]
            _w = df_raw.copy()
            for c in base_cols:
                _w[c] = _w[c].astype(str)
            _long = _w.melt(id_vars=base_cols, value_vars=list(scb_cols), var_name="scb_name", value_name="scb_value")
            _long["scb_name"] = _long["scb_name"].astype(str)
            _long["scb_value"] = pd.to_numeric(_long["scb_value"], errors="coerce")

            def _force_triplet_label(inv_stn: object, inv: object, scb: object) -> str:
                def _digits(x: object) -> Optional[str]:
                    m = re.search(r"(\d+)", str(x).strip(), flags=re.IGNORECASE)
                    return m.group(1) if m else None

                is_n = _digits(inv_stn)
                inv_n = _digits(inv)
                scb_n = _digits(scb)
                is_part = f"IS{is_n}" if is_n is not None else str(inv_stn).strip()
                inv_part = f"INV{inv_n}" if inv_n is not None else str(inv).strip()
                scb_part = f"SCB{scb_n}" if scb_n is not None else str(scb).strip()
                return f"{is_part}-{inv_part}-{scb_part}"

            _long["scb_label"] = _long.apply(lambda r: _force_triplet_label(r["inv_stn_name"], r["inv_name"], r["scb_name"]), axis=1)
            nn = (
                _long.groupby(["inv_stn_name", "inv_name", "scb_name", "scb_label"], dropna=False)["scb_value"]
                .apply(lambda s: int(pd.to_numeric(s, errors="coerce").notna().sum()))
                .reset_index(name="n_nonnull")
            )
            null_only_labels = set(nn.loc[pd.to_numeric(nn["n_nonnull"], errors="coerce").fillna(0).astype(int) == 0, "scb_label"].astype(str).tolist())
        except Exception:
            null_only_labels = set()

        low_energy_kpi = set(low_energy) - set(null_only_labels)

        return {
            "total_scb": int(total_scb),
            # Computed via TOTAL_SCB_SET − ELIMINATED_SCB_SET (independent of deviation/threshold/plots).
            "communicating_scb": communicating_cnt,
            "disconnected_strings": int(len(disconnected)),
            "night_time_bad_value": int(len(night_bad)),
            "constant_value": int(len(const_bad)),
            "low_energy": int(len(low_energy_kpi)),
            "bad_availability": int(len(bad_avail)),
        }

    kpis = _compute_kpis_for_run()
    st.markdown("### 📌 KPIs")
    _render_kpi_cards(kpis)

    # Persist KPIs early so they remain stable on threshold-only reruns (Plot Now not clicked).
    try:
        existing = st.session_state.get("scb_ot_last_kpis") or {}
        merged = dict(existing)
        merged.update(kpis)
        st.session_state["scb_ot_last_kpis"] = merged
    except Exception:
        pass

    st.markdown("### 🧠 SCB Health Insights")
    insight_df = build_scb_insight_summary(remarks_df)
    if insight_df.empty:
        st.info("No eliminations or anomalies detected for the selected window.")
    else:
        st.dataframe(insight_df, hide_index=True, width="stretch")

    # Remarks system (MANDATORY): eliminations, skips, insights
    # Requirement: remarks table must be collapsed by default, but still visible/accessible.
    if "remarks_df" in locals() and remarks_df is not None and not remarks_df.empty:
        # Inline summary (keeps page informative without forcing expand)
        sev_counts = remarks_df["severity"].fillna("").astype(str).str.lower().value_counts().to_dict()
        n_elim = int(sev_counts.get("eliminated", 0))
        n_skip = int(sev_counts.get("skipped", 0))
        n_ins = int(sev_counts.get("insight", 0))
        st.markdown(
            f"**Remarks summary:** "
            f":red[{n_elim} eliminated]  "
            f":orange[{n_skip} skipped]  "
            f":blue[{n_ins} insights]"
        )
        with st.expander("Remarks (why SCBs were eliminated/skipped + insights)", expanded=False):
            st.caption(
                "Elimination (red): SCB removed from computation. "
                "Missing data (amber): SCB skipped due to invalid/missing string_num. "
                "Insight (blue): possible disconnected strings detected."
            )
            # Tooltips: Streamlit doesn't support per-row hover tooltips in dataframes;
            # we include a dedicated Explanation column and header help text instead.
            if "message" in remarks_df.columns:
                rem = remarks_df.copy()
                rem["Explanation"] = rem["message"]
                rem = rem.drop(columns=["message"])
            else:
                rem = remarks_df.copy()
                rem["Explanation"] = ""
            st.dataframe(
                _style_remarks(rem),
                width="stretch",
                hide_index=True,
                column_config={
                    "severity": st.column_config.TextColumn(
                        "Type",
                        help="eliminated=red, skipped=amber, insight=blue",
                    ),
                    "scb_label": st.column_config.TextColumn(
                        "SCB",
                        help="Deterministic label: ISx-INVy-SCBz",
                    ),
                    "Explanation": st.column_config.TextColumn(
                        "Explanation",
                        help="Reason for elimination/skip, or insight interpretation.",
                    ),
                },
            )

            pass

    # Threshold behavior:
    # Threshold acts as FILTER for the diagnostic map.
    if threshold_val is None:
        dev_plot = dev.copy()
        title = f"SCB Deviation % — {site_name}"
    else:
        dev_plot = dev[dev["deviation_pct"] <= float(threshold_val)].copy()
        title = f"SCB Deviation % (≤ {threshold_val}%) — {site_name}"

    if dev_plot.empty:
        st.info("No SCBs meet the threshold filter.")
        with st.expander("Show full table (unfiltered)", expanded=False):
            show_cols = ["scb_label", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings", "DA%"]
            present = [c for c in show_cols if c in dev.columns]
            display_df = dev[present].drop(columns=["scb_median"], errors="ignore").copy()
            if "scb_label" in display_df.columns:
                display_df["_sort_key"] = display_df["scb_label"].map(_parse_scb_label)
                display_df = display_df.sort_values("_sort_key").drop(columns="_sort_key")
            st.dataframe(display_df, width="stretch", hide_index=True)
        return

    # -----------------------------
    # SCB Deviation Section (VISUALIZATION ONLY)
    # Uses the reusable helper _build_inverter_scb_diagnostic_map() which handles:
    # - Responsive column widths
    # - No overlapping text
    # - Top X-axis via annotation (Plotly constraint)
    # - Proper standoff for axis titles
    # - Zero-value bar filtering + descending sort
    #
    # IMPORTANT: Computation is sacrosanct. We ONLY use the already computed dev_plot.
    # -----------------------------

    with st.container(border=True):
        # Structural parsing from scb_label (MANDATORY)
        df = dev_plot[["scb_label", "deviation_pct", "disconnected_strings"]].copy()
        if df.empty:
            st.info("No SCB deviation or fault data available for the selected filters.")
        else:
            df["IS_num"] = pd.to_numeric(df["scb_label"].astype(str).str.extract(r"IS(\d+)", expand=False), errors="coerce")
            df["INV_num"] = pd.to_numeric(df["scb_label"].astype(str).str.extract(r"INV(\d+)", expand=False), errors="coerce")
            df["SCB_id"] = pd.to_numeric(df["scb_label"].astype(str).str.extract(r"SCB(\d+)", expand=False), errors="coerce")
            df["Block_Inv"] = df["scb_label"].astype(str).str.rsplit("-", n=1).str[0]

            # Keep only numeric SCB positions for the heatmap (SCBML etc. are excluded)
            df = df.dropna(subset=["IS_num", "INV_num", "SCB_id", "Block_Inv"]).copy()
            if df.empty:
                st.info("Diagnostic map needs numeric labels (ISx-INVy-SCBz). No numeric SCBs found after filtering.")
            else:
                df["IS_num"] = df["IS_num"].astype(int)
                df["INV_num"] = df["INV_num"].astype(int)
                df["SCB_id"] = df["SCB_id"].astype(int)

                df = df.sort_values(["IS_num", "INV_num", "SCB_id"])
                inverter_order = df["Block_Inv"].unique().tolist()
                scb_order = sorted(df["SCB_id"].unique().tolist())

                # Pivot for heatmap and faults
                pivot_dev = df.pivot(index="Block_Inv", columns="SCB_id", values="deviation_pct").reindex(inverter_order).reindex(columns=scb_order)
                pivot_disc = df.pivot(index="Block_Inv", columns="SCB_id", values="disconnected_strings").reindex(inverter_order).reindex(columns=scb_order)

                # Avg DA% per inverter (visualization only)
                # IMPORTANT: Must use final pipeline output (eligible SCBs only) and must not depend on threshold filtering.
                avg_da_by_inv: dict[str, float] = {}
                valid_scb_n_by_inv: dict[str, int] = {}
                try:
                    base_out = dev.copy()
                    base_out["Block_Inv"] = base_out["scb_label"].astype(str).str.rsplit("-", n=1).str[0]
                    da_num = (
                        base_out.get("DA%", pd.Series(dtype=str))
                        .astype(str)
                        .str.replace("%", "", regex=False)
                    )
                    base_out["_da_num"] = pd.to_numeric(da_num, errors="coerce")
                    valid = base_out.dropna(subset=["Block_Inv", "_da_num"]).copy()
                    if not valid.empty:
                        avg_da_by_inv = valid.groupby("Block_Inv")["_da_num"].mean().to_dict()
                        valid_scb_n_by_inv = valid.groupby("Block_Inv")["_da_num"].count().astype(int).to_dict()
                except Exception:
                    avg_da_by_inv = {}
                    valid_scb_n_by_inv = {}

                # Build responsive diagnostic map using the reusable helper
                fig = _build_inverter_scb_diagnostic_map(
                    pivot_dev=pivot_dev,
                    pivot_disc=pivot_disc,
                    inverter_order=inverter_order,
                    scb_order=scb_order,
                    title=title,
                    avg_da_by_inverter=avg_da_by_inv,
                    valid_scb_count_by_inverter=valid_scb_n_by_inv,
                )
                st.plotly_chart(fig, width="stretch")

    with st.expander("Show table"):
        show_cols = ["scb_label", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings", "DA%"]
        present = [c for c in show_cols if c in dev_plot.columns]
        display_df = dev_plot[present].drop(columns=["scb_median"], errors="ignore").copy()
        if "scb_label" in display_df.columns:
            display_df["_sort_key"] = display_df["scb_label"].map(_parse_scb_label)
            display_df = display_df.sort_values("_sort_key").drop(columns="_sort_key")
        st.dataframe(display_df, width="stretch", hide_index=True)

    # -----------------------------
    # SCB Comments (Supabase) — shown BELOW the SCB OT show table
    # -----------------------------
    st.markdown("### SCB Comments")
    prog2 = st.progress(0, text="Fetching SCB comments…")
    comments_df = pd.DataFrame()
    try:
        import add_comments  # reuse existing Supabase logic + AgGrid config

        # Avoid cached "no comments" results; comments must reflect Supabase as source of truth.
        try:
            add_comments.fetch_comments_live.clear()  # type: ignore[attr-defined]
        except Exception:
            pass

        comments_df = add_comments.fetch_comments_live(site_name=str(site_name), start_date=from_date, end_date=to_date, limit=2000)
    except Exception:
        comments_df = pd.DataFrame()
    prog2.progress(1.0, text="SCB comments ready")

    # Filter SCB-only comments (equipment_names contains SCB label)
    scb_comments = pd.DataFrame()
    if comments_df is not None and not comments_df.empty:
        tmp = comments_df.copy()
        # explode equipment_names into a single equipment_names cell per row
        if "equipment_names" in tmp.columns:
            tmp = tmp.explode("equipment_names")
        else:
            tmp["equipment_names"] = ""
        tmp["equipment_names"] = tmp["equipment_names"].astype(str)
        tmp = tmp[tmp["equipment_names"].str.contains("SCB", case=False, na=False)].copy()

        # normalize dates for display
        if "start_date" in tmp.columns:
            # AgGrid renders Python date objects as [object Object] in some setups; use ISO strings for clean UI.
            tmp["start_date"] = pd.to_datetime(tmp["start_date"], errors="coerce").dt.date.astype(str)
        if "end_date" in tmp.columns:
            tmp["end_date"] = pd.to_datetime(tmp["end_date"], errors="coerce").dt.date.astype(str)

        # reasons is typically a list; keep as string for clean UI while preserving content
        if "reasons" in tmp.columns:
            tmp["reasons"] = tmp["reasons"].apply(lambda x: ", ".join([str(v) for v in x]) if isinstance(x, list) else ("" if x is None else str(x)))

        # Required display columns (exact order)
        cols_order = ["site_name", "deviation", "start_date", "end_date", "equipment_names", "reasons", "remarks"]
        present_cols = [c for c in cols_order if c in tmp.columns]
        scb_comments = tmp[present_cols].copy()
        # Deterministic sort by SCB label (hierarchical)
        if "equipment_names" in scb_comments.columns:
            scb_comments["_sort_key"] = scb_comments["equipment_names"].map(_parse_scb_label)
            scb_comments = scb_comments.sort_values("_sort_key").drop(columns="_sort_key")

    # KPI bar (SCB-count-based)
    try:
        if threshold_val is None:
            dev_below = dev.copy()
        else:
            dev_below = dev[dev["deviation_pct"] <= float(threshold_val)].copy()
        below_set = set(dev_below["scb_label"].astype(str).tolist()) if not dev_below.empty and "scb_label" in dev_below.columns else set()
        comment_set = set(scb_comments["equipment_names"].astype(str).tolist()) if scb_comments is not None and not scb_comments.empty and "equipment_names" in scb_comments.columns else set()
        k1, k2 = st.columns(2)
        with k1:
            st.metric("Total SCBs below threshold", f"{len(below_set)}")
        with k2:
            st.metric("Total SCBs with comments", f"{len(below_set.intersection(comment_set))}")
        # Merge into existing KPI dict (cards are independent from this bar).
        existing = st.session_state.get("scb_ot_last_kpis") or {}
        merged = dict(existing)
        merged.update(
            {
                "below_threshold": len(below_set),
                "with_comments": len(below_set.intersection(comment_set)),
            }
        )
        st.session_state["scb_ot_last_kpis"] = merged
    except Exception:
        pass

    if scb_comments is None or scb_comments.empty:
        st.info("No SCB comments found for the selected site and date range.")
    else:
        try:
            # AgGrid (same configuration as Add Comments)
            add_comments._render_aggrid_table(scb_comments, key="scb_ot_scb_comments", height=360)  # type: ignore[attr-defined]
        except Exception:
            st.dataframe(scb_comments, width="stretch", hide_index=True)

    # Persist state for tab-switch preservation (no recompute unless Plot Now clicked again)
    try:
        st.session_state["scb_ot_last_fig"] = fig
    except Exception:
        pass
    try:
        st.session_state["scb_ot_last_table"] = display_df
    except Exception:
        pass
    try:
        st.session_state["scb_ot_last_comments"] = scb_comments
    except Exception:
        pass
    try:
        st.session_state["scb_ot_last_insights"] = insight_df
    except Exception:
        pass
    try:
        st.session_state["scb_ot_last_meta"] = {
            "site_name": site_name,
            "from_date": str(from_date) if from_date else "",
            "to_date": str(to_date) if to_date else "",
            "threshold": threshold_val,
        }
    except Exception:
        pass


