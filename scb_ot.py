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
from typing import Iterable, Optional

import numpy as np
import duckdb
import pandas as pd
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
    inv_disc_sum = inv_disc_sum[inv_disc_sum > 0].sort_values(ascending=False)

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
    disc_vals = pivot_disc.fillna(0).to_numpy()
    disc_int = disc_vals.astype(int, copy=False)
    text_vals = np.where(disc_int > 0, disc_int.astype(str), "")

    fig = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        column_widths=[heatmap_width_ratio, bar_width_ratio],
        horizontal_spacing=0.12,  # Increased gap to prevent text overlap
    )

    # Disable hover on "gaps" (NaN/None cells). This prevents tooltips for SCBs
    # that are not present after threshold filtering (or missing per inverter).
    z_vals = pd.to_numeric(pd.DataFrame(pivot_dev).to_numpy().ravel(), errors="coerce").reshape(pivot_dev.shape)

    fig.add_trace(
        go.Heatmap(
            z=z_vals,
            x=scb_order,
            y=inverter_order,
            colorscale="RdYlGn",
            zmid=0,
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
            hoverongaps=False,
            showscale=False,  # Remove colorbar completely
        ),
        row=1,
        col=1,
    )

    # -----------------------------
    # RIGHT: BAR CHART
    # -----------------------------
    if not inv_disc_sum.empty:
        fig.add_trace(
            go.Bar(
                x=inv_disc_sum.values,
                y=inv_disc_sum.index,
                orientation="h",
                marker=dict(color=[_severity_color(int(v)) for v in inv_disc_sum.values]),
                text=[int(v) for v in inv_disc_sum.values],
                textposition="outside",
                hovertemplate=(
                    "<span style='font-size:15px'><b>Inverter:</b> %{y}</span><br>"
                    "<span style='font-size:14px'><b>Total Disconnected Strings:</b> %{x}</span>"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=2,
        )

    # Axes configuration
    fig.update_xaxes(
        title=dict(text="<b>SCB Number (Position)</b>", standoff=12, font=dict(size=16)),
        automargin=True,
        row=1,
        col=1,
    )

    # Left heatmap: Y-axis
    fig.update_yaxes(
        title=dict(text="<b>Inverter (IS–INV)</b>", standoff=30),
        autorange="reversed",
        automargin=True,
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
    bold_ticks = [f"<b>{y}</b>" for y in inverter_order]
    fig.update_yaxes(tickfont=dict(size=13, color="black"), row=1, col=1)
    fig.update_yaxes(tickfont=dict(size=13, color="black"), row=1, col=2)
    fig.update_layout(
        yaxis=dict(ticktext=bold_ticks, tickvals=inverter_order),
        yaxis2=dict(ticktext=bold_ticks, tickvals=inverter_order),
    )

    # Right bar chart: Bottom X-axis
    fig.update_xaxes(
        title=dict(text="<b>Total Disconnected Strings</b>", standoff=12, font=dict(size=16)),
        automargin=True,
        row=1,
        col=2,
    )

    # Top X-axis labels removed per user request

    # Layout safety net
    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.01, font=dict(size=18)),
        template="plotly_white",
        height=calculated_height,
        margin=dict(l=80, r=50, t=110, b=bottom_margin),
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


def _compute_scb_ot_peak_pipeline(
    *,
    df_raw: pd.DataFrame,
    site_name: str,
    table: str,
    from_date: date,
    to_date: date,
    scb_cols: list[str],
    db_path: str,
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

    base_cols = ["inv_stn_name", "inv_name"]
    if df_raw is None or df_raw.empty or not scb_cols:
        return (
            pd.DataFrame(columns=[*base_cols, "scb_name", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings", "scb_label"]),
            pd.DataFrame(columns=["severity", "scb_label", "message"]),
            "no_data",
        )

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
    if not eligible_for_33.empty:
        # Pull operational-window data via SQL (keeps time filtering at SQL level).
        op_df = _fetch_raw_scb_data_in_time_window(
            db_path=db_path,
            table=table,
            from_date=from_date,
            to_date=to_date,
            scb_cols=scb_cols,
            start_time="06:00",
            end_time="18:00",
        )
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
                    op_w = op_df.copy()
                    for c in ["inv_stn_name", "inv_name"]:
                        op_w[c] = op_w[c].astype(str)
                    op_long = op_w.melt(
                        id_vars=["inv_stn_name", "inv_name"],
                        value_vars=scb_cols,
                        var_name="scb_name",
                        value_name="scb_value",
                    )
                    op_long["scb_name"] = op_long["scb_name"].astype(str)
                    op_long["scb_value"] = pd.to_numeric(op_long["scb_value"], errors="coerce")
                    op_long["scb_label"] = op_long.apply(lambda r: _force_triplet_label(r["inv_stn_name"], r["inv_name"], r["scb_name"]), axis=1)

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
    for rr in eligible.itertuples(index=False):
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

    out = joined[[*base_cols, "scb_name", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings", "scb_label"]].copy()
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

    allowed_sites = allowed_sites_for_user(username)
    if allowed_sites:
        allowed_l = {str(x).strip().lower() for x in allowed_sites}
        sites = [s for s in sites if str(s).strip().lower() in allowed_l]

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
            selected_sites = st.multiselect("Site Name", options=sites, default=[], key="scb_ot_site_multiselect")

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

    with c_btn:
        # Mandatory: disabled until at least one site is selected.
        plot_now = st.button("Plot Now", type="primary", disabled=(len(selected_sites) == 0), use_container_width=True)

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

    if not plot_now:
        st.caption("Click Plot Now to run the computation.")
        return

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

    prog = st.progress(0, text="🩺 Initializing SCB Operation Theatre…")
    time.sleep(0.3)

    prog.progress(0.12, text="📥 Loading inverter & SCB telemetry…")
    df_raw = _fetch_raw_scb_data(
        db_path=db_path,
        table=table,
        from_date=from_date,
        to_date=to_date,
        scb_cols=scb_cols,
    )
    time.sleep(0.3)

    if df_raw.empty:
        prog.progress(1.0, text="No data")
        st.info("No data found for the selected filters/time window.")
        return

    prog.progress(0.32, text="🧹 Cleaning raw SCB signals & removing noise…")
    time.sleep(0.2)

    prog.progress(0.52, text="⚙️ Applying SCB elimination & validation rules…")
    dev, remarks_df, abort_reason = _compute_scb_ot_peak_pipeline(
        df_raw=df_raw,
        site_name=site_name,
        table=table,
        from_date=from_date,
        to_date=to_date,
        scb_cols=list(scb_cols),
        db_path=db_path,
    )
    time.sleep(0.3)
    if abort_reason == "median_zero":
        prog.progress(1.0, text="Aborted")
        st.warning("Abort: median(normalized_value) is zero. Cannot compute deviations.")
        # Still show remarks if present (mandatory).
        if remarks_df is not None and not remarks_df.empty:
            st.markdown("### Remarks")
            st.dataframe(_style_remarks(remarks_df), width="stretch", hide_index=True)
        return
    if dev is None or dev.empty:
        prog.progress(1.0, text="No results")
        st.warning("Unable to compute deviations (no SCBs remained after elimination/skip rules).")
        if remarks_df is not None and not remarks_df.empty:
            st.markdown("### Remarks")
            st.dataframe(_style_remarks(remarks_df), width="stretch", hide_index=True)
        return

    prog.progress(0.72, text="📊 Computing deviation benchmarks & fault signals…")
    time.sleep(0.2)

    prog.progress(0.90, text="🎨 Preparing diagnostic visuals…")
    # Strict hierarchical sorting (IS -> INV -> SCB) before plotting.
    dev = dev.copy()
    dev["sort_key"] = dev["scb_label"].apply(_parse_scb_label)
    dev = dev.sort_values(["sort_key"], ascending=True).reset_index(drop=True)
    time.sleep(0.2)
    prog.progress(1.0, text="✅ SCB OT results ready")

    st.markdown("### Results")
    st.caption("Outliers are removed per SCB (cells nullified), peaks are selected per SCB, and normalization uses string_num (no DB writes).")

    # KPIs above remarks table (SCB-count-based, not reason-count-based)
    below_threshold_count = 0
    with_comments_count = 0

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
            show_cols = ["scb_label", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings"]
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

                # Build responsive diagnostic map using the reusable helper
                fig = _build_inverter_scb_diagnostic_map(
                    pivot_dev=pivot_dev,
                    pivot_disc=pivot_disc,
                    inverter_order=inverter_order,
                    scb_order=scb_order,
                    title=title,
                )

                st.plotly_chart(fig, width="stretch")

    with st.expander("Show table"):
        show_cols = ["scb_label", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings"]
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


