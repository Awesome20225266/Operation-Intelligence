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

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from aws_duckdb import get_duckdb_connection


TIME_START = "06:00"
# Updated requirement: time filter is 06:00 onward (end of day). This remains enforced at SQL level.
TIME_END = "23:59:59"

# Absolute threshold used for:
# - Step 4: cell-level outlier nullification
# - Step 5: peak rejection
ABS_SCB_MAX = 1000.0


def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    # Centralized read-only connection (same approach used by dashboard.py)
    return get_duckdb_connection(db_local=db_path)


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
    """
    con = _connect(db_path)
    try:
        info = con.execute("pragma table_info('array_details')").fetchdf()
        cols = [str(x) for x in info.get("name", pd.Series(dtype=str)).tolist()]
        required = {"site_name", "inv_stn_name", "inv_name", "scb_name", "string_num"}
        missing = sorted(required - set(cols))
        if missing:
            raise RuntimeError(f"`array_details` is missing required columns: {missing}")

        df = con.execute(
            """
            select
              site_name,
              inv_stn_name,
              inv_name,
              scb_name,
              string_num
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
        string_df[["site_name_key", "inv_stn_name_key", "inv_name_key", "scb_name_key", "string_num"]],
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

    # STEP 7 — MEDIAN BENCHMARK (UNCHANGED LOGIC)
    norm = pd.to_numeric(joined["normalized_value"], errors="coerce").dropna()
    if norm.empty:
        return (
            pd.DataFrame(columns=[*base_cols, "scb_name", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings", "scb_label"]),
            pd.DataFrame(remarks, columns=["severity", "scb_label", "message"]),
            "no_normalized",
        )

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

    # STEP 9 — DISCONNECTED STRING CHECK (NEW INSIGHT)
    # disconnected_check = ((median_value * string_num) - SCB_PEAK) / median_value
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
            remarks.append({"severity": "insight", "scb_label": lbl, "message": f"Possible disconnected strings detected: {ds}"})

    out = joined[[*base_cols, "scb_name", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings", "scb_label"]].copy()
    remarks_df = pd.DataFrame(remarks, columns=["severity", "scb_label", "message"])
    return out, remarks_df, None


def render_scb_ot(db_path: str) -> None:
    st.markdown("## SCB OT (SCB Operation Theatre)")
    st.caption("Read-only SCB deviation analysis (06:00–18:00 window). No DB writes.")

    try:
        sites = list_sites_from_array_details(db_path)
    except Exception as e:
        st.error(f"Failed to read sites from array_details.site_name. {e}")
        return

    if not sites:
        st.info("No sites found in array_details.site_name.")
        return

    # UI REQUIREMENTS (STRICT):
    # - Site Name multiselect (MUST default empty)
    # - Threshold % (numeric, optional)
    # - Continuous Days (UI only, not used yet)
    # - From Date, To Date (with availability hints)
    # - Compute deviation ONLY on Plot Now
    # We render in two rows so date availability can be shown as soon as a site is selected.
    c_site, c_thr, c_n = st.columns([3.8, 1.8, 1.8], vertical_alignment="bottom")

    with c_site:
        # Mandatory: default selection must be EMPTY.
        selected_sites = st.multiselect("Site Name", options=sites, default=[], key="scb_ot_site_multiselect")

    with c_thr:
        default_threshold = st.session_state.get("scb_ot_threshold", "")
        thr_txt = st.text_input("Threshold (%) (optional)", value=default_threshold, placeholder="e.g. -3", key="scb_ot_threshold_input")
        st.session_state["scb_ot_threshold"] = thr_txt
        threshold_val: Optional[float]
        if thr_txt.strip() == "":
            threshold_val = None
        else:
            try:
                threshold_val = float(thr_txt)
            except Exception:
                threshold_val = None
                st.warning("Threshold must be a number.")

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

    with st.spinner("Fetching and processing SCB data…"):
        df_raw = _fetch_raw_scb_data(
            db_path=db_path,
            table=table,
            from_date=from_date,
            to_date=to_date,
            scb_cols=scb_cols,
        )

        if df_raw.empty:
            st.info("No data found for the selected filters/time window.")
            return

        dev, remarks_df, abort_reason = _compute_scb_ot_peak_pipeline(
            df_raw=df_raw,
            site_name=site_name,
            table=table,
            from_date=from_date,
            to_date=to_date,
            scb_cols=list(scb_cols),
            db_path=db_path,
        )
        if abort_reason == "median_zero":
            st.warning("Abort: median(normalized_value) is zero. Cannot compute deviations.")
            # Still show remarks if present (mandatory).
            if remarks_df is not None and not remarks_df.empty:
                st.markdown("### Remarks")
                st.dataframe(_style_remarks(remarks_df), width="stretch", hide_index=True)
            return
        if dev is None or dev.empty:
            st.warning("Unable to compute deviations (no SCBs remained after elimination/skip rules).")
            if remarks_df is not None and not remarks_df.empty:
                st.markdown("### Remarks")
                st.dataframe(_style_remarks(remarks_df), width="stretch", hide_index=True)
            return

        # Strict hierarchical sorting (IS -> INV -> SCB) before plotting.
        dev = dev.copy()
        dev["sort_key"] = dev["scb_label"].apply(_parse_scb_label)
        dev = dev.sort_values(["sort_key"], ascending=True).reset_index(drop=True)

    st.markdown("### Results")
    st.caption("Outliers are removed per SCB (cells nullified), peaks are selected per SCB, and normalization uses string_num (no DB writes).")

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

    # Threshold behavior (IMPORTANT CORRECTION):
    # Threshold acts as FILTER + RED COLOR.
    if threshold_val is None:
        dev_plot = dev.copy()
        bar_color = "#60a5fa"  # default
        title = f"SCB Deviation % — {site_name}"
    else:
        dev_plot = dev[dev["deviation_pct"] <= float(threshold_val)].copy()
        bar_color = "#ef4444"  # all displayed bars must be red
        title = f"SCB Deviation % (≤ {threshold_val}%) — {site_name}"

    # Preserve deterministic x-axis order
    x_order = dev_plot["scb_label"].tolist()

    if dev_plot.empty:
        st.info("No SCBs meet the threshold filter.")
        with st.expander("Show full table (unfiltered)", expanded=False):
            show_cols = ["scb_label", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings"]
            present = [c for c in show_cols if c in dev.columns]
            st.dataframe(dev[present], width="stretch", hide_index=True)
        return

    fig = px.bar(
        dev_plot,
        x="scb_label",
        y="deviation_pct",
        title=title,
        labels={"scb_label": "SCB (inv_stn_name-inv_name-SCBx)", "deviation_pct": "Deviation (%)"},
        category_orders={"scb_label": x_order},
    )
    fig.update_traces(marker_color=bar_color)
    fig.update_layout(height=520, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Show table"):
        show_cols = ["scb_label", "scb_median", "scb_peak", "string_num", "normalized_value", "median_value", "deviation_pct", "disconnected_strings"]
        present = [c for c in show_cols if c in dev_plot.columns]
        st.dataframe(dev_plot[present], width="stretch", hide_index=True)


