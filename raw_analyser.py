from __future__ import annotations

import time
from datetime import date
from typing import Optional

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from access_control import allowed_sites_for_user
from aws_duckdb import get_duckdb_connection


# -----------------------------------------------------------------------------
# Raw Analyser (read-only) — master.duckdb
# -----------------------------------------------------------------------------

TIME_START = "06:00"
TIME_END = "18:00"


def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    return get_duckdb_connection(db_local=db_path)


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sanitize_table_guess(site_name: str) -> str:
    # Requirement: table == selected_site_name.lower(), but we keep a safe sanitizer
    # for cases like "GSPL-GAP" -> "gspl_gap" while still being deterministic.
    s = str(site_name).strip().lower()
    out: list[str] = []
    prev_us = False
    for ch in s:
        if ch.isalnum():
            out.append(ch)
            prev_us = False
        else:
            if not prev_us:
                out.append("_")
                prev_us = True
    return "".join(out).strip("_") or s


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    try:
        con.execute(f"select 1 from {_quote_ident(table)} limit 1")
        return True
    except Exception:
        return False


def _sql_date_expr(col: str) -> str:
    """
    Support both DD-MM-YYYY (Apollo tables) and YYYY-MM-DD.
    Returns a SQL expression yielding DATE.
    """
    c = _quote_ident(col)
    return f"coalesce(try_cast({c} as date), try_strptime(cast({c} as varchar), '%d-%m-%Y')::date)"


def _sql_time_expr(col: str) -> str:
    c = _quote_ident(col)
    return (
        f"coalesce("
        f"try_cast({c} as time),"
        f"try_strptime(cast({c} as varchar), '%H:%M:%S')::time,"
        f"try_strptime(cast({c} as varchar), '%H:%M')::time"
        f")"
    )


def _num_suffix(s: str) -> int:
    """Natural sort helper: IS1 < IS2 < IS10, INV1 < INV2 < INV10, SCB1 < SCB2 < SCB10."""
    s = str(s)
    num = ""
    for ch in reversed(s):
        if ch.isdigit():
            num = ch + num
        else:
            break
    try:
        return int(num) if num else 10**9
    except Exception:
        return 10**9


def _sort_key_is_inv_unit_scb(inv_stn: str, inv: str, unit: str, scb: str) -> tuple:
    return (
        _num_suffix(inv_stn),
        str(inv_stn),
        _num_suffix(inv),
        str(inv),
        _num_suffix(unit or ""),
        str(unit or ""),
        _num_suffix(scb),
        str(scb),
    )


@st.cache_data(show_spinner=False)
def _list_sites(db_path: str) -> list[str]:
    con = _connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT DISTINCT trim(site_name) AS site_name
            FROM array_details
            WHERE site_name IS NOT NULL AND trim(site_name) != ''
            ORDER BY 1
            """
        ).fetchall()
        return [str(r[0]) for r in rows if r and r[0] is not None and str(r[0]).strip() != ""]
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def _list_inv_stations(db_path: str, site_name: str) -> list[str]:
    con = _connect(db_path)
    try:
        rows = con.execute(
            """
            SELECT DISTINCT trim(inv_stn_name) AS inv_stn_name
            FROM array_details
            WHERE lower(trim(site_name)) = lower(trim(?))
              AND inv_stn_name IS NOT NULL AND trim(inv_stn_name) != ''
            ORDER BY 1
            """,
            [site_name],
        ).fetchall()
        out = [str(r[0]) for r in rows if r and r[0] is not None and str(r[0]).strip() != ""]
        return sorted(out, key=lambda x: (_num_suffix(x), str(x)))
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def _list_inverters(db_path: str, site_name: str, inv_stations: tuple[str, ...]) -> list[str]:
    if not inv_stations:
        return []
    con = _connect(db_path)
    try:
        placeholders = ", ".join(["?"] * len(inv_stations))
        df = con.execute(
            f"""
            SELECT DISTINCT trim(inv_stn_name) AS inv_stn_name, trim(inv_name) AS inv_name
            FROM array_details
            WHERE lower(trim(site_name)) = lower(trim(?))
              AND trim(inv_stn_name) IN ({placeholders})
              AND inv_name IS NOT NULL AND trim(inv_name) != ''
            """,
            [site_name, *inv_stations],
        ).df()
        if df is None or df.empty:
            return []
        df["inv_stn_name"] = df["inv_stn_name"].astype(str)
        df["inv_name"] = df["inv_name"].astype(str)
        df["_k"] = df.apply(lambda r: (_num_suffix(r["inv_stn_name"]), str(r["inv_stn_name"]), _num_suffix(r["inv_name"]), str(r["inv_name"])), axis=1)
        df = df.sort_values("_k", ignore_index=True)
        return df["inv_name"].drop_duplicates().tolist()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def _list_units(
    db_path: str, site_name: str, inv_stations: tuple[str, ...], inverters: tuple[str, ...]
) -> list[str]:
    if not inverters:
        return []
    con = _connect(db_path)
    try:
        inv_stn_filter = ""
        params: list[str] = [site_name]
        if inv_stations:
            placeholders = ", ".join(["?"] * len(inv_stations))
            inv_stn_filter = f" AND trim(inv_stn_name) IN ({placeholders})"
            params.extend(list(inv_stations))

        inv_placeholders = ", ".join(["?"] * len(inverters))
        params.extend(list(inverters))
        rows = con.execute(
            f"""
            SELECT DISTINCT trim(inv_unit_name) AS inv_unit_name
            FROM array_details
            WHERE lower(trim(site_name)) = lower(trim(?))
              {inv_stn_filter}
              AND trim(inv_name) IN ({inv_placeholders})
              AND inv_unit_name IS NOT NULL AND trim(inv_unit_name) != ''
            ORDER BY 1
            """,
            params,
        ).fetchall()
        out = [str(r[0]) for r in rows if r and r[0] is not None and str(r[0]).strip() != ""]
        return sorted(out, key=lambda x: (_num_suffix(x), str(x)))
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def _list_scbs_with_display(
    db_path: str,
    site_name: str,
    inv_stations: tuple[str, ...],
    inverters: tuple[str, ...],
    units: tuple[str, ...],
) -> pd.DataFrame:
    """
    Returns a dataframe with columns:
    - inv_stn_name, inv_name, inv_unit_name, scb_name, display
    Display format depends on whether units were selected (CASE A/B).
    """
    if not inverters:
        return pd.DataFrame(columns=["inv_stn_name", "inv_name", "inv_unit_name", "scb_name", "display"])

    con = _connect(db_path)
    try:
        inv_stn_filter = ""
        params: list[str] = [site_name]
        if inv_stations:
            placeholders = ", ".join(["?"] * len(inv_stations))
            inv_stn_filter = f" AND trim(inv_stn_name) IN ({placeholders})"
            params.extend(list(inv_stations))

        inv_placeholders = ", ".join(["?"] * len(inverters))
        params.extend(list(inverters))

        unit_filter = ""
        if units:
            unit_placeholders = ", ".join(["?"] * len(units))
            unit_filter = f" AND trim(inv_unit_name) IN ({unit_placeholders})"
            params.extend(list(units))

        df = con.execute(
            f"""
            SELECT
              trim(inv_stn_name) AS inv_stn_name,
              trim(inv_name) AS inv_name,
              trim(inv_unit_name) AS inv_unit_name,
              trim(scb_name) AS scb_name
            FROM array_details
            WHERE lower(trim(site_name)) = lower(trim(?))
              {inv_stn_filter}
              AND trim(inv_name) IN ({inv_placeholders})
              {unit_filter}
              AND scb_name IS NOT NULL AND trim(scb_name) != ''
            """,
            params,
        ).df()
    finally:
        con.close()

    if df is None or df.empty:
        return pd.DataFrame(columns=["inv_stn_name", "inv_name", "inv_unit_name", "scb_name", "display"])

    if units:
        # CASE A: UNIT selected
        df["inv_unit_name"] = df["inv_unit_name"].where(df["inv_unit_name"].notna(), "")
        df["display"] = (
            df["inv_stn_name"].astype(str)
            + "-"
            + df["inv_name"].astype(str)
            + "-"
            + df["inv_unit_name"].astype(str)
            + "-"
            + df["scb_name"].astype(str)
        )
        df = df.drop_duplicates(subset=["inv_stn_name", "inv_name", "inv_unit_name", "scb_name", "display"]).copy()
    else:
        # CASE B: UNIT not selected
        # Force unit blank to avoid duplicate display labels across units
        df["inv_unit_name"] = ""
        df["display"] = df["inv_stn_name"].astype(str) + "-" + df["inv_name"].astype(str) + "-" + df["scb_name"].astype(str)
        df = df.drop_duplicates(subset=["inv_stn_name", "inv_name", "scb_name", "display"]).copy()

    df["_k"] = df.apply(
        lambda r: _sort_key_is_inv_unit_scb(
            str(r.get("inv_stn_name") or ""),
            str(r.get("inv_name") or ""),
            str(r.get("inv_unit_name") or ""),
            str(r.get("scb_name") or ""),
        ),
        axis=1,
    )
    df = df.sort_values("_k", ignore_index=True).drop(columns=["_k"])
    return df


@st.cache_data(show_spinner=False)
def _table_columns(db_path: str, table: str) -> list[str]:
    con = _connect(db_path)
    try:
        info = con.execute(f"pragma table_info({_quote_ident(table)})").fetchdf()
        cols = [str(x) for x in info.get("name", pd.Series(dtype=str)).tolist()]
        return cols
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def _table_date_bounds(db_path: str, table: str) -> tuple[Optional[date], Optional[date]]:
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
def _available_dates_for_site_table_06_18(
    db_path: str, table: str, scb_cols: tuple[str, ...]
) -> set[date]:
    """
    Availability definition (STRICT for Raw Analyser):
    A date D is "available" if there exists ANY timestamp between 06:00–18:00
    on that date where ANY SCB column has value > 0.
    """
    if not scb_cols:
        return set()

    date_expr = _sql_date_expr("date")
    time_expr = _sql_time_expr("timestamp")

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


@st.cache_data(show_spinner=False)
def _available_dates_for_selected_scbs_06_18(db_path: str, table: str, scb_cols: tuple[str, ...]) -> set[date]:
    # Strict: availability is based on selected SCB columns only.
    return _available_dates_for_site_table_06_18(db_path, table, scb_cols)


def _fetch_raw_timeseries(
    *,
    db_path: str,
    table: str,
    from_date: date,
    to_date: date,
    inv_stations: tuple[str, ...],
    inverters: tuple[str, ...],
    units: tuple[str, ...],
    scb_cols: tuple[str, ...],
) -> pd.DataFrame:
    if not scb_cols:
        return pd.DataFrame()

    # Force typed date/time via SQL expressions (prevents pandas inference warnings)
    date_expr = _sql_date_expr("date")
    time_expr = _sql_time_expr("timestamp")

    cols = ["inv_stn_name", "inv_name"]
    tbl_cols = set([c.lower() for c in _table_columns(db_path, table)])
    has_unit_col = "inv_unit_name" in tbl_cols
    if has_unit_col:
        cols.append("inv_unit_name")
    cols.extend(list(scb_cols))

    select_list = ", ".join([_quote_ident(c) for c in cols])

    where_parts: list[str] = [
        f"{date_expr} between ? and ?",
        f"{time_expr} between time '{TIME_START}' and time '{TIME_END}'",
    ]
    params: list[object] = [from_date, to_date]

    if inv_stations:
        placeholders = ", ".join(["?"] * len(inv_stations))
        where_parts.append(f"trim(inv_stn_name) in ({placeholders})")
        params.extend(list(inv_stations))

    if inverters:
        placeholders = ", ".join(["?"] * len(inverters))
        where_parts.append(f"trim(inv_name) in ({placeholders})")
        params.extend(list(inverters))

    if units and has_unit_col:
        placeholders = ", ".join(["?"] * len(units))
        where_parts.append(f"trim(inv_unit_name) in ({placeholders})")
        params.extend(list(units))

    sql = f"""
    select
      {date_expr} as date,
      {time_expr} as timestamp,
      {select_list}
    from {_quote_ident(table)}
    where {' and '.join(where_parts)}
    order by {date_expr}, {time_expr}
    """

    con = _connect(db_path)
    try:
        return con.execute(sql, params).fetchdf()
    finally:
        con.close()


def _build_plot(
    *,
    df: pd.DataFrame,
    site_name: str,
    display_map: dict[tuple[str, str, str, str], str],
    scb_cols: tuple[str, ...],
    string_num_map: Optional[dict[tuple[str, str, str, str], float]] = None,
    y_axis_title: str = "<b>Raw SCB Value</b>",
) -> go.Figure:
    """
    df is wide with SCB columns; we melt to long and plot one trace per SCB identifier.
    display_map key: (inv_stn_name, inv_name, inv_unit_name, scb_name)
    """
    if df is None or df.empty:
        fig = go.Figure()
        fig.update_layout(template="plotly_white", height=420, margin=dict(l=60, r=30, t=60, b=60))
        return fig

    # Combine date/time without ambiguous inference (SQL already casts them)
    x = pd.to_datetime(
        df["date"].astype(str) + " " + df["timestamp"].astype(str),
        errors="coerce",
        format="%Y-%m-%d %H:%M:%S",
    )
    if x.isna().any():
        x2 = pd.to_datetime(
            df["date"].astype(str) + " " + df["timestamp"].astype(str),
            errors="coerce",
            format="%Y-%m-%d %H:%M",
        )
        x = x.fillna(x2)

    base_cols = ["inv_stn_name", "inv_name"]
    unit_present = "inv_unit_name" in df.columns
    if unit_present:
        base_cols.append("inv_unit_name")
    tmp = df.copy()
    tmp["__x"] = x

    long = tmp.melt(
        id_vars=["__x", "date", "timestamp", *base_cols],
        value_vars=list(scb_cols),
        var_name="scb_name",
        value_name="value",
    )
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["__x"])

    palette = [
        "#2563eb",
        "#16a34a",
        "#dc2626",
        "#7c3aed",
        "#0891b2",
        "#ea580c",
        "#0f766e",
        "#be185d",
        "#4f46e5",
        "#059669",
        "#b91c1c",
        "#6d28d9",
    ]

    fig = go.Figure()
    group_cols = ["inv_stn_name", "inv_name", "scb_name"]
    if unit_present:
        group_cols.insert(2, "inv_unit_name")
    keys = long.groupby(group_cols).size().reset_index()
    # Build a stable set of trace keys
    trace_keys: list[tuple[str, str, str, str]] = []
    for _, r in keys.iterrows():
        inv_stn = str(r["inv_stn_name"])
        inv = str(r["inv_name"])
        unit = str(r["inv_unit_name"]) if unit_present else ""
        scb = str(r["scb_name"])
        trace_keys.append((inv_stn, inv, unit, scb))

    trace_keys = sorted(trace_keys, key=lambda t: _sort_key_is_inv_unit_scb(t[0], t[1], t[2], t[3]))
    for i, (inv_stn, inv, unit, scb) in enumerate(trace_keys):
        mask = (long["inv_stn_name"].astype(str) == inv_stn) & (long["inv_name"].astype(str) == inv) & (
            long["scb_name"].astype(str) == scb
        )
        if unit_present:
            mask = mask & (long["inv_unit_name"].astype(str).fillna("") == unit)

        g = long.loc[mask].sort_values("__x")
        if g.empty:
            continue

        color = palette[i % len(palette)]
        fillcolor = f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.12)"

        base_label = display_map.get((inv_stn, inv, unit, scb)) or (
            f"{inv_stn}-{inv}-{unit}-{scb}" if unit else f"{inv_stn}-{inv}-{scb}"
        )
        sn = None
        if string_num_map is not None:
            sn = string_num_map.get((inv_stn, inv, unit, scb))
        if sn is not None and pd.notna(sn):
            try:
                snf = float(sn)
                if abs(snf - round(snf)) < 1e-9:
                    label = f"{base_label} (S:{int(round(snf))})"
                else:
                    label = f"{base_label} (S:{snf:.3g})"
            except Exception:
                label = base_label
        else:
            label = base_label

        # customdata must be row-aligned with points; build arrays (avoid scalar-only DataFrame error)
        n = int(len(g))
        custom = pd.DataFrame(
            {
                "inv_stn": [inv_stn] * n,
                "inv": [inv] * n,
                "unit": [unit] * n,
                "scb": [scb] * n,
            }
        ).values

        fig.add_trace(
            go.Scatter(
                x=g["__x"],
                y=g["value"],
                name=label,
                mode="lines",
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=fillcolor,
                customdata=custom,
                hovertemplate=(
                    "<b>Inverter Station:</b> %{customdata[0]}<br>"
                    "<b>Inverter:</b> %{customdata[1]}<br>"
                    "<b>Unit:</b> %{customdata[2]}<br>"
                    "<b>SCB:</b> %{customdata[3]}<br>"
                    "<b>Value:</b> %{y}<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        title=dict(text="<b>Raw SCB Time Series (06:00–18:00)</b>", x=0.01),
        margin=dict(l=70, r=35, t=70, b=70),
        height=560,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        hovermode="x unified",
    )
    # Hover axis / crosshair
    fig.update_xaxes(
        title=dict(text="<b>Timestamp</b>", standoff=10),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="rgba(0,0,0,0.35)",
    )
    fig.update_yaxes(
        title=dict(text=y_axis_title, standoff=10),
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikethickness=1,
        spikecolor="rgba(0,0,0,0.25)",
    )
    return fig


@st.cache_data(show_spinner=False)
def _fetch_normalization_map(
    db_path: str,
    *,
    site_name: str,
    inv_stations: tuple[str, ...],
    inverters: tuple[str, ...],
    units: tuple[str, ...],
    scb_cols: tuple[str, ...],
) -> pd.DataFrame:
    """
    Per-SCB denominators from array_details using STRICT keys:
    site_name, inv_stn_name, inv_name, inv_unit_name (if selected/available), scb_name
    """
    if not (site_name and inv_stations and inverters and scb_cols):
        return pd.DataFrame()

    con = _connect(db_path)
    try:
        inv_stn_ph = ", ".join(["?"] * len(inv_stations))
        inv_ph = ", ".join(["?"] * len(inverters))
        scb_ph = ", ".join(["?"] * len(scb_cols))

        unit_filter = ""
        params: list[object] = [site_name, *inv_stations, *inverters, *scb_cols]
        if units:
            unit_ph = ", ".join(["?"] * len(units))
            unit_filter = f" AND trim(inv_unit_name) IN ({unit_ph})"
            params.extend(list(units))

        df = con.execute(
            f"""
            SELECT
              trim(inv_stn_name) AS inv_stn_name,
              trim(inv_name) AS inv_name,
              trim(inv_unit_name) AS inv_unit_name,
              trim(scb_name) AS scb_name,
              try_cast(string_num as double) AS string_num,
              try_cast(load_kwp as double) AS load_kwp
            FROM array_details
            WHERE lower(trim(site_name)) = lower(trim(?))
              AND trim(inv_stn_name) IN ({inv_stn_ph})
              AND trim(inv_name) IN ({inv_ph})
              AND trim(scb_name) IN ({scb_ph})
              {unit_filter}
            """,
            params,
        ).df()
    finally:
        con.close()

    if df is None or df.empty:
        return pd.DataFrame()

    df["inv_unit_name"] = df["inv_unit_name"].where(df["inv_unit_name"].notna(), "")
    df = df.drop_duplicates(subset=["inv_stn_name", "inv_name", "inv_unit_name", "scb_name"]).copy()
    df["_k"] = df.apply(
        lambda r: _sort_key_is_inv_unit_scb(
            str(r.get("inv_stn_name") or ""),
            str(r.get("inv_name") or ""),
            str(r.get("inv_unit_name") or ""),
            str(r.get("scb_name") or ""),
        ),
        axis=1,
    )
    df = df.sort_values("_k", ignore_index=True).drop(columns=["_k"])
    return df


@st.cache_data(show_spinner=False)
def _fetch_array_details_rows(
    db_path: str,
    *,
    site_name: str,
    inv_stations: tuple[str, ...],
    inverters: tuple[str, ...],
    units: tuple[str, ...],
    scb_names: tuple[str, ...],
) -> pd.DataFrame:
    """
    Return raw `array_details` rows (ALL columns as-is) for the selected SCBs.
    Filters are design-driven and optional on Unit (if not selected, match across units).
    """
    if not (site_name and inv_stations and inverters and scb_names):
        return pd.DataFrame()

    con = _connect(db_path)
    try:
        inv_stn_ph = ", ".join(["?"] * len(inv_stations))
        inv_ph = ", ".join(["?"] * len(inverters))
        scb_ph = ", ".join(["?"] * len(scb_names))

        unit_filter = ""
        params: list[object] = [site_name, *inv_stations, *inverters, *scb_names]
        if units:
            unit_ph = ", ".join(["?"] * len(units))
            unit_filter = f" AND trim(inv_unit_name) IN ({unit_ph})"
            params.extend(list(units))

        df = con.execute(
            f"""
            SELECT *
            FROM array_details
            WHERE lower(trim(site_name)) = lower(trim(?))
              AND trim(inv_stn_name) IN ({inv_stn_ph})
              AND trim(inv_name) IN ({inv_ph})
              AND trim(scb_name) IN ({scb_ph})
              {unit_filter}
            """,
            params,
        ).df()
        return df if df is not None else pd.DataFrame()
    finally:
        con.close()


def _apply_normalization_wide(
    df: pd.DataFrame,
    *,
    norm_map: pd.DataFrame,
    scb_cols: tuple[str, ...],
    method: str,  # "string" | "load"
    use_unit_key: bool,
) -> pd.DataFrame:
    if df is None or df.empty or norm_map is None or norm_map.empty or not scb_cols:
        return df

    denom_col = "string_num" if method == "string" else "load_kwp"
    if denom_col not in norm_map.columns:
        return df

    unit_present = "inv_unit_name" in df.columns
    id_cols = ["date", "timestamp", "inv_stn_name", "inv_name"] + (["inv_unit_name"] if unit_present else [])

    tmp = df.copy()
    if unit_present:
        tmp["inv_unit_name"] = tmp["inv_unit_name"].where(tmp["inv_unit_name"].notna(), "")

    long = tmp.melt(id_vars=id_cols, value_vars=list(scb_cols), var_name="scb_name", value_name="value")
    long["value"] = pd.to_numeric(long["value"], errors="coerce")

    nm = norm_map.copy()
    nm["inv_unit_name"] = nm["inv_unit_name"].where(nm["inv_unit_name"].notna(), "")

    # Decide join keys:
    # - If Unit was selected by user AND the table has inv_unit_name: join includes unit
    # - Else: join excludes unit, but only if mapping is unambiguous across units
    if (not unit_present) or (not use_unit_key):
        # Normalize both sides to a "no-unit" join
        if "inv_unit_name" in long.columns:
            long["inv_unit_name"] = ""
        nm["inv_unit_name"] = nm["inv_unit_name"].astype(str).fillna("")

        # If array_details has multiple units for the same (IS, INV, SCB), normalization is ambiguous.
        dup_check = nm.groupby(["inv_stn_name", "inv_name", "scb_name"], dropna=False)["inv_unit_name"].nunique().reset_index(name="unit_n")
        if (dup_check["unit_n"] > 1).any():
            # Return original df; caller will warn the user to select Unit.
            return df

        join_cols = ["inv_stn_name", "inv_name", "scb_name"]
        merged = long.merge(nm[join_cols + [denom_col]], on=join_cols, how="left")
    else:
        join_cols = ["inv_stn_name", "inv_name", "inv_unit_name", "scb_name"]
        merged = long.merge(nm[join_cols + [denom_col]], on=join_cols, how="left")

    denom = pd.to_numeric(merged[denom_col], errors="coerce")
    denom = denom.where(denom.notna() & (denom != 0))
    merged["value_norm"] = merged["value"] / denom

    wide = merged.pivot_table(index=id_cols, columns="scb_name", values="value_norm", aggfunc="first").reset_index()
    for c in scb_cols:
        if c not in wide.columns:
            wide[c] = pd.NA
    wide = wide[id_cols + list(scb_cols)]
    return wide


def render(db_path: str) -> None:
    st.markdown("## 🧪 Raw Analyser")
    st.caption("Read-only raw time-series explorer (DuckDB: `master.duckdb`). Queries run only when you click **Plot Now**.")

    entered = bool(st.session_state.pop("_entered_raw_analyser", False))

    user_info = st.session_state.get("user_info") or {}
    username = (user_info.get("username") or "").strip()
    allowed_sites = allowed_sites_for_user(username)

    all_sites = _list_sites(db_path)
    if allowed_sites:
        # restricted user: only their site (case-insensitive)
        all_sites = [s for s in all_sites if s.strip().lower() in {x.lower() for x in allowed_sites}]

    # Normalize site labels (avoid losing selection due to whitespace/case mismatches)
    all_sites = sorted({str(s).strip() for s in all_sites if str(s).strip()})

    if not all_sites:
        st.warning("No sites available for your account.")
        return

    # Site selection (single) — key is required so tab switching preserves selection
    if allowed_sites and len(all_sites) == 1:
        site_name = all_sites[0]
        # Also keep a canonical site value in session_state (used by other widgets / cached render)
        st.session_state["raw_analyser_site"] = site_name
        st.selectbox("Site Name", options=all_sites, index=0, disabled=True, key="raw_analyser_site_locked")
    else:
        # Restore site ONLY on tab-enter (never on normal reruns; avoids fighting user intent).
        if entered:
            last_site = st.session_state.get("raw_analyser_last_site")
            cur_site = st.session_state.get("raw_analyser_site")
            if (cur_site is None or cur_site == "(select)") and isinstance(last_site, str) and last_site.strip():
                ls = last_site.strip()
                if ls in all_sites:
                    st.session_state["raw_analyser_site"] = ls
                else:
                    hit = {s.lower(): s for s in all_sites}.get(ls.lower())
                    if hit:
                        st.session_state["raw_analyser_site"] = hit

        # Clamp current selection to valid options (case-insensitive)
        cur_site = st.session_state.get("raw_analyser_site")
        if isinstance(cur_site, str) and cur_site not in {"", "(select)"} and cur_site not in all_sites:
            hit = {s.lower(): s for s in all_sites}.get(cur_site.strip().lower())
            st.session_state["raw_analyser_site"] = hit if hit else "(select)"

        site_name = st.selectbox("Site Name", options=["(select)"] + all_sites, index=0, key="raw_analyser_site")
        if site_name == "(select)":
            st.info("Select a Site Name to continue.")
            return

    # Resolve site table name strictly from site name (lowercase/sanitized).
    table_guess = _sanitize_table_guess(site_name)
    con = _connect(db_path)
    try:
        if not _table_exists(con, table_guess):
            st.error(f'No time-series table found for site "{site_name}" (expected table "{table_guess}").')
            return
    finally:
        con.close()

    table = table_guess

    # Cascading filters
    inv_stations = _list_inv_stations(db_path, site_name)
    last_filters = st.session_state.get("raw_analyser_last_filters") or {}
    # Restore ONLY on tab-enter AND only if widget key is missing (not if user cleared selections)
    if entered and "raw_analyser_is" not in st.session_state and isinstance(last_filters, dict) and last_filters.get("inv_stations"):
        st.session_state["raw_analyser_is"] = list(last_filters.get("inv_stations") or [])

    # Clamp IS selection to current options (case-insensitive)
    cur_is = st.session_state.get("raw_analyser_is", [])
    if isinstance(cur_is, list) and cur_is:
        opt_by_lower = {str(s).strip().lower(): str(s) for s in inv_stations}
        mapped = []
        for v in cur_is:
            vv = str(v).strip()
            if vv in inv_stations:
                mapped.append(vv)
            else:
                hit = opt_by_lower.get(vv.lower())
                if hit:
                    mapped.append(hit)
        st.session_state["raw_analyser_is"] = mapped

    sel_inv_stations = st.multiselect(
        "Inverter Station Name (IS)",
        options=inv_stations,
        default=[],
        key="raw_analyser_is",
        disabled=not bool(site_name),
    )

    inverters = _list_inverters(db_path, site_name, tuple(sel_inv_stations)) if sel_inv_stations else []
    # Restore ONLY on tab-enter AND only if widget key is missing
    if entered and "raw_analyser_inv" not in st.session_state and isinstance(last_filters, dict) and last_filters.get("inverters"):
        st.session_state["raw_analyser_inv"] = list(last_filters.get("inverters") or [])

    # Clamp INV selection to current options (case-insensitive)
    cur_inv = st.session_state.get("raw_analyser_inv", [])
    if isinstance(cur_inv, list) and cur_inv:
        opt_by_lower = {str(s).strip().lower(): str(s) for s in inverters}
        mapped = []
        for v in cur_inv:
            vv = str(v).strip()
            if vv in inverters:
                mapped.append(vv)
            else:
                hit = opt_by_lower.get(vv.lower())
                if hit:
                    mapped.append(hit)
        st.session_state["raw_analyser_inv"] = mapped

    sel_inverters = st.multiselect(
        "Inverter Name (INV)",
        options=inverters,
        default=[],
        key="raw_analyser_inv",
        disabled=not bool(sel_inv_stations),
    )

    units = _list_units(db_path, site_name, tuple(sel_inv_stations), tuple(sel_inverters)) if sel_inverters else []
    # Restore ONLY on tab-enter AND only if widget key is missing
    if entered and "raw_analyser_unit" not in st.session_state and isinstance(last_filters, dict) and last_filters.get("units"):
        st.session_state["raw_analyser_unit"] = list(last_filters.get("units") or [])

    # Clamp UNIT selection to current options (case-insensitive)
    cur_unit = st.session_state.get("raw_analyser_unit", [])
    if isinstance(cur_unit, list) and cur_unit and units:
        opt_by_lower = {str(s).strip().lower(): str(s) for s in units}
        mapped = []
        for v in cur_unit:
            vv = str(v).strip()
            if vv in units:
                mapped.append(vv)
            else:
                hit = opt_by_lower.get(vv.lower())
                if hit:
                    mapped.append(hit)
        st.session_state["raw_analyser_unit"] = mapped

    sel_units = st.multiselect(
        "Unit Name (Optional)",
        options=units,
        default=[],
        key="raw_analyser_unit",
        disabled=not bool(sel_inverters) or not bool(units),
    )

    scb_df = (
        _list_scbs_with_display(db_path, site_name, tuple(sel_inv_stations), tuple(sel_inverters), tuple(sel_units))
        if sel_inverters
        else pd.DataFrame(columns=["inv_stn_name", "inv_name", "inv_unit_name", "scb_name", "display"])
    )
    scb_display_options = scb_df["display"].tolist() if scb_df is not None and not scb_df.empty else []
    # Restore SCB selection ONLY on tab-enter AND only if widget key is missing
    if entered and "raw_analyser_scb_sel" not in st.session_state and isinstance(last_filters, dict) and last_filters.get("scb_cols") and scb_df is not None and not scb_df.empty:
        want_cols = {str(x) for x in (last_filters.get("scb_cols") or [])}
        q = scb_df.copy()
        q["scb_name"] = q["scb_name"].astype(str)
        q["display"] = q["display"].astype(str)
        restore_displays = [d for (c, d) in zip(q["scb_name"].tolist(), q["display"].tolist()) if c in want_cols]
        restore_displays = [d for d in restore_displays if d in set(scb_display_options)]
        if restore_displays:
            st.session_state["raw_analyser_scb_sel"] = restore_displays

    # Prevent StreamlitAPIException: default selections must exist in options
    existing_sel = st.session_state.get("raw_analyser_scb_sel", [])
    if isinstance(existing_sel, list):
        st.session_state["raw_analyser_scb_sel"] = [x for x in existing_sel if x in scb_display_options]
    else:
        st.session_state["raw_analyser_scb_sel"] = []

    c_scb_a, c_scb_b = st.columns([1, 1], vertical_alignment="bottom")
    with c_scb_a:
        if st.button(
            "Select All SCBs",
            use_container_width=True,
            disabled=not bool(sel_inverters) or not bool(scb_display_options),
        ):
            st.session_state["raw_analyser_scb_sel"] = list(scb_display_options)
    with c_scb_b:
        if st.button(
            "Clear SCBs",
            use_container_width=True,
            disabled=not bool(sel_inverters) or not bool(scb_display_options),
        ):
            st.session_state["raw_analyser_scb_sel"] = []

    sel_scb_displays = st.multiselect(
        "SCB Name",
        options=scb_display_options,
        key="raw_analyser_scb_sel",
        help="SCB names may appear truncated in the pills; expand 'Selected SCBs (full)' below to view full identifiers.",
        disabled=not bool(sel_inverters) or not bool(scb_display_options),
    )

    # Make full SCB identifiers easy to read (no truncation confusion)
    if sel_scb_displays:
        with st.expander("Selected SCBs (full)", expanded=False):
            st.code("\n".join([str(x) for x in sel_scb_displays]))

    # Metadata: show array_details rows for selected SCBs (collapsible)
    if site_name and sel_inv_stations and sel_inverters and sel_scb_displays:
        sel_meta_rows = scb_df[scb_df["display"].isin(sel_scb_displays)].copy() if scb_df is not None and not scb_df.empty else pd.DataFrame()
        scb_names = tuple(sorted({str(x) for x in sel_meta_rows.get("scb_name", pd.Series(dtype=str)).tolist()}))
        with st.expander("Show Design Metadata (array_details)", expanded=False):
            meta_df = _fetch_array_details_rows(
                db_path,
                site_name=site_name,
                inv_stations=tuple(sel_inv_stations),
                inverters=tuple(sel_inverters),
                units=tuple(sel_units),
                scb_names=scb_names,
            )
            if meta_df is None or meta_df.empty:
                st.info("No array_details rows found for the selected SCBs.")
            else:
                st.dataframe(meta_df, width="stretch", hide_index=True)

    # Date availability logic (06:00–18:00, from site table)
    cols = _table_columns(db_path, table)
    dmin, dmax = _table_date_bounds(db_path, table)

    # STRICT: availability is based on the SELECTED SCB columns only.
    selected_scb_cols: tuple[str, ...] = ()
    if sel_scb_displays and (scb_df is not None) and (not scb_df.empty):
        sel_rows_preview = scb_df[scb_df["display"].isin(sel_scb_displays)].copy()
        selected_scb_cols = tuple(sorted({str(x) for x in sel_rows_preview["scb_name"].tolist()}))
        selected_scb_cols = tuple([c for c in selected_scb_cols if c in set(cols)])

    available_dates: set[date] = set()
    if selected_scb_cols:
        available_dates = _available_dates_for_selected_scbs_06_18(db_path, table, selected_scb_cols)

    if not selected_scb_cols:
        st.caption("Select one or more SCBs to enable date availability checks (06:00–18:00).")
    else:
        if available_dates:
            st.caption(
                f"Available dates for selected SCBs (06:00–18:00): **{min(available_dates).isoformat()} → {max(available_dates).isoformat()}**"
            )
            st.caption("Pick **From Date** and **To Date** (no defaults). Then click **Plot Now**.")
        elif dmin and dmax:
            st.caption(f"Table date bounds: {dmin.isoformat()} → {dmax.isoformat()}")
            st.caption("No available dates detected for the selected SCBs (06:00–18:00).")
        else:
            st.caption("No date information could be derived for this site table.")

    # Restore dates ONLY on tab-enter AND only if widget keys are missing (not fighting user edits)
    if entered and isinstance(last_filters, dict):
        if "raw_analyser_from" not in st.session_state and last_filters.get("from_date") is not None:
            st.session_state["raw_analyser_from"] = last_filters.get("from_date")
        if "raw_analyser_to" not in st.session_state and last_filters.get("to_date") is not None:
            st.session_state["raw_analyser_to"] = last_filters.get("to_date")

    c1, c2, c3 = st.columns([1.1, 1.1, 1.2], vertical_alignment="bottom")
    with c1:
        from_date = st.date_input(
            "From Date",
            # Do NOT auto-select a date. User must explicitly pick dates.
            value=st.session_state.get("raw_analyser_from", None),
            disabled=not bool(selected_scb_cols),
            key="raw_analyser_from",
        )
    with c2:
        to_date = st.date_input(
            "To Date",
            # Do NOT auto-select a date. User must explicitly pick dates.
            value=st.session_state.get("raw_analyser_to", None),
            disabled=not bool(selected_scb_cols),
            key="raw_analyser_to",
        )
    with c3:
        plot_now = st.button(
            "Plot Now",
            type="primary",
            use_container_width=True,
            disabled=not (bool(selected_scb_cols) and (from_date is not None) and (to_date is not None)),
        )


    invalid_date_selected = False
    if available_dates and selected_scb_cols:
        if from_date is not None and from_date not in available_dates:
            st.warning("Selected From Date has no SCB data between 06:00–18:00 for the selected SCBs")
            invalid_date_selected = True
        if to_date is not None and to_date not in available_dates:
            st.warning("Selected To Date has no SCB data between 06:00–18:00 for the selected SCBs")
            invalid_date_selected = True
    if from_date is not None and to_date is not None and from_date > to_date:
        st.warning("From Date must be <= To Date.")
        invalid_date_selected = True

    def _render_from_state() -> None:
        last_df = st.session_state.get("raw_analyser_last_df")
        last_display_map = st.session_state.get("raw_analyser_last_display_map")
        last_scb_cols = st.session_state.get("raw_analyser_last_scb_cols")
        last_site = st.session_state.get("raw_analyser_last_site")
        last_string_num_map = st.session_state.get("raw_analyser_last_string_num_map") or {}

        if last_df is None or last_display_map is None or last_scb_cols is None or last_site is None:
            st.info("Filters are ready. Click **Plot Now** to run the query and plot.")
            return

        applied_method = st.session_state.get("raw_analyser_applied_norm_method") or "(none)"
        if "raw_analyser_norm_choice" not in st.session_state:
            st.session_state["raw_analyser_norm_choice"] = applied_method

        plot_df = last_df
        y_title = "<b>Raw SCB Value</b>"
        if applied_method in {"String", "Load"} and st.session_state.get("raw_analyser_norm_df") is not None:
            plot_df = st.session_state["raw_analyser_norm_df"]
            y_title = "<b>Normalized SCB Value</b>"

        st.plotly_chart(
            _build_plot(
                df=plot_df,
                site_name=str(last_site),
                display_map=dict(last_display_map),
                scb_cols=tuple(last_scb_cols),
                string_num_map=dict(last_string_num_map),
                y_axis_title=y_title,
            ),
            width="stretch",
            key="raw_analyser_plot",
        )

        st.markdown("### Normalization (optional)")
        norm_choice = st.selectbox(
            "Normalize by",
            options=["(none)", "String", "Load"],
            key="raw_analyser_norm_choice",
            help="Select a method, then click Apply Normalization. Plot will not change until you apply.",
        )

        apply_norm = st.button(
            "Apply Normalization",
            type="secondary",
            disabled=False,
            use_container_width=True,
        )

        if apply_norm:
            if norm_choice == "(none)":
                st.session_state.pop("raw_analyser_norm_df", None)
                st.session_state["raw_analyser_applied_norm_method"] = "(none)"
                st.success("Normalization cleared.")
            else:
                plot_ph = st.empty()
                with plot_ph.container():
                    p2 = st.progress(0, text="Starting…")
                    t1 = time.perf_counter()
                    elapsed2 = st.empty()

                    def _tick2(frac: float, msg: str) -> None:
                        p2.progress(int(max(0, min(1, frac)) * 100), text=msg)
                        elapsed2.caption(f"Elapsed: **{time.perf_counter() - t1:.1f}s**")

                    _tick2(0.15, "Fetching normalization mapping from array_details…")
                    fmeta = st.session_state.get("raw_analyser_last_filters") or {}
                    norm_map = _fetch_normalization_map(
                        db_path,
                        site_name=str(last_site),
                        inv_stations=tuple(fmeta.get("inv_stations") or ()),
                        inverters=tuple(fmeta.get("inverters") or ()),
                        units=tuple(fmeta.get("units") or ()),
                        scb_cols=tuple(fmeta.get("scb_cols") or ()),
                    )

                    _tick2(0.55, "Applying per-SCB normalization…")
                    norm_df = _apply_normalization_wide(
                        last_df,
                        norm_map=norm_map,
                        scb_cols=tuple(last_scb_cols),
                        method="string" if norm_choice == "String" else "load",
                        use_unit_key=bool((st.session_state.get("raw_analyser_last_filters") or {}).get("units")),
                    )

                    # Guard: if normalization is ambiguous/missing denom, keep raw plot and warn.
                    scb_cols_local = list(last_scb_cols)
                    all_nan = False
                    try:
                        if scb_cols_local and all(pd.to_numeric(norm_df[c], errors="coerce").isna().all() for c in scb_cols_local if c in norm_df.columns):
                            all_nan = True
                    except Exception:
                        all_nan = False

                    if norm_df is None or norm_df.empty or all_nan:
                        _tick2(1.00, "Done")
                        st.warning(
                            "Normalization could not be applied (missing/ambiguous denominators). "
                            "If your plant has multiple units per inverter, please select Unit before normalizing."
                        )
                    else:
                        st.session_state["raw_analyser_norm_df"] = norm_df
                        st.session_state["raw_analyser_applied_norm_method"] = norm_choice
                        _tick2(1.00, "Done")

                st.rerun()

        with st.expander("Show Raw Data", expanded=False):
            show_cols = ["date", "timestamp", "inv_stn_name", "inv_name"]
            if "inv_unit_name" in plot_df.columns:
                show_cols.append("inv_unit_name")
            show_cols.extend(list(last_scb_cols))
            show_cols = [c for c in show_cols if c in plot_df.columns]
            st.dataframe(plot_df[show_cols], width="stretch", hide_index=True)

    if not plot_now:
        # Keep the last plot visible even if filters change; refresh only on Plot Now.
        if st.session_state.get("raw_analyser_last_df") is not None:
            st.caption("Showing last plotted result. Update filters and click **Plot Now** to refresh.")
        _render_from_state()
        return

    if from_date > to_date:
        st.error("From Date must be <= To Date.")
        return

    if not sel_inv_stations or not sel_inverters or not sel_scb_displays:
        st.warning("Select at least one Inverter Station, one Inverter, and one SCB.")
        return
    if invalid_date_selected:
        st.warning("Pick only available dates for the selected SCBs before plotting.")
        return

    # Resolve selected SCB columns + display labels from array_details mapping
    sel_rows = scb_df[scb_df["display"].isin(sel_scb_displays)].copy() if not scb_df.empty else pd.DataFrame()
    if sel_rows.empty:
        st.warning("No SCBs found for the current selections.")
        return

    # SCB column names must exist in the site table
    sel_scb_cols = tuple(sorted({str(x) for x in sel_rows["scb_name"].tolist()}))
    existing_cols_set = {c for c in cols}
    sel_scb_cols = tuple([c for c in sel_scb_cols if c in existing_cols_set])
    if not sel_scb_cols:
        st.warning("Selected SCBs do not exist as columns in the site time-series table.")
        return

    display_map: dict[tuple[str, str, str, str], str] = {}
    for _, r in sel_rows.iterrows():
        key = (str(r["inv_stn_name"]), str(r["inv_name"]), str(r.get("inv_unit_name") or ""), str(r["scb_name"]))
        display_map[key] = str(r["display"])

    # Progress + elapsed timer
    p = st.progress(0, text="Starting…")
    t0 = time.perf_counter()
    elapsed = st.empty()

    def _tick(frac: float, msg: str) -> None:
        p.progress(int(max(0, min(1, frac)) * 100), text=msg)
        elapsed.caption(f"Elapsed: **{time.perf_counter() - t0:.1f}s**")

    try:
        _tick(0.10, "Querying raw time-series (06:00–18:00)…")
        df = _fetch_raw_timeseries(
            db_path=db_path,
            table=table,
            from_date=from_date,
            to_date=to_date,
            inv_stations=tuple(sel_inv_stations),
            inverters=tuple(sel_inverters),
            units=tuple(sel_units),
            scb_cols=sel_scb_cols,
        )

        _tick(0.60, "Preparing plot data…")
        if df is None or df.empty:
            _tick(1.00, "Done")
            st.warning("No data found for the selected filters in 06:00–18:00.")
            return

        # Order raw rows: IS -> INV -> UNIT -> then time (for table + stable plotting)
        sort_cols = ["inv_stn_name", "inv_name"]
        if "inv_unit_name" in df.columns:
            sort_cols.append("inv_unit_name")
        sort_cols.extend(["date", "timestamp"])
        df["_k_is"] = df["inv_stn_name"].astype(str).map(_num_suffix)
        df["_k_inv"] = df["inv_name"].astype(str).map(_num_suffix)
        if "inv_unit_name" in df.columns:
            df["_k_unit"] = df["inv_unit_name"].astype(str).map(_num_suffix)
            df = df.sort_values(["_k_is", "inv_stn_name", "_k_inv", "inv_name", "_k_unit", "inv_unit_name", "date", "timestamp"], ignore_index=True)
        else:
            df = df.sort_values(["_k_is", "inv_stn_name", "_k_inv", "inv_name", "date", "timestamp"], ignore_index=True)
        df = df.drop(columns=[c for c in ["_k_is", "_k_inv", "_k_unit"] if c in df.columns])

        _tick(0.85, "Building Plotly figure…")
        # Fetch mapping for string_num (legend annotation), strictly by selected keys
        norm_map_for_legend = _fetch_normalization_map(
            db_path,
            site_name=site_name,
            inv_stations=tuple(sel_inv_stations),
            inverters=tuple(sel_inverters),
            units=tuple(sel_units),
            scb_cols=tuple(sel_scb_cols),
        )
        string_num_map: dict[tuple[str, str, str, str], float] = {}
        if norm_map_for_legend is not None and not norm_map_for_legend.empty:
            for _, r in norm_map_for_legend.iterrows():
                k = (
                    str(r.get("inv_stn_name") or ""),
                    str(r.get("inv_name") or ""),
                    str(r.get("inv_unit_name") or ""),
                    str(r.get("scb_name") or ""),
                )
                try:
                    v = r.get("string_num")
                    if v is not None and pd.notna(v):
                        string_num_map[k] = float(v)
                except Exception:
                    pass

        fig = _build_plot(
            df=df,
            site_name=site_name,
            display_map=display_map,
            scb_cols=sel_scb_cols,
            string_num_map=string_num_map,
            y_axis_title="<b>Raw SCB Value</b>",
        )

        _tick(1.00, "Done")

        # Store last raw context for stateful rendering across reruns
        st.session_state["raw_analyser_last_df"] = df
        st.session_state["raw_analyser_last_site"] = site_name
        st.session_state["raw_analyser_last_table"] = table
        st.session_state["raw_analyser_last_filters"] = {
            "inv_stations": tuple(sel_inv_stations),
            "inverters": tuple(sel_inverters),
            "units": tuple(sel_units),
            "scb_cols": tuple(sel_scb_cols),
            "from_date": from_date,
            "to_date": to_date,
        }
        st.session_state["raw_analyser_last_display_map"] = display_map
        st.session_state["raw_analyser_last_scb_cols"] = tuple(sel_scb_cols)
        st.session_state["raw_analyser_last_string_num_map"] = string_num_map

        # Reset normalization state on new plot
        st.session_state.pop("raw_analyser_norm_df", None)
        st.session_state["raw_analyser_applied_norm_method"] = "(none)"
        if "raw_analyser_norm_choice" in st.session_state:
            st.session_state["raw_analyser_norm_choice"] = "(none)"

        # Render plot + normalization + table from stored state (prevents disappearing on reruns)
        _render_from_state()
    except Exception as e:
        _tick(1.00, "Failed")
        st.error(f"Raw Analyser error: {e}")

