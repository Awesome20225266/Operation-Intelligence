from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st

from supabase_link import get_supabase_client
from aws_duckdb import get_duckdb_connection

try:
    from st_aggrid import AgGrid  # type: ignore
    from st_aggrid.grid_options_builder import GridOptionsBuilder  # type: ignore
    from st_aggrid.shared import GridUpdateMode  # type: ignore

    _HAS_AGGRID = True
except Exception:
    AgGrid = None  # type: ignore
    GridOptionsBuilder = None  # type: ignore
    GridUpdateMode = None  # type: ignore
    _HAS_AGGRID = False


def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    return get_duckdb_connection(db_local=db_path)


@st.cache_data(show_spinner=False)
def list_sites_from_syd(db_path: str) -> list[str]:
    con = _connect(db_path)
    try:
        rows = con.execute("select distinct site_name from syd order by site_name").fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def date_bounds_from_syd(db_path: str) -> tuple[Optional[date], Optional[date]]:
    con = _connect(db_path)
    try:
        row = con.execute("select min(date) as dmin, max(date) as dmax from syd").fetchone()
        if not row:
            return None, None
        return row[0], row[1]
    finally:
        con.close()


def _sql_in_list(n: int) -> str:
    return "(" + ",".join(["?"] * n) + ")"


@st.cache_data(show_spinner=False)
def fetch_latest_syd(db_path: str, sites: list[str]) -> pd.DataFrame:
    """
    Case A: No date range selected.
    Use the most recent available date per site as the 'raw' day, and show equipment for that day.
    """
    if not sites:
        return pd.DataFrame()
    con = _connect(db_path)
    try:
        in_clause = _sql_in_list(len(sites))
        params: list[Any] = [*sites]
        return con.execute(
            f"""
            select
              s.site_name,
              s.equipment_name,
              s.date,
              s.syd_percent * 100.0 as syd_dev_pct
            from syd s
            join (
              select site_name, max(date) as max_date
              from syd
              where site_name in {in_clause}
              group by 1
            ) m
              on m.site_name = s.site_name
             and m.max_date = s.date
            where s.site_name in {in_clause}
            order by s.site_name, s.equipment_name
            """,
            [*params, *params],
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def fetch_syd_for_date(db_path: str, sites: list[str], d: date) -> pd.DataFrame:
    """
    Case A': Single date explicitly selected.
    Return equipment SYD deviation for the selected date.
    """
    if not sites:
        return pd.DataFrame()
    con = _connect(db_path)
    try:
        in_clause = _sql_in_list(len(sites))
        params: list[Any] = [*sites, d]
        return con.execute(
            f"""
            select
              site_name,
              equipment_name,
              date,
              syd_percent * 100.0 as syd_dev_pct
            from syd
            where site_name in {in_clause}
              and date = ?
            order by 1,2
            """,
            params,
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def fetch_median_syd(db_path: str, sites: list[str], d1: date, d2: date) -> pd.DataFrame:
    """
    Case B: Date range selected.
    Aggregate by median(syd_percent*100) per (site_name, equipment_name) in the range.
    """
    if not sites:
        return pd.DataFrame()
    con = _connect(db_path)
    try:
        in_clause = _sql_in_list(len(sites))
        params: list[Any] = [*sites, d1, d2]
        return con.execute(
            f"""
            select
              site_name,
              equipment_name,
              median(syd_percent * 100.0) as syd_dev_pct
            from syd
            where site_name in {in_clause}
              and date between ? and ?
            group by 1,2
            order by 1,2
            """,
            params,
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def fetch_latest_pr_for_sites(db_path: str, sites: list[str]) -> pd.DataFrame:
    """
    PR values aligned to each site's most recent SYD date (Case A).
    """
    if not sites:
        return pd.DataFrame()
    con = _connect(db_path)
    try:
        in_clause = _sql_in_list(len(sites))
        params: list[Any] = [*sites]
        return con.execute(
            f"""
            select
              p.site_name,
              p.equipment_name,
              p.date,
              p.pr_percent * 100.0 as pr_pct
            from pr p
            join (
              select site_name, max(date) as max_date
              from syd
              where site_name in {in_clause}
              group by 1
            ) m
              on m.site_name = p.site_name
             and m.max_date = p.date
            where p.site_name in {in_clause}
            order by p.site_name, p.equipment_name
            """,
            [*params, *params],
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def fetch_pr_for_date(db_path: str, sites: list[str], d: date) -> pd.DataFrame:
    """
    PR values for a specific selected date (aligned to fetch_syd_for_date).
    """
    if not sites:
        return pd.DataFrame()
    con = _connect(db_path)
    try:
        in_clause = _sql_in_list(len(sites))
        params: list[Any] = [*sites, d]
        return con.execute(
            f"""
            select
              site_name,
              equipment_name,
              date,
              pr_percent * 100.0 as pr_pct
            from pr
            where site_name in {in_clause}
              and date = ?
            order by 1,2
            """,
            params,
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def fetch_median_pr(db_path: str, sites: list[str], d1: date, d2: date) -> pd.DataFrame:
    """
    Case B PR aggregation: median(pr_percent*100) per site/equipment in date range.
    """
    if not sites:
        return pd.DataFrame()
    con = _connect(db_path)
    try:
        in_clause = _sql_in_list(len(sites))
        params: list[Any] = [*sites, d1, d2]
        return con.execute(
            f"""
            select
              site_name,
              equipment_name,
              median(pr_percent * 100.0) as pr_pct
            from pr
            where site_name in {in_clause}
              and date between ? and ?
            group by 1,2
            order by 1,2
            """,
            params,
        ).fetchdf()
    finally:
        con.close()


def _apply_threshold(df: pd.DataFrame, threshold: Optional[float], value_col: str) -> pd.DataFrame:
    if df.empty:
        return df
    if threshold is None:
        return df
    # Only underperforming equipment below threshold
    return df[df[value_col] < float(threshold)].copy()


def _build_composite_chart(
    df: pd.DataFrame,
    *,
    title: str,
    overlay_mode: str,
    asof_label: str,
    threshold: Optional[float],
) -> go.Figure:
    """
    Composite chart:
    - Bars on primary axis: SYD deviation (%)
    - Optional line on secondary axis: PR% or ΔPR%
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    if df.empty:
        fig.update_layout(
            title=title,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=560,
        )
        return fig

    df = df.copy()
    df["equipment_key"] = df["site_name"].astype(str) + " | " + df["equipment_name"].astype(str)

    # Color palette per site
    sites = list(dict.fromkeys(df["site_name"].tolist()))
    palette = ["#60a5fa", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#fb7185", "#22c55e", "#f97316"]
    color_by_site = {s: palette[i % len(palette)] for i, s in enumerate(sites)}

    # Bars: group by site
    for s in sites:
        sdf = df[df["site_name"] == s].copy()
        base_color = color_by_site[s]
        if "is_continuously_deviating" in sdf.columns:
            bar_colors = ["#ef4444" if bool(v) else base_color for v in sdf["is_continuously_deviating"].tolist()]
        else:
            bar_colors = base_color
        custom = list(
            zip(
                sdf["site_name"].tolist(),
                sdf["equipment_name"].tolist(),
                sdf["syd_dev_pct"].tolist(),
                sdf.get("overlay_val", pd.Series([None] * len(sdf))).tolist(),
            )
        )
        if overlay_mode != "None":
            overlay_part = f"{overlay_mode}: %{{customdata[3]:.2f}}%<br>"
        else:
            overlay_part = ""
        hover_tmpl = (
            "<b>%{customdata[0]}</b><br>"
            "Equipment: %{customdata[1]}<br>"
            "SYD deviation: %{customdata[2]:.2f}%<br>"
            + overlay_part
            + f"{asof_label}<extra></extra>"
        )
        fig.add_trace(
            go.Bar(
                x=sdf["equipment_key"].tolist(),
                y=sdf["syd_dev_pct"].tolist(),
                name=f"{s} (SYD)",
                marker_color=bar_colors,
                customdata=custom,
                hovertemplate=hover_tmpl,
            ),
            secondary_y=False,
        )

    # Line overlay (per site), only for equipment in bars
    if overlay_mode != "None" and "overlay_val" in df.columns:
        overlay_color = "#06b6d4" if overlay_mode == "PR%" else "#a855f7"  # PR=cyan, ΔPR=purple
        dashes = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
        for s in sites:
            sdf = df[df["site_name"] == s].copy()
            vals = [float(v) if v is not None else 0.0 for v in sdf["overlay_val"].tolist()]
            labels = [f"{v:.1f}%" for v in vals]
            fig.add_trace(
                go.Scatter(
                    x=sdf["equipment_key"].tolist(),
                    y=vals,
                    name=f"{s} ({overlay_mode})",
                    mode="lines+markers+text",
                    line=dict(color=overlay_color, width=2, dash=dashes[sites.index(s) % len(dashes)]),
                    marker=dict(size=7, color=overlay_color),
                    text=labels,
                    textposition="top center",
                    # Dark, larger labels so PR / ΔPR values are clearly visible
                    textfont=dict(color="black", size=13),
                    hovertemplate=(
                        "<b>%{customdata}</b><br>"
                        f"{overlay_mode}: %{{y:.2f}}%<br>"
                        + f"{asof_label}<extra></extra>"
                    ),
                    customdata=[f"{s} | {e}" for e in sdf["equipment_name"].tolist()],
                    cliponaxis=False,
                ),
                secondary_y=True,
            )

    # Threshold line on SYD axis
    if threshold is not None:
        try:
            fig.add_hline(y=float(threshold), line_color="red", line_dash="dot", annotation_text="Threshold", annotation_position="top left")
        except Exception:
            pass

    # Layout / axes
    fig.update_layout(
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10),
        height=620,
        legend_title_text="Site",
        barmode="group",
    )
    fig.update_yaxes(
        title_text="SYD deviation (%)",
        zeroline=True,
        zerolinecolor="rgba(15, 23, 42, 0.55)",
        gridcolor="rgba(148, 163, 184, 0.25)",
        secondary_y=False,
    )
    fig.update_yaxes(
        title_text=f"{overlay_mode} (%)" if overlay_mode != "None" else "",
        zeroline=True,
        zerolinecolor="rgba(15, 23, 42, 0.25)",
        gridcolor="rgba(148, 163, 184, 0.15)",
        secondary_y=True,
        showgrid=False,
    )
    fig.update_xaxes(
        title="Equipment (site | equipment)",
        tickangle=-45,
        tickfont=dict(size=10),
        automargin=True,
    )
    return fig


def _meta_universe_from_plot_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Meta "universe" must match the plotted equipment exactly (post-threshold).
    If a concrete date is available (Case A), include it so Meta can fetch
    capacity for the same date used by the plot.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["site_name", "equipment_name", "date"])
    cols = ["site_name", "equipment_name"]
    if "date" in df.columns:
        cols.append("date")
    u = df[cols].drop_duplicates().copy()
    # Normalize expected columns so downstream code is simple
    if "date" not in u.columns:
        u["date"] = pd.NaT
    return u[["site_name", "equipment_name", "date"]]


def _fetch_dc_capacity_latest_for_universe(db_path: str, universe: pd.DataFrame) -> pd.DataFrame:
    """
    Case A (no range): fetch dc_capacity_kwp for the exact date used in the plot.
    Universe must contain: site_name, equipment_name, date (per-row).
    """
    if universe is None or universe.empty:
        return pd.DataFrame(columns=["site_name", "equipment_name", "dc_capacity_kwp"])
    u = universe.dropna(subset=["site_name", "equipment_name", "date"]).copy()
    if u.empty:
        return pd.DataFrame(columns=["site_name", "equipment_name", "dc_capacity_kwp"])
    con = _connect(db_path)
    try:
        con.register("universe", u[["site_name", "equipment_name", "date"]])
        return con.execute(
            """
            select
              u.site_name,
              u.equipment_name,
              dc.dc_capacity_kwp
            from universe u
            left join dc_capacity dc
              on dc.site_name = u.site_name
             and dc.equipment_name = u.equipment_name
             and dc.date = u.date
            order by 1,2
            """
        ).fetchdf()
    finally:
        con.close()


def _fetch_dc_capacity_avg_for_universe_range(db_path: str, universe: pd.DataFrame, d1: date, d2: date) -> pd.DataFrame:
    """
    Case B (range): compute AVG(dc_capacity_kwp) over the selected date range.
    Universe must contain: site_name, equipment_name (date ignored).
    """
    if universe is None or universe.empty:
        return pd.DataFrame(columns=["site_name", "equipment_name", "dc_capacity_kwp"])
    u = universe[["site_name", "equipment_name"]].dropna().drop_duplicates().copy()
    if u.empty:
        return pd.DataFrame(columns=["site_name", "equipment_name", "dc_capacity_kwp"])
    con = _connect(db_path)
    try:
        con.register("universe", u)
        return con.execute(
            """
            select
              u.site_name,
              u.equipment_name,
              avg(dc.dc_capacity_kwp) as dc_capacity_kwp
            from universe u
            left join dc_capacity dc
              on dc.site_name = u.site_name
             and dc.equipment_name = u.equipment_name
             and dc.date between ? and ?
            group by 1,2
            order by 1,2
            """,
            [d1, d2],
        ).fetchdf()
    finally:
        con.close()


def _render_meta_panel(db_path: str, *, universe: pd.DataFrame, d1: Optional[date], d2: Optional[date]) -> None:
    """
    UI: Meta is informational and must match the plotted universe exactly.
    """
    # Render control (right above chart). Hidden unless a plot exists.
    c_meta, c_hint = st.columns([1.3, 8.7], vertical_alignment="center")
    with c_meta:
        show = st.toggle("Meta", value=bool(st.session_state.get("ot_meta_open", False)), key="ot_meta_open")
    with c_hint:
        st.caption("Contextual DC capacity for exactly the equipment shown in the current plot.")

    if not show:
        return

    with st.container():
        st.markdown("### Meta (DC Capacity)")
        if universe is None or universe.empty:
            st.info("No equipment is currently plotted, so Meta is unavailable.")
            return

        is_range = bool(d1 and d2)
        if is_range:
            meta_df = _fetch_dc_capacity_avg_for_universe_range(db_path, universe, d1, d2)
        else:
            meta_df = _fetch_dc_capacity_latest_for_universe(db_path, universe)

        if meta_df.empty:
            st.info("No DC capacity rows found for the plotted equipment.")
            return

        out = meta_df.rename(
            columns={
                "site_name": "Site Name",
                "equipment_name": "Equipment Name",
                "dc_capacity_kwp": "DC Capacity (kWp)",
            }
        ).copy()
        out["DC Capacity (kWp)"] = pd.to_numeric(out["DC Capacity (kWp)"], errors="coerce")
        st.dataframe(out, use_container_width=True, hide_index=True, height=320)


def _render_aggrid_table(df: pd.DataFrame, *, key: str, height: int = 360) -> None:
    if df is None or df.empty:
        st.info("No comments found for the current selection.")
        return
    if not _HAS_AGGRID:
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)
        return

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(sortable=True, filter=True, resizable=True, wrapText=True, autoHeight=True)
    gb.configure_grid_options(domLayout="normal")
    grid_options = gb.build()
    custom_css = {
        ".ag-row-hover": {"background-color": "rgba(255, 179, 0, 0.14) !important"},
        ".ag-theme-balham": {"font-family": "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"},
        ".ag-header-cell-label": {"font-weight": "700"},
    }
    AgGrid(  # type: ignore[misc]
        df,
        gridOptions=grid_options,
        theme="balham",
        height=height,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=False,
        custom_css=custom_css,
        enable_enterprise_modules=False,
        update_mode=GridUpdateMode.NO_UPDATE,  # type: ignore[union-attr]
        key=key,
    )


def _parse_listish(v: Any) -> list[str]:
    """
    Supabase may return arrays as Python lists; handle string fallbacks too.
    """
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip() != ""]
    s = str(v).strip()
    if s == "":
        return []
    # naive parse for "{a,b}" or "a,b"
    s = s.strip("{}")
    parts = [p.strip().strip('"').strip("'") for p in s.split(",")]
    return [p for p in parts if p]


@st.cache_data(show_spinner=False)
def _fetch_supabase_comments_for_range(sites: tuple[str, ...], start: date, end: date) -> pd.DataFrame:
    """
    Fetch comments that overlap the selected range (inclusive):
      comment.start_date <= end AND comment.end_date >= start
    """
    if not sites:
        return pd.DataFrame()
    sb = get_supabase_client(prefer_service_role=True)
    resp = (
        sb.table("zelestra_comments")
        .select("*")
        .in_("site_name", list(sites))
        .lte("start_date", end.isoformat())
        .gte("end_date", start.isoformat())
        .execute()
    )
    rows = resp.data or []
    return pd.DataFrame(rows)


def _enrich_comments_with_dc_capacity(db_path: str, grouped: pd.DataFrame) -> pd.DataFrame:
    """
    Add DC Capacity (kWp) as AVG(dc_capacity_kwp) for each (site,equipment) over Start Date → End Date.
    Expects grouped to contain: site_name, equipment_name, start_date, end_date.
    """
    if grouped is None or grouped.empty:
        grouped = grouped.copy() if grouped is not None else pd.DataFrame()
        grouped["dc_capacity_kwp"] = pd.NA
        return grouped

    g = grouped.copy()
    # Ensure dates are DATE-like
    g["_sd"] = pd.to_datetime(g["start_date"], errors="coerce").dt.date
    g["_ed"] = pd.to_datetime(g["end_date"], errors="coerce").dt.date
    g = g.dropna(subset=["site_name", "equipment_name", "_sd", "_ed"])
    if g.empty:
        grouped["dc_capacity_kwp"] = pd.NA
        return grouped

    intervals = g[["site_name", "equipment_name", "_sd", "_ed"]].copy()
    intervals["idx"] = intervals.index.astype(int)

    con = _connect(db_path)
    try:
        con.register("intervals", intervals)
        cap = con.execute(
            """
            select
              i.idx,
              avg(dc.dc_capacity_kwp) as dc_capacity_kwp
            from intervals i
            left join dc_capacity dc
              on dc.site_name = i.site_name
             and dc.equipment_name = i.equipment_name
             and dc.date between i._sd and i._ed
            group by 1
            """
        ).fetchdf()
    finally:
        con.close()

    cap = cap.set_index("idx")
    grouped = grouped.copy()
    grouped["dc_capacity_kwp"] = grouped.index.to_series().map(cap["dc_capacity_kwp"] if "dc_capacity_kwp" in cap.columns else pd.Series(dtype=float))
    return grouped


def _build_comments_view_for_plot(
    db_path: str,
    plot_df: pd.DataFrame,
    *,
    selected_sites: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Below-plot comments view:
    - Only equipment present in the plot
    - Grouped by (site, equipment, reason)
    - Merge range start/end per group
    - Deviation shown as MEDIAN of stored deviations for that group
    - Past column shows counts per reason for the same equipment within the range
    """
    if plot_df is None or plot_df.empty:
        return pd.DataFrame()
    # 1) Universe (plotted equipment)
    universe = plot_df[["site_name", "equipment_name"]].drop_duplicates().copy()
    if universe.empty:
        return pd.DataFrame()

    # 2) Fetch comments
    sup = _fetch_supabase_comments_for_range(tuple(selected_sites), start, end)
    if sup.empty:
        return pd.DataFrame()

    # 3) Normalize required columns
    for col in ["site_name", "start_date", "end_date", "equipment_names", "reasons", "remarks", "deviation"]:
        if col not in sup.columns:
            sup[col] = None

    # 4) Parse + filter by date overlap (vectorized)
    sup["_sd"] = pd.to_datetime(sup["start_date"], errors="coerce").dt.date
    sup["_ed"] = pd.to_datetime(sup["end_date"], errors="coerce").dt.date
    sup = sup.dropna(subset=["site_name", "_sd", "_ed"]).copy()
    if sup.empty:
        return pd.DataFrame()
    sup = sup[(sup["_ed"] >= start) & (sup["_sd"] <= end)].copy()
    if sup.empty:
        return pd.DataFrame()

    def _ensure_list(x: Any) -> list[str]:
        if isinstance(x, list):
            return [str(v) for v in x]
        return [str(v) for v in _parse_listish(x)]

    sup["equipment_names"] = sup["equipment_names"].map(_ensure_list)
    sup["reasons"] = sup["reasons"].map(_ensure_list)

    # 5) Explode to (site, equipment, reason) rows and inner-join to plotted universe
    exp = sup.explode("equipment_names").rename(columns={"equipment_names": "equipment_name"})
    exp = exp.explode("reasons").rename(columns={"reasons": "reason"})
    exp["site_name"] = exp["site_name"].astype(str)
    exp["equipment_name"] = exp["equipment_name"].astype(str)
    exp["reason"] = exp["reason"].fillna("").astype(str)
    universe["site_name"] = universe["site_name"].astype(str)
    universe["equipment_name"] = universe["equipment_name"].astype(str)
    exp = exp.merge(universe, on=["site_name", "equipment_name"], how="inner")
    if exp.empty:
        return pd.DataFrame()

    # Standardize fields
    exp["start_date"] = exp["_sd"]
    exp["end_date"] = exp["_ed"]
    exp["remarks"] = exp["remarks"].fillna("").astype(str).str.strip()
    exp["deviation"] = pd.to_numeric(exp["deviation"], errors="coerce")

    # Past summary: counts by reason for each equipment (vectorized)
    counts = exp.groupby(["site_name", "equipment_name", "reason"]).size().reset_index(name="cnt")
    counts = counts.sort_values(["site_name", "equipment_name", "cnt", "reason"], ascending=[True, True, False, True])
    counts["part"] = counts.apply(
        lambda r: f"{r['reason']}({int(r['cnt'])})" if str(r["reason"]).strip() != "" else "",
        axis=1,
    )
    past_series = (
        counts[counts["part"] != ""]
        .groupby(["site_name", "equipment_name"])["part"]
        .apply(lambda s: ", ".join(s.tolist()))
    )
    past_map: dict[tuple[str, str], str] = past_series.to_dict()

    # Merge overlapping/touching intervals per (site, equipment, reason).
    grouped_rows: list[dict[str, Any]] = []
    for (sname, eq, reason), g in exp.groupby(["site_name", "equipment_name", "reason"], dropna=False):
        g = g.sort_values(["start_date", "end_date"])
        cur_sd: Optional[date] = None
        cur_ed: Optional[date] = None
        cur_devs: list[float] = []
        cur_remarks: list[str] = []

        def flush() -> None:
            nonlocal cur_sd, cur_ed, cur_devs, cur_remarks
            if cur_sd is None or cur_ed is None:
                return
            med = float(pd.Series(cur_devs).median()) if cur_devs else None
            uniq = [x for x in dict.fromkeys([str(x).strip() for x in cur_remarks if str(x).strip() != ""])]
            grouped_rows.append(
                {
                    "site_name": sname,
                    "equipment_name": eq,
                    "reason": reason,
                    "start_date": cur_sd,
                    "end_date": cur_ed,
                    "deviation": med,
                    "remarks": " | ".join(uniq[:4]),
                }
            )
            cur_sd, cur_ed, cur_devs, cur_remarks = None, None, [], []

        for rr in g.itertuples(index=False):
            sd = getattr(rr, "start_date")
            ed = getattr(rr, "end_date")
            dv = getattr(rr, "deviation")
            rm = getattr(rr, "remarks")

            if cur_sd is None:
                cur_sd, cur_ed = sd, ed
            else:
                if sd <= (cur_ed + timedelta(days=1)):
                    if ed > cur_ed:
                        cur_ed = ed
                else:
                    flush()
                    cur_sd, cur_ed = sd, ed

            if dv is not None and pd.notna(dv):
                cur_devs.append(float(dv))
            if rm:
                cur_remarks.append(str(rm))

        flush()

    grouped = pd.DataFrame(grouped_rows)
    if grouped.empty:
        return pd.DataFrame()
    grouped["past"] = grouped.apply(lambda x: past_map.get((str(x["site_name"]), str(x["equipment_name"])), ""), axis=1)

    # DC Capacity enrichment (pure metadata)
    grouped = _enrich_comments_with_dc_capacity(db_path, grouped)

    # Convert dates to strings for proper display (source of truth from Supabase)
    if "start_date" in grouped.columns:
        grouped["start_date"] = grouped["start_date"].apply(lambda x: x.isoformat() if isinstance(x, date) else str(x) if x is not None else "")
    if "end_date" in grouped.columns:
        grouped["end_date"] = grouped["end_date"].apply(lambda x: x.isoformat() if isinstance(x, date) else str(x) if x is not None else "")

    out = grouped.rename(
        columns={
            "site_name": "Site Name",
            "equipment_name": "Equipment Name",
            "reason": "Reason",
            "start_date": "Start Date",
            "end_date": "End Date",
            "deviation": "Deviation (median)",
            "dc_capacity_kwp": "DC Capacity (kWp)",
            "remarks": "Remarks",
            "past": "Past",
        }
    )
    # Nice sorting
    out["Deviation (median)"] = pd.to_numeric(out["Deviation (median)"], errors="coerce")
    if "DC Capacity (kWp)" in out.columns:
        out["DC Capacity (kWp)"] = pd.to_numeric(out["DC Capacity (kWp)"], errors="coerce")
        out["DC Capacity (kWp)"] = out["DC Capacity (kWp)"].apply(lambda x: f"{x:.2f} kWp" if pd.notna(x) else "")
    out = out.sort_values(["Site Name", "Equipment Name", "Reason"], ascending=[True, True, True])
    return out


def _render_comments_panel_for_plot(
    db_path: str,
    plot_df: pd.DataFrame,
    *,
    selected_sites: list[str],
    start: Optional[date],
    end: Optional[date],
) -> None:
    if plot_df is None or plot_df.empty:
        return
    if not selected_sites:
        return
    # Determine window
    if start is None or end is None:
        # Fallback: use plotted dates if present
        if "date" in plot_df.columns:
            ds = pd.to_datetime(plot_df["date"], errors="coerce").dt.date.dropna()
            if not ds.empty:
                start = min(ds)
                end = max(ds)
    if start is None or end is None:
        return

    # Persist expander state across reruns (e.g., after download clicks)
    # Use session_state to remember if user has expanded it
    expander_key = "ot_comments_expander_open"
    # If expander was previously opened, keep it open; otherwise collapsed by default
    was_open = st.session_state.get(expander_key, False)
    
    with st.expander("Comments for plotted equipment (Supabase)", expanded=was_open):
        # Once inside the expander, mark it as open (will persist on next rerun)
        st.session_state[expander_key] = True
        st.caption("Filtered to exactly the equipment shown in the chart. Grouped by equipment + reason; deviation shown as median.")
        try:
            table = _build_comments_view_for_plot(db_path, plot_df, selected_sites=selected_sites, start=start, end=end)
            if table.empty:
                st.info("No comments found for the plotted equipment in this date range.")
            else:
                _render_aggrid_table(table, key="ot_comments_for_plot", height=360)
                
                # Download buttons for comments table (always visible when table has data)
                col_csv, col_png = st.columns(2)
                with col_csv:
                    st.download_button(
                        "Download Comments (CSV)",
                        data=table.to_csv(index=False).encode("utf-8"),
                        file_name="operation_theatre_comments.csv",
                        mime="text/csv",
                        key="ot_comments_download_csv",
                    )
                with col_png:
                    try:
                        # Create a simple table visualization as PNG
                        import matplotlib.pyplot as plt
                        import matplotlib
                        matplotlib.use("Agg")
                        fig_table, ax = plt.subplots(figsize=(12, max(6, len(table) * 0.3)))
                        ax.axis("tight")
                        ax.axis("off")
                        table_display = table.copy()
                        # Truncate long text for display
                        for col in table_display.columns:
                            if table_display[col].dtype == "object":
                                table_display[col] = table_display[col].astype(str).str[:30] + "..."
                        tbl = ax.table(cellText=table_display.values, colLabels=table_display.columns, cellLoc="left", loc="center")
                        tbl.auto_set_font_size(False)
                        tbl.set_fontsize(8)
                        tbl.scale(1, 1.5)
                        plt.tight_layout()
                        from io import BytesIO
                        buf = BytesIO()
                        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                        buf.seek(0)
                        plt.close(fig_table)
                        st.download_button(
                            "Download Comments (PNG)",
                            data=buf.getvalue(),
                            file_name="operation_theatre_comments.png",
                            mime="image/png",
                            key="ot_comments_download_png",
                        )
                    except Exception:
                        st.caption("PNG download unavailable (matplotlib not installed).")
        except Exception as e:
            st.error(f"Failed to build comments table: {e}")
            import traceback
            st.code(traceback.format_exc())


def _fetch_continuous_deviation_flags(
    db_path: str,
    *,
    universe_with_anchor: pd.DataFrame,
    continuous_days: int,
    threshold_to_use: float,
) -> pd.DataFrame:
    """
    For each (site_name, equipment_name) in the plotted universe, check if SYD
    is below threshold_to_use for ALL of the last N days prior to anchor_date:
      anchor_date-1 ... anchor_date-N
    Missing days break continuity.
    """
    if universe_with_anchor is None or universe_with_anchor.empty:
        return pd.DataFrame(columns=["site_name", "equipment_name", "is_continuously_deviating"])
    n = int(continuous_days)
    if n <= 0:
        return pd.DataFrame(columns=["site_name", "equipment_name", "is_continuously_deviating"])

    u = universe_with_anchor[["site_name", "equipment_name", "anchor_date"]].dropna(subset=["site_name", "equipment_name", "anchor_date"]).copy()
    if u.empty:
        return pd.DataFrame(columns=["site_name", "equipment_name", "is_continuously_deviating"])

    con = _connect(db_path)
    try:
        con.register("universe", u)
        # DuckDB supports date - integer arithmetic for DATE types.
        flags = con.execute(
            """
            select
              u.site_name,
              u.equipment_name,
              case
                when count(distinct s.date) = ?
                 and sum(case when (s.syd_percent * 100.0) < ? then 1 else 0 end) = ?
                then true else false
              end as is_continuously_deviating
            from universe u
            left join syd s
              on s.site_name = u.site_name
             and s.equipment_name = u.equipment_name
             and s.date between (u.anchor_date - ?) and (u.anchor_date - 1)
            group by 1,2
            """,
            [n, float(threshold_to_use), n, n],
        ).fetchdf()
        return flags
    finally:
        con.close()


def render(db_path: str) -> None:
    st.markdown("## Operation Theatre")

    sites = list_sites_from_syd(db_path)
    if not sites:
        st.error("No sites found in `syd`. Load data first.")
        return

    dmin, dmax = date_bounds_from_syd(db_path)

    # Controls (in main page, horizontal)
    c_site, c_thr, c_n, c_overlay, c_from, c_to, c_btn = st.columns([4.6, 1.8, 1.8, 1.8, 1.8, 1.8, 1.4], vertical_alignment="bottom")

    with c_site:
        options = ["Portfolio", *sites]
        # Persist selection across tabs
        default_sites = st.session_state.get("ot_site_selection", [])
        selected = st.multiselect("Site Name", options=options, default=default_sites, key="ot_site_multiselect")
        st.session_state["ot_site_selection"] = selected  # Persist
        if "Portfolio" in selected:
            selected_sites = sites
        else:
            selected_sites = [s for s in selected if s in sites]

    with c_thr:
        # Persist threshold across tabs
        default_threshold = st.session_state.get("ot_threshold", "")
        thr_txt = st.text_input("Threshold (%) (optional)", value=default_threshold, placeholder="e.g. -3 or 2", key="ot_threshold_input")
        st.session_state["ot_threshold"] = thr_txt  # Persist
        threshold: Optional[float]
        try:
            threshold = float(thr_txt) if thr_txt.strip() != "" else None
        except Exception:
            threshold = None
            if thr_txt.strip() != "":
                st.warning("Threshold must be a number.")

    with c_n:
        # Persist continuous days across tabs
        default_continuous = st.session_state.get("ot_continuous_days", 7)
        continuous_days = st.number_input("Continuous Days (N)", min_value=1, max_value=60, value=default_continuous, step=1, key="ot_continuous_days_input")
        st.session_state["ot_continuous_days"] = continuous_days  # Persist

    with c_overlay:
        # Persist overlay mode across tabs
        default_overlay_idx = {"None": 0, "PR%": 1, "ΔPR%": 2}.get(st.session_state.get("ot_overlay_mode", "None"), 0)
        overlay_mode = st.selectbox("Overlay", options=["None", "PR%", "ΔPR%"], index=default_overlay_idx, key="ot_overlay_select")
        st.session_state["ot_overlay_mode"] = overlay_mode  # Persist

    with c_from:
        # Persist date across tabs (no default on initial load)
        default_d1 = st.session_state.get("ot_from_date", None)
        d1 = st.date_input("From", value=default_d1, min_value=dmin, max_value=dmax, key="ot_from", format="YYYY-MM-DD") if dmin and dmax else None
        if d1 is not None:
            st.session_state["ot_from_date"] = d1  # Persist

    with c_to:
        # Persist date across tabs (no default on initial load)
        default_d2 = st.session_state.get("ot_to_date", None)
        d2 = st.date_input("To", value=default_d2, min_value=dmin, max_value=dmax, key="ot_to", format="YYYY-MM-DD") if dmin and dmax else None
        if d2 is not None:
            st.session_state["ot_to_date"] = d2  # Persist

    valid = bool(selected_sites)
    with c_btn:
        plot = st.button("Plot Now", type="primary", disabled=not valid, use_container_width=True)

    # Reuse cached plot on reruns (e.g., after download clicks), but ONLY if cache is valid.
    # `dashboard.py` initializes these keys to None, so checking mere presence is not enough.
    fig_last = st.session_state.get("ot_last_fig")
    df_last = st.session_state.get("ot_last_df")
    has_valid_cache = isinstance(fig_last, go.Figure) and isinstance(df_last, pd.DataFrame) and not df_last.empty

    if not plot and has_valid_cache:
        # Meta (only appears once a plot exists) — rendered right above the chart
        universe = st.session_state.get("ot_last_meta_universe", _meta_universe_from_plot_df(df_last))
        _render_meta_panel(
            db_path,
            universe=universe,
            d1=st.session_state.get("ot_last_meta_d1"),
            d2=st.session_state.get("ot_last_meta_d2"),
        )

        st.plotly_chart(fig_last, use_container_width=True, config={"displayModeBar": True})

        # Comments table below plot (collapsed by default)
        _render_comments_panel_for_plot(
            db_path,
            df_last,
            selected_sites=st.session_state.get("ot_last_plot_sites", []),
            start=st.session_state.get("ot_last_plot_d1"),
            end=st.session_state.get("ot_last_plot_d2"),
        )

        # Download buttons (always shown when plot exists)
        st.download_button(
            "Download Raw Data (CSV)",
            data=df_last.to_csv(index=False).encode("utf-8"),
            file_name="operation_theatre_syd.csv",
            mime="text/csv",
        )
        try:
            png_bytes = pio.to_image(fig_last, format="png", width=1400, height=700, scale=2)
            st.download_button(
                "Download Chart (PNG)",
                data=png_bytes,
                file_name="operation_theatre.png",
                mime="image/png",
            )
        except Exception:
            st.caption("PNG download unavailable (kaleido not installed).")
        return

    if not plot:
        st.caption("Select at least one Site Name to enable Plot Now.")
        return

    progress = st.progress(0)
    with st.spinner("Loading SYD / PR deviations..."):
        progress.progress(10)
        if d1 and d2:
            # Case B: median aggregation
            syd_df = fetch_median_syd(db_path, selected_sites, d1, d2)
            pr_df = fetch_median_pr(db_path, selected_sites, d1, d2) if overlay_mode != "None" else pd.DataFrame()
            asof_label = f"Range: {d1} → {d2}"
            title = "Median SYD Deviation (%) by Equipment"
            anchor_mode = "from_date"
        elif d1 and not d2:
            # Case A': single selected date
            syd_df = fetch_syd_for_date(db_path, selected_sites, d1)
            pr_df = fetch_pr_for_date(db_path, selected_sites, d1) if overlay_mode != "None" else pd.DataFrame()
            asof_label = f"As of: {d1}"
            title = "SYD Deviation (%) by Equipment"
            anchor_mode = "from_date"
        else:
            # Case A: most recent day per site
            syd_df = fetch_latest_syd(db_path, selected_sites)
            pr_df = fetch_latest_pr_for_sites(db_path, selected_sites) if overlay_mode != "None" else pd.DataFrame()
            # best-effort label (dates can vary per site; show 'Latest per site')
            asof_label = "As of: latest per site"
            title = "SYD Deviation (%) by Equipment (Latest per Site)"
            anchor_mode = "per_row_date"
        progress.progress(45)

        # Threshold applies to SYD universe (after aggregation if used)
        syd_df = _apply_threshold(syd_df, threshold, value_col="syd_dev_pct")
        progress.progress(60)

        # Equipment universe: only equipment present in SYD bars
        if syd_df.empty:
            df = syd_df
        else:
            df = syd_df.copy()
            if overlay_mode != "None" and not pr_df.empty:
                # Join PR only for equipment in SYD
                join_cols = ["site_name", "equipment_name"]
                if "date" in syd_df.columns and "date" in pr_df.columns:
                    join_cols = ["site_name", "equipment_name", "date"]
                df = df.merge(pr_df, on=join_cols, how="left")

                # Compute overlay value
                df["pr_pct"] = pd.to_numeric(df.get("pr_pct"), errors="coerce").fillna(0.0)
                if overlay_mode == "PR%":
                    df["overlay_val"] = df["pr_pct"]
                else:
                    # ΔPR%: per-site max reference
                    max_pr = df.groupby("site_name")["pr_pct"].transform("max").replace({0.0: pd.NA})
                    df["overlay_val"] = ((df["pr_pct"] - max_pr) / max_pr) * 100.0
                    df["overlay_val"] = df["overlay_val"].fillna(0.0)
            progress.progress(80)

            # Continuous deviation highlighting (purely visual; does not change numeric values)
            threshold_to_use = float(threshold) if threshold is not None else 0.0
            u = df[["site_name", "equipment_name"]].drop_duplicates().copy()
            if anchor_mode == "per_row_date" and "date" in df.columns:
                # Use the plotted date as anchor per equipment (latest per site mode).
                u = df[["site_name", "equipment_name", "date"]].drop_duplicates().rename(columns={"date": "anchor_date"})
            else:
                # Use the selected FROM date (single date or range).
                anchor_date = d1
                if anchor_date is not None:
                    u["anchor_date"] = anchor_date
                else:
                    # No anchor available -> skip flags
                    u["anchor_date"] = pd.NaT
            flags = _fetch_continuous_deviation_flags(
                db_path,
                universe_with_anchor=u,
                continuous_days=int(continuous_days),
                threshold_to_use=threshold_to_use,
            )
            if not flags.empty:
                df = df.merge(flags, on=["site_name", "equipment_name"], how="left")
                df["is_continuously_deviating"] = df["is_continuously_deviating"].fillna(False)
        progress.progress(100)
    progress.empty()

    fig = _build_composite_chart(df, title=title, overlay_mode=overlay_mode, asof_label=asof_label, threshold=threshold)

    meta_universe = _meta_universe_from_plot_df(df)
    st.session_state["ot_last_meta_universe"] = meta_universe
    st.session_state["ot_last_meta_d1"] = d1 if (d1 and d2) else None
    st.session_state["ot_last_meta_d2"] = d2 if (d1 and d2) else None
    st.session_state["ot_last_plot_sites"] = list(selected_sites)
    if d1 and d2:
        st.session_state["ot_last_plot_d1"] = d1
        st.session_state["ot_last_plot_d2"] = d2
    elif d1 and not d2:
        st.session_state["ot_last_plot_d1"] = d1
        st.session_state["ot_last_plot_d2"] = d1
    else:
        # Latest-per-site: derive window from plotted dates if present; else None
        st.session_state["ot_last_plot_d1"] = None
        st.session_state["ot_last_plot_d2"] = None

    # Persist state for reuse on reruns (e.g., after download clicks) - BEFORE rendering panels
    # NOTE: ot_last_plot_* is already computed above (handles range/single/latest-per-site).
    st.session_state["ot_last_df"] = df
    st.session_state["ot_last_fig"] = fig

    # Meta (only appears once a plot exists), rendered right above chart
    _render_meta_panel(db_path, universe=meta_universe, d1=st.session_state["ot_last_meta_d1"], d2=st.session_state["ot_last_meta_d2"])

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})

    # Comments table below plot (collapsed by default) - uses persisted state so it survives download clicks
    _render_comments_panel_for_plot(
        db_path,
        df,
        selected_sites=list(selected_sites),
        start=d1,
        end=d2,
    )

    st.download_button(
        "Download Raw Data (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="operation_theatre_syd.csv",
        mime="text/csv",
    )

    try:
        png_bytes = pio.to_image(fig, format="png", width=1400, height=700, scale=2)
        st.download_button(
            "Download Chart (PNG)",
            data=png_bytes,
            file_name="operation_theatre.png",
            mime="image/png",
        )
    except Exception:
        st.caption("PNG download unavailable (kaleido not installed).")

    # (intentionally no second comments panel render here)
