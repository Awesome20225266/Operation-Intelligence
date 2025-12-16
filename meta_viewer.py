from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO
from typing import Any, Optional

import duckdb
import pandas as pd
import streamlit as st

from aws_duckdb import get_duckdb_connection


def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    return get_duckdb_connection(db_local=db_path)


@st.cache_data(show_spinner=False)
def _list_sites(db_path: str) -> list[str]:
    con = _connect(db_path)
    try:
        rows = con.execute("select distinct site_name from daily_kpi order by site_name").fetchall()
        return [str(r[0]) for r in rows]
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def _date_bounds(db_path: str, *, site_name: str) -> tuple[Optional[date], Optional[date]]:
    con = _connect(db_path)
    try:
        row = con.execute(
            "select min(date) as dmin, max(date) as dmax from daily_kpi where site_name = ?",
            [site_name],
        ).fetchone()
        if not row:
            return (None, None)
        return (row[0], row[1])
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def _list_equipment(db_path: str, *, site_name: str) -> list[str]:
    con = _connect(db_path)
    try:
        rows = con.execute(
            "select distinct equipment_name from syd where site_name = ? order by equipment_name",
            [site_name],
        ).fetchall()
        return [str(r[0]) for r in rows]
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def _table_columns(db_path: str, table: str) -> list[str]:
    con = _connect(db_path)
    try:
        df = con.execute(f"pragma table_info('{table}')").fetchdf()
        return df["name"].astype(str).tolist() if "name" in df.columns else []
    finally:
        con.close()


@dataclass(frozen=True)
class TagUniverse:
    mode: str  # "site" | "equipment"
    tag_to_source: dict[str, tuple[str, str]]  # tag -> (table, col)
    tags: list[str]


def _build_tag_universe(db_path: str, *, equipment_selected: bool) -> TagUniverse:
    if not equipment_selected:
        # Site-level tags: budget_kpi (except site_name,date) + daily_kpi (except site_name,date,days)
        daily_cols = [c for c in _table_columns(db_path, "daily_kpi") if c not in {"site_name", "date", "days"}]
        budget_cols = [c for c in _table_columns(db_path, "budget_kpi") if c not in {"site_name", "date"}]
        tag_to_source: dict[str, tuple[str, str]] = {}
        for c in daily_cols:
            tag_to_source[c] = ("daily_kpi", c)
        for c in budget_cols:
            tag_to_source[c] = ("budget_kpi", c)
        tags = sorted(tag_to_source.keys())
        return TagUniverse(mode="site", tag_to_source=tag_to_source, tags=tags)

    # Equipment-level tags: dc_capacity/pr/syd (except site_name,date)
    syd_cols = [c for c in _table_columns(db_path, "syd") if c not in {"site_name", "date"}]
    pr_cols = [c for c in _table_columns(db_path, "pr") if c not in {"site_name", "date"}]
    dc_cols = [c for c in _table_columns(db_path, "dc_capacity") if c not in {"site_name", "date"}]
    tag_to_source = {}
    for c in syd_cols:
        tag_to_source[c] = ("syd", c)
    for c in pr_cols:
        tag_to_source[c] = ("pr", c)
    for c in dc_cols:
        tag_to_source[c] = ("dc_capacity", c)
    tags = sorted(tag_to_source.keys())
    return TagUniverse(mode="equipment", tag_to_source=tag_to_source, tags=tags)


def _format_for_display(df: pd.DataFrame, *, tags: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in tags:
        if c not in out.columns:
            continue
        # numeric coercion
        out[c] = pd.to_numeric(out[c], errors="coerce")
        if c.endswith("_percent"):
            out[c] = out[c] * 100.0
        out[c] = out[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "")
    return out


def _fetch_meta_view(
    db_path: str,
    *,
    site_name: str,
    start_date: date,
    end_date: date,
    equipment_names: list[str],
    tags: list[str],
    universe: TagUniverse,
) -> pd.DataFrame:
    con = _connect(db_path)
    try:
        # Register equipment selection (optional)
        eq_df = pd.DataFrame({"equipment_name": equipment_names})
        con.register("equip_sel", eq_df)

        # Build select list
        select_cols: list[str] = ["b.date as date"]
        if universe.mode == "equipment":
            select_cols.append("b.equipment_name as equipment_name")

        # Add tag columns
        for t in tags:
            table, col = universe.tag_to_source[t]
            alias = {"daily_kpi": "dk", "budget_kpi": "bk", "syd": "sy", "pr": "pr", "dc_capacity": "dc"}[table]
            select_cols.append(f"{alias}.{col} as {col}")

        select_sql = ",\n  ".join(select_cols)

        if universe.mode == "site":
            sql = f"""
            with dates as (
              select * from generate_series(?::date, ?::date, interval 1 day) as t(date)
            ),
            base as (
              select date as date from dates
            )
            select
              {select_sql}
            from base b
            left join daily_kpi dk
              on dk.site_name = ? and dk.date = b.date
            left join budget_kpi bk
              on bk.site_name = ? and bk.date = b.date
            order by 1;
            """
            return con.execute(sql, [start_date, end_date, site_name, site_name]).fetchdf()

        # Equipment-level: one row per day per equipment (selected list)
        sql = f"""
        with dates as (
          select * from generate_series(?::date, ?::date, interval 1 day) as t(date)
        ),
        base as (
          select d.date, e.equipment_name
          from dates d
          join equip_sel e on true
        )
        select
          {select_sql}
        from base b
        left join syd sy
          on sy.site_name = ? and sy.date = b.date and sy.equipment_name = b.equipment_name
        left join pr pr
          on pr.site_name = ? and pr.date = b.date and pr.equipment_name = b.equipment_name
        left join dc_capacity dc
          on dc.site_name = ? and dc.date = b.date and dc.equipment_name = b.equipment_name
        order by 1, 2;
        """
        return con.execute(sql, [start_date, end_date, site_name, site_name, site_name]).fetchdf()
    finally:
        con.close()


def render(db_path: str) -> None:
    st.markdown("## Meta Viewer")
    st.caption("Controlled DuckDB browser (no analytics).")

    sites = _list_sites(db_path)
    if not sites:
        st.error("No sites found in `daily_kpi`. Load data first.")
        return

    # ---- Controls (top) ----
    c1, c2, c3, c4 = st.columns([2.2, 3.0, 3.6, 3.2], vertical_alignment="bottom")

    with c1:
        site_opt = st.selectbox("Site Name", options=["(select)", *sites], index=0, key="mv_site")
        site_name = None if site_opt == "(select)" else str(site_opt)

    dmin, dmax = (None, None)
    if site_name:
        dmin, dmax = _date_bounds(db_path, site_name=site_name)

    with c2:
        start_date = st.date_input("From", value=st.session_state.get("mv_from"), min_value=dmin, max_value=dmax, key="mv_from")
    with c3:
        end_date = st.date_input("To", value=st.session_state.get("mv_to"), min_value=dmin, max_value=dmax, key="mv_to")

    with c4:
        equipment_names: list[str] = []
        if site_name:
            eq = _list_equipment(db_path, site_name=site_name)
            equipment_names = st.multiselect(
                "Equipment Name (optional)",
                options=eq,
                default=st.session_state.get("mv_equipment", []),
                key="mv_equipment",
                help="If empty → site-level view. If selected → equipment-level view (one row per day per equipment).",
            )
        else:
            st.caption("Select a site to enable equipment list.")

    if not site_name:
        st.info("Select **Site Name** to proceed.")
        return

    if start_date is None or end_date is None:
        st.info("Select **From** and **To** dates to proceed.")
        return

    if start_date > end_date:
        st.error("From date cannot be after To date.")
        return

    equipment_selected = len(equipment_names) > 0
    universe = _build_tag_universe(db_path, equipment_selected=equipment_selected)

    # ---- Tag selector ----
    st.markdown("### Tags")
    t1, t2 = st.columns([1.2, 8.8], vertical_alignment="bottom")
    with t1:
        select_all = st.checkbox("Select All", value=False, key="mv_select_all")
    with t2:
        default_tags = universe.tags if select_all else st.session_state.get("mv_tags", [])
        selected_tags = st.multiselect(
            "Tag(s)",
            options=universe.tags,
            default=[t for t in default_tags if t in universe.tags],
            key="mv_tags",
            help=("Site-level tags from daily_kpi/budget_kpi." if universe.mode == "site" else "Equipment-level tags from syd/pr/dc_capacity."),
        )

    if not selected_tags:
        st.info("Select at least one tag to view data.")
        return

    # If equipment-level mode but no equipment selected, treat as site-level
    if universe.mode == "equipment" and not equipment_names:
        st.info("Select at least one equipment to use equipment-level tags.")
        return

    # ---- Fetch + display ----
    with st.spinner("Fetching from DuckDB..."):
        df = _fetch_meta_view(
            db_path,
            site_name=site_name,
            start_date=start_date,
            end_date=end_date,
            equipment_names=equipment_names if equipment_names else [],
            tags=selected_tags,
            universe=universe,
        )

    # Format and display
    df_display = _format_for_display(df, tags=selected_tags)

    # Put id columns first
    id_cols = ["date"] + (["equipment_name"] if "equipment_name" in df_display.columns else [])
    df_display = df_display[id_cols + [c for c in selected_tags if c in df_display.columns]]

    st.markdown("### Output")
    st.dataframe(df_display, use_container_width=True, hide_index=True, height=520)

    # ---- Downloads ----
    st.markdown("### Download")
    d1, d2 = st.columns([1.2, 1.2])
    with d1:
        st.download_button(
            "Download CSV",
            data=df_display.to_csv(index=False).encode("utf-8"),
            file_name="meta_viewer.csv",
            mime="text/csv",
            key="mv_download_csv",
        )
    with d2:
        try:
            buf = BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                df_display.to_excel(writer, index=False, sheet_name="meta_viewer")
            buf.seek(0)
            st.download_button(
                "Download Excel",
                data=buf.getvalue(),
                file_name="meta_viewer.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="mv_download_xlsx",
            )
        except Exception:
            st.caption("Excel download unavailable.")


