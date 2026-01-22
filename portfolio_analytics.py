from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Optional

import duckdb
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import plotly.io as pio

from aws_duckdb import get_duckdb_connection


@dataclass(frozen=True)
class PortfolioInputs:
    sites: list[str]
    date_from: date
    date_to: date


@dataclass(frozen=True)
class PortfolioAgg:
    b_energy: float
    a_energy: float
    b_poa: float
    a_poa: float
    b_pa: float
    a_pa: float
    b_ga: float
    a_ga: float


def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    return get_duckdb_connection(db_local=db_path)


@st.cache_data(show_spinner=False)
def list_sites_from_budget(db_path: str) -> list[str]:
    con = _connect(db_path)
    try:
        rows = con.execute("select distinct site_name from budget_kpi order by site_name").fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def date_bounds_from_budget(db_path: str) -> tuple[Optional[date], Optional[date]]:
    con = _connect(db_path)
    try:
        row = con.execute("select min(date) as dmin, max(date) as dmax from budget_kpi").fetchone()
        if not row:
            return None, None
        return row[0], row[1]
    finally:
        con.close()


def _sql_in_list(n: int) -> str:
    return "(" + ",".join(["?"] * n) + ")"


def _parse_date_maybe(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    for fmt in ("%d-%m-%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            continue
    return None


@st.cache_data(show_spinner=False)
def fetch_raw_data(db_path: str, sites: list[str], d1: date, d2: date) -> pd.DataFrame:
    """
    Raw data (site/date level) used for the calculation.
    Returned as a single joined frame for download.
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
              b.site_name,
              b.date,
              b.b_energy_kwh,
              b.b_poa,
              b.b_pa_percent,
              b.b_ga_percent,
              d.abt_export_kwh,
              d.poa,
              d.pa_percent,
              d.ga_percent
            from budget_kpi b
            left join daily_kpi d
              on d.site_name = b.site_name and d.date = b.date
            where b.site_name in {in_clause}
              and b.date between ? and ?
            order by b.site_name, b.date
            """,
            params,
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def compute_aggregates(db_path: str, sites: list[str], d1: date, d2: date) -> PortfolioAgg:
    """
    Aggregation rules (as requested):
    - Energy sums:
      Total_Budget_Energy = SUM(b_energy_kwh)
      Total_Actual_Energy = SUM(abt_export_kwh)
    - POA weighted:
      b_poa_agg = SUM(b_poa*b_energy_kwh) / SUM(b_energy_kwh)
      poa_agg   = SUM(poa*abt_export_kwh) / SUM(abt_export_kwh)
    - PA%, GA% weighted similarly.
    """
    if not sites:
        return PortfolioAgg(0, 0, 0, 0, 0, 0, 0, 0)

    con = _connect(db_path)
    try:
        in_clause = _sql_in_list(len(sites))
        params: list[Any] = [*sites, d1, d2]

        b = con.execute(
            f"""
            select
              coalesce(sum(b_energy_kwh), 0) as b_energy,
              coalesce(sum(b_poa * b_energy_kwh), 0) as b_poa_w,
              coalesce(sum(b_pa_percent * b_energy_kwh), 0) as b_pa_w,
              coalesce(sum(b_ga_percent * b_energy_kwh), 0) as b_ga_w
            from budget_kpi
            where site_name in {in_clause}
              and date between ? and ?
            """,
            params,
        ).fetchone()

        a = con.execute(
            f"""
            select
              coalesce(sum(abt_export_kwh), 0) as a_energy,
              coalesce(sum(poa * abt_export_kwh), 0) as a_poa_w,
              coalesce(sum(pa_percent * abt_export_kwh), 0) as a_pa_w,
              coalesce(sum(ga_percent * abt_export_kwh), 0) as a_ga_w
            from daily_kpi
            where site_name in {in_clause}
              and date between ? and ?
            """,
            params,
        ).fetchone()

        b_energy = float(b[0] or 0.0)
        a_energy = float(a[0] or 0.0)

        b_poa = float(b[1] or 0.0) / b_energy if b_energy else 0.0
        b_pa = float(b[2] or 0.0) / b_energy if b_energy else 0.0
        b_ga = float(b[3] or 0.0) / b_energy if b_energy else 0.0

        a_poa = float(a[1] or 0.0) / a_energy if a_energy else 0.0
        a_pa = float(a[2] or 0.0) / a_energy if a_energy else 0.0
        a_ga = float(a[3] or 0.0) / a_energy if a_energy else 0.0

        return PortfolioAgg(
            b_energy=b_energy,
            a_energy=a_energy,
            b_poa=b_poa,
            a_poa=a_poa,
            b_pa=b_pa,
            a_pa=a_pa,
            b_ga=b_ga,
            a_ga=a_ga,
        )
    finally:
        con.close()


def _severity(actual: float, budget: float) -> float:
    if budget <= 0:
        return 0.0
    ratio = actual / budget
    return max(0.0, 1.0 - ratio)


def build_waterfall(agg: PortfolioAgg) -> tuple[pd.DataFrame, go.Figure]:
    """
    Waterfall:
    Design = 100%
      - POA loss
      - PA loss
      - GA loss
      - Unknown
    Actual = Actual Energy / Budget Energy
    """
    design = 1.0
    actual = (agg.a_energy / agg.b_energy) if agg.b_energy else 0.0
    actual = max(0.0, min(1.5, actual))  # clamp for safety; allow >100% if it happens
    gap = max(0.0, design - actual)

    poa_sev = _severity(agg.a_poa, agg.b_poa)
    pa_sev = _severity(agg.a_pa, agg.b_pa)
    ga_sev = _severity(agg.a_ga, agg.b_ga)
    sev_sum = poa_sev + pa_sev + ga_sev

    if sev_sum <= 0:
        poa_loss = 0.0
        pa_loss = 0.0
        ga_loss = 0.0
        unknown = gap
    else:
        poa_loss = gap * (poa_sev / sev_sum)
        pa_loss = gap * (pa_sev / sev_sum)
        ga_loss = gap * (ga_sev / sev_sum)
        unknown = max(0.0, gap - (poa_loss + pa_loss + ga_loss))

    df = pd.DataFrame(
        [
            {"step": "Design", "pct": design},
            {"step": "POA Loss", "pct": -poa_loss},
            {"step": "PA Loss", "pct": -pa_loss},
            {"step": "GA Loss", "pct": -ga_loss},
            {"step": "Unknown", "pct": -unknown},
            {"step": "Actual", "pct": actual},
        ]
    )

    # Convert each step to kWh attribution (based on Budget energy)
    # - Design is total budget energy
    # - Actual is total actual energy
    # - Loss buckets are % of budget energy
    def _step_kwh(step: str, pct: float) -> float:
        if step == "Design":
            return float(agg.b_energy)
        if step == "Actual":
            return float(agg.a_energy)
        return float(pct) * float(agg.b_energy)

    df["kwh"] = [_step_kwh(step, float(pct)) for step, pct in zip(df["step"].tolist(), df["pct"].tolist())]
    hover = [
        f"<b>{step}</b><br>kWh: {kwh:,.0f}<br>%: {pct*100:.2f}%<extra></extra>"
        for step, kwh, pct in zip(df["step"].tolist(), df["kwh"].tolist(), df["pct"].tolist())
    ]

    fig = go.Figure(
        go.Waterfall(
            name="Loss Walk",
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "relative", "total"],
            x=df["step"].tolist(),
            y=[v * 100 for v in df["pct"].tolist()],
            text=[f"{v*100:.1f}%" for v in df["pct"].tolist()],
            textposition="outside",
            hovertext=hover,
            hoverinfo="text",
            decreasing={"marker": {"color": "#ef4444"}},
            increasing={"marker": {"color": "#22c55e"}},
            totals={"marker": {"color": "#2563eb"}},
            connector={"line": {"color": "rgba(100,116,139,0.45)"}},
        )
    )
    fig.update_layout(
        title="Normalised Loss Walk (Design → Actual)",
        yaxis_title="Percent (%)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=60, b=10),
        height=520,
    )
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.25)")
    return df, fig


def render(db_path: str) -> None:
    st.markdown("## Portfolio Analytics")

    sites = list_sites_from_budget(db_path)
    if not sites:
        st.error("No sites found in `budget_kpi`. Load data first.")
        return

    dmin, dmax = date_bounds_from_budget(db_path)
    if not dmin or not dmax:
        st.error("No dates found in `budget_kpi`.")
        return

    # --- UI controls (horizontal) ---
    c_site, c_from, c_to, c_btn = st.columns([5, 2, 2, 1.6], vertical_alignment="bottom")

    with c_site:
        options = ["Portfolio", *sites]
        # Use stable key for multiselect to preserve selection across tab switches
        selected = st.multiselect("Site Name", options=options, default=[], key="pa_site_multiselect")
        # If Portfolio is chosen, it overrides other selections
        if "Portfolio" in selected:
            selected_sites = sites
        else:
            selected_sites = [s for s in selected if s in sites]

    with c_from:
        d1 = st.date_input("From", value=None, min_value=dmin, max_value=dmax, key="pa_from", format="YYYY-MM-DD")

    with c_to:
        d2 = st.date_input("To", value=None, min_value=dmin, max_value=dmax, key="pa_to", format="YYYY-MM-DD")

    valid_sites = bool(selected_sites)
    valid_dates = bool(d1 and d2 and d1 <= d2 and d1 >= dmin and d2 <= dmax)
    valid = valid_sites and valid_dates

    with c_btn:
        plot = st.button("Plot Now", type="primary", disabled=not valid, use_container_width=True)

    if not plot and st.session_state.get("pa_last_fig") is not None and st.session_state.get("pa_last_raw") is not None:
        # Show meta info about cached results
        meta = st.session_state.get("pa_last_meta") or {}
        meta_sites = meta.get("sites") or []
        meta_from = meta.get("from")
        meta_to = meta.get("to")
        if meta_sites and meta_from and meta_to:
            site_label = "Portfolio" if len(meta_sites) > 3 else ", ".join(meta_sites)
            st.caption(f"Showing last computed results: **{site_label}** ({meta_from} → {meta_to}). Change filters and click **Plot Now** to refresh.")
        else:
            st.caption("Showing last computed results. Change filters and click **Plot Now** to refresh.")

        st.download_button(
            "Download Raw Data (CSV)",
            data=st.session_state["pa_last_raw"].to_csv(index=False).encode("utf-8"),
            file_name="portfolio_analytics_raw.csv",
            mime="text/csv",
        )
        st.plotly_chart(st.session_state["pa_last_fig"], use_container_width=True, config={"displayModeBar": True})
        return

    if not plot:
        st.caption(f"Select Site Name(s) and Date Range (between {dmin:%d-%m-%Y} and {dmax:%d-%m-%Y}) to enable Plot Now.")
        return

    # --- Compute + plot ---
    with st.spinner("Computing portfolio aggregates..."):
        agg = compute_aggregates(db_path, selected_sites, d1, d2)
        raw = fetch_raw_data(db_path, selected_sites, d1, d2)

    if agg.b_energy <= 0:
        st.warning("Budget energy is 0 for the selected filters. Waterfall cannot be computed.")
        return

    df_steps, fig = build_waterfall(agg)

    st.session_state["pa_last_raw"] = raw
    st.session_state["pa_last_fig"] = fig
    st.session_state["pa_last_meta"] = {"sites": selected_sites, "from": d1, "to": d2}

    # Download raw data (for audit)
    st.download_button(
        "Download Raw Data (CSV)",
        data=raw.to_csv(index=False).encode("utf-8"),
        file_name="portfolio_analytics_raw.csv",
        mime="text/csv",
    )

    # Download chart as PNG (requires kaleido)
    try:
        png_bytes = pio.to_image(fig, format="png", width=1000, height=600, scale=2)
        st.download_button(
            "Download Waterfall (PNG)",
            data=png_bytes,
            file_name="portfolio_waterfall.png",
            mime="image/png",
        )
    except Exception:
        st.caption("PNG download unavailable (kaleido not installed).")

    # Only the waterfall (as requested)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": True})


