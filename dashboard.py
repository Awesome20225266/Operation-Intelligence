from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import portfolio_analytics
import operation_theatre
import add_comments
import meta_viewer
from aws_duckdb import get_duckdb_connection

# -----------------------------
# App config
# -----------------------------

APP_NAME = "Zelestra Energy"
DB_DEFAULT = "master.duckdb"

st.set_page_config(
    page_title=f"{APP_NAME} | Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)


# -----------------------------
# Styling
# -----------------------------

_BASE_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
  html, body, [class*="css"]  { font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }

  /* Hide Deploy + MainMenu + footer; keep toolbar for hamburger */
  .stDeployButton { display: none !important; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }

  /* Transparent header to remove white band, but keep native hamburger */
  header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
    border-bottom: none !important;
  }

  /* Ensure the toolbar (hamburger) stays visible */
  div[data-testid="stToolbar"] {
    display: flex !important;
    visibility: visible !important;
  }
  [data-testid="collapsedControl"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
  }

  /* Content padding */
  .block-container { padding-top: 1.2rem !important; }

  /* Page background */
  .stApp { background: #f6f8fb; }

  /* Reduce default top padding */
  .block-container { padding-top: 0.6rem; }

  /* Sidebar - styled (do NOT force width/visibility; allow Streamlit to collapse/expand) */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1020 0%, #070a14 100%) !important;
    padding-top: 0 !important;
  }
  section[data-testid="stSidebar"] > div {
    padding-top: 0 !important;
    margin-top: 0 !important;
  }
  section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0 !important;
    margin-top: 0 !important;
  }
  section[data-testid="stSidebar"] * { 
    color: #e9eefc; 
  }

  /* Sidebar header (like screenshot: icon box + Analytics) */
  .sb-header {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 0px 18px 14px 18px; /* remove top padding so title sits at top */
    margin-top: -10px;           /* counter Streamlit's internal top gap */
    text-align: center;
  }
  .sb-title {
    font-size: 18px;
    font-weight: 800;
    color: #ffffff;
    line-height: 1.1;
    margin: 0;
    white-space: nowrap;
  }
  .sb-title-top {
    font-size: 44px; /* doubled for prominence */
    font-weight: 900;
    color: rgba(255,255,255,0.98);
    line-height: 1.0;
    margin: 0 0 6px 0;
    white-space: nowrap;
  }
  /* Sidebar button text alignment tweaks */
  section[data-testid="stSidebar"] button p {
    margin: 0 !important;
    line-height: 1.0 !important;
  }
  .sb-divider {
    height: 1px;
    background: rgba(255,255,255,0.08);
    margin: 10px 0 12px 0;
  }

  /* Sidebar nav buttons - default: white text, NO highlight */
  section[data-testid="stSidebar"] button[kind="secondary"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    box-shadow: none !important;
    color: #ffffff !important;
    font-weight: 650 !important;
    font-size: 16px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 12px 14px !important;
    margin: 4px 14px !important;
    border-radius: 10px !important;
    transition: background 0.15s ease, border-color 0.15s ease !important;
    height: 46px !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
  }
  section[data-testid="stSidebar"] button[kind="secondary"] * {
    color: #ffffff !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 10px !important;
  }
  section[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: rgba(255,255,255,0.10) !important;
    border-color: rgba(255,255,255,0.12) !important;
  }
  
  /* Sidebar nav buttons - active (yellow, stays selected) */
  section[data-testid="stSidebar"] button[kind="primary"] {
    background: #ffb300 !important;           /* yellow active */
    border: 1px solid rgba(0,0,0,0.08) !important;
    color: #0b1020 !important;                /* dark text on yellow */
    box-shadow: 0 10px 26px rgba(255, 179, 0, 0.18) !important;
    font-weight: 800 !important;
    font-size: 16px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 12px 14px !important;
    margin: 4px 14px !important;
    border-radius: 10px !important;
    height: 46px !important;
    display: flex !important;
    align-items: center !important;
    gap: 10px !important;
  }
  section[data-testid="stSidebar"] button[kind="primary"] * {
    color: #0b1020 !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 10px !important;
  }
  section[data-testid="stSidebar"] button[kind="primary"]:hover {
    background: #ffc533 !important;
  }

  /* KPI cards */
  .kpi-card {
    background: #ffffff;
    border: 1px solid #e8edf6;
    border-radius: 16px;
    padding: 18px 18px 14px 18px;
    box-shadow: 0 8px 22px rgba(13, 25, 56, 0.06);
    height: 112px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  .kpi-label { color: #5c6b8a; font-size: 14px; margin-bottom: 8px; }
  .kpi-value { font-size: 36px; font-weight: 800; color: #0b1220; line-height: 1.1; }
  .kpi-unit { font-size: 20px; font-weight: 700; color: #0b1220; margin-left: 6px; }
  .kpi-sub { margin-top: 10px; display: flex; gap: 10px; align-items: baseline; }
  .kpi-delta { font-weight: 800; font-size: 13px; }
  .kpi-delta-pos { color: #14804a; }
  .kpi-delta-neg { color: #b42318; }
  .kpi-note { color: #5c6b8a; font-weight: 650; font-size: 12px; }

  /* Section cards */
  .panel {
    background: #ffffff;
    border: 1px solid #e8edf6;
    border-radius: 18px;
    padding: 14px 16px 6px 16px;
    box-shadow: 0 8px 22px rgba(13, 25, 56, 0.06);
  }
  .panel-title {
    font-size: 22px;
    font-weight: 800;
    color: #0b1220;
    margin: 4px 0 8px 0;
  }
  .panel-sub {
    color: #5c6b8a;
    font-weight: 650;
    font-size: 12px;
    margin: -2px 0 10px 0;
  }
  .muted { color: #5c6b8a; }

  /* Topbar (simple) */
  .topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin: 6px 0 10px 0;
  }
  .topbar-title {
    font-size: 28px;
    font-weight: 900;
    color: #0b1220;
  }
  .topbar-right {
    display: flex;
    gap: 14px;
    align-items: center;
    color: #5c6b8a;
    font-weight: 600;
  }
</style>
"""


def _inject_css() -> None:
    st.markdown(_BASE_CSS, unsafe_allow_html=True)


# -----------------------------
# Data access
# -----------------------------


@dataclass(frozen=True)
class Filters:
    site_name: str
    as_of_date: date
    tariff_inr_per_kwh: float


def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    # Always use the shared, read-only DuckDB connection helper (S3-backed if needed).
    # Keeps signatures stable so business logic/queries remain unchanged.
    return get_duckdb_connection(db_local=db_path)


@st.cache_data(show_spinner=False)
def list_sites(db_path: str) -> list[str]:
    con = _connect(db_path)
    try:
        rows = con.execute("select distinct site_name from daily_kpi order by site_name").fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def get_available_dates(db_path: str, site_name: str) -> list[date]:
    con = _connect(db_path)
    try:
        rows = con.execute(
            "select distinct date from daily_kpi where site_name = ? order by date",
            [site_name],
        ).fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def get_daily_row(db_path: str, site_name: str, d: date) -> pd.DataFrame:
    con = _connect(db_path)
    try:
        return con.execute(
            """
            select *
            from daily_kpi
            where site_name = ? and date = ?
            """,
            [site_name, d],
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def get_budget_row(db_path: str, site_name: str, d: date) -> pd.DataFrame:
    con = _connect(db_path)
    try:
        return con.execute(
            """
            select *
            from budget_kpi
            where site_name = ? and date = ?
            """,
            [site_name, d],
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def get_pr_equipment(db_path: str, site_name: str, d: date) -> pd.DataFrame:
    con = _connect(db_path)
    try:
        return con.execute(
            """
            select equipment_name, pr_percent
            from pr
            where site_name = ? and date = ?
            order by equipment_name
            """,
            [site_name, d],
        ).fetchdf()
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def get_syd_equipment(db_path: str, site_name: str, d: date) -> pd.DataFrame:
    con = _connect(db_path)
    try:
        return con.execute(
            """
            select equipment_name, syd_percent
            from syd
            where site_name = ? and date = ?
            order by equipment_name
            """,
            [site_name, d],
        ).fetchdf()
    finally:
        con.close()


def ensure_remarks_table(db_path: str) -> None:
    con = _connect(db_path)
    try:
        con.execute(
            """
            create table if not exists remarks (
              site_name text not null,
              date date not null,
              remark text not null,
              created_at timestamp not null
            );
            """
        )
        con.execute("create index if not exists idx_remarks_site_date on remarks(site_name, date);")
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def get_remarks(db_path: str, site_name: str, d: date) -> pd.DataFrame:
    ensure_remarks_table(db_path)
    con = _connect(db_path)
    try:
        return con.execute(
            """
            select created_at, remark
            from remarks
            where site_name = ? and date = ?
            order by created_at desc
            """,
            [site_name, d],
        ).fetchdf()
    finally:
        con.close()


def add_remark(db_path: str, site_name: str, d: date, remark: str) -> None:
    ensure_remarks_table(db_path)
    con = _connect(db_path)
    try:
        con.execute(
            "insert into remarks(site_name, date, remark, created_at) values (?, ?, ?, ?)",
            [site_name, d, remark, datetime.now()],
        )
    finally:
        con.close()
    # bust cache
    get_remarks.clear()


# -----------------------------
# Chart helpers (approx “last 24h”)
# -----------------------------


def _solar_profile_24h(total: float, *, sunrise_hour: int = 6, sunset_hour: int = 18) -> pd.Series:
    """
    Build a smooth 24-point profile that sums to `total`.
    This is used to approximate "last 24h" curves from daily totals.
    """
    total = float(total or 0.0)
    hours = np.arange(24)
    mask = (hours >= sunrise_hour) & (hours <= sunset_hour)
    x = np.linspace(-2.5, 2.5, mask.sum())
    bell = np.exp(-(x**2))
    y = np.zeros(24, dtype=float)
    if bell.sum() > 0:
        y[mask] = bell / bell.sum() * total
    return pd.Series(y, index=hours)


def _fmt_inr(x: float) -> str:
    x = float(x or 0.0)
    if abs(x) >= 1e7:
        return f"₹{x/1e7:.2f}Cr"
    if abs(x) >= 1e5:
        return f"₹{x/1e5:.2f}L"
    if abs(x) >= 1e3:
        return f"₹{x/1e3:.1f}K"
    return f"₹{x:.0f}"


def _fmt_kwh(x: float) -> str:
    x = float(x or 0.0)
    if abs(x) >= 1e6:
        return f"{x/1e6:.2f} GWh"
    if abs(x) >= 1e3:
        return f"{x/1e3:.2f} MWh"
    return f"{x:.0f} kWh"


# -----------------------------
# Plotly helpers
# -----------------------------


PLOTLY_COLORS = {
    "actual": "#2563eb",  # blue
    "actual_fill": "rgba(37, 99, 235, 0.20)",
    "target": "#6b7280",  # slate
    "target_fill": "rgba(107, 114, 128, 0.18)",
    "accent": "#f59e0b",  # amber
}


def _plotly_base_layout(title: str, *, x_title: str | None = None, y_title: str | None = None) -> dict[str, Any]:
    layout: dict[str, Any] = dict(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=18, color="#0b1220")),
        margin=dict(l=14, r=10, t=46, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0b1220"),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.25)",
            zeroline=False,
            tickfont=dict(color="#475569"),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(148, 163, 184, 0.25)",
            zeroline=False,
            tickfont=dict(color="#475569"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="x unified",
    )

    # Plotly v6+ uses axis.title.font (not titlefont)
    title_font = dict(color="#475569", size=12)
    if x_title:
        layout["xaxis"]["title"] = dict(text=x_title, font=title_font)
    if y_title:
        layout["yaxis"]["title"] = dict(text=y_title, font=title_font)
    return layout


def plot_area_actual_vs_target(
    *,
    x: list[str],
    actual: list[float],
    target: list[float],
    title: str,
    y_title: str,
) -> go.Figure:
    # Ensure all inputs are valid lists with same length
    if not x or not actual or not target:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(**_plotly_base_layout(title, y_title=y_title))
        return fig
    
    # Ensure lists are same length
    min_len = min(len(x), len(actual), len(target))
    x = x[:min_len]
    actual = actual[:min_len]
    target = target[:min_len]
    
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=target,
            name="Target",
            mode="lines",
            line=dict(color=PLOTLY_COLORS["target"], width=2, shape="spline", smoothing=1.05),
            fill="tozeroy",
            fillcolor=PLOTLY_COLORS["target_fill"],
            hovertemplate="%{x}<br>Target: %{y:.2f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=actual,
            name="Actual",
            mode="lines",
            line=dict(color=PLOTLY_COLORS["actual"], width=3, shape="spline", smoothing=1.05),
            fill="tozeroy",
            fillcolor=PLOTLY_COLORS["actual_fill"],
            hovertemplate="%{x}<br>Actual: %{y:.2f}<extra></extra>",
        )
    )

    fig.update_layout(**_plotly_base_layout(title, y_title=y_title))
    fig.update_xaxes(nticks=8)
    return fig


def plot_bar_top_n(
    *,
    x: list[float],
    y: list[str],
    title: str,
    x_title: str,
    color: str,
) -> go.Figure:
    # Handle empty data
    if not x or not y:
        fig = go.Figure()
        fig.update_layout(**_plotly_base_layout(title, x_title=x_title))
        return fig
    
    # Ensure lists are same length
    min_len = min(len(x), len(y))
    x = x[:min_len]
    y = y[:min_len]
    
    fig = go.Figure(
        data=[
            go.Bar(
                x=x,
                y=y,
                orientation="h",
                marker=dict(color=color),
                hovertemplate="%{y}<br>%{x:.2f}<extra></extra>",
            )
        ]
    )
    fig.update_layout(
        **_plotly_base_layout(title, x_title=x_title),
        margin=dict(l=120, r=10, t=46, b=12),
    )
    fig.update_yaxes(showgrid=False)
    return fig


# -----------------------------
# UI components
# -----------------------------


def kpi_card(label: str, value: str, unit: str = "", *, delta_text: str = "", delta_is_positive: bool | None = None) -> None:
    delta_html = ""
    if delta_text:
        if delta_is_positive is None:
            delta_class = ""
        else:
            delta_class = "kpi-delta-pos" if delta_is_positive else "kpi-delta-neg"
        delta_html = f'<div class="kpi-sub"><div class="kpi-delta {delta_class}">{delta_text}</div></div>'

    st.markdown(
        f"""
        <div class="kpi-card">
          <div>
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}<span class="kpi-unit">{unit}</span></div>
            {delta_html}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def topbar(email: str, selected_date: date) -> None:
    st.markdown(
        f"""
        <div class="topbar">
          <div class="topbar-title">Solar Performance Dashboard</div>
          <div class="topbar-right">
            <div class="muted">{selected_date.strftime('%B %d, %Y')}</div>
            <div>{email}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Pages
# -----------------------------


def page_dashboard(db_path: str, f: Filters) -> None:
    daily = get_daily_row(db_path, f.site_name, f.as_of_date)
    budget = get_budget_row(db_path, f.site_name, f.as_of_date)

    inv_gen_kwh = float(daily["inv_gen_kwh"].iloc[0]) if not daily.empty else 0.0
    pr_percent = float(daily["pr_percent"].iloc[0]) if not daily.empty else 0.0
    cuf_percent = float(daily["ac_cuf_percent"].iloc[0]) if not daily.empty else 0.0
    target_kwh = float(budget["b_energy_kwh"].iloc[0]) if not budget.empty else 0.0

    # “Real-time power” approximation: avg power over day
    rt_power_kw = inv_gen_kwh / 24.0
    revenue = inv_gen_kwh * float(f.tariff_inr_per_kwh or 0.0)

    st.markdown("## Performance Overview")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Real-Time Power", f"{rt_power_kw:,.0f}", "kW", delta_text="Estimated", delta_is_positive=None)
    with c2:
        kpi_card("Performance Ratio", f"{pr_percent:.1f}", "%")
    with c3:
        kpi_card("CUF", f"{cuf_percent:.1f}", "%")
    with c4:
        kpi_card("Today's Revenue", _fmt_inr(revenue), "", delta_text=f"Tariff: ₹{f.tariff_inr_per_kwh:.2f}/kWh", delta_is_positive=None)

    st.write("")
    left, right = st.columns(2)

    # Generation actual vs target (approx last 24h)
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        gap = target_kwh - inv_gen_kwh
        gap_txt = f"{_fmt_kwh(abs(gap))} {'below' if gap >= 0 else 'above'} target"
        st.markdown(
            f"""
            <div class="panel-title">Generation: Actual vs Target (Last 24h)</div>
            <div class="panel-sub">Actual: <b>{inv_gen_kwh:,.0f} kWh</b> • Target: <b>{target_kwh:,.0f} kWh</b> • Gap: <b>{gap_txt}</b></div>
            """,
            unsafe_allow_html=True,
        )

        x = [f"{h:02d}:00" for h in range(24)]
        try:
            actual_values = _solar_profile_24h(inv_gen_kwh).values.tolist()
            target_values = _solar_profile_24h(target_kwh).values.tolist()
            fig = plot_area_actual_vs_target(
                x=x,
                actual=actual_values,
                target=target_values,
                title="Generation: Actual vs Target (Last 24h)",
                y_title="kWh",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.error(f"Error rendering generation chart: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Irradiance actual vs target (approx last 24h)
    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        poa = float(daily["poa"].iloc[0]) if not daily.empty else 0.0
        b_poa = float(budget["b_poa"].iloc[0]) if not budget.empty else 0.0

        # Use arbitrary scaling to look like W/m² curves for UI.
        # (poa/b_poa are daily totals in kWh/m² in your files; this is an approximation.)
        target_wm2 = b_poa * 100.0
        actual_wm2 = poa * 100.0
        st.markdown(
            f"""
            <div class="panel-title">Irradiance: Actual vs Target (Last 24h)</div>
            <div class="panel-sub">Actual (scaled): <b>{actual_wm2:,.0f}</b> • Target (scaled): <b>{target_wm2:,.0f}</b></div>
            """,
            unsafe_allow_html=True,
        )
        x = [f"{h:02d}:00" for h in range(24)]
        try:
            actual_values = _solar_profile_24h(actual_wm2).values.tolist()
            target_values = _solar_profile_24h(target_wm2).values.tolist()
            fig2 = plot_area_actual_vs_target(
                x=x,
                actual=actual_values,
                target=target_values,
                title="Irradiance: Actual vs Target (Last 24h)",
                y_title="W/m²",
            )
            st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        except Exception as e:
            st.error(f"Error rendering irradiance chart: {e}")
        st.markdown("</div>", unsafe_allow_html=True)


def page_inverter_bd(db_path: str, f: Filters) -> None:
    st.markdown("## Inverter BD")
    df = get_pr_equipment(db_path, f.site_name, f.as_of_date)
    if df.empty:
        st.info("No inverter PR data available for the selected site/date.")
        return
    df = df.copy()
    df["pr_percent"] = pd.to_numeric(df["pr_percent"], errors="coerce").fillna(0.0)
    top_n = st.slider("Top N", min_value=5, max_value=50, value=30, step=5)
    df_sorted = df.sort_values("pr_percent", ascending=True).tail(top_n)

    st.dataframe(df.sort_values("pr_percent", ascending=False), width="stretch", height=420, hide_index=True)
    try:
        fig = plot_bar_top_n(
            x=df_sorted["pr_percent"].tolist(),
            y=df_sorted["equipment_name"].tolist(),
            title=f"Top {top_n} Inverters by PR (%)",
            x_title="PR (%)",
            color=PLOTLY_COLORS["accent"],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.error(f"Error rendering inverter chart: {e}")


def page_string_bd(db_path: str, f: Filters) -> None:
    st.markdown("## String BD")
    df = get_syd_equipment(db_path, f.site_name, f.as_of_date)
    if df.empty:
        st.info("No string SYD data available for the selected site/date.")
        return
    df = df.copy()
    df["syd_percent"] = pd.to_numeric(df["syd_percent"], errors="coerce").fillna(0.0)
    top_n = st.slider("Top N", min_value=5, max_value=50, value=30, step=5)
    df_sorted = df.sort_values("syd_percent", ascending=True).tail(top_n)

    st.dataframe(df.sort_values("syd_percent", ascending=False), width="stretch", height=420, hide_index=True)
    try:
        fig = plot_bar_top_n(
            x=df_sorted["syd_percent"].tolist(),
            y=df_sorted["equipment_name"].tolist(),
            title=f"Top {top_n} Strings by SYD (%)",
            x_title="SYD (%)",
            color=PLOTLY_COLORS["actual"],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.error(f"Error rendering string chart: {e}")


def page_losses(db_path: str, f: Filters) -> None:
    st.markdown("## Losses")
    daily = get_daily_row(db_path, f.site_name, f.as_of_date)
    budget = get_budget_row(db_path, f.site_name, f.as_of_date)
    if daily.empty or budget.empty:
        st.info("Loss calculations need both Daily KPI and Budget KPI for the selected site/date.")
        return

    actual = float(daily["inv_gen_kwh"].iloc[0] or 0.0)
    target = float(budget["b_energy_kwh"].iloc[0] or 0.0)
    gap = max(target - actual, 0.0)
    sl = float(budget["b_sl_percent"].iloc[0] or 0.0)
    est_soiling_loss_kwh = (sl / 100.0) * target if target else 0.0

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Target Energy", f"{target:,.0f}", "kWh")
    with c2:
        kpi_card("Actual Energy", f"{actual:,.0f}", "kWh")
    with c3:
        kpi_card("Energy Gap", f"{gap:,.0f}", "kWh")

    st.write("")
    loss_df = pd.DataFrame(
        [
            {"loss_component": "Estimated soiling loss (from budget)", "value_kwh": est_soiling_loss_kwh},
            {"loss_component": "Unattributed gap (target - actual - soiling)", "value_kwh": max(gap - est_soiling_loss_kwh, 0.0)},
        ]
    )
    st.dataframe(loss_df, width="stretch", hide_index=True)
    try:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=loss_df["loss_component"].tolist(),
                    y=loss_df["value_kwh"].tolist(),
                    marker=dict(color=[PLOTLY_COLORS["accent"], PLOTLY_COLORS["target"]]),
                    hovertemplate="%{x}<br>%{y:,.0f} kWh<extra></extra>",
                )
            ]
        )
        fig.update_layout(**_plotly_base_layout("Loss Breakdown", y_title="kWh"))
        fig.update_xaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as e:
        st.error(f"Error rendering loss chart: {e}")


def page_remarks(db_path: str, f: Filters) -> None:
    st.markdown("## Remarks")
    st.caption("Save notes for a site/day. This writes to a small `remarks` table inside your DuckDB.")

    with st.form("remarks_form", clear_on_submit=True):
        remark = st.text_area("Add remark", placeholder="Type your remark here...")
        submitted = st.form_submit_button("Save")
    if submitted:
        remark = (remark or "").strip()
        if not remark:
            st.warning("Remark is empty.")
        else:
            add_remark(db_path, f.site_name, f.as_of_date, remark)
            st.success("Saved.")

    df = get_remarks(db_path, f.site_name, f.as_of_date)
    if df.empty:
        st.info("No remarks for this site/date yet.")
        return
    st.dataframe(df, width="stretch", hide_index=True)


# -----------------------------
# Layout / routing
# -----------------------------


def render_sidebar_tab(icon: str, label: str, key: str, is_active: bool) -> bool:
    """Render a sidebar tab button and return True if clicked."""
    button_label = f"{icon} {label}"
    # Use different button types based on active state
    button_type = "primary" if is_active else "secondary"
    clicked = st.button(button_label, key=f"btn_{key}", use_container_width=True, type=button_type)
    if clicked:
        return True
    return False


def page_portfolio_analytics() -> None:
    st.markdown("# Portfolio Analytics")
    st.info("Portfolio Analytics page - Coming soon")


def page_operation_theatre() -> None:
    st.markdown("# Operation Theatre")
    st.info("Operation Theatre page - Coming soon")

def main() -> None:
    _inject_css()

    # Sidebar navigation state (matches the UI you shared)
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "portfolio"
    allowed_pages = {"portfolio", "operation", "add_comments", "dfm", "visual_analyser", "meta_viewer"}
    if st.session_state.nav_page not in allowed_pages:
        st.session_state.nav_page = "portfolio"

    # Use default values for data access
    db_path = DB_DEFAULT
    # Ensure DB is available locally (download from S3 once if missing)
    try:
        _ = get_duckdb_connection(db_local=db_path).close()
    except Exception as e:
        st.error(f"Failed to open DuckDB (S3 download may be required). {e}")
        st.stop()

    sites = list_sites(db_path)
    if not sites:
        st.error("No sites found in DB (daily_kpi is empty). Run the loader first.")
        st.stop()

    # Use defaults: first site, latest date, default tariff
    site_name = sites[0]
    dates = get_available_dates(db_path, site_name)
    if dates:
        as_of_date = dates[-1]
    else:
        as_of_date = date.today()
    tariff = 6.0

    try:
        with st.sidebar:
            # Sidebar header
            st.markdown(
                """
                <div class="sb-header">
                  <div>
                    <div class="sb-title-top">Zelestra</div>
                    <div class="sb-title">Operation Intelligence</div>
                  </div>
                </div>
                <div class="sb-divider"></div>
                """,
                unsafe_allow_html=True,
            )

            nav_items = [
                ("portfolio", "📊  Portfolio Analytics"),
                ("operation", "🏥  Operation Theatre"),
                ("add_comments", "📝  Add Comments"),
                ("dfm", "🛠️  Fault Detector"),
                ("visual_analyser", "🖥️  Visual Analyser"),
                ("meta_viewer", "🧭  Meta Viewer"),
            ]

            for key, label in nav_items:
                is_active = st.session_state.nav_page == key
                btn_type = "primary" if is_active else "secondary"
                if st.button(label, key=f"nav_{key}", type=btn_type, use_container_width=True):
                    st.session_state.nav_page = key
                    st.rerun()
    except Exception as e:
        st.error(f"Sidebar error: {e}")
        # Fallback: show sidebar content in main area for debugging
        st.sidebar.write("Sidebar content (fallback)")

    # Route to pages
    f = Filters(site_name=site_name, as_of_date=as_of_date, tariff_inr_per_kwh=float(tariff))
    page = st.session_state.nav_page

    if page == "portfolio":
        portfolio_analytics.render(db_path)
    elif page == "operation":
        operation_theatre.render(db_path)
    elif page == "add_comments":
        add_comments.render(db_path)
    elif page == "dfm":
        st.markdown("## Digital Fault Monitoring")
        st.info("Coming soon")
    elif page == "visual_analyser":
        st.markdown("## Visual Analyser")
        st.info("Coming soon")
    elif page == "meta_viewer":
        meta_viewer.render(db_path)
    else:
        st.error("Unknown page")


if __name__ == "__main__":
    main()


