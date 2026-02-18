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
import reconnect_dsm
import scb_ot
import scb_comment
import raw_analyser
import S1
import S2
import S3
from aws_duckdb import get_duckdb_connection
import auth
from access_control import user_allowed_pages

# -----------------------------
# App config
# -----------------------------

APP_NAME = "Zel - EYE: OI"
DB_DEFAULT = "master.duckdb"

# All available pages (key, label)
ALL_PAGES = [
    ("portfolio", "📊  Portfolio Analytics"),
    ("operation", "🏥  Operation Theatre"),
    ("reconnect", "🔌  Re Connect"),
    ("add_comments", "📝  Add Comments"),
    ("scb_comment", "🧾  SCB Comment"),
    ("dfm", "🛠️  Fault Detector"),
    ("visual_analyser", "🖥️  Visual Analyser"),
    ("meta_viewer", "🧭  Meta Viewer"),
    ("scb_ot", "⚡  SCB OT"),
    ("raw_analyser", "🧪  Raw Analyser"),
    ("s1", "📋  S1"),
    ("s2", "📝  S2"),
    ("s3", "✅  S3"),
]

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
# Authentication (Must be first)
# -----------------------------

if not auth.check_password():
    st.stop()  # Do not run the rest of the app if not authenticated


# -----------------------------
# Styling
# -----------------------------

_BASE_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800;900&display=swap');
  html, body, [class*="css"]  { font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }

  /* ============================================================
     HIDE ALL DEFAULT STREAMLIT WIDGETS (Production Lock-Down)
     ============================================================ */
  
  /* Hide Deploy button completely */
  .stDeployButton,
  [data-testid="stDeployButton"],
  button[kind="deploy"],
  /* Streamlit variants */
  header button[title="Deploy"],
  header button[aria-label="Deploy"] { 
    display: none !important; 
    visibility: hidden !important;
    pointer-events: none !important;
  }
  
  /* Hide hamburger menu / MainMenu */
  #MainMenu,
  [data-testid="stMainMenu"] {
    display: none !important;
    visibility: hidden !important;
  }

  /* Ensure sidebar collapse button is NEVER clipped or hidden */
  [data-testid="stSidebarCollapseButton"] {
    display: flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    position: relative !important;
    z-index: 9999 !important;
    pointer-events: auto !important;
  }
  section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
    position: absolute !important;
    top: 10px !important;
    right: 10px !important;
  }
  
  /* Hide footer ("Made with Streamlit") */
  footer,
  .stApp footer,
  [data-testid="stFooter"] { 
    display: none !important; 
    visibility: hidden !important;
  }
  
  /* Hide toolbar actions (settings, fullscreen, etc.) but keep sidebar toggle */
  div[data-testid="stToolbar"] {
    visibility: visible !important;
  }
  div[data-testid="stToolbarActions"] {
    display: none !important;
  }
  
  /* Keep ONLY the sidebar collapse/expand control visible */
  [data-testid="collapsedControl"],
  [data-testid="stSidebarCollapseButton"],
  [data-testid="stSidebarCollapsedControl"],
  button[aria-label*="sidebar"],
  button[title*="sidebar"] {
    display: inline-flex !important;
    visibility: visible !important;
    opacity: 1 !important;
    pointer-events: auto !important;
  }

  /* Transparent header to remove white band */
  header[data-testid="stHeader"] {
    background: transparent !important;
    box-shadow: none !important;
    border-bottom: none !important;
  }

  /* ============================================================
     SMOOTH TRANSITIONS - Prevent UI Ghosting
     ============================================================ */
  
  /* Smooth fade-in for main content area */
  .main .block-container {
    animation: contentFadeIn 0.25s ease-out;
  }
  
  @keyframes contentFadeIn {
    from { opacity: 0.3; }
    to { opacity: 1; }
  }
  
  /* Prevent flash of unstyled content */
  .stApp {
    opacity: 1;
    transition: opacity 0.15s ease-in-out;
  }
  
  /* Tab panel transitions */
  .stTabs [data-baseweb="tab-panel"] {
    animation: tabPanelFadeIn 0.2s ease-out;
  }
  
  @keyframes tabPanelFadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to { opacity: 1; transform: translateY(0); }
  }
  
  /* NOTE: Do NOT hide tab-panels based on aria-hidden.
     Streamlit doesn't set aria-hidden consistently, which can blank tab content.
     If we need to hide inactive panels, rely on the native [hidden] attribute inside
     individual pages (S1/S2/S3) where Streamlit actually sets it. */
  
  /* Ensure only one main content block is visible */
  .element-container:empty {
    display: none !important;
  }
  
  /* Prevent stale content flash during page transitions */
  .stale-content, .old-content {
    display: none !important;
  }

  /* Content padding */
  .block-container { padding-top: 1rem !important; }

  /* Page background */
  .stApp { background: #f6f8fb; }

  /* ============================================================
     ENTERPRISE SIDEBAR — FIXED HEADER + SCROLLABLE NAV
     ============================================================ */

  /* Main sidebar container */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1020 0%, #070a14 100%) !important;
  }

  /* Force sidebar to full viewport height */
  section[data-testid="stSidebar"] > div:first-child {
    height: 100vh !important;
    display: flex !important;
    flex-direction: column !important;
    padding: 0 !important;
  }

  /* Sidebar content wrapper */
  [data-testid="stSidebarContent"] {
    display: flex !important;
    flex-direction: column !important;
    height: 100% !important;
    padding: 0 !important;
  }

  /* Header (TOP FIXED) */
  .sidebar-top {
    flex-shrink: 0 !important;
    padding: 16px 18px 10px 18px !important;
    position: sticky !important;
    top: 0 !important;
    background: linear-gradient(180deg, #0b1020 0%, #070a14 100%) !important;
    z-index: 5 !important;
  }

  /* Scrollable NAV AREA */
  .sidebar-nav {
    flex: 1 1 auto !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    padding: 6px 10px !important;
  }

  /* Bottom fixed section */
  .sidebar-bottom {
    flex-shrink: 0 !important;
    padding: 12px 14px 18px 14px !important;
    border-top: 1px solid rgba(255,255,255,0.08) !important;
  }

  /* Scrollbar styling */
  .sidebar-nav::-webkit-scrollbar {
    width: 4px;
  }
  .sidebar-nav::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.25);
    border-radius: 4px;
  }
  .sidebar-nav::-webkit-scrollbar-thumb:hover {
    background: rgba(255,255,255,0.4);
  }

  .sb-welcome {
    font-size: 15px;
    font-weight: 600;
    color: #ffffff !important;
    margin: 4px 18px 10px 18px;
    opacity: 0.9;
  }
  .sb-avatar-row {
    display: flex; align-items: center; gap: 10px;
    padding: 6px 18px 12px 18px;
  }
  .sb-avatar {
    width: 34px; height: 34px; border-radius: 50%;
    background: linear-gradient(135deg, #ffb300, #ff6b35);
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 800; color: #0b1020; flex-shrink: 0;
  }
  .sb-user-name { font-size: 13px; font-weight: 700; color: #fff; }
  .sb-role-badge {
    display: inline-block; font-size: 9px; font-weight: 700;
    background: rgba(255,179,0,0.18); color: #ffb300;
    border: 1px solid rgba(255,179,0,0.3); border-radius: 4px;
    padding: 1px 6px; letter-spacing: 0.5px; text-transform: uppercase; margin-top: 2px;
  }
  section[data-testid="stSidebar"] input[type="text"] {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
    color: #ffffff !important; border-radius: 8px !important; font-size: 13px !important;
  }
  section[data-testid="stSidebar"] input[type="text"]::placeholder {
    color: rgba(255,255,255,0.4) !important;
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
    font-size: 14px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 10px 12px !important;
    margin: 2px 10px !important;
    border-radius: 10px !important;
    transition: background 0.15s ease, border-color 0.15s ease !important;
    height: 40px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
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
    font-size: 14px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 10px 12px !important;
    margin: 2px 10px !important;
    border-radius: 10px !important;
    height: 40px !important;
    display: flex !important;
    align-items: center !important;
    gap: 8px !important;
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
  .kpi-label { color: inherit; font-size: 14px; margin-bottom: 8px; }
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
    color: inherit;
    font-weight: 650;
    font-size: 12px;
    margin: -2px 0 10px 0;
  }
  .muted { color: inherit; }

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
    color: inherit;
    font-weight: 600;
  }
</style>
"""


def _inject_css() -> None:
    theme = st.session_state.get("app_theme", "light")
    if theme == "dark":
        bg, card, text, border, sub_text = "#0f172a", "#111827", "#f1f5f9", "#1f2937", "#94a3b8"
    else:
        bg, card, text, border, sub_text = "#f6f8fb", "#ffffff", "#0b1220", "#e8edf6", "#5c6b8a"

    dynamic = f"""<style>
.stApp {{ background: {bg} !important; color: {text} !important; }}
.panel, .kpi-card {{
    background: {card} !important;
    border: 1px solid {border} !important;
    color: {text} !important;
}}
.panel-title, .kpi-value, .kpi-unit, .topbar-title {{ color: {text} !important; }}
.kpi-label, .panel-sub, .muted, .topbar-right {{ color: {sub_text} !important; }}
[data-testid="stMarkdownContainer"] p {{ color: {text} !important; }}
body {{ color-scheme: {"dark" if theme == "dark" else "light"} !important; }}
</style>"""
    st.markdown(_BASE_CSS + dynamic, unsafe_allow_html=True)


def _inject_sidebar_freeze_layout() -> None:
    """
    Streamlit renders each widget into isolated containers, so HTML wrappers
    (sidebar-top/nav/bottom) won't automatically contain the buttons.
    This injects a DOM rearranger that directly moves button containers:
      - Last 2 buttons (theme + logout) → .sidebar-bottom
      - All others → .sidebar-nav (scrollable)
    """
    st.components.v1.html(
        """
        <script>
        (function () {
          const doc = window.parent.document;

          function restructure() {
            const sidebar = doc.querySelector('[data-testid="stSidebarContent"]');
            if (!sidebar) return;

            const nav = sidebar.querySelector('.sidebar-nav');
            const bottom = sidebar.querySelector('.sidebar-bottom');
            if (!nav || !bottom) return;

            // Already restructured
            if (nav.dataset.done === '1') return;
            nav.dataset.done = '1';

            // Collect all button containers in the sidebar
            const allBtns = Array.from(sidebar.querySelectorAll('.stButton'))
              .filter(btn => !nav.contains(btn) && !bottom.contains(btn));

            if (allBtns.length < 2) return;

            // Last 2 buttons are footer (theme + logout)
            const footerBtns = allBtns.slice(-2);
            const navBtns = allBtns.slice(0, -2);

            // Move nav buttons into scrollable area
            navBtns.forEach(btn => nav.appendChild(btn));
            
            // Move footer buttons into fixed bottom
            footerBtns.forEach(btn => bottom.appendChild(btn));
          }

          // Run immediately and after a short delay (to ensure buttons are rendered)
          restructure();
          setTimeout(restructure, 100);
          setTimeout(restructure, 300);

          // Watch for Streamlit rerenders
          const sidebar = doc.querySelector('section[data-testid="stSidebar"]');
          if (!sidebar) return;

          const obs = new MutationObserver(() => {
            const nav = doc.querySelector('.sidebar-nav');
            if (nav) nav.dataset.done = '0';
            setTimeout(restructure, 50);
          });
          obs.observe(sidebar, { childList: true, subtree: true });
        })();
        </script>
        """,
        height=0,
    )


# =============================================================================
# REUSABLE SECTION-LEVEL LOADER UTILITY
# =============================================================================

def create_section_loader(container=None):
    """
    Create a section-level loader that shows a progress bar instead of ghosting.
    
    Usage:
        loader = create_section_loader()
        loader.start("Loading data...")
        # ... do work ...
        loader.update(50, "Processing...")
        # ... more work ...
        loader.finish()
    
    Returns:
        SectionLoader instance with start(), update(), finish() methods.
    """
    import time as _time
    
    class SectionLoader:
        def __init__(self, container):
            self._container = container or st.empty()
            self._progress = None
            self._progress_slot = None
            self._started = False
        
        def start(self, message: str = "Loading...") -> None:
            """Clear any existing content and show progress bar at 0%."""
            self._container.empty()  # Clear old content immediately
            with self._container.container():
                self._progress_slot = st.empty()
                self._progress = self._progress_slot.progress(0, text=f"{message} (0%)")
            self._started = True
        
        def update(self, percent: int, message: str = "") -> None:
            """Update progress bar to given percentage."""
            if self._progress and self._started:
                text = f"{message} ({percent}%)" if message else f"Loading... ({percent}%)"
                self._progress_slot.progress(min(100, max(0, percent)), text=text)
        
        def smooth_progress(self, start: int, end: int, message: str = "", duration: float = 0.3) -> None:
            """Smoothly animate progress from start to end percentage."""
            if not self._started:
                return
            steps = max(1, abs(end - start) // 3)
            delay = duration / steps if steps > 0 else 0.01
            current = start
            step_size = (end - start) / steps if steps > 0 else end - start
            for _ in range(steps):
                current = min(end, current + step_size)
                self.update(int(current), message)
                _time.sleep(delay)
            self.update(end, message)
        
        def finish(self) -> None:
            """Clear the loader - content will be rendered after this."""
            if self._started:
                self._container.empty()
                self._started = False
        
        def get_container(self):
            """Get the container for rendering final content."""
            return self._container
    
    return SectionLoader(container)


def _show_page_transition(page: str) -> None:
    """
    Show a smooth, modern page transition animation when switching tabs.
    Uses an incremental progress bar (1% → 100%) for professional UX.
    """
    import time
    
    # Page display names for the loading message
    page_names = {
        "portfolio": "Portfolio Analytics",
        "operation": "Operation Theatre",
        "reconnect": "Reconnect DSM",
        "add_comments": "Add Comments",
        "dfm": "Digital Fault Monitoring",
        "visual_analyser": "Visual Analyser",
        "meta_viewer": "Meta Viewer",
        "scb_ot": "SCB Operation Theatre",
        "scb_comment": "SCB Comment",
        "raw_analyser": "Raw Analyser",
        "s1": "S1 Portal",
        "s2": "S2 Portal",
        "s3": "S3 Portal",
    }
    display_name = page_names.get(page, page.replace("_", " ").title())
    
    # Page icons
    page_icons = {
        "portfolio": "📊",
        "operation": "🏥",
        "reconnect": "🔌",
        "add_comments": "📝",
        "dfm": "🛠️",
        "visual_analyser": "🖥️",
        "meta_viewer": "🧭",
        "scb_ot": "⚡",
        "scb_comment": "🧾",
        "raw_analyser": "🧪",
        "s1": "📋",
        "s2": "📝",
        "s3": "✅",
    }
    icon = page_icons.get(page, "⚡")
    
    # Create container for the entire transition (prevents ghosting)
    transition_container = st.empty()
    
    # Show initial loading state with styled container
    with transition_container.container():
        # Inject transition-specific CSS
        st.markdown(
            """
            <style>
            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(12px); }
                to { opacity: 1; transform: translateY(0); }
            }
            @keyframes pulse {
                0%, 100% { opacity: 0.7; transform: scale(1); }
                50% { opacity: 1; transform: scale(1.05); }
            }
            .page-loader-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                padding: 2.5rem 2rem;
                animation: fadeInUp 0.2s ease-out;
                background: linear-gradient(135deg, rgba(241,245,249,0.95) 0%, rgba(248,250,252,0.95) 100%);
                border-radius: 16px;
                margin: 1rem 0;
                box-shadow: 0 4px 20px rgba(15,23,42,0.06);
            }
            .page-loader-icon {
                font-size: 2.5rem;
                margin-bottom: 0.75rem;
                animation: pulse 1.2s ease-in-out infinite;
            }
            .page-loader-title {
                font-size: 1.35rem;
                font-weight: 700;
                color: #1e293b;
                margin-bottom: 0.25rem;
            }
            .page-loader-subtitle {
                font-size: 0.9rem;
                color: #64748b;
                margin-bottom: 1rem;
            }
            .page-loader-percent {
                font-size: 0.85rem;
                font-weight: 600;
                color: #3b82f6;
                margin-top: 0.5rem;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        st.markdown(
            f"""
            <div class="page-loader-container">
                <div class="page-loader-icon">{icon}</div>
                <div class="page-loader-title">{display_name}</div>
                <div class="page-loader-subtitle">Loading your workspace...</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Show actual progress bar with percentage updates
        progress_bar = st.progress(0, text="Initializing...")
        
        # Smooth incremental progress (perceived progress)
        stages = [
            (15, "Loading components..."),
            (35, "Preparing interface..."),
            (60, "Fetching data..."),
            (85, "Rendering content..."),
            (100, "Ready!"),
        ]
        
        current = 0
        for target, msg in stages:
            # Smooth incremental updates
            step = max(1, (target - current) // 5)
            while current < target:
                current = min(current + step, target)
                progress_bar.progress(current, text=f"{msg} ({current}%)")
                time.sleep(0.02)  # Fast but visible increments
    
    # Clear the transition - new page content will render
    transition_container.empty()


def _render_global_alerts() -> None:
    alerts = st.session_state.get("_global_alerts", [])
    if not alerts:
        return
    for alert in alerts:
        t = alert.get("type", "info")
        msg = alert.get("msg", "")
        if t == "warning": st.warning(msg, icon="⚠️")
        elif t == "error": st.error(msg, icon="🚨")
        elif t == "success": st.success(msg, icon="✅")
        else: st.info(msg, icon="ℹ️")
    st.session_state["_global_alerts"] = []


def _inject_session_timeout(timeout_minutes: int = 30, warn_before_minutes: int = 5) -> None:
    timeout_ms = timeout_minutes * 60 * 1000
    warn_ms = (timeout_minutes - warn_before_minutes) * 60 * 1000
    st.components.v1.html(
        f"""
        <script>
        (function() {{
            let warnTimer, logoutTimer;
            function resetTimers() {{
                clearTimeout(warnTimer); clearTimeout(logoutTimer);
                warnTimer = setTimeout(() => {{
                    if (confirm("⚠️ You will be logged out in {warn_before_minutes} minutes due to inactivity.\\n\\nClick OK to stay logged in.")) {{
                        resetTimers();
                    }}
                }}, {warn_ms});
                logoutTimer = setTimeout(() => {{
                    const buttons = window.parent.document.querySelectorAll('button');
                    for (let b of buttons) {{ if (b.innerText.includes('Logout')) {{ b.click(); break; }} }}
                }}, {timeout_ms});
            }}
            ['mousemove','keydown','mousedown','touchstart'].forEach(evt =>
                window.parent.document.addEventListener(evt, resetTimers, true)
            );
            resetTimers();
        }})();
        </script>
        """,
        height=0,
    )


_PAGE_META = {
    "portfolio":       ("📊", "Portfolio Analytics",    "Performance overview across all sites"),
    "operation":       ("🏥", "Operation Theatre",      "Inverter and string-level diagnostics"),
    "reconnect":       ("🔌", "Re Connect",             "DSM reconnection analysis"),
    "add_comments":    ("📝", "Add Comments",           "Annotate site events"),
    "scb_comment":     ("🧾", "SCB Comment",            "SCB-level comment management"),
    "dfm":             ("🛠️",  "Fault Detector",         "Digital fault monitoring"),
    "visual_analyser": ("🖥️",  "Visual Analyser",        "Visual data explorer"),
    "meta_viewer":     ("🧭", "Meta Viewer",            "Metadata inspection"),
    "scb_ot":          ("⚡", "SCB OT",                "SCB operation theatre"),
    "raw_analyser":    ("🧪", "Raw Analyser",           "Raw data deep-dive"),
    "s1":              ("📋", "S1",                    "S1 portal"),
    "s2":              ("📝", "S2",                    "S2 portal"),
    "s3":              ("✅", "S3",                    "S3 portal"),
}


def render_breadcrumb(page: str, as_of_date: date) -> None:
    icon, label, desc = _PAGE_META.get(page, ("📄", page.title(), ""))
    date_str = as_of_date.strftime("%d %b %Y")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    margin-bottom:12px;padding:10px 16px;border-radius:10px;
                    background:rgba(37,99,235,0.06);border:1px solid rgba(37,99,235,0.1);">
            <div style="display:flex;align-items:center;gap:10px;">
                <span style="font-size:20px;">{icon}</span>
                <div>
                    <span style="font-size:11px;color:#64748b;font-weight:600;">Zelestra EYE &rsaquo; </span>
                    <span style="font-size:15px;font-weight:800;">{label}</span>
                    <div style="font-size:11px;color:#94a3b8;">{desc}</div>
                </div>
            </div>
            <div style="font-size:12px;color:#64748b;font-weight:600;">📅 As of {date_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_coming_soon(icon: str, title: str, description: str, eta: str = "Q3 2025", contact: str = "ops-team@zelestra.com") -> None:
    st.markdown(
        f"""
        <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                    padding:60px 20px;text-align:center;border-radius:18px;
                    border:2px dashed rgba(148,163,184,0.25);margin:20px 0;">
            <div style="font-size:56px;margin-bottom:16px;">{icon}</div>
            <div style="font-size:24px;font-weight:800;margin-bottom:8px;">{title}</div>
            <div style="font-size:14px;color:#64748b;max-width:420px;line-height:1.6;margin-bottom:20px;">{description}</div>
            <div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center;">
                <div style="background:rgba(37,99,235,0.08);border:1px solid rgba(37,99,235,0.2);
                            border-radius:8px;padding:8px 16px;font-size:12px;font-weight:600;color:#2563eb;">
                    🗓️ Expected: {eta}
                </div>
                <div style="background:rgba(148,163,184,0.08);border:1px solid rgba(148,163,184,0.2);
                            border-radius:8px;padding:8px 16px;font-size:12px;font-weight:600;color:#64748b;">
                    ✉️ {contact}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# Data access
# -----------------------------


@dataclass(frozen=True)
class Filters:
    site_name: str
    as_of_date: date
    tariff_inr_per_kwh: float


def _connect(db_path: str) -> duckdb.DuckDBPyConnection:
    return get_duckdb_connection(db_local=db_path)


@st.cache_data(show_spinner=False, ttl=300)
def list_sites(db_path: str) -> list[str]:
    con = _connect(db_path)
    try:
        rows = con.execute("select distinct site_name from daily_kpi order by site_name").fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


@st.cache_data(show_spinner=False, ttl=300)
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


@st.cache_data(show_spinner=False, ttl=300)
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


@st.cache_data(show_spinner=False, ttl=300)
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


@st.cache_data(show_spinner=False, ttl=300)
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


@st.cache_data(show_spinner=False, ttl=300)
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

PLOTLY_CONFIG = {
    "displayModeBar": "hover",
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d", "hoverCompareCartesian"],
    "modeBarButtonsToAdd": ["toggleSpikelines"],
    "toImageButtonOptions": {
        "format": "png", "filename": "zelestra_chart",
        "height": 500, "width": 900, "scale": 2
    },
    "displaylogo": False,
    "scrollZoom": False,
}


def _plotly_base_layout(title: str, *, x_title: str | None = None, y_title: str | None = None) -> dict[str, Any]:
    _font_color = "#f1f5f9" if st.session_state.get("app_theme", "dark") == "dark" else "#0b1220"
    layout: dict[str, Any] = dict(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=18, color=_font_color)),
        margin=dict(l=14, r=10, t=46, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=_font_color),
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
    if not x or not actual or not target:
        fig = go.Figure()
        fig.update_layout(**_plotly_base_layout(title, y_title=y_title))
        return fig
    
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
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
        except Exception as e:
            st.error(f"Error rendering generation chart: {e}")
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        poa = float(daily["poa"].iloc[0]) if not daily.empty else 0.0
        b_poa = float(budget["b_poa"].iloc[0]) if not budget.empty else 0.0

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
            st.plotly_chart(fig2, width="stretch", config=PLOTLY_CONFIG)
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
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
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
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
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
        st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)
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

def _init_app_state() -> None:
    """
    Initialize global application state for all tabs.
    This ensures each tab has its own isolated state namespace.
    Tab switching NEVER resets state - only explicit user actions do.
    """
    # Global defaults (only set if not already present)
    defaults = {
        # Navigation
        "nav_page": "portfolio",

        # Portfolio Analytics state
        "pa_last_fig": None,
        "pa_last_raw": None,
        "pa_last_meta": None,

        # Operation Theatre state
        "ot_last_fig": None,
        "ot_last_df": None,
        "ot_last_meta_universe": None,
        "ot_last_meta_d1": None,
        "ot_last_meta_d2": None,
        "ot_last_plot_sites": None,
        "ot_last_plot_d1": None,
        "ot_last_plot_d2": None,

        # SCB OT state
        "scb_ot_last_fig": None,
        "scb_ot_last_table": None,
        "scb_ot_last_insights": None,
        "scb_ot_last_comments": None,
        "scb_ot_last_meta": None,

        # Raw Analyser state
        "raw_analyser_last_df": None,
        "raw_analyser_last_site": None,
        "raw_analyser_last_table": None,
        "raw_analyser_last_filters": None,
        "raw_analyser_last_display_map": None,
        "raw_analyser_last_scb_cols": None,
        "raw_analyser_last_string_num_map": None,
        "raw_analyser_norm_df": None,
        "raw_analyser_applied_norm_method": None,

        # Reconnect DSM state
        "dsm_results_df": None,
        "dsm_detailed_df": None,
        "dsm_from_date": None,
        "dsm_to_date": None,
        "dsm_last_fig": None,
        "dsm_last_summary": None,

        # Meta Viewer state
        "mv_last_df": None,
        "mv_last_meta": None,

        # Add Comments state (form persistence)
        "comments_edit_id": None,
        "comments_edit_row": None,

        "app_theme": "light",
        "_global_alerts": [],
    }

    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value


def main() -> None:
    _inject_css()
    _inject_sidebar_freeze_layout()
    _inject_session_timeout(timeout_minutes=30, warn_before_minutes=5)
    _render_global_alerts()

    # Initialize global state for ALL tabs (bidirectional persistence)
    _init_app_state()

    # Sidebar navigation state (matches the UI you shared)
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "portfolio"

    user_info = st.session_state.get("user_info", {}) or {}
    # Prefer normalized username set by auth.check_password(); fall back to user_info
    username = (st.session_state.get("username") or user_info.get("username") or "").strip().lower() or None

    allowed_pages = user_allowed_pages()

    # Deny access if no roles assigned
    if not allowed_pages:
        st.error("Access Denied")
        st.stop()

    # Set default page to first allowed page (stable order)
    default_page = "portfolio"
    for key, _ in ALL_PAGES:
        if key in allowed_pages:
            default_page = key
            break

    if st.session_state.nav_page not in allowed_pages:
        st.session_state.nav_page = default_page

    db_path = DB_DEFAULT
    try:
        _ = get_duckdb_connection(db_local=db_path).close()
    except Exception as e:
        st.error(f"Failed to open DuckDB (S3 download may be required). {e}")
        st.stop()

    sites = list_sites(db_path)
    if not sites:
        st.error("No sites found in DB (daily_kpi is empty). Run the loader first.")
        st.stop()

    site_name = sites[0]
    dates = get_available_dates(db_path, site_name)
    if dates:
        as_of_date = dates[-1]
    else:
        as_of_date = date.today()
    tariff = 6.0

    try:
        with st.sidebar:
            username_disp = username or "Unknown"
            
            # Fixed header at top
            st.markdown(
                f"""
                <div class="sidebar-top">
                  <div class="sb-header">
                    <div>
                      <div class="sb-title-top">Zelestra</div>
                      <div class="sb-title">Operation Intelligence</div>
                    </div>
                  </div>
                  <div class="sb-welcome">Welcome, {username_disp}</div>
                  <div class="sb-divider"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Empty scrollable nav container (JS will move buttons here)
            st.markdown('<div class="sidebar-nav"></div>', unsafe_allow_html=True)

            # Nav buttons (will be moved into .sidebar-nav by JS)
            nav_items = [
                (key, label)
                for key, label in ALL_PAGES
                if key in allowed_pages
            ]

            for key, label in nav_items:
                is_active = st.session_state.nav_page == key
                btn_type = "primary" if is_active else "secondary"
                
                # Add subtle group separator before S-series pages
                if key == "s1":
                    st.markdown(
                        '<div style="height:1px;background:rgba(255,255,255,0.07);margin:8px 14px;"></div>',
                        unsafe_allow_html=True
                    )
                
                if st.button(label, key=f"nav_{key}", type=btn_type, use_container_width=True):
                    st.session_state.nav_page = key
                    st.rerun()

            # Empty footer container (JS will move last 2 buttons here)
            st.markdown('<div class="sidebar-bottom"></div>', unsafe_allow_html=True)

            # Footer buttons (will be moved into .sidebar-bottom by JS)
            current_theme = st.session_state.get("app_theme", "light")
            if current_theme == "light":
                theme_label = "🌙 Switch to Dark Mode"
            else:
                theme_label = "☀️ Switch to Light Mode"
            if st.button(theme_label, use_container_width=True, type="secondary"):
                st.session_state.app_theme = (
                    "light" if st.session_state.get("app_theme", "light") == "dark" else "dark"
                )
                st.rerun()

            if st.button("🚪 Logout", use_container_width=True, type="secondary"):
                auth.logout()
    except Exception as e:
        st.error(f"Sidebar error: {e}")
        # Fallback: show sidebar content in main area for debugging
        st.sidebar.write("Sidebar content (fallback)")

    # Tab-enter hooks (run-once per navigation)
    prev_page = st.session_state.get("_last_nav_page")
    page = st.session_state.nav_page
    entered_new_page = prev_page != page
    st.session_state["_last_nav_page"] = page

    def _reset_scb_comment_state_on_enter() -> None:
        """
        SCB Comment should ALWAYS reload fresh on tab entry (per requirement),
        because it is an input/write workflow.
        """
        for k in list(st.session_state.keys()):
            if k.startswith("scb_comment_"):
                st.session_state.pop(k, None)
        # Also clear non-prefixed SCB-comment-specific flags if any
        st.session_state.pop("scb_comment_success_until", None)

        # Force clean defaults for SCB Comment widgets (prevents any stale widget restoration)
        st.session_state["scb_comment_site"] = "(select)"
        st.session_state["scb_comment_site_locked"] = None
        st.session_state["scb_comment_threshold"] = -3.0
        st.session_state["scb_comment_from"] = None
        st.session_state["scb_comment_to"] = None

        # View/Edit section defaults
        st.session_state["scb_comment_ve_site"] = "(select)"
        st.session_state["scb_comment_ve_site_locked"] = None
        st.session_state["scb_comment_ve_from"] = None
        st.session_state["scb_comment_ve_to"] = None

        # Rotate file_uploader key so uploaded file is ALWAYS cleared on re-entry
        st.session_state["scb_comment_uploader_counter"] = int(st.session_state.get("scb_comment_uploader_counter", 0) or 0) + 1

    if entered_new_page and page == "scb_comment":
        _reset_scb_comment_state_on_enter()

    # Broadcast a per-page "entered" flag for feature modules to consume (pop) on render.
    # This enables restore-on-tab-enter without fighting widget ❌ behavior on normal reruns.
    if entered_new_page:
        st.session_state[f"_entered_{page}"] = True
        # Show smooth page transition animation
        _show_page_transition(page)

    # Route to pages
    f = Filters(site_name=site_name, as_of_date=as_of_date, tariff_inr_per_kwh=float(tariff))
    page = st.session_state.nav_page

    # Defensive routing safety: block hidden pages even if nav_page is tampered.
    if page not in allowed_pages:
        st.error("Access Denied")
        st.stop()

    # =========================================================================
    # ANTI-GHOSTING: Create isolated page container
    # This prevents old page content from persisting during navigation
    # =========================================================================
    
    # Clear any stale page content from previous navigation
    # by using a unique key per page that forces Streamlit to re-create container
    page_container_key = f"_page_container_{page}"
    
    # Track which page was last rendered to detect navigation
    last_rendered_page = st.session_state.get("_last_rendered_page")
    if last_rendered_page != page:
        # Page changed - clear cached content keys to prevent ghosting
        for key in list(st.session_state.keys()):
            # Clear page-specific UI caches but preserve form data
            if key.startswith("_page_container_") and key != page_container_key:
                st.session_state.pop(key, None)
        st.session_state["_last_rendered_page"] = page
    
    # Route to page content (only ONE page renders per execution)
    if page not in allowed_pages:
        st.error("Access Denied")
        st.stop()
    if page == "portfolio":
        portfolio_analytics.render(db_path)
    elif page == "operation":
        operation_theatre.render(db_path)
    elif page == "reconnect":
        reconnect_dsm.render(db_path)
    elif page == "add_comments":
        add_comments.render(db_path)
    elif page == "dfm":
        render_coming_soon(
            icon="🛠️", title="Digital Fault Monitoring",
            description="Automated fault detection across inverters and strings using rule-based and ML anomaly detection.",
            eta="Q3 2025",
        )
    elif page == "visual_analyser":
        render_coming_soon(
            icon="🖥️", title="Visual Analyser",
            description="Interactive visual exploration of raw sensor data with heatmaps, correlation matrices, and trend overlays.",
            eta="Q4 2025",
        )
    elif page == "meta_viewer":
        meta_viewer.render(db_path)
    elif page == "scb_ot":
        scb_ot.render_scb_ot(db_path)
    elif page == "scb_comment":
        scb_comment.render(db_path)
    elif page == "raw_analyser":
        raw_analyser.render(db_path)
    elif page == "s1":
        S1.render(db_path)
    elif page == "s2":
        S2.render(db_path)
    elif page == "s3":
        S3.render(db_path)
    else:
        st.error("Unknown page")


if __name__ == "__main__":
    main()


