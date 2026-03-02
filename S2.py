from __future__ import annotations

"""
S2 Portal - PTW Review & Forwarding Stage

This module implements:
- View Work Order: Read-only view (reuses S1 logic)
- View Submitted PTW: Review, edit, and forward PTWs submitted from S1

Key responsibilities:
- Review PTWs submitted in S1
- Add Permit Holder Name
- Confirm Isolation Requirement (with evidence upload)
- Confirm Tool Box Talk (with evidence upload)
- Forward PTW to S3 for final approval

All evidence files are stored in Supabase Storage under ptw-evidence bucket.
"""

import os
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import time as _time
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from PyPDF2 import PdfReader, PdfWriter  # type: ignore

from supabase_link import get_supabase_client

# Import shared functions from S1
from S1 import (
    TABLE_WORK_ORDERS,
    TABLE_PTW_REQUESTS,
    TABLE_PTW_TEMPLATES,
    UI_STATUSES,
    DB_STATUS_TO_UI,
    STATUS_ORDER,
    derive_ptw_status,
    _list_sites_from_work_orders,
    _list_locations_from_work_orders,
    _list_statuses_from_work_orders,
    _fetch_work_orders,
    _highlight_status,
    fetch_ptw_requests,
    _download_template_from_supabase,
    build_doc_data,
    generate_ptw_pdf,
)

# Shared approval/PDF helpers (breaks S2<->S3 circular imports)
from ptw_approval_utils import add_floating_approval_stamp, get_ptw_approval_times
from ptw_pdf_pipeline import (
    EVIDENCE_BUCKET,
    _get_content_type,
    download_evidence_file as _download_evidence_file,
    generate_ptw_pdf_with_attachments,
    list_evidence_files as _list_evidence_files,
)


# =============================================================================
# CONSTANTS
# =============================================================================

# S2-specific session state prefix
S2_PREFIX = "s2_"


# =============================================================================
# UX HELPERS (UI-only)
# =============================================================================


def _smooth_progress(prog: Any, start: int, end: int, *, text: str, step_delay_s: float = 0.008) -> None:
    """UI-only helper to animate progress smoothly."""
    start_i = int(start)
    end_i = int(end)
    if end_i < start_i:
        start_i, end_i = end_i, start_i
    for p in range(start_i, end_i + 1):
        prog.progress(p, text=text)
        _time.sleep(step_delay_s)


def modern_section_selector(options: list[str], *, key: str) -> str:
    """
    UI-only segmented (pill) selector built with Streamlit buttons.
    Uses session_state[key] as the single source of truth for selection.
    """
    if not options:
        raise ValueError("options must be a non-empty list")

    if key not in st.session_state:
        st.session_state[key] = options[0]

    cols = st.columns(len(options))
    changed = False
    for i, opt in enumerate(options):
        is_active = st.session_state.get(key) == opt
        btn_kind = "primary" if is_active else "secondary"
        if cols[i].button(
            opt,
            key=f"{key}_{i}",
            use_container_width=True,
            type=btn_kind,
        ):
            if st.session_state.get(key) != opt:
                st.session_state[key] = opt
                changed = True

    if changed:
        st.rerun()

    return str(st.session_state.get(key, options[0]))


def _apply_modern_tabs_css() -> None:
    """Match S1 tab UX (UI-only) + Anti-Ghosting CSS."""
    st.markdown(
        """
        <style>
          /* ==============================================
             ANTI-GHOSTING: Prevent old tab content flash
             ============================================== */
          
          /* Hide all tab panels by default, show only active */
          .stTabs [data-baseweb="tab-panel"] {
            opacity: 0;
            animation: tabContentFadeIn 0.2s ease-out forwards;
          }
          
          @keyframes tabContentFadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
          }
          
          /* Ensure clean transition between tabs */
          .stTabs [data-baseweb="tab-panel"][hidden] {
            display: none !important;
            visibility: hidden !important;
          }
          
          /* ==============================================
             TABS STYLING
             ============================================== */
          
          .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            padding: 6px 4px 10px 4px;
            border-bottom: 1px solid rgba(148,163,184,0.45);
          }
          .stTabs [data-baseweb="tab"] {
            font-size: 18px;
            font-weight: 800;
            padding: 10px 16px;
            border-radius: 12px;
            background: rgba(226,232,240,0.35);
            color: #0f172a;
            border: 1px solid rgba(148,163,184,0.28);
            transition: all 0.15s ease;
          }
          .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(37,99,235,0.14), rgba(59,130,246,0.09));
            border: 1px solid rgba(37,99,235,0.35);
            color: #0b2a6f;
            box-shadow: 0 8px 20px rgba(15,23,42,0.08);
          }
          .stTabs [data-baseweb="tab-panel"] {
            padding-top: 12px;
          }
          .kpi-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0 14px 0; }
          .kpi-card { flex: 1 1 160px; background: white; border: 1px solid rgba(148,163,184,0.35);
                      border-radius: 14px; padding: 14px 16px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); }
          .kpi-title { font-size: 14px; color: #475569; margin-bottom: 6px; font-weight: 700; }
          .kpi-value { font-size: 34px; font-weight: 900; line-height: 1.05; }

          /* Reduce space below page title */
          h1 {
            margin-bottom: 10px !important;
          }
          /* Reduce space above tabs */
          .stTabs {
            margin-top: -10px !important;
          }
          /* Remove extra gap inside tab container */
          .stTabs [data-baseweb="tab-list"] {
            padding-top: 0px !important;
            margin-top: 0px !important;
          }
          /* Remove large default Streamlit top padding */
          .block-container {
            padding-top: 1rem !important;
          }

          /* ==============================================
             SEGMENTED NAV (pill-style buttons)
             ============================================== */
          /* Segmented button container spacing */
          div[data-testid="column"] {
              padding: 0px 4px !important;
          }

          /* Button styling */
          .stButton > button {
              border-radius: 12px !important;
              font-weight: 700 !important;
              height: 45px !important;
              border: 1px solid rgba(148,163,184,0.35) !important;
              transition: all 0.2s ease-in-out !important;
          }

          /* Secondary style (inactive) */
          .stButton > button[kind="secondary"] {
              background-color: rgba(226,232,240,0.35) !important;
              color: #0f172a !important;
          }

          /* Primary style (active) */
          .stButton > button[kind="primary"] {
              background: linear-gradient(135deg, rgba(37,99,235,0.14), rgba(59,130,246,0.09)) !important;
              border: 1px solid rgba(37,99,235,0.35) !important;
              color: #0b2a6f !important;
              box-shadow: 0 8px 20px rgba(15,23,42,0.08) !important;
          }

          .stButton > button:hover {
              transform: translateY(-1px);
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _compute_s2_kpis_from_work_orders(*, start_date: date, end_date: date) -> dict[str, int]:
    """
    UI-only KPI computation for S2 "View Submitted PTW".

    Requirements:
    - Must be computed from `work_orders` only (NOT ptw_requests, NOT derive_ptw_status)
    - Must respect the selected date range, using stage-specific lifecycle dates:
      - Pending @ S2 uses date_s1_created
      - Pending @ S3 uses date_s2_forwarded
      - Approved uses date_s3_approved
    - Must be recalculated only when the user clicks "Fetch PTWs" (same rerun that fetches data)
    - Rejected KPI is intentionally omitted from the UI.
    """
    start_iso = start_date.strftime("%Y-%m-%d")
    end_iso = (end_date + timedelta(days=1)).strftime("%Y-%m-%d")

    sb = get_supabase_client(prefer_service_role=True)

    # ----------------------------
    # Pending at S2
    # ----------------------------
    res_s2 = (
        sb.table(TABLE_WORK_ORDERS)
        .select("work_order_id")
        .gte("date_s1_created", start_iso)
        .lt("date_s1_created", end_iso)
        .is_("date_s2_forwarded", "null")
        .execute()
    )
    pending_s2 = len(getattr(res_s2, "data", None) or [])

    # ----------------------------
    # Pending at S3
    # ----------------------------
    res_s3 = (
        sb.table(TABLE_WORK_ORDERS)
        .select("work_order_id")
        .gte("date_s2_forwarded", start_iso)
        .lt("date_s2_forwarded", end_iso)
        .is_("date_s3_approved", "null")
        .execute()
    )
    pending_s3 = len(getattr(res_s3, "data", None) or [])

    # ----------------------------
    # Approved
    # ----------------------------
    res_approved = (
        sb.table(TABLE_WORK_ORDERS)
        .select("work_order_id")
        .gte("date_s3_approved", start_iso)
        .lt("date_s3_approved", end_iso)
        .execute()
    )
    approved = len(getattr(res_approved, "data", None) or [])

    total = int(pending_s2 + pending_s3 + approved)

    return {
        "total": int(total),
        "pending_s2": int(pending_s2),
        "pending_s3": int(pending_s3),
        "approved": int(approved),
    }


# =============================================================================
# SUPABASE STORAGE HELPERS
# =============================================================================


def _ensure_bucket_exists() -> bool:
    """
    Ensure the evidence bucket exists in Supabase Storage.
    Returns True if bucket exists or was created, False if failed.
    """
    sb = get_supabase_client(prefer_service_role=True)
    
    try:
        # Try to list buckets to check if our bucket exists
        buckets = sb.storage.list_buckets()
        bucket_names = [b.name if hasattr(b, 'name') else b.get('name', '') for b in buckets]
        
        if EVIDENCE_BUCKET in bucket_names:
            return True
        
        # Try to create the bucket
        try:
            sb.storage.create_bucket(EVIDENCE_BUCKET, options={"public": False})
            return True
        except Exception:
            # Bucket might already exist or we don't have permission
            return True  # Assume it exists and let upload fail if not
            
    except Exception:
        return True  # Optimistic - let the actual operation fail if needed


def _upload_evidence_file(
    work_order_id: str,
    evidence_type: str,  # "isolation" or "toolbox"
    file_bytes: bytes,
    file_name: str,
) -> str:
    """
    Upload evidence file to Supabase Storage.
    
    Args:
        work_order_id: The work order ID (used for folder path)
        evidence_type: Type of evidence ("isolation" or "toolbox")
        file_bytes: The file content as bytes
        file_name: Original file name (for extension)
    
    Returns:
        The storage path of the uploaded file
    
    Raises:
        RuntimeError: If upload fails
    """
    sb = get_supabase_client(prefer_service_role=True)
    
    # Ensure bucket exists
    _ensure_bucket_exists()
    
    # Extract extension
    ext = os.path.splitext(file_name)[1].lower() or ".bin"
    
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    storage_filename = f"{work_order_id}_{evidence_type}_{timestamp}{ext}"
    storage_path = f"{work_order_id}/{evidence_type}/{storage_filename}"
    
    try:
        # Upload to Supabase Storage
        resp = sb.storage.from_(EVIDENCE_BUCKET).upload(
            path=storage_path,
            file=file_bytes,
            file_options={"content-type": _get_content_type(ext)}
        )
        
        # Check for errors
        if hasattr(resp, "error") and resp.error:
            raise RuntimeError(f"Storage upload error: {resp.error}")
        
        return storage_path
    
    except Exception as e:
        error_msg = str(e)
        if "Bucket not found" in error_msg:
            raise RuntimeError(
                f"Storage bucket '{EVIDENCE_BUCKET}' not found. "
                "Please create it in Supabase Dashboard > Storage > Create a new bucket."
            ) from e
        raise RuntimeError(f"Failed to upload evidence file: {e}") from e


# =============================================================================
# DATABASE HELPERS
# =============================================================================


def _fetch_ptw_for_s2(
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Fetch PTW requests that are visible in S2 (multi-WO safe).

    Lifecycle engine:
    - Extract work_order_ids from form_data["work_order_ids"] if present
      else legacy form_data["work_order_id"]
      else fallback to permit_no (and split on "-" if it looks like multi-WO)
    - Query work_orders once using .in_("work_order_id", all_ids)
    - Derive per-WO status using derive_ptw_status()
    - Aggregate to an S2 display status (3-state + rejected):
        REJECTED
        APPROVED_BY_S3
        PENDING_AT_S3
        PENDING_AT_S2
        OPEN

    Visibility rule (date-driven):
    - Show permits that have been submitted from S1 (date_s1_created exists on at least one covered WO)
    - Date filter uses min(date_s1_created) across covered WOs (lifecycle clock)
    """
    sb = get_supabase_client(prefer_service_role=True)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt_excl = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    # Fetch PTW requests (do NOT derive lifecycle from ptw_requests.status)
    resp = (
        sb.table(TABLE_PTW_REQUESTS)
        .select("ptw_id,permit_no,site_name,created_at,created_by,form_data")
        .order("created_at", desc=True)
        .execute()
    )
    err = getattr(resp, "error", None)
    if err:
        raise RuntimeError(f"Failed to fetch PTW requests: {err}")
    rows: list[dict[str, Any]] = getattr(resp, "data", None) or []
    if not rows:
        return pd.DataFrame()

    def _extract_ids(r: dict) -> list[str]:
        fd = r.get("form_data") if isinstance(r.get("form_data"), dict) else {}
        if isinstance(fd, dict):
            ids = fd.get("work_order_ids")
            if isinstance(ids, list) and ids:
                out = [str(x).strip() for x in ids if str(x).strip()]
                if out:
                    return out
            legacy = str(fd.get("work_order_id") or "").strip()
            if legacy:
                return [legacy]
        pn = str(r.get("permit_no") or "").strip()
        if pn and "-" in pn:
            parts = [p.strip() for p in pn.split("-") if p.strip()]
            return parts or [pn]
        return [pn] if pn else []

    ids_per_ptw: list[list[str]] = []
    all_ids: set[str] = set()
    for r in rows:
        ids = _extract_ids(r if isinstance(r, dict) else {})
        ids_per_ptw.append(ids)
        all_ids.update(ids)

    wo_lookup: dict[str, dict] = {}
    if all_ids:
        wo_resp = (
            sb.table(TABLE_WORK_ORDERS)
            .select(
                "work_order_id,site_name,location,equipment,isolation_requirement,"
                "date_s1_created,date_s2_forwarded,date_s3_approved,date_s2_rejected,date_s3_rejected"
            )
            .in_("work_order_id", list(all_ids))
            .execute()
        )
        wo_err = getattr(wo_resp, "error", None)
        if wo_err:
            raise RuntimeError(f"Failed to fetch work orders: {wo_err}")
        wo_rows = getattr(wo_resp, "data", None) or []
        for w in wo_rows:
            if isinstance(w, dict) and w.get("work_order_id"):
                wo_lookup[str(w["work_order_id"]).strip()] = w

    def _has(v) -> bool:
        return v is not None and str(v).strip() != ""

    def _agg_s2_display_status(wo_rows: list[dict]) -> str:
        if not wo_rows:
            return "OPEN"

        derived = []
        for w in wo_rows:
            d = derive_ptw_status(w)
            derived.append(d)

        sset = {str(x).strip().upper() for x in derived if str(x).strip()}
        if "REJECTED" in sset:
            return "REJECTED"
        if "CLOSED" in sset:
            return "CLOSED"
        if "APPROVED" in sset:
            return "APPROVED"
        if "WIP" in sset:
            return "WIP"
        return "OPEN"

    out_rows: list[dict[str, Any]] = []
    for r, ids in zip(rows, ids_per_ptw):
        if not isinstance(r, dict):
            continue
        wo_rows = [wo_lookup.get(i) for i in ids if i in wo_lookup]
        wo_rows = [w for w in wo_rows if isinstance(w, dict)]
        display_status = _agg_s2_display_status(wo_rows)

        # Date filter uses work_orders.date_s1_created (lifecycle clock): min s1_created across permit
        s1_vals = [w.get("date_s1_created") for w in wo_rows if w.get("date_s1_created") is not None]
        s1_min = None
        try:
            ts = pd.to_datetime(s1_vals, errors="coerce")
            ts = ts.dropna() if hasattr(ts, "dropna") else ts
            if len(ts) > 0:
                s1_min = ts.min()
        except Exception:
            s1_min = s1_vals[0] if s1_vals else None

        try:
            if s1_min is not None:
                s1_dt = pd.to_datetime(s1_min, errors="coerce")
                if pd.isna(s1_dt) or not (s1_dt >= start_dt and s1_dt < end_dt_excl):
                    continue
        except Exception:
            pass

        # Visibility gate: only show permits that have an S1-submitted lifecycle clock
        if s1_min is None:
            continue

        # Aggregate lifecycle timestamps for KPIs and display
        s2_vals = [w.get("date_s2_forwarded") for w in wo_rows if _has(w.get("date_s2_forwarded"))]
        s3_vals = [w.get("date_s3_approved") for w in wo_rows if _has(w.get("date_s3_approved"))]
        s2_min = None
        s3_min = None
        try:
            ts2 = pd.to_datetime(s2_vals, errors="coerce")
            ts2 = ts2.dropna() if hasattr(ts2, "dropna") else ts2
            if len(ts2) > 0:
                s2_min = ts2.min()
        except Exception:
            s2_min = s2_vals[0] if s2_vals else None
        try:
            ts3 = pd.to_datetime(s3_vals, errors="coerce")
            ts3 = ts3.dropna() if hasattr(ts3, "dropna") else ts3
            if len(ts3) > 0:
                s3_min = ts3.min()
        except Exception:
            s3_min = s3_vals[0] if s3_vals else None

        # Aggregate work_location for display
        locs = []
        for w in wo_rows:
            loc = str(w.get("location", "") or "").strip()
            eq = str(w.get("equipment", "") or "").strip()
            combo = f"{loc}-{eq}".strip("-")
            if combo:
                locs.append(combo)
        work_location = " & ".join(sorted(set(locs))) if locs else ""

        out_rows.append(
            {
                "ptw_id": r.get("ptw_id"),
                "permit_no": r.get("permit_no"),
                "site_name": r.get("site_name") or (wo_rows[0].get("site_name") if wo_rows else ""),
                "created_at": r.get("created_at"),
                "created_by": r.get("created_by"),
                "form_data": r.get("form_data") or {},
                "work_order_ids": ids,
                "work_location": work_location,
                "status": display_status,
                "date_s1_created": s1_min,
                "date_s2_forwarded": s2_min,
                "date_s3_approved": s3_min,
            }
        )

    df = pd.DataFrame(out_rows)
    if df.empty:
        return df
    try:
        df = df.sort_values(by=["date_s1_created"], ascending=[False], na_position="last")
    except Exception:
        pass
    return df


def _update_ptw_form_data(ptw_id: str, form_data: dict) -> None:
    """Update the form_data JSON in ptw_requests."""
    sb = get_supabase_client(prefer_service_role=True)
    
    resp = (
        sb.table(TABLE_PTW_REQUESTS)
        .update({"form_data": form_data})
        .eq("ptw_id", ptw_id)
        .execute()
    )
    
    err = getattr(resp, "error", None)
    if err:
        raise RuntimeError(f"Failed to update PTW form data: {err}")


def _update_work_order_s2_forwarded(
    work_order_id: str,
    isolation_requirement: str,
) -> None:
    """
    Update work_orders when S2 forwards PTW.
    
    Sets:
    - date_s2_forwarded = current timestamp
    - isolation_requirement = 'YES' or 'NO'
    """
    sb = get_supabase_client(prefer_service_role=True)
    
    resp = (
        sb.table(TABLE_WORK_ORDERS)
        .update(
            {
                "date_s2_forwarded": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(sep=" ", timespec="seconds"),
                "isolation_requirement": isolation_requirement,
            }
        )
        .in_("work_order_id", [str(work_order_id).strip()])
        .execute()
    )
    
    err = getattr(resp, "error", None)
    if err:
        raise RuntimeError(f"Failed to update work order: {err}")


def _revoke_s2_submission(work_order_id: str) -> None:
    """
    Revoke S2 submission by clearing date_s2_forwarded.
    
    This allows the PTW to be edited again.
    """
    sb = get_supabase_client(prefer_service_role=True)
    
    resp = (
        sb.table(TABLE_WORK_ORDERS)
        .update({"date_s2_forwarded": None})
        .in_("work_order_id", [str(work_order_id).strip()])
        .execute()
    )
    
    err = getattr(resp, "error", None)
    if err:
        raise RuntimeError(f"Failed to revoke submission: {err}")


def _extract_work_order_ids_from_form_data(*, work_order_id: str, form_data: dict | None) -> list[str]:
    """Multi-WO support: prefer form_data['work_order_ids'] if present, else fallback to work_order_id."""
    fd = form_data if isinstance(form_data, dict) else {}
    ids = fd.get("work_order_ids")
    if isinstance(ids, list) and ids:
        out = [str(x).strip() for x in ids if str(x).strip()]
        return out if out else ([work_order_id] if work_order_id else [])
    return [work_order_id] if work_order_id else []


def _update_work_orders_s2_forwarded(*, work_order_ids: list[str], isolation_requirement: str) -> None:
    from ptw_lifecycle_utils import _update_all_work_orders_lifecycle

    _update_all_work_orders_lifecycle(
        work_order_ids=[str(x).strip() for x in (work_order_ids or []) if str(x).strip()],
        update_fields={
            "date_s2_forwarded": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(sep=" ", timespec="seconds"),
            "isolation_requirement": isolation_requirement,
        },
    )


def _revoke_s2_submissions(*, work_order_ids: list[str]) -> None:
    from ptw_lifecycle_utils import _update_all_work_orders_lifecycle

    _update_all_work_orders_lifecycle(
        work_order_ids=[str(x).strip() for x in (work_order_ids or []) if str(x).strip()],
        update_fields={"date_s2_forwarded": None},
    )


# =============================================================================
# CREATE WORK ORDER — DATA HELPERS
# =============================================================================


@st.cache_data(show_spinner=False, ttl=60)
def _cwo_fetch_sites() -> list[str]:
    """Return distinct site_name values from work_orders."""
    sb = get_supabase_client(prefer_service_role=True)
    resp = sb.table(TABLE_WORK_ORDERS).select("site_name").execute()
    rows = getattr(resp, "data", None) or []
    return sorted({str(r["site_name"]).strip() for r in rows if r.get("site_name") and str(r["site_name"]).strip()})


@st.cache_data(show_spinner=False, ttl=60)
def _cwo_fetch_locations() -> list[str]:
    """Return distinct location tokens from work_orders (handles comma-separated storage)."""
    sb = get_supabase_client(prefer_service_role=True)
    resp = sb.table(TABLE_WORK_ORDERS).select("location").execute()
    rows = getattr(resp, "data", None) or []
    values: set[str] = set()
    for r in rows:
        for part in str(r.get("location") or "").split(","):
            part = part.strip()
            if part:
                values.add(part)
    return sorted(values)


@st.cache_data(show_spinner=False, ttl=60)
def _cwo_fetch_equipment() -> list[str]:
    """Return distinct equipment tokens from work_orders (handles comma-separated storage)."""
    sb = get_supabase_client(prefer_service_role=True)
    resp = sb.table(TABLE_WORK_ORDERS).select("equipment").execute()
    rows = getattr(resp, "data", None) or []
    values: set[str] = set()
    for r in rows:
        for part in str(r.get("equipment") or "").split(","):
            part = part.strip()
            if part:
                values.add(part)
    return sorted(values)


_CWO_FREQUENCIES = ["D", "W", "Q", "HY", "Y", "UP"]


def _cwo_insert_work_order(
    *,
    site_name: str,
    location: list[str],
    equipment: list[str],
    frequency: str,
    isolation_requirement: str,
    date_planned: date,
) -> str:
    """
    Insert a new work_orders row.
    work_order_id is NOT sent — auto-generated by trg_work_orders_generate_id.
    Multi-select values stored as comma-separated strings.
    Returns the auto-generated work_order_id or empty string.
    """
    sb = get_supabase_client(prefer_service_role=True)
    date_planned_str = datetime.combine(date_planned, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "site_name": site_name.strip(),
        "location": ",".join(x.strip() for x in location if x.strip()),
        "equipment": ",".join(x.strip() for x in equipment if x.strip()),
        "frequency": frequency,
        "isolation_requirement": isolation_requirement,
        "date_planned": date_planned_str,
        "status": "OPEN",
    }
    resp = sb.table(TABLE_WORK_ORDERS).insert(payload).execute()
    err = getattr(resp, "error", None)
    if err:
        raise RuntimeError(f"Failed to create Work Order: {err}")
    rows = getattr(resp, "data", None) or []
    if rows and isinstance(rows[0], dict):
        return str(rows[0].get("work_order_id") or "")
    return ""


def _cwo_update_work_order(
    *,
    work_order_id: str,
    site_name: str,
    location: list[str],
    equipment: list[str],
    frequency: str,
    isolation_requirement: str,
    date_planned: date,
) -> None:
    """
    Update work order ONLY when date_s1_created IS NULL (atomic guard).
    Raises RuntimeError if the update is blocked or the DB returns an error.
    """
    sb = get_supabase_client(prefer_service_role=True)
    date_planned_str = datetime.combine(date_planned, datetime.min.time()).strftime("%Y-%m-%d %H:%M:%S")
    payload = {
        "site_name": site_name.strip(),
        "location": ",".join(x.strip() for x in location if x.strip()),
        "equipment": ",".join(x.strip() for x in equipment if x.strip()),
        "frequency": frequency,
        "isolation_requirement": isolation_requirement,
        "date_planned": date_planned_str,
    }
    resp = (
        sb.table(TABLE_WORK_ORDERS)
        .update(payload)
        .eq("work_order_id", work_order_id)
        .is_("date_s1_created", "null")
        .execute()
    )
    err = getattr(resp, "error", None)
    if err:
        raise RuntimeError(f"Update failed: {err}")
    rows = getattr(resp, "data", None) or []
    if not rows:
        raise RuntimeError(
            "Update blocked: Work Order may already have an initiated PTW (date_s1_created is set)."
        )


# =============================================================================
# UI COMPONENTS
# =============================================================================


def _render_view_work_order_s2() -> None:
    """
    Render the View Work Order tab for S2.
    
    This is a read-only view that reuses S1's logic.
    """
    st.markdown("## View Work Order")
    st.caption("Read-only view of work orders and their current status")
    
    # Row hover styling
    st.markdown(
        """
        <style>
        .stDataFrame tbody tr:hover {
          background-color: rgba(148, 163, 184, 0.18) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    sites = _list_sites_from_work_orders()
    if not sites:
        st.warning(
            "No Site Names could be loaded from `work_orders`.\n\n"
            "Common reasons:\n"
            "- `work_orders` is empty (ingestion not done)\n"
            "- Supabase RLS/policies are blocking SELECT access"
        )
        return

    # Initialize session state
    if "s2_wo_last_df" not in st.session_state:
        st.session_state["s2_wo_last_df"] = None
        st.session_state["s2_wo_last_meta"] = None
    if "s2_wo_last_kpis" not in st.session_state:
        st.session_state["s2_wo_last_kpis"] = None
    if "s2_wo_fetch_requested" not in st.session_state:
        st.session_state["s2_wo_fetch_requested"] = False
    
    site_options = ["(select)"] + sites
    
    ss_site = st.session_state.get("s2_wo_site")
    ss_start = st.session_state.get("s2_wo_start")
    ss_end = st.session_state.get("s2_wo_end")
    have_site = ss_site not in (None, "(select)", "")
    have_dates = isinstance(ss_start, date) and isinstance(ss_end, date)
    if have_site and have_dates:
        locs = _list_locations_from_work_orders(site_name=str(ss_site), start_date=ss_start, end_date=ss_end)
        statuses_ui = _list_statuses_from_work_orders(site_name=str(ss_site), start_date=ss_start, end_date=ss_end)
    else:
        locs = []
        statuses_ui = []
    loc_options = ["(all)"] + locs
    status_options = ["(all)"] + (statuses_ui if statuses_ui else UI_STATUSES)

    with st.form("s2_view_work_orders_filters", clear_on_submit=False):
        c1, c2, c3, c4 = st.columns([2.0, 1.5, 1.5, 1.0], vertical_alignment="bottom")
        with c1:
            site_name = st.selectbox("Site Name", options=site_options, index=0, key="s2_wo_site")
        with c2:
            start_date = st.date_input("Start Date", value=None, key="s2_wo_start")
        with c3:
            end_date = st.date_input("End Date", value=None, key="s2_wo_end")
        with c4:
            submitted = st.form_submit_button("Submit", use_container_width=True)
    
    if submitted:
        st.session_state["s2_wo_fetch_requested"] = True
    
    if st.session_state.get("s2_wo_fetch_requested"):
        st.session_state["s2_wo_fetch_requested"] = False
        if not site_name or site_name == "(select)":
            st.error("Please select a Site Name.")
        elif start_date is None or end_date is None:
            st.error("Please select both Start Date and End Date.")
        elif start_date > end_date:
            st.error("Start Date must be on or before End Date.")
        else:
            progress_slot = st.empty()
            prog = progress_slot.progress(0, text="Safety First: Initializing...")
            _smooth_progress(prog, 0, 18, text="Validating filters...")
            _smooth_progress(prog, 18, 55, text="Fetching work orders...")
            df = _fetch_work_orders(
                site_name=site_name,
                start_date=start_date,
                end_date=end_date,
                status_ui=None,
                location=None,
            )
            _smooth_progress(prog, 55, 88, text="Preparing results...")
            _smooth_progress(prog, 88, 100, text="Results ready")
            progress_slot.empty()
            st.session_state["s2_wo_last_df"] = df
            st.session_state["s2_wo_last_meta"] = {
                "site_name": site_name,
                "start_date": start_date,
                "end_date": end_date,
                "status": None,
                "location": None,
            }
            if isinstance(df, pd.DataFrame) and not df.empty:
                total = int(len(df))
                c_rej = int((df["status"].astype("string").str.upper() == "REJECTED").sum())
                c_open = int((df["status"].astype("string").str.upper() == "OPEN").sum())
                c_wip = int((df["status"].astype("string").str.upper() == "WIP").sum())
                c_approved = int(df["status"].astype("string").str.upper().isin(["APPROVED", "CLOSED"]).sum())
                c_closed = int((df["status"].astype("string").str.upper() == "CLOSED").sum())
            else:
                total = c_rej = c_open = c_wip = c_approved = c_closed = 0
            st.session_state["s2_wo_last_kpis"] = {
                "total": total,
                "rejected": c_rej,
                "open": c_open,
                "wip": c_wip,
                "approved": c_approved,
                "closed": c_closed,
            }
    
    # Render cached results (outside button/action block)
    df_last = st.session_state.get("s2_wo_last_df")
    if isinstance(df_last, pd.DataFrame) and df_last.empty:
        st.info("No work orders found for the selected filters.")
    elif isinstance(df_last, pd.DataFrame) and not df_last.empty:
        df = df_last
        
        # KPI cards
        k = st.session_state.get("s2_wo_last_kpis") or {}
        total = int(k.get("total", 0) or 0)
        c_rej = int(k.get("rejected", 0) or 0)
        c_open = int(k.get("open", 0) or 0)
        c_wip = int(k.get("wip", 0) or 0)
        c_approved = int(k.get("approved", 0) or 0)
        c_closed = int(k.get("closed", 0) or 0)
        COLOR_MAP = {
            "total": "#0f172a",
            "open": "#2563eb",
            "wip": "#f97316",
            "approved": "#10b981",
            "closed": "#065f46",
            "rejected": "#dc2626",
        }

        st.markdown(
            f"""
            <div class="kpi-row">
              <div class="kpi-card"><div class="kpi-title">Work Orders</div><div class="kpi-value" style="color:{COLOR_MAP['total']};">{total}</div></div>
              <div class="kpi-card"><div class="kpi-title">Rejected</div><div class="kpi-value" style="color:{COLOR_MAP['rejected']};">{c_rej}</div></div>
              <div class="kpi-card"><div class="kpi-title">Open</div><div class="kpi-value" style="color:{COLOR_MAP['open']};">{c_open}</div></div>
              <div class="kpi-card"><div class="kpi-title">Awaiting Approval</div><div class="kpi-value" style="color:{COLOR_MAP['wip']};">{c_wip}</div></div>
              <div class="kpi-card"><div class="kpi-title">Approved</div><div class="kpi-value" style="color:{COLOR_MAP['approved']};">{c_approved}</div></div>
              <div class="kpi-card"><div class="kpi-title">Closed</div><div class="kpi-value" style="color:{COLOR_MAP['closed']};">{c_closed}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.divider()
        
        # Format date for display
        if "date_planned" in df.columns:
            df["date_planned"] = pd.to_datetime(df["date_planned"]).dt.strftime("%Y-%m-%d")
        
        # Table with styling (display labels only)
        df_display = df.copy()
        df_display["status"] = df_display["status"].astype("string").fillna("").map(
            lambda s: DB_STATUS_TO_UI.get(str(s).strip().upper(), str(s))
        )
        styled = df_display.style.map(_highlight_status, subset=["status"]).set_table_styles(
            [{"selector": "th", "props": [("font-weight", "800"), ("color", "#0f172a")]}]
        )
        st.dataframe(styled, width="stretch", hide_index=True)


def _render_view_submitted_ptw() -> None:
    """
    Render the View Submitted PTW tab for S2.
    
    Features:
    - Date range filter
    - PTW listing as expandable accordions
    - Editable form fields
    - Mandatory inputs: Holder Name, Isolation (with file), Toolbox (with file)
    - Submit and Revoke buttons
    """
    # IMPORTANT UX: When using Streamlit fragments, only the fragment reruns.
    # If we fetch inside a fragment but render the list outside it, the list won't update
    # until some other rerun occurs (e.g., tab switch). So we render the entire tab inside
    # a single fragment when available.
    _frag = getattr(st, "fragment", None)
    if callable(_frag):
        def _impl() -> None:
            _render_view_submitted_ptw_body()
        _frag(_impl)()
        return

    _render_view_submitted_ptw_body()


def _render_view_submitted_ptw_body() -> None:
    st.markdown("## View Submitted PTW")
    st.caption("Review and forward PTWs submitted from S1")
    
    # Custom CSS for better UI
    st.markdown("""
        <style>
        .success-card {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            padding: 1.5rem;
            border-radius: 12px;
            color: white;
            margin: 1rem 0;
        }
        .success-card h3 { margin: 0 0 0.5rem 0; color: white; }
        .success-card p { margin: 0.3rem 0; opacity: 0.95; }
        </style>
    """, unsafe_allow_html=True)
    
    if "s2_active_ptw_id" not in st.session_state:
        st.session_state["s2_active_ptw_id"] = None
    if "s2_ptw_fetch_requested" not in st.session_state:
        st.session_state["s2_ptw_fetch_requested"] = False

    # Date range filter + fetch (button only sets state; heavy work in separate block)
    def _filters_block() -> None:
        col1, col2, col3 = st.columns([1.5, 1.5, 1])
        with col1:
            start_date = st.date_input(
                "From Date",
                value=st.session_state.get("s2_ptw_start_val", date.today() - timedelta(days=30)),
                key="s2_ptw_start",
            )
            st.session_state["s2_ptw_start_val"] = start_date
        with col2:
            end_date = st.date_input(
                "To Date",
                value=st.session_state.get("s2_ptw_end_val", date.today()),
                key="s2_ptw_end",
            )
            st.session_state["s2_ptw_end_val"] = end_date
        with col3:
            st.write("")
            st.write("")
            if st.button("Fetch PTWs", type="primary", key="s2_fetch_ptw"):
                st.session_state["s2_ptw_fetch_requested"] = True

        # Heavy work: only when requested or after refresh
        if st.session_state.get("s2_ptw_fetch_requested") or st.session_state.get("refresh_s2"):
            fetch_clicked = bool(st.session_state.get("s2_ptw_fetch_requested"))
            st.session_state["s2_ptw_fetch_requested"] = False
            st.session_state["refresh_s2"] = False
            progress_slot = st.empty()
            prog = progress_slot.progress(0, text="Fetching submitted PTWs...")
            try:
                _smooth_progress(prog, 0, 20, text="Validating date range...")
                _smooth_progress(prog, 20, 70, text="Loading from database...")
                df = _fetch_ptw_for_s2(start_date=start_date, end_date=end_date)
                st.session_state["s2_ptw_view_df"] = df
                # KPI snapshot: ONLY recompute on explicit Fetch PTWs click (requirement).
                if fetch_clicked:
                    try:
                        st.session_state["s2_vsp_kpis"] = _compute_s2_kpis_from_work_orders(
                            start_date=start_date, end_date=end_date
                        )
                    except Exception:
                        st.session_state["s2_vsp_kpis"] = {
                            "total": 0,
                            "pending_s2": 0,
                            "pending_s3": 0,
                            "approved": 0,
                        }
                _smooth_progress(prog, 70, 100, text="PTWs ready")
            except Exception as e:
                st.error(f"Failed to fetch PTW requests: {e}")
                st.session_state["s2_ptw_view_df"] = pd.DataFrame()
                if fetch_clicked:
                    st.session_state["s2_vsp_kpis"] = {
                        "total": 0,
                        "pending_s2": 0,
                        "pending_s3": 0,
                        "approved": 0,
                    }
            finally:
                progress_slot.empty()

    _filters_block()
    
    df = st.session_state.get("s2_ptw_view_df")
    
    if df is None:
        st.info("Select a date range and click 'Fetch PTWs' to load submitted permits.")
        return
    
    if df.empty:
        st.info("No submitted PTWs found for the selected date range.")
        return
    
    # Summary metrics (modern KPI cards) — computed from work_orders only.
    k = st.session_state.get("s2_vsp_kpis") or {}
    total = int(k.get("total", 0) or 0)
    pending_s2_count = int(k.get("pending_s2", 0) or 0)
    pending_s3_count = int(k.get("pending_s3", 0) or 0)
    approved_count = int(k.get("approved", 0) or 0)

    st.markdown(
        f"""
        <div class="kpi-row">
          <div class="kpi-card">
            <div class="kpi-title">Total PTWs</div>
            <div class="kpi-value" style="color:#2563eb;">{total}</div>
          </div>

          <div class="kpi-card">
            <div class="kpi-title">Pending at S2</div>
            <div class="kpi-value" style="color:#f97316;">{pending_s2_count}</div>
          </div>

          <div class="kpi-card">
            <div class="kpi-title">Pending at S3</div>
            <div class="kpi-value" style="color:#dc2626;">{pending_s3_count}</div>
          </div>

          <div class="kpi-card">
            <div class="kpi-title">Approved</div>
            <div class="kpi-value" style="color:#10b981;">{approved_count}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.divider()
    
    # BUILD DROPDOWN OPTIONS FROM df
    def _build_ptw_option(row):
        """
        Build dropdown label for PTW selection.
        Required format:
          {permit_no} | {site_name} | {STATUS} | {date_s1_created}
        
        IMPORTANT: Use date_s1_created (lifecycle clock) NOT created_at.
        """
        permit_no = str(row.get("permit_no", "") or "")
        site = row.get("site_name", "")
        # Use lifecycle clock (S1 submission time), not ptw_requests.created_at
        s1_created = row.get("date_s1_created", "")
        status = str(row.get("status") or "").strip() or "PENDING_AT_S2"

        s = status.strip().upper()
        badge = "⚪"
        if s == "PENDING_AT_S2":
            badge = "🔴"
        elif s == "PENDING_AT_S3":
            badge = "🟠"
        elif s in ("APPROVED_BY_S3", "APPROVED", "CLOSED"):
            badge = "🟢"
        elif s == "REJECTED":
            badge = "⛔"

        try:
            s1_created = pd.to_datetime(s1_created).strftime("%d-%m-%Y %H:%M")
        except Exception:
            pass
        # Keep permit_no first so downstream parsing remains stable.
        return f"{permit_no} | {site} | {badge} {status} | {s1_created}"
    
    ptw_options = {
        _build_ptw_option(row): row
        for _, row in df.iterrows()
    }
    
    # SINGLE-PTW SELECTOR
    st.markdown("### Select PTW to Review")
    selected_label = st.selectbox(
        "Choose a PTW from the list",
        options=["(select PTW)"] + list(ptw_options.keys()),
        index=0,
        key="s2_selected_ptw_label",
        label_visibility="collapsed",
    )
    
    # RENDER ONLY SELECTED PTW
    if selected_label and selected_label != "(select PTW)":
        row = ptw_options[selected_label]
        
        # Normalize IDs to string so UI state keys are stable across pandas/numpy dtypes.
        ptw_id = str(row.get("ptw_id", ""))
        work_order_id = str(row.get("permit_no", "") or "")
        site_name = row.get("site_name", "")
        work_location = row.get("work_location", "")
        status = str(row.get("status", "OPEN") or "").strip() or "OPEN"
        form_data = row.get("form_data", {}) or {}
        date_s2_forwarded = row.get("date_s2_forwarded")
        
        # Check if forwarded based on S2 aggregated display status.
        # Important: mixed partial-forward / partial-approve is intentionally actionable at S2
        # and is represented as PENDING_AT_S2 even if some WOs have date_s2_forwarded.
        is_forwarded = status.strip().upper() == "PENDING_AT_S3"

        # Streamlit-safe status badge (selectbox items can't be reliably colored)
        st_status = status.strip().upper()
        if st_status == "PENDING_AT_S2":
            st.markdown(
                "<div class='status-badge' style='background:#fee2e2;color:#7f1d1d;border:1px solid #fecaca;'>🔴 Pending at S2</div>",
                unsafe_allow_html=True,
            )
        elif st_status == "PENDING_AT_S3":
            st.markdown(
                "<div class='status-badge' style='background:#ffedd5;color:#7c2d12;border:1px solid #fed7aa;'>🟠 Pending at S3</div>",
                unsafe_allow_html=True,
            )
        elif st_status in ("APPROVED_BY_S3", "APPROVED"):
            st.markdown(
                "<div class='status-badge' style='background:#dcfce7;color:#065f46;border:1px solid #bbf7d0;'>🟢 Approved</div>",
                unsafe_allow_html=True,
            )
        elif st_status == "CLOSED":
            st.markdown(
                "<div class='status-badge' style='background:#d1fae5;color:#065f46;border:1px solid #a7f3d0;'>🟢 Closed</div>",
                unsafe_allow_html=True,
            )
        elif st_status == "REJECTED":
            st.markdown(
                "<div class='status-badge' style='background:#fee2e2;color:#7f1d1d;border:1px solid #fecaca;'>⛔ Rejected</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div class='status-badge' style='background:#e0f2fe;color:#0f172a;border:1px solid #bae6fd;'>⚪ Open</div>",
                unsafe_allow_html=True,
            )
        
        # Check if just submitted (show success state)
        just_submitted_key = f"s2_wo_{work_order_id}_just_submitted"
        just_submitted = bool(st.session_state.get(just_submitted_key, False))
        
        st.divider()
        
        if just_submitted:
            # Show success message and download option
            _render_post_submit_view(
                ptw_id=ptw_id,
                work_order_id=work_order_id,
                site_name=site_name,
                form_data=form_data,
                just_submitted_key=just_submitted_key,
            )
        elif is_forwarded:
            # Show read-only summary card for forwarded PTW
            _render_forwarded_ptw_card(
                ptw_id=ptw_id,
                work_order_id=work_order_id,
                site_name=site_name,
                work_location=work_location,
                form_data=form_data,
            )
        else:
            # Show editable PTW detail form
            _render_ptw_detail(
                ptw_id=ptw_id,
                work_order_id=work_order_id,
                site_name=site_name,
                work_location=work_location,
                form_data=form_data,
                is_forwarded=is_forwarded,
                status=status,
            )


def _render_forwarded_ptw_card(
    *,
    ptw_id: str,
    work_order_id: str,
    site_name: str,
    work_location: str,
    form_data: dict,
) -> None:
    """
    Render read-only summary card for PTWs that have been forwarded to S3.
    
    This prevents editing and provides clear visual confirmation of PTW state.
    Only non-destructive actions (Download PDF) are allowed.
    """
    st.info("🔒 This PTW has been forwarded to S3. Editing is disabled.")
    
    # Styled container for the summary card
    st.markdown(
        """
        <style>
        .forwarded-card {
            background-color: #f1f7ff;
            border: 1px solid #cfe3ff;
            padding: 16px;
            border-radius: 10px;
            margin-bottom: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    with st.container():
        st.markdown('<div class="forwarded-card">', unsafe_allow_html=True)
        st.markdown("### 🧾 PTW Summary")
        
        # Key metadata in two columns
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**Work Order ID:** {work_order_id}")
            st.markdown(f"**Site Name:** {site_name}")
        
        with c2:
            st.markdown(f"**Work Location:** {work_location}")
            permit_holder = form_data.get("holder_name", "N/A")
            st.markdown(f"**Permit Holder:** {permit_holder}")
        
        st.markdown("---")
        
        # Additional details
        isolation_req = form_data.get("isolation_required", "N/A")
        toolbox_conducted = form_data.get("toolbox_conducted", False)
        st.markdown(f"**Isolation Required:** {isolation_req}")
        st.markdown(f"**Toolbox Talk Conducted:** {'✓ Yes' if toolbox_conducted else '✗ No'}")
        
        remarks = form_data.get("s2_remarks", "")
        if remarks:
            st.markdown(f"**Supervisor Remarks:** {remarks}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Non-destructive action: Download PDF
    st.markdown("### 📥 Available Actions")

    dl_req_key = f"s2_vsp_fwd_dl_{work_order_id}_requested"
    dl_cache_key = f"s2_vsp_fwd_dl_{work_order_id}_bytes"
    if dl_req_key not in st.session_state:
        st.session_state[dl_req_key] = False
    if dl_cache_key not in st.session_state:
        st.session_state[dl_cache_key] = None

    def _on_dl() -> None:
        st.session_state[dl_req_key] = True

    st.button("⬇️ Generate PDF", key=f"s2_vsp_fwd_dl_{ptw_id}", use_container_width=True, on_click=_on_dl)

    if st.session_state.get(dl_req_key):
        st.session_state[dl_req_key] = False  # RESET IMMEDIATELY
        prog_slot = st.empty()
        prog = prog_slot.progress(0, text="Generating PDF...")
        try:
            def progress_callback(pct, msg):
                prog.progress(int(pct), text=msg)

            pdf_bytes = generate_ptw_pdf_with_attachments(
                form_data=form_data,
                work_order_id=work_order_id,
                progress_callback=progress_callback,
            )
            # If already S3-approved, apply stamp AFTER attachments merge.
            try:
                approval_times = get_ptw_approval_times(work_order_id)
                if approval_times.get("date_s3_approved_raw"):
                    pdf_bytes = add_floating_approval_stamp(
                        pdf_bytes, approved_on=approval_times.get("issuer_datetime", "")
                    )
            except Exception:
                pass

            st.session_state[dl_cache_key] = pdf_bytes
        except Exception as e:
            st.error(f"❌ Failed to generate PDF: {e}")
        finally:
            prog_slot.empty()

    cached_pdf = st.session_state.get(dl_cache_key)
    if isinstance(cached_pdf, (bytes, bytearray)) and len(cached_pdf) > 0:
        st.download_button(
            label="📄 Download PTW PDF",
            data=cached_pdf,
            file_name=f"{work_order_id}_PTW.pdf",
            mime="application/pdf",
            key=f"s2_vsp_fwd_dl_btn_{ptw_id}",
            use_container_width=True,
        )


def _render_post_submit_view(
    *,
    ptw_id: str,
    work_order_id: str,
    site_name: str,
    form_data: dict,
    just_submitted_key: str,
) -> None:
    """Render the view shown after successful submission."""
    
    st.markdown("""
        <div class="success-card">
            <h3>✅ PTW Successfully Submitted for Approval</h3>
            <p><strong>Work Order:</strong> {}</p>
            <p><strong>Site:</strong> {}</p>
            <p>This PTW has been forwarded to S3 for final approval.</p>
        </div>
    """.format(work_order_id, site_name), unsafe_allow_html=True)
    
    st.markdown("### Download PTW Document")
    st.caption("The PDF includes all evidence attachments on the last page")
    
    # Download button with progress (queued to show progress immediately on click rerun)
    # Use work_order_id for stability (avoid ptw_id dtype key mismatches)
    download_key = f"s2_download_{work_order_id}"
    req_key = f"{download_key}_requested"
    cache_key = f"{download_key}_bytes"

    if req_key not in st.session_state:
        st.session_state[req_key] = False
    if cache_key not in st.session_state:
        st.session_state[cache_key] = None

    def _on_gen_pdf() -> None:
        st.session_state["s2_active_ptw_id"] = work_order_id
        st.session_state[req_key] = True
        # Explicitly preserve the just_submitted flag to prevent accidental clearing
        st.session_state[just_submitted_key] = True

    st.button("Generate & Download PDF", key=download_key, type="primary", on_click=_on_gen_pdf)

    if st.session_state.get(req_key):
        st.session_state[req_key] = False
        # Explicitly preserve just_submitted flag BEFORE any operation
        st.session_state[just_submitted_key] = True
        st.session_state["s2_active_ptw_id"] = work_order_id
        
        prog_slot = st.empty()
        prog = prog_slot.progress(0, text="Preparing PDF...")
        try:
            def progress_callback(pct, msg):
                prog.progress(int(pct), text=msg)

            pdf_bytes = generate_ptw_pdf_with_attachments(
                form_data=form_data,
                work_order_id=work_order_id,
                progress_callback=progress_callback,
            )
            st.session_state[cache_key] = pdf_bytes
            # Preserve the just_submitted state after PDF generation (redundant but safe)
            st.session_state[just_submitted_key] = True
            st.session_state["s2_active_ptw_id"] = work_order_id
            prog_slot.empty()
        except Exception as e:
            prog_slot.empty()
            st.error(f"Failed to generate PDF: {e}")
            # Even on error, preserve success view state
            st.session_state[just_submitted_key] = True

    cached = st.session_state.get(cache_key)
    if isinstance(cached, (bytes, bytearray)) and len(cached) > 0:
        # Callback to preserve success view state during download
        def _on_download() -> None:
            st.session_state[just_submitted_key] = True
            st.session_state["s2_active_ptw_id"] = work_order_id
        
        st.download_button(
            label=f"📥 Download {work_order_id}.pdf",
            data=cached,
            file_name=f"{work_order_id}.pdf",
            mime="application/pdf",
            key=f"{download_key}_btn",
            on_click=_on_download,
        )
    
    st.divider()
    
    # Option to proceed to another PTW
    def _on_review_another() -> None:
        # Clear the just_submitted flag so this PTW shows as editable again (if user re-opens it)
        st.session_state[just_submitted_key] = False
        # Reset dropdown to "(select PTW)"
        st.session_state["s2_selected_ptw_label"] = "(select PTW)"
        # Clear active PTW
        st.session_state["s2_active_ptw_id"] = None
        # Clear cached PDF bytes for this PTW to free memory
        st.session_state.pop(cache_key, None)
    
    st.button("📋 Review Another PTW", key=f"s2_another_{work_order_id}", on_click=_on_review_another)


def _render_ptw_detail(
    *,
    ptw_id: str,
    work_order_id: str,
    site_name: str,
    work_location: str,
    form_data: dict,
    is_forwarded: bool,
    status: str,
) -> None:
    """Render detailed PTW view with editing capabilities."""
    
    # NOTE: Removed nested fragment to avoid state management issues with success view.
    # The outer fragment (wrapping the entire View Submitted PTW tab) is sufficient.
    _render_ptw_detail_body(
        ptw_id=ptw_id,
        work_order_id=work_order_id,
        site_name=site_name,
        work_location=work_location,
        form_data=form_data,
        is_forwarded=is_forwarded,
        status=status,
    )


def _render_ptw_detail_body(
    *,
    ptw_id: str,
    work_order_id: str,
    site_name: str,
    work_location: str,
    form_data: dict,
    is_forwarded: bool,
    status: str,
) -> None:
    """Body for PTW details (split to allow fragment isolation)."""
    # Normalize IDs to string so session_state keys don't break on dtype changes
    ptw_id = str(ptw_id)
    work_order_id = str(work_order_id)
    key_prefix = f"s2_ptw_{ptw_id}_"

    def _set_active() -> None:
        # Track active PTW by work_order_id (stable across reruns)
        st.session_state["s2_active_ptw_id"] = work_order_id

    # PRE-RUN ACTION HANDLING (progress must appear BEFORE the form is rebuilt)
    # If an action was requested by an on_change or button click, handle it first,
    # show progress immediately, and stop rendering the rest of the expander body.
    prev_req = f"{key_prefix}preview_requested"
    prev_cache = f"{key_prefix}preview_bytes"
    if prev_cache not in st.session_state:
        st.session_state[prev_cache] = None
    submit_req = f"{key_prefix}submit_requested"
    if submit_req not in st.session_state:
        st.session_state[submit_req] = False

    # Handle preview request FIRST (avoid rebuilding the whole form before progress appears)
    if st.session_state.get(prev_req):
        st.session_state[prev_req] = False
        prog_slot = st.empty()
        prog = prog_slot.progress(0, text="Generating preview...")
        try:
            _smooth_progress(prog, 0, 20, text="Preparing data...")
            updated_form = dict(form_data or {})
            updated_form["holder_name"] = st.session_state.get(f"{key_prefix}holder_name", updated_form.get("holder_name", ""))
            updated_form["isolation_required"] = st.session_state.get(
                f"{key_prefix}isolation_required", updated_form.get("isolation_required", "")
            )
            # Toolbox is evidence-driven (no autosave checkbox); treat as conducted if files exist
            try:
                existing_toolbox = _list_evidence_files(work_order_id, "toolbox")
            except Exception:
                existing_toolbox = []
            uploaded_toolbox = st.session_state.get(f"{key_prefix}toolbox_files", []) or []
            updated_form["toolbox_conducted"] = bool(existing_toolbox or uploaded_toolbox)
            updated_form["s2_remarks"] = st.session_state.get(f"{key_prefix}s2_remarks", updated_form.get("s2_remarks", ""))

            def progress_callback(pct, msg):
                prog.progress(int(pct), text=msg)

            pdf_bytes = generate_ptw_pdf_with_attachments(
                form_data=updated_form,
                work_order_id=work_order_id,
                progress_callback=progress_callback,
            )
            st.session_state[prev_cache] = pdf_bytes
        except Exception as e:
            st.error(f"Failed to generate preview: {e}")
        finally:
            prog_slot.empty()
        # Continue rendering so the expander stays populated; download button is shown below.

    # Handle submit request FIRST (avoid rebuilding before progress appears)
    if st.session_state.get(submit_req):
        st.session_state[submit_req] = False
        # Reuse current widget values (already in session_state)
        holder_name = st.session_state.get(f"{key_prefix}holder_name", (form_data or {}).get("holder_name", ""))
        isolation_required = st.session_state.get(
            f"{key_prefix}isolation_required", (form_data or {}).get("isolation_required", "")
        )
        try:
            existing_toolbox = _list_evidence_files(work_order_id, "toolbox")
        except Exception:
            existing_toolbox = []
        toolbox_files_uploaded = st.session_state.get(f"{key_prefix}toolbox_files", []) or []
        toolbox_conducted = bool(existing_toolbox or toolbox_files_uploaded)
        s2_remarks = st.session_state.get(f"{key_prefix}s2_remarks", (form_data or {}).get("s2_remarks", ""))

        isolation_files_uploaded = st.session_state.get(f"{key_prefix}isolation_files", [])

        updated_form = _handle_s2_submit(
            ptw_id=ptw_id,
            work_order_id=work_order_id,
            form_data=form_data,
            holder_name=holder_name,
            isolation_required=isolation_required,
            toolbox_conducted=toolbox_conducted,
            s2_remarks=s2_remarks,
            isolation_files=isolation_files_uploaded,
            toolbox_files=toolbox_files_uploaded,
            key_prefix=key_prefix,
        )
        # Show PTW summary immediately after progress completes (no tab switch required)
        if isinstance(updated_form, dict):
            _render_post_submit_view(
                ptw_id=ptw_id,
                work_order_id=work_order_id,
                site_name=site_name,
                form_data=updated_form,
                just_submitted_key=f"s2_wo_{work_order_id}_just_submitted",
            )
            return
    
    if is_forwarded or status in ("CLOSED", "APPROVED", "APPROVED_BY_S3", "REJECTED"):
        st.info(
            f"This PTW has been {'forwarded to S3' if is_forwarded else 'finalized'}. "
            "Editing is disabled."
        )
        
        # Show current form data (read-only)
        with st.container():
            st.markdown("### PTW Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Work Order ID:** {work_order_id}")
                st.write(f"**Site Name:** {site_name}")
            with col2:
                st.write(f"**Work Location:** {work_location}")
                st.write(f"**Permit Holder:** {form_data.get('holder_name', 'N/A')}")
        
        # Action buttons
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            # Revoke button (only if forwarded but not yet approved/rejected)
            if is_forwarded and status == "PENDING_AT_S3":
                req_key = f"{key_prefix}revoke_requested"

                def _on_revoke() -> None:
                    st.session_state["s2_active_ptw_id"] = work_order_id
                    st.session_state[req_key] = True

                st.button("🔄 Revoke Submission", key=f"{key_prefix}revoke", type="secondary", on_click=_on_revoke)

                if st.session_state.get(req_key):
                    st.session_state[req_key] = False
                    prog_slot = st.empty()
                    prog = prog_slot.progress(0, text="Revoking...")
                    try:
                        _smooth_progress(prog, 0, 60, text="Updating database...")
                        from ptw_lifecycle_utils import _get_all_work_order_ids_for_ptw
                        wo_ids = _get_all_work_order_ids_for_ptw(
                            ptw_id=str(ptw_id),
                            permit_no=str(work_order_id),
                            form_data=form_data,
                        )
                        if len(wo_ids) > 1:
                            st.info(f"Atomic lifecycle update: {len(wo_ids)} linked work orders updated together.")
                        _revoke_s2_submissions(work_order_ids=wo_ids)
                        _smooth_progress(prog, 60, 100, text="Done")
                        prog_slot.empty()
                        st.success("Submission revoked. You can now edit the PTW.")
                        # Clear the just_submitted flag so the editable form appears immediately
                        st.session_state[f"s2_wo_{work_order_id}_just_submitted"] = False
                        # Trigger a refresh (refetch) without forcing a hard rerun
                        st.session_state["refresh_s2"] = True
                        # Ensure this PTW stays expanded after refresh
                        st.session_state["s2_active_ptw_id"] = work_order_id
                        return
                    except Exception as e:
                        prog_slot.empty()
                        st.error(f"Failed to revoke: {e}")
        
        with btn_col2:
            # Download PDF button
            dl_req = f"{key_prefix}dl_requested"
            dl_cache = f"{key_prefix}dl_bytes"
            if dl_cache not in st.session_state:
                st.session_state[dl_cache] = None

            def _on_dl() -> None:
                st.session_state["s2_active_ptw_id"] = work_order_id
                st.session_state[dl_req] = True

            st.button("📥 Download PDF", key=f"{key_prefix}download_pdf", on_click=_on_dl)

            if st.session_state.get(dl_req):
                st.session_state[dl_req] = False
                prog_slot = st.empty()
                prog = prog_slot.progress(0, text="Generating PDF...")
                try:
                    def progress_callback(pct, msg):
                        prog.progress(int(pct), text=msg)

                    pdf_bytes = generate_ptw_pdf_with_attachments(
                        form_data=form_data,
                        work_order_id=work_order_id,
                        progress_callback=progress_callback,
                    )
                    # If already S3-approved, apply stamp AFTER attachments merge.
                    try:
                        approval_times = get_ptw_approval_times(work_order_id)
                        if approval_times.get("date_s3_approved_raw"):
                            pdf_bytes = add_floating_approval_stamp(
                                pdf_bytes, approved_on=approval_times.get("issuer_datetime", "")
                            )
                    except Exception:
                        pass
                    st.session_state[dl_cache] = pdf_bytes
                except Exception as e:
                    st.error(f"Failed to generate PDF: {e}")
                finally:
                    prog_slot.empty()

            cached = st.session_state.get(dl_cache)
            if isinstance(cached, (bytes, bytearray)) and len(cached) > 0:
                st.download_button(
                    label=f"Download {work_order_id}.pdf",
                    data=cached,
                    file_name=f"{work_order_id}.pdf",
                    mime="application/pdf",
                    key=f"{key_prefix}pdf_btn",
                )
        
        return
    
    # Editable form
    st.markdown("### A. Permit Details")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Work Order ID", value=work_order_id, disabled=True, key=f"{key_prefix}wo_id")
        st.text_input("Site Name", value=site_name, disabled=True, key=f"{key_prefix}site")
    with col2:
        st.text_input("Work Location", value=work_location, disabled=True, key=f"{key_prefix}location")
        st.text_input("Created By", value=form_data.get("receiver_name", ""), disabled=True, key=f"{key_prefix}created_by")
    
    st.divider()
    
    # Section B: Permit Holder (MANDATORY for S2)
    st.markdown("### B. Permit Holder (Required)")
    
    holder_name = st.text_area(
        "Permit Holder Name *",
        value=form_data.get("holder_name", ""),
        key=f"{key_prefix}holder_name",
        placeholder="Enter the name of the permit holder",
        height=38,
    )
    
    st.divider()
    
    # Section C: Isolation Requirement (MANDATORY)
    st.markdown("### C. Isolation Requirement (Required)")

    # Enforce work_order-level isolation: if ANY linked work_order has
    # isolation_requirement == "YES", lock the reviewer's choice to YES.
    _wo_isolation_required = "NO"
    try:
        from ptw_lifecycle_utils import _get_all_work_order_ids_for_ptw
        _linked_ids = _get_all_work_order_ids_for_ptw(
            ptw_id=ptw_id,
            permit_no=work_order_id,
            form_data=form_data,
        )
        _sb = get_supabase_client(prefer_service_role=True)
        _iso_resp = (
            _sb.table(TABLE_WORK_ORDERS)
            .select("isolation_requirement")
            .in_("work_order_id", _linked_ids or [work_order_id])
            .execute()
        )
        _iso_rows = getattr(_iso_resp, "data", None) or []
        # If ANY linked WO requires isolation, mandate it for the whole PTW
        if any(str(r.get("isolation_requirement") or "").upper().strip() == "YES" for r in _iso_rows):
            _wo_isolation_required = "YES"
    except Exception:
        pass

    wo_mandates_isolation = _wo_isolation_required == "YES"

    if wo_mandates_isolation:
        st.info("ℹ️ Isolation required as per Work Order. This cannot be changed.")
        isolation_required = "YES"
        # Keep session state in sync so downstream logic reads the correct value
        st.session_state[f"{key_prefix}isolation_required"] = "YES"
    else:
        current_isolation = form_data.get("isolation_required", "")
        isolation_idx = 0 if current_isolation.upper() != "NO" else 1
        isolation_required = st.radio(
            "Is Isolation Required? *",
            options=["YES", "NO"],
            index=isolation_idx,
            key=f"{key_prefix}isolation_required",
            horizontal=True,
        )
    
    # File upload for isolation evidence (mandatory if YES)
    existing_isolation_files = _list_evidence_files(work_order_id, "isolation")
    isolation_files_key = f"{key_prefix}isolation_files"
    
    if isolation_required == "YES":
        st.caption("Upload isolation evidence (images or PDFs) - Required")
        
        # Show existing files
        if existing_isolation_files:
            st.success(f"✓ {len(existing_isolation_files)} isolation file(s) already uploaded")
            with st.expander("View uploaded files"):
                for f in existing_isolation_files:
                    st.write(f"  📎 {f['name']}")
        
        # New file upload
        isolation_upload = st.file_uploader(
            "Upload Isolation Evidence",
            type=["pdf", "jpg", "jpeg", "png", "gif", "webp"],
            accept_multiple_files=True,
            key=f"{key_prefix}isolation_upload",
        )
        
        # Store in session state
        st.session_state[isolation_files_key] = isolation_upload if isolation_upload else []
    else:
        st.session_state[isolation_files_key] = []
    
    st.divider()
    
    # Section D: Tool Box Talk (MANDATORY)
    st.markdown("### D. Tool Box Talk Confirmation (Required)")

    # Toolbox evidence (always visible; conducted is evidence-driven)
    existing_toolbox_files = _list_evidence_files(work_order_id, "toolbox")
    toolbox_files_key = f"{key_prefix}toolbox_files"

    st.caption("Upload toolbox talk evidence (images or PDFs) - Required")

    # Show existing files
    if existing_toolbox_files:
        st.success(f"✓ {len(existing_toolbox_files)} toolbox file(s) already uploaded")
        with st.expander("View uploaded files"):
            for f in existing_toolbox_files:
                st.write(f"  📎 {f['name']}")

    # New file upload (always visible)
    toolbox_upload = st.file_uploader(
        "Upload Toolbox Talk Evidence",
        type=["pdf", "jpg", "jpeg", "png", "gif", "webp"],
        accept_multiple_files=True,
        key=f"{key_prefix}toolbox_upload",
    )
    st.session_state[toolbox_files_key] = toolbox_upload if toolbox_upload else []

    # Modern upload UX (only when user selects new files)
    ack_key = f"{key_prefix}toolbox_upload_ack"
    try:
        curr_names = [getattr(f, "name", "") for f in (toolbox_upload or [])]
        curr_names = [n for n in curr_names if str(n).strip()]
    except Exception:
        curr_names = []
    if curr_names and st.session_state.get(ack_key) != curr_names:
        prog = st.progress(0, text="Uploading toolbox evidence…")
        for i in range(0, 101, 20):
            prog.progress(i, text="Uploading toolbox evidence…")
            _time.sleep(0.02)
        prog.empty()
        st.success("Files selected. They’ll upload on Submit.")
        st.session_state[ack_key] = curr_names
    
    st.divider()
    
    # Section E: Additional Remarks (Optional)
    st.markdown("### E. Supervisor Remarks (Optional)")
    
    s2_remarks = st.text_area(
        "Remarks / Notes",
        value=form_data.get("s2_remarks", ""),
        key=f"{key_prefix}s2_remarks",
        placeholder="Add any additional remarks or notes...",
    )
    
    st.divider()
    
    # Validation
    validation_errors = []
    
    if not holder_name or not holder_name.strip():
        validation_errors.append("Permit Holder Name is required")
    
    isolation_files_uploaded = st.session_state.get(isolation_files_key, [])
    toolbox_files_uploaded = st.session_state.get(toolbox_files_key, [])
    
    if isolation_required == "YES":
        if not existing_isolation_files and not isolation_files_uploaded:
            validation_errors.append("Isolation evidence file is required when Isolation = YES")

    # Toolbox evidence is mandatory in S2
    if not existing_toolbox_files and not toolbox_files_uploaded:
        validation_errors.append("Toolbox Talk evidence file is required")
    
    # Show validation status
    if validation_errors:
        st.warning("Please complete the following before submitting:")
        for err in validation_errors:
            st.write(f"  ⚠️ {err}")
    
    # Action buttons
    btn_col1, btn_col2 = st.columns(2)
    
    with btn_col1:
        submit_disabled = len(validation_errors) > 0
        submit_req = f"{key_prefix}submit_requested"
        if submit_req not in st.session_state:
            st.session_state[submit_req] = False

        def _on_submit() -> None:
            _set_active()
            st.session_state[submit_req] = True

        st.button(
            "✅ Submit for Approval",
            type="primary",
            disabled=submit_disabled,
            key=f"{key_prefix}submit",
            on_click=_on_submit,
        )
    
    with btn_col2:
        # Preview PDF (queued; generation handled at the top of this function for progress-first UX)
        def _on_preview() -> None:
            _set_active()
            st.session_state[f"{key_prefix}preview_requested"] = True

        st.button("👁️ Preview PDF", key=f"{key_prefix}preview", on_click=_on_preview)

    cached_prev = st.session_state.get(f"{key_prefix}preview_bytes")
    if isinstance(cached_prev, (bytes, bytearray)) and len(cached_prev) > 0:
        st.download_button(
            label="📥 Download Preview PDF",
            data=cached_prev,
            file_name=f"{work_order_id}_preview.pdf",
            mime="application/pdf",
            key=f"{key_prefix}preview_pdf_btn",
        )
    
    # Submit is handled via queued `submit_req` at the top of this function (progress-first UX).


def _handle_s2_submit(
    *,
    ptw_id: str,
    work_order_id: str,
    form_data: dict,
    holder_name: str,
    isolation_required: str,
    toolbox_conducted: bool,
    s2_remarks: str,
    isolation_files: list,
    toolbox_files: list,
    key_prefix: str,
) -> dict | None:
    """Handle the S2 submit action."""
    
    progress = st.progress(0, text="Validating...")
    status_msg = st.empty()
    
    try:
        # Step 1: Upload evidence files
        if isolation_files or toolbox_files:
            progress.progress(10, text="Uploading evidence files...")
            status_msg.info("Uploading evidence files to secure storage...")
        
        upload_errors = []
        
        if isolation_files:
            for idx, f in enumerate(isolation_files):
                progress.progress(10 + (idx * 10 // max(len(isolation_files), 1)), 
                                text=f"Uploading isolation file {idx + 1}/{len(isolation_files)}...")
                try:
                    file_bytes = f.read()
                    f.seek(0)  # Reset for potential re-read
                    _upload_evidence_file(work_order_id, "isolation", file_bytes, f.name)
                except Exception as e:
                    upload_errors.append(f"Isolation file '{f.name}': {e}")
        
        if toolbox_files:
            for idx, f in enumerate(toolbox_files):
                progress.progress(30 + (idx * 10 // max(len(toolbox_files), 1)), 
                                text=f"Uploading toolbox file {idx + 1}/{len(toolbox_files)}...")
                try:
                    file_bytes = f.read()
                    f.seek(0)  # Reset for potential re-read
                    _upload_evidence_file(work_order_id, "toolbox", file_bytes, f.name)
                except Exception as e:
                    upload_errors.append(f"Toolbox file '{f.name}': {e}")
        
        if upload_errors:
            progress.empty()
            status_msg.empty()
            st.error("Some files failed to upload:")
            for err in upload_errors:
                st.write(f"  ❌ {err}")
            return
        
        # --- SUCCESS FEEDBACK FOR EVIDENCE UPLOADS ---
        success_msgs = []
        
        if isolation_files:
            success_msgs.append(f"✅ Isolation evidence uploaded successfully ({len(isolation_files)} file(s))")
        
        if toolbox_files:
            success_msgs.append(f"✅ Toolbox Talk evidence uploaded successfully ({len(toolbox_files)} file(s))")
        
        for msg in success_msgs:
            st.success(msg)
        
        # Step 2: Update form_data in ptw_requests
        progress.progress(50, text="Updating PTW record...")
        status_msg.info("Saving changes to database...")
        
        updated_form = form_data.copy()
        updated_form["holder_name"] = holder_name
        updated_form["isolation_required"] = isolation_required
        updated_form["toolbox_conducted"] = toolbox_conducted
        updated_form["s2_remarks"] = s2_remarks
        updated_form["s2_submitted_at"] = datetime.now().isoformat()
        
        _update_ptw_form_data(ptw_id, updated_form)
        
        # Step 3: Update work_orders (date_s2_forwarded, isolation_requirement)
        progress.progress(75, text="Forwarding to S3...")
        status_msg.info("Forwarding PTW for final approval...")
        
        from ptw_lifecycle_utils import (
            _get_all_work_order_ids_for_ptw,
            _update_all_work_orders_lifecycle,
        )

        wo_ids = _get_all_work_order_ids_for_ptw(
            ptw_id=str(ptw_id),
            permit_no=str(work_order_id),
            form_data=updated_form,
        )
        if len(wo_ids) > 1:
            st.info(f"Atomic lifecycle update: {len(wo_ids)} linked work orders updated together.")
        _update_all_work_orders_lifecycle(
            work_order_ids=wo_ids,
            update_fields={
                "date_s2_forwarded": datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(sep=" ", timespec="seconds"),
                "isolation_requirement": isolation_required,
            },
        )
        
        # Step 4: Done
        progress.progress(100, text="Complete!")
        progress.empty()
        status_msg.empty()
        
        # Mark as just submitted - success view will render on the normal button-click rerun
        # Use work_order_id-based key (stable) so success view doesn't disappear on reruns.
        st.session_state[f"s2_wo_{work_order_id}_just_submitted"] = True
        st.session_state["s2_active_ptw_id"] = str(work_order_id)
        # Refresh list without "vanishing" items (keep same date filters)
        st.session_state["refresh_s2"] = True
        
        return updated_form
        
    except Exception as e:
        progress.empty()
        status_msg.empty()
        st.error(f"Failed to submit: {e}")
        # Avoid showing stack traces to end users in production UI
        return None


# =============================================================================
# CREATE WORK ORDER — UI
# =============================================================================

_CWO_PREFIX = "s2_cwo_"

# Session state key for the last successfully created/updated WO id
_CWO_LAST_CREATED_KEY = f"{_CWO_PREFIX}last_created"
_CWO_LAST_UPDATED_KEY = f"{_CWO_PREFIX}last_updated"


def _cwo_reset_form_state(*, keep_success: bool = False) -> None:
    """
    Clear all Create Work Order form keys so the form renders blank.

    Bumps the form version counter so Streamlit constructs a truly fresh
    st.form instance (just deleting widget keys is not enough for st.form).

    keep_success=True preserves _CWO_LAST_CREATED_KEY / _CWO_LAST_UPDATED_KEY
    so the success card can remain visible while the form below it is blank
    (used after successful insert so the card shows but form is empty).
    """
    field_keys = [
        f"{_CWO_PREFIX}site",
        f"{_CWO_PREFIX}location",
        f"{_CWO_PREFIX}equipment",
        f"{_CWO_PREFIX}freq",
        f"{_CWO_PREFIX}isolation",
        f"{_CWO_PREFIX}date_planned",
        f"{_CWO_PREFIX}edit_id",
    ]
    filter_keys = [
        f"{_CWO_PREFIX}filter_site",
        f"{_CWO_PREFIX}filter_start",
        f"{_CWO_PREFIX}filter_end",
        f"{_CWO_PREFIX}preview_df",
    ]
    for k in field_keys + filter_keys:
        st.session_state.pop(k, None)
    if not keep_success:
        st.session_state.pop(_CWO_LAST_CREATED_KEY, None)
        st.session_state.pop(_CWO_LAST_UPDATED_KEY, None)
    # Bump form version → forces a fresh st.form with a new key on next render
    st.session_state[f"{_CWO_PREFIX}form_version"] = (
        st.session_state.get(f"{_CWO_PREFIX}form_version", 0) + 1
    )


def _cwo_check_edit_eligibility_live(work_order_id: str) -> bool:
    """
    Live Supabase read: returns True (edit allowed) when date_s1_created IS NULL.
    Does NOT rely on any cached dataframe or derived status.
    """
    try:
        sb = get_supabase_client(prefer_service_role=True)
        resp = (
            sb.table(TABLE_WORK_ORDERS)
            .select("date_s1_created")
            .eq("work_order_id", work_order_id)
            .execute()
        )
        rows = getattr(resp, "data", None) or []
        if not rows:
            return True  # Row not found — allow optimistically; DB update guard will protect
        d1c = rows[0].get("date_s1_created")
        return d1c is None or str(d1c).strip() in ("", "None", "NaT", "nan")
    except Exception:
        return True  # Fail-open; DB update's .is_("date_s1_created", "null") is the real guard


def _render_create_work_order() -> None:
    """
    S2 — Create Work Order sub-section.

    Design contract:
    - NO st.rerun() after a successful create/update → prevents flicker.
    - Success state is persisted in session_state so the card survives reruns.
    - Progress bar renders in the SAME run as the submit (form reruns on submit).
    - Preview fetch has its own progress animation.
    - Edit eligibility uses a live DB read, NOT cached df or derived status.
    """
    st.markdown("## Create Work Order")
    st.caption("Create and manage work orders. Once an S1 PTW is initiated, editing is locked.")

    # ── Load dropdown options (cached, progress shown only on cache-miss) ──────
    _opt_slot = st.empty()
    try:
        sites = _cwo_fetch_sites()
        all_locations = _cwo_fetch_locations()
        all_equipment = _cwo_fetch_equipment()
    except Exception as e:
        _opt_slot.empty()
        st.error(f"Failed to load options: {e}")
        return
    _opt_slot.empty()

    # ── Persistent success card (survives reruns until "Create Another") ───────
    last_created = st.session_state.get(_CWO_LAST_CREATED_KEY)
    last_updated = st.session_state.get(_CWO_LAST_UPDATED_KEY)

    if last_created:
        st.markdown(
            f"""
            <div style="
                padding:20px;border-radius:12px;
                background:linear-gradient(135deg,#e0f2fe,#f0fdf4);
                border:1px solid #10b981;
                box-shadow:0 6px 18px rgba(15,23,42,0.08);
                margin-bottom:12px;
            ">
                <h3 style="margin:0;color:#065f46;">✅ Work Order Created Successfully</h3>
                <p style="margin-top:8px;font-size:18px;font-weight:600;">
                    Work Order ID: <span style="color:#2563eb;">{last_created}</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("➕ Create Another Work Order", key=f"{_CWO_PREFIX}create_another", type="primary"):
            _cwo_reset_form_state()
            st.cache_data.clear()
            st.rerun()
        st.divider()

    elif last_updated:
        st.success(f"✅ Work Order **{last_updated}** updated successfully.")
        if st.button("✏️ Edit Another / Create New", key=f"{_CWO_PREFIX}after_update", type="secondary"):
            _cwo_reset_form_state()
            st.cache_data.clear()
            st.rerun()
        st.divider()

    # ── Edit mode banner ───────────────────────────────────────────────────────
    edit_key = f"{_CWO_PREFIX}edit_id"
    is_edit = bool(st.session_state.get(edit_key))

    if is_edit:
        st.info(f"✏️ Editing Work Order: **{st.session_state[edit_key]}**")
        if st.button("✖ Cancel Edit", key=f"{_CWO_PREFIX}cancel_edit"):
            _cwo_reset_form_state()
            st.rerun()
        st.divider()

    st.markdown("### Work Order Details")

    # ── Form ──────────────────────────────────────────────────────────────────
    # Versioned form key: each time _cwo_reset_form_state() is called it bumps
    # the version counter, which forces Streamlit to construct a brand-new
    # st.form instance with all default values — regardless of prior widget state.
    _form_version = st.session_state.get(f"{_CWO_PREFIX}form_version", 0)
    form_key = f"{_CWO_PREFIX}form_v{_form_version}"
    with st.form(form_key, clear_on_submit=False):
        col_a, col_b = st.columns(2)
        with col_a:
            site_default = st.session_state.get(f"{_CWO_PREFIX}site", "(select)")
            site_options = ["(select)"] + sites
            site_idx = site_options.index(site_default) if site_default in site_options else 0
            site_name = st.selectbox(
                "Site Name *",
                options=site_options,
                index=site_idx,
                key=f"{_CWO_PREFIX}site",
            )

            frequency_default = st.session_state.get(f"{_CWO_PREFIX}freq", _CWO_FREQUENCIES[0])
            freq_idx = _CWO_FREQUENCIES.index(frequency_default) if frequency_default in _CWO_FREQUENCIES else 0
            frequency = st.selectbox(
                "Frequency *",
                options=_CWO_FREQUENCIES,
                index=freq_idx,
                key=f"{_CWO_PREFIX}freq",
                help="D=Daily, W=Weekly, Q=Quarterly, HY=Half-Yearly, Y=Yearly, UP=Unplanned",
            )

        with col_b:
            date_planned = st.date_input(
                "Planned Date *",
                value=st.session_state.get(f"{_CWO_PREFIX}date_planned", date.today()),
                key=f"{_CWO_PREFIX}date_planned",
            )
            isolation_default = st.session_state.get(f"{_CWO_PREFIX}isolation", "NO")
            isolation_idx = 0 if isolation_default == "YES" else 1
            isolation_requirement = st.radio(
                "Isolation Required? *",
                options=["YES", "NO"],
                index=isolation_idx,
                horizontal=True,
                key=f"{_CWO_PREFIX}isolation",
            )

        location_default = st.session_state.get(f"{_CWO_PREFIX}location", [])
        location_sel = st.multiselect(
            "Location(s) *",
            options=all_locations,
            default=[v for v in location_default if v in all_locations],
            key=f"{_CWO_PREFIX}location",
        )

        equipment_default = st.session_state.get(f"{_CWO_PREFIX}equipment", [])
        equipment_sel = st.multiselect(
            "Equipment *",
            options=all_equipment,
            default=[v for v in equipment_default if v in all_equipment],
            key=f"{_CWO_PREFIX}equipment",
        )

        submit_label = "💾 Update Work Order" if is_edit else "✅ Create Work Order"
        submitted = st.form_submit_button(submit_label, type="primary", use_container_width=True)

    # ── Validation + DB write (runs in same rerun as form submit) ─────────────
    # Progress container is placed directly below the form; it renders
    # immediately because we are already inside the post-submit rerun.
    prog_container = st.empty()
    msg_container = st.empty()

    if submitted:
        errors: list[str] = []
        if not site_name or site_name == "(select)":
            errors.append("Site Name is required.")
        if not location_sel:
            errors.append("At least one Location is required.")
        if not equipment_sel:
            errors.append("At least one Equipment is required.")

        if errors:
            for err_msg in errors:
                msg_container.error(err_msg)
        else:
            prog = prog_container.progress(0, text="Validating inputs...")
            try:
                _smooth_progress(prog, 0, 25, text="Validating inputs...")

                if is_edit:
                    wo_being_edited = str(st.session_state.get(edit_key, ""))
                    # Live eligibility check — no cached df, no derived status
                    _smooth_progress(prog, 25, 40, text="Checking eligibility...")
                    if not _cwo_check_edit_eligibility_live(wo_being_edited):
                        prog_container.empty()
                        msg_container.warning(
                            f"⚠️ Work Order **{wo_being_edited}** already has an initiated PTW "
                            "and cannot be edited."
                        )
                    else:
                        _smooth_progress(prog, 40, 70, text="Updating Work Order...")
                        _cwo_update_work_order(
                            work_order_id=wo_being_edited,
                            site_name=site_name,
                            location=location_sel,
                            equipment=equipment_sel,
                            frequency=frequency,
                            isolation_requirement=isolation_requirement,
                            date_planned=date_planned,
                        )
                        _smooth_progress(prog, 70, 100, text="Done")
                        prog_container.empty()
                        # Reset form fields (blank) and persist update success card.
                        _saved_updated = wo_being_edited
                        _cwo_reset_form_state(keep_success=True)
                        st.session_state[_CWO_LAST_UPDATED_KEY] = _saved_updated
                        st.cache_data.clear()
                        st.rerun()
                else:
                    _smooth_progress(prog, 25, 65, text="Creating Work Order...")
                    new_id = _cwo_insert_work_order(
                        site_name=site_name,
                        location=location_sel,
                        equipment=equipment_sel,
                        frequency=frequency,
                        isolation_requirement=isolation_requirement,
                        date_planned=date_planned,
                    )
                    _smooth_progress(prog, 65, 90, text="Syncing Database...")
                    _smooth_progress(prog, 90, 100, text="Done")
                    prog_container.empty()
                    # Persist the new WO id for the success card, then clear
                    # all form fields + filters so the form re-renders blank.
                    # keep_success=True ensures the card survives the rerun.
                    _saved_id = new_id or "(auto-generated)"
                    _cwo_reset_form_state(keep_success=True)
                    st.session_state[_CWO_LAST_CREATED_KEY] = _saved_id
                    st.cache_data.clear()
                    st.rerun()

            except RuntimeError as exc:
                prog_container.empty()
                msg_container.error(str(exc))
            except Exception as exc:
                prog_container.empty()
                msg_container.error(f"Unexpected error: {exc}")

    # ── Preview / filter section ───────────────────────────────────────────────
    st.divider()
    st.markdown("### Work Orders Preview")
    st.caption("Filter and view existing work orders. Click Edit to modify (only available before S1 initiates PTW).")

    with st.form(f"{_CWO_PREFIX}filter_form", clear_on_submit=False):
        fc1, fc2, fc3, fc4 = st.columns([2, 1.5, 1.5, 1])
        with fc1:
            filter_sites = ["(all)"] + sites
            _fsite_default = st.session_state.get(f"{_CWO_PREFIX}filter_site", "(all)")
            _fsite_idx = filter_sites.index(_fsite_default) if _fsite_default in filter_sites else 0
            filter_site = st.selectbox("Site", options=filter_sites, index=_fsite_idx, key=f"{_CWO_PREFIX}filter_site")
        with fc2:
            filter_start = st.date_input(
                "Start Date",
                value=st.session_state.get(f"{_CWO_PREFIX}filter_start", date.today() - timedelta(days=30)),
                key=f"{_CWO_PREFIX}filter_start",
            )
        with fc3:
            filter_end = st.date_input(
                "End Date",
                value=st.session_state.get(f"{_CWO_PREFIX}filter_end", date.today()),
                key=f"{_CWO_PREFIX}filter_end",
            )
        with fc4:
            st.write("")
            st.write("")
            fetch_submitted = st.form_submit_button("Fetch", use_container_width=True)

    # Progress for preview fetch — renders in same rerun as the Fetch click
    fetch_prog_slot = st.empty()
    fetch_flag = f"{_CWO_PREFIX}fetch_requested"

    if fetch_submitted:
        st.session_state[fetch_flag] = True

    if st.session_state.get(fetch_flag):
        st.session_state[fetch_flag] = False
        fetch_prog = fetch_prog_slot.progress(0, text="Validating filters...")
        try:
            _smooth_progress(fetch_prog, 0, 20, text="Validating filters...")
            _smooth_progress(fetch_prog, 20, 60, text="Fetching Work Orders...")
            site_arg = "" if filter_site == "(all)" else filter_site
            df_prev = _fetch_work_orders(
                site_name=site_arg,
                start_date=filter_start,
                end_date=filter_end,
                status_ui=None,
                location=None,
            )
            _smooth_progress(fetch_prog, 60, 90, text="Preparing table...")
            st.session_state[f"{_CWO_PREFIX}preview_df"] = df_prev
            _smooth_progress(fetch_prog, 90, 100, text="Done")
        except Exception as e:
            st.session_state[f"{_CWO_PREFIX}preview_df"] = pd.DataFrame()
            fetch_prog_slot.empty()
            st.error(f"Failed to fetch work orders: {e}")
        finally:
            fetch_prog_slot.empty()

    # Only render the table once the user has explicitly clicked Fetch
    if f"{_CWO_PREFIX}preview_df" not in st.session_state:
        st.info("Select filters above and click **Fetch** to load work orders.")
        return

    df_view: pd.DataFrame = st.session_state.get(f"{_CWO_PREFIX}preview_df", pd.DataFrame())

    if df_view is None or (hasattr(df_view, "empty") and df_view.empty):
        st.info("No work orders found for the selected filters.")
        return

    # Display columns
    show_cols = [
        "work_order_id", "site_name", "location", "equipment", "frequency",
        "isolation_requirement", "date_planned", "status", "date_s1_created",
    ]
    for c in show_cols:
        if c not in df_view.columns:
            df_view[c] = pd.NA

    df_display = df_view[show_cols].copy()
    if "status" in df_display.columns:
        df_display["status"] = df_display["status"].astype("string").fillna("").map(
            lambda s: DB_STATUS_TO_UI.get(str(s).strip().upper(), str(s))
        )

    styled = (
        df_display.style
        .map(_highlight_status, subset=["status"])
        .set_table_styles([{"selector": "th", "props": [("font-weight", "800"), ("color", "#0f172a")]}])
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── Edit section ───────────────────────────────────────────────────────────
    st.markdown("#### Edit a Work Order")
    st.caption(
        "Select a Work Order to edit. Only rows where no PTW has been initiated "
        "(date_s1_created is empty) can be modified."
    )

    wo_options = df_view["work_order_id"].astype(str).tolist()
    if not wo_options:
        return

    selected_wo = st.selectbox(
        "Select Work Order to Edit",
        options=["(select)"] + wo_options,
        index=0,
        key=f"{_CWO_PREFIX}edit_select",
    )

    if selected_wo and selected_wo != "(select)":
        # Live DB eligibility check — authoritative, not cached-df-based
        can_edit = _cwo_check_edit_eligibility_live(selected_wo)

        if not can_edit:
            st.warning(
                f"⚠️ Work Order **{selected_wo}** already has an initiated PTW and cannot be edited."
            )
        else:
            # Try to pre-fill from the cached df for convenience
            row_matches = df_view[df_view["work_order_id"].astype(str) == selected_wo]
            row = row_matches.iloc[0] if not row_matches.empty else None

            def _safe(v, fallback: str = "") -> str:
                """Convert a pandas-row value to str, handling pd.NA / NaT safely."""
                try:
                    if pd.isna(v):
                        return fallback
                except (TypeError, ValueError):
                    pass
                return str(v) if v is not None else fallback

            def _on_edit() -> None:
                # Clear any previous success card before entering edit mode
                st.session_state.pop(_CWO_LAST_CREATED_KEY, None)
                st.session_state.pop(_CWO_LAST_UPDATED_KEY, None)
                st.session_state[edit_key] = selected_wo
                if row is not None:
                    st.session_state[f"{_CWO_PREFIX}site"] = _safe(row.get("site_name"))
                    st.session_state[f"{_CWO_PREFIX}location"] = [
                        p.strip() for p in _safe(row.get("location")).split(",") if p.strip()
                    ]
                    st.session_state[f"{_CWO_PREFIX}equipment"] = [
                        p.strip() for p in _safe(row.get("equipment")).split(",") if p.strip()
                    ]
                    freq_val = _safe(row.get("frequency"))
                    st.session_state[f"{_CWO_PREFIX}freq"] = (
                        freq_val if freq_val in _CWO_FREQUENCIES else _CWO_FREQUENCIES[0]
                    )
                    iso_val = _safe(row.get("isolation_requirement"), "NO").upper()
                    st.session_state[f"{_CWO_PREFIX}isolation"] = iso_val if iso_val in ("YES", "NO") else "NO"
                    try:
                        st.session_state[f"{_CWO_PREFIX}date_planned"] = pd.to_datetime(
                            row.get("date_planned")
                        ).date()
                    except Exception:
                        st.session_state[f"{_CWO_PREFIX}date_planned"] = date.today()

            st.button(
                f"✏️ Edit Work Order {selected_wo}",
                key=f"{_CWO_PREFIX}edit_btn_{selected_wo}",
                type="secondary",
                on_click=_on_edit,
            )


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================


def render(db_path: str) -> None:
    """
    S2 Portal - Main render function.
    
    Tabs:
    - View Work Order: Read-only view of work orders
    - View Submitted PTW: Review and forward PTWs from S1
    """
    # Hard access guard (prevents manual bypass via session_state tampering)
    from access_control import user_allowed_pages

    allowed_pages = user_allowed_pages()
    if "s2" not in allowed_pages:
        st.error("Access Denied - S2 Only")
        st.stop()

    st.markdown("# S2 Portal")
    st.caption("PTW Review & Forwarding Stage")

    # Import modern UI styles
    try:
        from modern_ui_styles import MODERN_UI_CSS
        st.markdown(MODERN_UI_CSS, unsafe_allow_html=True)
    except Exception:
        pass  # Fallback if import fails

    _apply_modern_tabs_css()
    
    # Initialize session state
    if "s2_ptw_view_df" not in st.session_state:
        st.session_state["s2_ptw_view_df"] = None
    if "refresh_s2" not in st.session_state:
        st.session_state["refresh_s2"] = False
    
    SECTION_KEY = "s2_section"
    sections = ["View Work Order", "View Submitted PTW", "Create Work Order"]
    section = modern_section_selector(sections, key=SECTION_KEY)

    if section == "View Work Order":
        _render_view_work_order_s2()
    elif section == "View Submitted PTW":
        _render_view_submitted_ptw()
    elif section == "Create Work Order":
        _render_create_work_order()
