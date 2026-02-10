from __future__ import annotations

"""
S3 Portal - Final PTW Approval Stage

S3 responsibilities:
- View Work Order (read-only, same behavior as S1/S2; status derived from dates)
- View Approvals (final approval of PTWs that are S1-created + S2-forwarded + not yet S3-approved)

Rules:
- No PTW creation in S3
- No editing except providing Permit Issuer Name and approving
- Approval is date-driven: set work_orders.date_s3_approved only
- PDFs are always generated dynamically from latest ptw_requests.form_data + latest template
- Approved PTWs remain visible with post-approval card (can view/revoke)
- {{holder_datetime}} auto-filled from work_orders.date_s2_forwarded
- {{issuer_datetime}} auto-filled from work_orders.date_s3_approved
- Approved PDFs include floating APPROVED stamp overlay
"""

import time
from datetime import date, datetime, timedelta
from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st

from supabase_link import get_supabase_client

# PDF manipulation for approval stamp
try:
    from PyPDF2 import PdfReader, PdfWriter
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.colors import Color
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    _HAS_PDF_LIBS = True
except ImportError:
    _HAS_PDF_LIBS = False

# Reuse S1 helpers (logic stays single-source-of-truth)
from S1 import (
    TABLE_WORK_ORDERS,
    TABLE_PTW_REQUESTS,
    UI_STATUSES,
    DB_STATUS_TO_UI,
    derive_ptw_status,
    _list_sites_from_work_orders,
    _list_locations_from_work_orders,
    _list_statuses_from_work_orders,
    _fetch_work_orders,
    _highlight_status,
    build_doc_data,
    generate_ptw_pdf,
    _download_template_from_supabase,
)

# Prefer the same "PDF with evidence" behavior as S2 if available.
try:
    from S2 import generate_ptw_pdf_with_attachments  # type: ignore
except Exception:  # pragma: no cover
    generate_ptw_pdf_with_attachments = None  # type: ignore


# =============================================================================
# Helper Functions for Approval Timestamps & PDF Stamp
# =============================================================================


def get_ptw_approval_times(work_order_id: str) -> dict:
    """
    Fetch S2 and S3 approval timestamps from work_orders.
    
    Returns:
        {
            "holder_datetime": "DD-MM-YYYY HH:MM" (from date_s2_forwarded),
            "issuer_datetime": "DD-MM-YYYY HH:MM" (from date_s3_approved),
            "date_s2_forwarded_raw": original timestamp or None,
            "date_s3_approved_raw": original timestamp or None,
        }
    """
    sb = get_supabase_client(prefer_service_role=True)
    
    resp = (
        sb.table(TABLE_WORK_ORDERS)
        .select("date_s2_forwarded,date_s3_approved")
        .eq("work_order_id", work_order_id)
        .limit(1)
        .execute()
    )
    
    err = getattr(resp, "error", None)
    if err:
        raise RuntimeError(f"Failed to fetch approval times: {err}")
    
    rows = getattr(resp, "data", None) or []
    if not rows:
        return {
            "holder_datetime": "",
            "issuer_datetime": "",
            "date_s2_forwarded_raw": None,
            "date_s3_approved_raw": None,
        }
    
    row = rows[0]
    s2_raw = row.get("date_s2_forwarded")
    s3_raw = row.get("date_s3_approved")
    
    def format_datetime(val) -> str:
        """Convert to DD-MM-YYYY HH:MM format."""
        if val is None or str(val).strip() == "":
            return ""
        try:
            dt = pd.to_datetime(val)
            return dt.strftime("%d-%m-%Y %H:%M")
        except Exception:
            return str(val)
    
    return {
        "holder_datetime": format_datetime(s2_raw),
        "issuer_datetime": format_datetime(s3_raw),
        "date_s2_forwarded_raw": s2_raw,
        "date_s3_approved_raw": s3_raw,
    }


def add_floating_approval_stamp(pdf_bytes: bytes, *, approved_on: str) -> bytes:
    """
    Add a floating APPROVED stamp overlay to every page of a PDF.
    
    Args:
        pdf_bytes: Original PDF as bytes
        approved_on: Approval date string (DD-MM-YYYY HH:MM format)
    
    Returns:
        Modified PDF bytes with APPROVED stamp
    """
    if not _HAS_PDF_LIBS:
        # If PDF libraries not available, return original PDF
        return pdf_bytes
    
    try:
        # Create the stamp overlay on a canvas
        stamp_buffer = BytesIO()
        c = canvas.Canvas(stamp_buffer, pagesize=A4)
        
        # A4 dimensions: 595.28 x 841.89 points
        page_width, page_height = A4
        
        # Stamp positioning - bottom right area near Permit Issuer section
        # Adjust these coordinates as needed for exact placement
        stamp_x = page_width - 200  # ~395 points from left
        stamp_y = 120  # ~120 points from bottom
        
        # Stamp dimensions
        stamp_width = 160
        stamp_height = 60
        
        # Draw stamp border (red rectangle with rounded-ish appearance)
        stamp_color = Color(0.8, 0.1, 0.1, alpha=0.85)  # Dark red with slight transparency
        c.setStrokeColor(stamp_color)
        c.setLineWidth(3)
        c.rect(stamp_x, stamp_y, stamp_width, stamp_height, stroke=1, fill=0)
        
        # Draw inner border for visual effect
        c.setLineWidth(1.5)
        c.rect(stamp_x + 4, stamp_y + 4, stamp_width - 8, stamp_height - 8, stroke=1, fill=0)
        
        # Set text color
        c.setFillColor(stamp_color)
        
        # Draw "APPROVED" text (bold, large)
        c.setFont("Helvetica-Bold", 18)
        text_x = stamp_x + stamp_width / 2
        c.drawCentredString(text_x, stamp_y + 35, "APPROVED")
        
        # Draw date text (smaller)
        c.setFont("Helvetica", 9)
        if approved_on:
            c.drawCentredString(text_x, stamp_y + 18, f"ON: {approved_on}")
        
        c.save()
        stamp_buffer.seek(0)
        
        # Read original PDF
        original_pdf = PdfReader(BytesIO(pdf_bytes))
        stamp_pdf = PdfReader(stamp_buffer)
        
        # Create output PDF
        output = PdfWriter()
        
        # Merge stamp with EVERY page (requirement: post S3 approval stamp on each page)
        stamp_page = stamp_pdf.pages[0]
        for i in range(len(original_pdf.pages)):
            page = original_pdf.pages[i]
            page.merge_page(stamp_page)
            output.add_page(page)
        
        # Write to bytes
        output_buffer = BytesIO()
        output.write(output_buffer)
        output_buffer.seek(0)
        
        return output_buffer.read()
    
    except Exception as e:
        # If stamping fails, return original PDF rather than crashing
        import traceback
        traceback.print_exc()
        return pdf_bytes


def _inject_approval_times_into_form_data(form_data: dict, work_order_id: str, is_approved: bool = False) -> dict:
    """
    Inject holder_datetime and issuer_datetime into form_data from work_orders.
    
    Args:
        form_data: Existing form_data dict
        work_order_id: The work order ID
        is_approved: Whether this is for an approved PTW (determines if issuer_datetime should be set)
    
    Returns:
        Updated form_data dict with datetime fields
    """
    approval_times = get_ptw_approval_times(work_order_id)
    
    updated = dict(form_data) if form_data else {}
    
    # Always inject holder_datetime from S2 forwarding date
    if approval_times["holder_datetime"]:
        updated["holder_datetime"] = approval_times["holder_datetime"]
    
    # Inject issuer_datetime only if PTW is approved
    if is_approved and approval_times["issuer_datetime"]:
        updated["issuer_datetime"] = approval_times["issuer_datetime"]
    
    return updated


def _smooth_progress(progress_bar, start: int, end: int, text: str) -> None:
    """Animate progress bar smoothly from start to end."""
    steps = 8
    for i in range(steps + 1):
        val = start + (end - start) * i // steps
        progress_bar.progress(val, text=text)
        time.sleep(0.04)


# =============================================================================
# View Work Order (read-only) — reuse S1 helpers, avoid S1 widget keys
# =============================================================================


def _render_view_work_order_s3() -> None:
    st.markdown("## View Work Order")
    st.caption("Read-only view. Status is derived from PTW lifecycle dates (S1/S2/S3).")

    # Modern tabs styling (same as S1/S2) + Anti-Ghosting CSS
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
        
        /* Tab styling */
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
        
        /* Table hover */
        .stDataFrame tbody tr:hover {
          background-color: rgba(148, 163, 184, 0.18) !important;
        }
        
        /* KPI cards */
        .kpi-row {
            display: flex;
            gap: 1rem;
            margin: 1.5rem 0;
        }
        .kpi-card {
            flex: 1;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .kpi-title {
            font-size: 0.875rem;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    sites = _list_sites_from_work_orders()
    if not sites:
        st.warning("No Site Names found in `work_orders` (or permissions/RLS block SELECT).")
        return

    if "s3_wo_last_df" not in st.session_state:
        st.session_state["s3_wo_last_df"] = None
        st.session_state["s3_wo_last_kpis"] = None
        st.session_state["s3_wo_last_meta"] = None
    if "s3_wo_run_fetch" not in st.session_state:
        st.session_state["s3_wo_run_fetch"] = False

    site_options = ["(select)"] + sites

    ss_site = st.session_state.get("s3_wo_site")
    ss_start = st.session_state.get("s3_wo_start")
    ss_end = st.session_state.get("s3_wo_end")

    have_site = ss_site not in (None, "(select)", "")
    have_dates = isinstance(ss_start, date) and isinstance(ss_end, date)

    if have_site and have_dates:
        locs = _list_locations_from_work_orders(site_name=str(ss_site), start_date=ss_start, end_date=ss_end)
        statuses_ui = _list_statuses_from_work_orders(site_name=str(ss_site), start_date=ss_start, end_date=ss_end)
    else:
        locs, statuses_ui = [], []

    loc_options = ["(all)"] + locs
    status_options = ["(all)"] + (statuses_ui if statuses_ui else UI_STATUSES)

    # Filters section
    c1, c2, c3, c4, c5, c6 = st.columns([2.0, 1.3, 1.3, 1.4, 1.4, 1.0], vertical_alignment="bottom")
    with c1:
        site_name = st.selectbox("Site Name", options=site_options, index=0, key="s3_wo_site")
    with c2:
        start_date = st.date_input("Start Date", value=None, key="s3_wo_start")
    with c3:
        end_date = st.date_input("End Date", value=None, key="s3_wo_end")
    with c4:
        location = st.selectbox("Location", options=loc_options, index=0, key="s3_wo_location")
    with c5:
        status_ui = st.selectbox("Status", options=status_options, index=0, key="s3_wo_status")
    with c6:
        def _on_submit() -> None:
            st.session_state["s3_wo_run_fetch"] = True
        st.button("Submit", type="primary", key="s3_wo_submit", on_click=_on_submit, use_container_width=True)

    # Handle queued fetch action (progress-first UX)
    if st.session_state.get("s3_wo_run_fetch"):
        st.session_state["s3_wo_run_fetch"] = False
        
        if not site_name or site_name == "(select)":
            st.error("Please select a Site Name.")
        elif start_date is None or end_date is None:
            st.error("Please select both Start Date and End Date.")
        elif start_date > end_date:
            st.error("Start Date must be on or before End Date.")
        else:
            prog_slot = st.empty()
            prog = prog_slot.progress(0, text="Validating filters...")
            try:
                _smooth_progress(prog, 0, 20, text="Validating filters...")
                _smooth_progress(prog, 20, 70, text="Fetching work orders from database...")

                loc_val = None if location in (None, "(all)") else location
                st_val = None if status_ui in (None, "(all)") else status_ui

                df = _fetch_work_orders(
                    site_name=site_name,
                    start_date=start_date,
                    end_date=end_date,
                    status_ui=st_val,
                    location=loc_val,
                )

                _smooth_progress(prog, 70, 90, text="Calculating KPIs...")

                st.session_state["s3_wo_last_df"] = df
                st.session_state["s3_wo_last_meta"] = {
                    "site": site_name,
                    "start": start_date,
                    "end": end_date,
                    "location": location,
                    "status": status_ui,
                }

                if isinstance(df, pd.DataFrame) and not df.empty:
                    total = int(len(df))
                    c_rej = int((df["status"].astype("string").str.upper() == "REJECTED").sum())
                    c_open = int((df["status"].astype("string").str.upper() == "OPEN").sum())
                    c_wip = int((df["status"].astype("string").str.upper() == "WIP").sum())
                    # Approved KPI must use derived status (single source of truth)
                    c_approved = int((df["status"].astype("string").str.upper() == "APPROVED").sum())
                else:
                    total = c_rej = c_open = c_wip = c_approved = 0

                st.session_state["s3_wo_last_kpis"] = {
                    "total": total,
                    "rejected": c_rej,
                    "open": c_open,
                    "wip": c_wip,
                    "approved": c_approved,
                }

                _smooth_progress(prog, 90, 100, text="Work orders ready")
            except Exception as e:
                st.error(f"Failed to fetch work orders: {e}")
            finally:
                prog_slot.empty()

    # Display results
    df_last = st.session_state.get("s3_wo_last_df")
    if isinstance(df_last, pd.DataFrame) and not df_last.empty:
        kpis = st.session_state.get("s3_wo_last_kpis") or {}
        
        # Modern KPI cards (same as S2)
        st.markdown(
            f"""
            <div class="kpi-row">
              <div class="kpi-card"><div class="kpi-title">Work Orders</div><div class="kpi-value" style="color:#2563eb;">{kpis.get("total", 0)}</div></div>
              <div class="kpi-card"><div class="kpi-title">Rejected</div><div class="kpi-value" style="color:#dc2626;">{kpis.get("rejected", 0)}</div></div>
              <div class="kpi-card"><div class="kpi-title">Open</div><div class="kpi-value" style="color:#10b981;">{kpis.get("open", 0)}</div></div>
              <div class="kpi-card"><div class="kpi-title">Awaiting Approval</div><div class="kpi-value" style="color:#f97316;">{kpis.get("wip", 0)}</div></div>
              <div class="kpi-card"><div class="kpi-title">Approved</div><div class="kpi-value" style="color:#10b981;">{kpis.get("approved", 0)}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        df = df_last.copy()
        if "date_planned" in df.columns:
            df["date_planned"] = pd.to_datetime(df["date_planned"]).dt.strftime("%Y-%m-%d")

        df_display = df.copy()
        df_display["status"] = df_display["status"].astype("string").fillna("").map(
            lambda s: DB_STATUS_TO_UI.get(str(s).strip().upper(), str(s))
        )
        styled = (
            df_display.style.map(_highlight_status, subset=["status"])
            .set_table_styles([{"selector": "th", "props": [("font-weight", "800"), ("color", "#0f172a")]}])
        )
        st.dataframe(styled, width="stretch", hide_index=True)


# =============================================================================
# View Approvals — final approval
# =============================================================================


def _fetch_all_s3_ptws(*, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetch ALL PTWs that have reached S3 (both pending approval and already approved).
    S1 created + S2 forwarded (regardless of S3 approval status).
    Filtered by date_s2_forwarded within date range.
    """
    sb = get_supabase_client(prefer_service_role=True)

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt_excl = datetime.combine(end_date + timedelta(days=1), datetime.min.time())

    cols = (
        "work_order_id,site_name,location,equipment,"
        "date_s1_created,date_s2_forwarded,date_s3_approved,date_s2_rejected,date_s3_rejected"
    )

    q = (
        sb.table(TABLE_WORK_ORDERS)
        .select(cols)
        .not_.is_("date_s1_created", "null")
        .not_.is_("date_s2_forwarded", "null")
        .gte("date_s2_forwarded", start_dt.isoformat(sep=" ", timespec="seconds"))
        .lt("date_s2_forwarded", end_dt_excl.isoformat(sep=" ", timespec="seconds"))
        .order("date_s2_forwarded", desc=True)
    )

    resp = q.execute()
    err = getattr(resp, "error", None)
    if err:
        raise RuntimeError(f"Failed to fetch PTWs: {err}")

    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["work_location"] = df.apply(
        lambda r: f"{(r.get('location') or '').strip()}-{(r.get('equipment') or '').strip()}".strip("-"),
        axis=1,
    )
    df["derived_status"] = df.apply(lambda r: derive_ptw_status(r.to_dict()), axis=1)
    
    # Add approval status flag for UI display
    df["is_approved"] = df["date_s3_approved"].notna() & (df["date_s3_approved"].astype(str).str.strip() != "")
    
    return df


def _fetch_ptw_requests_for_work_orders(work_order_ids: list[str]) -> dict[str, dict]:
    """Return mapping work_order_id -> ptw_requests row (latest by created_at if duplicates exist)."""
    if not work_order_ids:
        return {}

    sb = get_supabase_client(prefer_service_role=True)
    resp = (
        sb.table(TABLE_PTW_REQUESTS)
        .select("ptw_id,permit_no,site_name,created_at,created_by,form_data")
        .in_("permit_no", work_order_ids)
        .order("created_at", desc=True)
        .execute()
    )
    err = getattr(resp, "error", None)
    if err:
        raise RuntimeError(f"Failed to fetch PTW requests: {err}")

    rows: list[dict[str, Any]] = getattr(resp, "data", None) or []
    out: dict[str, dict] = {}
    for r in rows:
        woid = str(r.get("permit_no") or "")
        if not woid:
            continue
        # keep the first (latest) row if duplicates exist
        if woid not in out:
            out[woid] = r
    return out


def _update_ptw_issuer_name(*, work_order_id: str, issuer_name: str) -> tuple[bool, str]:
    """Update PTW form_data with issuer_name (autosave)."""
    sb = get_supabase_client(prefer_service_role=True)
    
    ptw_resp = (
        sb.table(TABLE_PTW_REQUESTS)
        .select("ptw_id,form_data")
        .eq("permit_no", work_order_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    ptw_err = getattr(ptw_resp, "error", None)
    if ptw_err:
        return False, f"Failed to load PTW request: {ptw_err}"

    ptw_rows = getattr(ptw_resp, "data", None) or []
    if not ptw_rows:
        return False, "No PTW request found for this work_order_id."

    form_data = (ptw_rows[0].get("form_data") or {}) if isinstance(ptw_rows[0], dict) else {}
    if not isinstance(form_data, dict):
        form_data = {}
    form_data["issuer_name"] = issuer_name

    upd_ptw = (
        sb.table(TABLE_PTW_REQUESTS)
        .update({"form_data": form_data})
        .eq("ptw_id", ptw_rows[0].get("ptw_id"))
        .execute()
    )
    upd_ptw_err = getattr(upd_ptw, "error", None)
    if upd_ptw_err:
        return False, f"Failed to update PTW issuer info: {upd_ptw_err}"

    return True, "Saved"


def _approve_work_order(*, work_order_id: str, issuer_name: str) -> tuple[bool, str]:
    """Atomic approval: set date_s3_approved only if still NULL. Returns (ok, message/timestamp)."""
    sb = get_supabase_client(prefer_service_role=True)
    ts = datetime.now().isoformat(sep=" ", timespec="seconds")

    # Update PTW form_data with issuer_name + issuer_datetime (so template can render it)
    ptw_resp = (
        sb.table(TABLE_PTW_REQUESTS)
        .select("ptw_id,form_data")
        .eq("permit_no", work_order_id)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
    )
    ptw_err = getattr(ptw_resp, "error", None)
    if ptw_err:
        return False, f"Failed to load PTW request for approval: {ptw_err}"

    ptw_rows = getattr(ptw_resp, "data", None) or []
    if not ptw_rows:
        return False, "No PTW request found for this work_order_id."

    form_data = (ptw_rows[0].get("form_data") or {}) if isinstance(ptw_rows[0], dict) else {}
    if not isinstance(form_data, dict):
        form_data = {}
    form_data["issuer_name"] = issuer_name
    form_data["issuer_datetime"] = ts

    upd_ptw = (
        sb.table(TABLE_PTW_REQUESTS)
        .update({"form_data": form_data})
        .eq("ptw_id", ptw_rows[0].get("ptw_id"))
        .execute()
    )
    upd_ptw_err = getattr(upd_ptw, "error", None)
    if upd_ptw_err:
        return False, f"Failed to update PTW issuer info: {upd_ptw_err}"

    # Approve only if not already approved (prevents duplicate approvals)
    upd = (
        sb.table(TABLE_WORK_ORDERS)
        .update({"date_s3_approved": ts})
        .eq("work_order_id", work_order_id)
        .is_("date_s3_approved", "null")
        .execute()
    )
    upd_err = getattr(upd, "error", None)
    if upd_err:
        return False, f"Failed to approve work order: {upd_err}"

    updated_rows = getattr(upd, "data", None) or []
    if not updated_rows:
        return False, "This PTW appears to already be approved (date_s3_approved is set)."

    return True, ts


def _revoke_s3_approval(*, work_order_id: str) -> tuple[bool, str]:
    """Revoke S3 approval by clearing date_s3_approved."""
    sb = get_supabase_client(prefer_service_role=True)
    
    resp = (
        sb.table(TABLE_WORK_ORDERS)
        .update({"date_s3_approved": None})
        .eq("work_order_id", work_order_id)
        .execute()
    )
    
    err = getattr(resp, "error", None)
    if err:
        return False, f"Failed to revoke approval: {err}"
    
    return True, "Approval revoked successfully"


def _render_view_approvals() -> None:
    st.markdown("## View Approvals")
    st.caption("Final approval stage (S3). View all PTWs submitted from S1 and forwarded from S2.")

    # Success card and KPI styling (matching S2)
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
        
        .kpi-row {
            display: flex;
            gap: 1rem;
            margin: 1.5rem 0;
        }
        .kpi-card {
            flex: 1;
            background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        }
        .kpi-title {
            font-size: 0.875rem;
            color: #64748b;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.5rem;
        }
        .kpi-value {
            font-size: 2rem;
            font-weight: 700;
            line-height: 1;
        }
        </style>
    """, unsafe_allow_html=True)

    if "s3_pending_df" not in st.session_state:
        st.session_state["s3_pending_df"] = None
    if "s3_ptw_map" not in st.session_state:
        st.session_state["s3_ptw_map"] = {}
    if "s3_appr_run_fetch" not in st.session_state:
        st.session_state["s3_appr_run_fetch"] = False
    if "s3_active_approval_id" not in st.session_state:
        st.session_state["s3_active_approval_id"] = None

    # Filters (matching S2 layout)
    col1, col2, col3 = st.columns([1.5, 1.5, 1])
    with col1:
        start_date = st.date_input(
            "From Date",
            value=st.session_state.get("s3_appr_start_val", date.today() - timedelta(days=30)),
            key="s3_appr_from"
        )
        st.session_state["s3_appr_start_val"] = start_date
    with col2:
        end_date = st.date_input(
            "To Date",
            value=st.session_state.get("s3_appr_end_val", date.today()),
            key="s3_appr_to"
        )
        st.session_state["s3_appr_end_val"] = end_date
    with col3:
        st.write("")
        st.write("")
        def _on_fetch() -> None:
            st.session_state["s3_appr_run_fetch"] = True
        st.button("Fetch PTWs", type="primary", key="s3_fetch", on_click=_on_fetch)

    # Handle queued fetch (progress-first UX)
    if st.session_state.get("s3_appr_run_fetch"):
        st.session_state["s3_appr_run_fetch"] = False
        prog_slot = st.empty()
        prog = prog_slot.progress(0, text="Fetching PTWs...")
        try:
            _smooth_progress(prog, 0, 20, text="Validating date range...")
            _smooth_progress(prog, 20, 70, text="Loading from database...")
            df = _fetch_all_s3_ptws(start_date=start_date, end_date=end_date)
            ptw_map = _fetch_ptw_requests_for_work_orders(df["work_order_id"].astype(str).tolist() if not df.empty else [])
            st.session_state["s3_pending_df"] = df
            st.session_state["s3_ptw_map"] = ptw_map
            _smooth_progress(prog, 70, 100, text="PTWs ready")
        except Exception as e:
            st.error(f"Failed to fetch PTWs: {e}")
            st.session_state["s3_pending_df"] = pd.DataFrame()
            st.session_state["s3_ptw_map"] = {}
        finally:
            prog_slot.empty()

    df = st.session_state.get("s3_pending_df")
    ptw_map = st.session_state.get("s3_ptw_map") or {}

    if df is None:
        st.info("Select a date range and click 'Fetch PTWs' to load submitted permits.")
        return
    if not isinstance(df, pd.DataFrame) or df.empty:
        st.info("No PTWs found for the selected date range.")
        return

    # KPI Cards - S3 decision metrics only (no WIP, no redundant "Approved Total")
    # Single source of truth: Approved = date_s3_approved IS NOT NULL
    total = int(len(df))
    pending_count = int((~df["is_approved"]).sum())
    approved_count = int(df["is_approved"].sum())

    st.markdown(
        f"""
        <div class="kpi-row">
          <div class="kpi-card"><div class="kpi-title">Total PTWs</div><div class="kpi-value" style="color:#2563eb;">{total}</div></div>
          <div class="kpi-card"><div class="kpi-title">Pending Approval</div><div class="kpi-value" style="color:#f97316;">{pending_count}</div></div>
          <div class="kpi-card"><div class="kpi-title">Approved</div><div class="kpi-value" style="color:#10b981;">{approved_count}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.divider()

    # ---------------------------------------------------------------------
    # Selection-first UX: build PTW dropdown (no mass rendering)
    # ---------------------------------------------------------------------

    def _build_s3_ptw_label(row: pd.Series) -> str:
        """Build label for S3 PTW selectbox."""
        work_order_id = str(row.get("work_order_id") or "")
        site_name = str(row.get("site_name") or "")

        # Prefer explicit approval flag; fall back to derived status
        status = "APPROVED" if bool(row.get("is_approved", False)) else str(row.get("derived_status") or "")

        fwd = row.get("date_s2_forwarded")
        fwd_str = ""
        try:
            if fwd is not None and str(fwd).strip():
                fwd_str = pd.to_datetime(fwd).strftime("%Y-%m-%d %H:%M")
        except Exception:
            fwd_str = str(fwd)

        return f"{work_order_id} | {site_name} | {status} | Forwarded: {fwd_str}"

    options = ["(select PTW)"] + [
        _build_s3_ptw_label(r) for _, r in df.iterrows()
    ]

    selected_label = st.selectbox(
        "Select PTW for Approval",
        options=options,
        key="s3_selected_ptw_label",
    )

    # Map selection back to active work_order_id
    if selected_label and selected_label != "(select PTW)":
        selected_wo = selected_label.split("|")[0].strip()
        st.session_state["s3_active_approval_id"] = selected_wo
    else:
        st.session_state["s3_active_approval_id"] = None

    active_wo = st.session_state.get("s3_active_approval_id")
    if not active_wo:
        return

    # Locate the active row in the current dataset
    active_row = df[df["work_order_id"].astype(str) == str(active_wo)]
    if active_row.empty:
        st.warning("Selected PTW is no longer available in the current filter set.")
        return

    r = active_row.iloc[0]
    work_order_id = str(r.get("work_order_id") or "")
    site_name = str(r.get("site_name") or "")
    status = str(r.get("derived_status") or "")
    is_db_approved = bool(r.get("is_approved", False))

    fwd = r.get("date_s2_forwarded")
    fwd_str = ""
    try:
        if fwd is not None and str(fwd).strip():
            fwd_str = pd.to_datetime(fwd).strftime("%Y-%m-%d %H:%M")
    except Exception:
        fwd_str = str(fwd)

    appr = r.get("date_s3_approved")
    appr_str = ""
    try:
        if appr is not None and str(appr).strip():
            appr_str = pd.to_datetime(appr).strftime("%Y-%m-%d %H:%M")
    except Exception:
        appr_str = str(appr)

    # Check if just approved in this session (show success state)
    just_approved_key = f"s3_wo_{work_order_id}_just_approved"
    just_approved = bool(st.session_state.get(just_approved_key, False))

    # Progress-first loading for PTW details
    prog_slot = st.empty()
    prog = prog_slot.progress(0, text="Loading PTW details...")
    try:
        _smooth_progress(prog, 0, 40, text="Fetching PTW form data...")
        ptw_data = ptw_map.get(work_order_id)

        _smooth_progress(prog, 40, 70, text="Preparing approval view...")
        # (ptw_data is passed via ptw_map into the render helpers)

        _smooth_progress(prog, 70, 100, text="Ready")
    finally:
        prog_slot.empty()

    # Render read-only or approval form based on state
    if just_approved or is_db_approved:
        _render_post_approval_view(
            work_order_id=work_order_id,
            site_name=site_name,
            ptw_map=ptw_map,
            just_approved_key=just_approved_key,
            is_db_approved=is_db_approved,
        )
    else:
        _render_approval_form(
            work_order_id=work_order_id,
            site_name=site_name,
            fwd_str=fwd_str,
            ptw_map=ptw_map,
            just_approved_key=just_approved_key,
        )


def _render_post_approval_view(*, work_order_id: str, site_name: str, ptw_map: dict, just_approved_key: str, is_db_approved: bool = False) -> None:
    """Render the view shown after successful approval (or for already-approved PTWs)."""
    ptw_row = ptw_map.get(work_order_id) or {}
    form_data = ptw_row.get("form_data") or {}
    if not isinstance(form_data, dict):
        form_data = {}
    
    approval_ts = form_data.get("issuer_datetime", "")
    issuer_name = form_data.get("issuer_name", "N/A")
    
    st.markdown("""
        <div class="success-card">
            <h3>✅ PTW Approved</h3>
            <p><strong>Work Order:</strong> {}</p>
            <p><strong>Site:</strong> {}</p>
            <p><strong>Issuer:</strong> {}</p>
            <p><strong>Approved At:</strong> {}</p>
            <p>This PTW has been approved successfully.</p>
        </div>
    """.format(work_order_id, site_name, issuer_name, approval_ts or "N/A"), unsafe_allow_html=True)
    
    st.markdown("### Download Approved PTW")
    st.caption("The PDF includes issuer name, approval timestamp, and all evidence attachments")
    
    # Download button with progress (queued)
    download_key = f"s3_download_{work_order_id}"
    req_key = f"{download_key}_requested"
    cache_key = f"{download_key}_bytes"

    if req_key not in st.session_state:
        st.session_state[req_key] = False
    if cache_key not in st.session_state:
        st.session_state[cache_key] = None

    def _on_gen_pdf() -> None:
        st.session_state["s3_active_approval_id"] = work_order_id
        st.session_state[req_key] = True
        st.session_state[just_approved_key] = True

    st.button("Generate & Download PDF", key=download_key, type="primary", on_click=_on_gen_pdf)

    if st.session_state.get(req_key):
        st.session_state[req_key] = False
        st.session_state[just_approved_key] = True
        st.session_state["s3_active_approval_id"] = work_order_id
        
        prog_slot = st.empty()
        prog = prog_slot.progress(0, text="Preparing PDF...")
        try:
            def progress_callback(pct, msg):
                prog.progress(int(pct), text=msg)

            # Step 1: Fetch approval timestamps from work_orders (single source of truth)
            progress_callback(5, "Fetching approval timestamps...")
            approval_times = get_ptw_approval_times(work_order_id)
            
            # Step 2: Inject timestamps into form_data for template placeholders
            updated_form = _inject_approval_times_into_form_data(form_data, work_order_id, is_approved=True)
            
            # Step 3: Generate the PDF with injected timestamps
            if callable(generate_ptw_pdf_with_attachments):
                pdf_bytes = generate_ptw_pdf_with_attachments(
                    form_data=updated_form,
                    work_order_id=work_order_id,
                    progress_callback=progress_callback,
                )
            else:
                progress_callback(30, "Downloading template...")
                tpl = _download_template_from_supabase()
                progress_callback(70, "Rendering PTW PDF...")
                pdf_bytes = generate_ptw_pdf(tpl, build_doc_data(updated_form), progress_callback=progress_callback)
                progress_callback(90, "Done")
            
            # Step 4: Apply APPROVED stamp overlay to the PDF (S3 approved PDFs only)
            progress_callback(95, "Applying approval stamp...")
            stamped_pdf = add_floating_approval_stamp(
                pdf_bytes,
                approved_on=approval_times["issuer_datetime"],
            )
            
            progress_callback(100, "Complete")
            st.session_state[cache_key] = stamped_pdf
            st.session_state[just_approved_key] = True
            st.session_state["s3_active_approval_id"] = work_order_id
            prog_slot.empty()
        except Exception as e:
            prog_slot.empty()
            st.error(f"Failed to generate PDF: {e}")
            st.session_state[just_approved_key] = True

    cached = st.session_state.get(cache_key)
    if isinstance(cached, (bytes, bytearray)) and len(cached) > 0:
        def _on_download() -> None:
            st.session_state[just_approved_key] = True
            st.session_state["s3_active_approval_id"] = work_order_id
        
        st.download_button(
            label=f"📥 Download {work_order_id}.pdf",
            data=cached,
            file_name=f"{work_order_id}.pdf",
            mime="application/pdf",
            key=f"{download_key}_btn",
            on_click=_on_download,
        )
    
    st.divider()
    
    # Option to revoke or view another
    col1, col2 = st.columns(2)
    with col1:
        def _on_revoke() -> None:
            st.session_state["s3_active_approval_id"] = work_order_id
            st.session_state[f"s3_wo_{work_order_id}_revoke_requested"] = True
        st.button("🔄 Revoke Approval", key=f"s3_revoke_{work_order_id}", on_click=_on_revoke)
    
    with col2:
        def _on_review_another() -> None:
            # Clear approval state
            st.session_state[just_approved_key] = False
            st.session_state["s3_active_approval_id"] = None
            st.session_state.pop(cache_key, None)
            
            # Reset dropdown to "(select PTW)" - soft reset, no page reload
            st.session_state["s3_selected_ptw_label"] = "(select PTW)"
        st.button("📋 Review Another PTW", key=f"s3_another_{work_order_id}", on_click=_on_review_another)
    
    # Handle revoke request
    if st.session_state.get(f"s3_wo_{work_order_id}_revoke_requested"):
        st.session_state[f"s3_wo_{work_order_id}_revoke_requested"] = False
        prog_slot = st.empty()
        prog = prog_slot.progress(0, text="Revoking approval...")
        try:
            _smooth_progress(prog, 0, 60, text="Updating database...")
            ok, msg = _revoke_s3_approval(work_order_id=work_order_id)
            _smooth_progress(prog, 60, 100, text="Done")
            prog_slot.empty()
            if ok:
                st.success("Approval revoked. You can now re-approve this PTW.")
                st.session_state[just_approved_key] = False
                st.session_state.pop("s3_pending_df", None)
                st.session_state["s3_appr_run_fetch"] = True
                st.session_state["s3_active_approval_id"] = work_order_id
                st.rerun()
            else:
                st.error(msg)
        except Exception as e:
            prog_slot.empty()
            st.error(f"Failed to revoke: {e}")


def _render_approval_form(*, work_order_id: str, site_name: str, fwd_str: str, ptw_map: dict, just_approved_key: str) -> None:
    """Render the approval form (permit issuer name + preview + approve)."""
    ptw_row = ptw_map.get(work_order_id) or {}
    form_data = ptw_row.get("form_data") or {}
    if not isinstance(form_data, dict):
        form_data = {}

    st.markdown("### Final PTW Preview (read-only)")
    st.caption("PDF is generated dynamically from latest `ptw_requests.form_data` and the active template.")

    key_prefix = f"s3_appr_{work_order_id}_"
    issuer_key = f"{key_prefix}issuer"
    preview_req_key = f"{key_prefix}preview_requested"
    preview_cache_key = f"{key_prefix}preview_bytes"
    approve_req_key = f"{key_prefix}approve_requested"

    if preview_req_key not in st.session_state:
        st.session_state[preview_req_key] = False
    if preview_cache_key not in st.session_state:
        st.session_state[preview_cache_key] = None
    if approve_req_key not in st.session_state:
        st.session_state[approve_req_key] = False

    # Handle preview request FIRST (progress-first UX)
    if st.session_state.get(preview_req_key):
        st.session_state[preview_req_key] = False
        issuer_value = st.session_state.get(issuer_key, "")
        
        prog_slot = st.empty()
        prog = prog_slot.progress(0, text="Generating PDF preview...")
        try:
            def progress_callback(pct, msg):
                prog.progress(int(pct), text=msg)

            # Inject holder_datetime from work_orders (S2 forwarding date)
            # Preview does NOT get issuer_datetime or APPROVED stamp (not yet approved)
            progress_callback(5, "Fetching approval timestamps...")
            updated = _inject_approval_times_into_form_data(form_data, work_order_id, is_approved=False)
            updated["issuer_name"] = issuer_value.strip()
            # Leave issuer_datetime empty for preview (will be set on actual approval)
            updated.setdefault("issuer_datetime", "")

            if callable(generate_ptw_pdf_with_attachments):
                pdf_bytes = generate_ptw_pdf_with_attachments(updated, work_order_id, progress_callback=progress_callback)
            else:
                progress_callback(30, "Downloading template...")
                tpl = _download_template_from_supabase()
                progress_callback(70, "Rendering PTW PDF...")
                pdf_bytes = generate_ptw_pdf(tpl, build_doc_data(updated), progress_callback=progress_callback)
                progress_callback(100, "Done")
            
            # Note: Preview PDF does NOT get APPROVED stamp (only approved PDFs do)
            st.session_state[preview_cache_key] = pdf_bytes
            prog_slot.empty()
        except Exception as e:
            prog_slot.empty()
            st.error(f"Failed to generate preview: {e}")

    # Handle approve request FIRST (progress-first UX)
    if st.session_state.get(approve_req_key):
        st.session_state[approve_req_key] = False
        issuer_value = st.session_state.get(issuer_key, "")
        
        if not issuer_value.strip():
            st.error("Permit Issuer Name is required.")
        else:
            prog_slot = st.empty()
            prog = prog_slot.progress(0, text="Approving PTW...")
            ok = False
            try:
                _smooth_progress(prog, 0, 40, text="Saving issuer name...")
                ok, approval_ts = _approve_work_order(work_order_id=work_order_id, issuer_name=issuer_value.strip())
                if ok:
                    _smooth_progress(prog, 40, 70, text="Approved")
                    
                    # Update form_data with approval timestamp
                    ptw_map[work_order_id]["form_data"]["issuer_name"] = issuer_value.strip()
                    ptw_map[work_order_id]["form_data"]["issuer_datetime"] = approval_ts
                    st.session_state["s3_ptw_map"] = ptw_map
                    
                    # Refresh the PTW list so dropdown and KPIs reflect latest status
                    _smooth_progress(prog, 70, 90, text="Refreshing PTW list...")
                    try:
                        start_date = st.session_state.get("s3_appr_start_val")
                        end_date = st.session_state.get("s3_appr_end_val")
                        if start_date and end_date:
                            df_refreshed = _fetch_all_s3_ptws(start_date=start_date, end_date=end_date)
                            st.session_state["s3_pending_df"] = df_refreshed
                            ptw_map_refreshed = _fetch_ptw_requests_for_work_orders(
                                df_refreshed["work_order_id"].astype(str).tolist() if not df_refreshed.empty else []
                            )
                            st.session_state["s3_ptw_map"] = ptw_map_refreshed
                    except Exception as refresh_err:
                        # Do not fail approval if refresh fails
                        print(f"Failed to refresh PTW list after approval: {refresh_err}")
                    
                    _smooth_progress(prog, 90, 100, text="Complete")
                    prog_slot.empty()
                    
                    # Mark as just approved
                    st.session_state[just_approved_key] = True
                    st.session_state["s3_active_approval_id"] = work_order_id
                    
                    # Render success view immediately, using refreshed map
                    _render_post_approval_view(
                        work_order_id=work_order_id,
                        site_name=site_name,
                        ptw_map=st.session_state.get("s3_ptw_map", ptw_map),
                        just_approved_key=just_approved_key,
                    )
                    return
                else:
                    prog_slot.empty()
                    st.error(str(approval_ts))
            except Exception as e:
                prog_slot.empty()
                st.error(f"Approval failed: {e}")

    # Render form - NO autosave, purely local state until "Approve" is clicked
    issuer_name = st.text_area(
        "Permit Issuer Name *",
        value=str(form_data.get("issuer_name") or ""),
        key=issuer_key,
        height=38,
        placeholder="Enter Permit Issuer name (will save on approval only)",
        # NO on_change callback - no backend write until approval
    )

    btn1, btn2 = st.columns([1, 1], vertical_alignment="bottom")
    with btn1:
        def _on_preview() -> None:
            st.session_state["s3_active_approval_id"] = work_order_id
            st.session_state[preview_req_key] = True
        st.button("Generate Preview PDF", use_container_width=True, key=f"{key_prefix}preview", on_click=_on_preview)
    with btn2:
        approve_disabled = not issuer_name.strip()
        def _on_approve() -> None:
            st.session_state["s3_active_approval_id"] = work_order_id
            st.session_state[approve_req_key] = True
        st.button(
            "Approve",
            type="primary",
            disabled=approve_disabled,
            use_container_width=True,
            key=f"{key_prefix}approve",
            on_click=_on_approve,
        )

    # Show download button if preview is cached
    cached_preview = st.session_state.get(preview_cache_key)
    if isinstance(cached_preview, (bytes, bytearray)) and len(cached_preview) > 0:
        st.download_button(
            "Download Preview PDF",
            data=cached_preview,
            file_name=f"{work_order_id}_preview.pdf",
            mime="application/pdf",
            use_container_width=True,
            key=f"{key_prefix}preview_dl",
        )


def render(db_path: str) -> None:
    st.markdown("# S3 Portal")
    st.caption("Final PTW Approval Stage (read-only except approval).")

    # Modern tabs styling + Anti-Ghosting CSS
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
        
        /* KPI cards styling */
        .kpi-row { display: flex; gap: 12px; flex-wrap: wrap; margin: 10px 0 14px 0; }
        .kpi-card { flex: 1 1 160px; background: white; border: 1px solid rgba(148,163,184,0.35);
                    border-radius: 14px; padding: 14px 16px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); }
        .kpi-title { font-size: 14px; color: #475569; margin-bottom: 6px; font-weight: 700; }
        .kpi-value { font-size: 34px; font-weight: 900; line-height: 1.05; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Wrap entire tab in fragment for better performance
    _frag = getattr(st, "fragment", None)
    if callable(_frag):
        def _impl() -> None:
            tab1, tab2 = st.tabs(["View Work Order", "View Approvals"])
            with tab1:
                _render_view_work_order_s3()
            with tab2:
                _render_view_approvals()
        _frag(_impl)()
        return
    
    tab1, tab2 = st.tabs(["View Work Order", "View Approvals"])
    with tab1:
        _render_view_work_order_s3()
    with tab2:
        _render_view_approvals()
