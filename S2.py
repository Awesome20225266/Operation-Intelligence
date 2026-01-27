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
from PyPDF2 import PdfReader, PdfWriter  # type: ignore

from supabase_link import get_supabase_client

# Import shared functions from S1
from S1 import (
    TABLE_WORK_ORDERS,
    TABLE_PTW_REQUESTS,
    TABLE_PTW_TEMPLATES,
    UI_STATUSES,
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

# Import S3 approval stamp functions (for applying stamp when PTW is approved)
try:
    from S3 import (
        get_ptw_approval_times,
        add_floating_approval_stamp,
        _inject_approval_times_into_form_data,
    )
    _HAS_S3_STAMP = True
except ImportError:
    _HAS_S3_STAMP = False


# =============================================================================
# CONSTANTS
# =============================================================================

EVIDENCE_BUCKET = "ptw-evidence"

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
        </style>
        """,
        unsafe_allow_html=True,
    )


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


def _get_content_type(ext: str) -> str:
    """Get MIME content type from file extension."""
    content_types = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return content_types.get(ext.lower(), "application/octet-stream")


def _list_evidence_files(work_order_id: str, evidence_type: str) -> list[dict]:
    """
    List existing evidence files for a work order.
    
    Returns list of dicts with 'name', 'path', and 'url' keys.
    """
    sb = get_supabase_client(prefer_service_role=True)
    
    try:
        folder_path = f"{work_order_id}/{evidence_type}"
        resp = sb.storage.from_(EVIDENCE_BUCKET).list(folder_path)
        
        if not resp:
            return []
        
        files = []
        for item in resp:
            if isinstance(item, dict) and item.get("name"):
                file_path = f"{folder_path}/{item['name']}"
                files.append({
                    "name": item["name"],
                    "path": file_path,
                })
        return files
    
    except Exception:
        return []


def _download_evidence_file(file_path: str) -> bytes | None:
    """Download an evidence file from Supabase Storage."""
    sb = get_supabase_client(prefer_service_role=True)
    
    try:
        resp = sb.storage.from_(EVIDENCE_BUCKET).download(file_path)
        return resp
    except Exception:
        return None


def _get_evidence_public_url(file_path: str) -> str | None:
    """Get public URL for an evidence file (signed URL for private buckets)."""
    sb = get_supabase_client(prefer_service_role=True)
    
    try:
        # Create signed URL valid for 1 hour
        resp = sb.storage.from_(EVIDENCE_BUCKET).create_signed_url(file_path, 3600)
        if resp and "signedURL" in resp:
            return resp["signedURL"]
        return None
    except Exception:
        return None


# =============================================================================
# PDF WITH ATTACHMENTS
# =============================================================================


def _create_attachments_page(
    isolation_files: list[dict],
    toolbox_files: list[dict],
    work_order_id: str,
) -> bytes | None:
    """
    Create a PDF page with attachment thumbnails/references.
    
    Args:
        isolation_files: List of isolation evidence files
        toolbox_files: List of toolbox evidence files
        work_order_id: The work order ID for reference
    
    Returns:
        PDF bytes for the attachments page, or None if no files
    """
    if not isolation_files and not toolbox_files:
        return None
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    y_position = height - 50
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_position, "EVIDENCE ATTACHMENTS")
    y_position -= 10
    
    c.setFont("Helvetica", 10)
    c.drawString(50, y_position, f"Work Order: {work_order_id}")
    y_position -= 5
    c.drawString(50, y_position, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y_position -= 30
    
    # Draw line separator
    c.line(50, y_position, width - 50, y_position)
    y_position -= 20
    
    def add_section(title: str, files: list[dict], y_pos: float) -> float:
        if not files:
            return y_pos
        
        c.setFont("Helvetica-Bold", 12)
        c.drawString(50, y_pos, title)
        y_pos -= 20
        
        for idx, file_info in enumerate(files):
            file_name = file_info.get("name", "Unknown")
            file_path = file_info.get("path", "")
            
            # Check if we need a new page
            if y_pos < 150:
                c.showPage()
                y_pos = height - 50
            
            # Try to embed image if it's an image file
            ext = os.path.splitext(file_name)[1].lower()
            is_image = ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]
            
            if is_image:
                # Download and embed the image
                file_bytes = _download_evidence_file(file_path)
                if file_bytes:
                    try:
                        img = Image.open(BytesIO(file_bytes))
                        
                        # Resize to fit (max 200x150)
                        max_width, max_height = 200, 150
                        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
                        
                        # Convert to RGB if necessary
                        if img.mode in ('RGBA', 'P'):
                            img = img.convert('RGB')
                        
                        img_buffer = BytesIO()
                        img.save(img_buffer, format='JPEG', quality=85)
                        img_buffer.seek(0)
                        
                        img_reader = ImageReader(img_buffer)
                        
                        # Draw image
                        c.drawImage(img_reader, 50, y_pos - img.height, 
                                   width=img.width, height=img.height)
                        
                        # Draw file name below image
                        c.setFont("Helvetica", 9)
                        c.drawString(50, y_pos - img.height - 15, f"{idx + 1}. {file_name}")
                        
                        y_pos -= (img.height + 35)
                        
                    except Exception:
                        # Fallback to text reference
                        c.setFont("Helvetica", 10)
                        c.drawString(70, y_pos, f"{idx + 1}. {file_name} (image)")
                        y_pos -= 20
                else:
                    c.setFont("Helvetica", 10)
                    c.drawString(70, y_pos, f"{idx + 1}. {file_name} (image - unable to load)")
                    y_pos -= 20
            else:
                # For PDFs and other documents, just list them
                c.setFont("Helvetica", 10)
                c.drawString(70, y_pos, f"{idx + 1}. {file_name}")
                y_pos -= 20
        
        y_pos -= 10
        return y_pos
    
    # Add sections
    y_position = add_section("ISOLATION EVIDENCE", isolation_files, y_position)
    y_position = add_section("TOOLBOX TALK EVIDENCE", toolbox_files, y_position)
    
    c.save()
    buffer.seek(0)
    return buffer.read()


def _merge_pdfs(main_pdf: bytes, attachments_pdf: bytes | None) -> bytes:
    """Merge main PTW PDF with attachments page."""
    if not attachments_pdf:
        return main_pdf
    
    try:
        writer = PdfWriter()
        
        # Add main PDF pages
        main_reader = PdfReader(BytesIO(main_pdf))
        for page in main_reader.pages:
            writer.add_page(page)
        
        # Add attachments page(s)
        attachments_reader = PdfReader(BytesIO(attachments_pdf))
        for page in attachments_reader.pages:
            writer.add_page(page)
        
        # Write merged PDF
        output = BytesIO()
        writer.write(output)
        output.seek(0)
        return output.read()
        
    except Exception:
        # If merge fails, return original PDF
        return main_pdf


def generate_ptw_pdf_with_attachments(
    form_data: dict,
    work_order_id: str,
    progress_callback=None,
) -> bytes:
    """
    Generate PTW PDF with evidence attachments on the last page.
    
    If the PTW is S3-approved (date_s3_approved is set), the APPROVED stamp
    will be applied to every page of the PDF.
    
    Args:
        form_data: The PTW form data
        work_order_id: Work order ID for fetching attachments
        progress_callback: Optional callback function for progress updates
    
    Returns:
        Complete PDF bytes with attachments (and APPROVED stamp if approved)
    """
    if progress_callback:
        progress_callback(5, "Checking approval status...")
    
    # Check if PTW is S3-approved and inject approval timestamps
    is_s3_approved = False
    approval_times = {}
    updated_form_data = dict(form_data) if form_data else {}
    
    if _HAS_S3_STAMP:
        try:
            approval_times = get_ptw_approval_times(work_order_id)
            is_s3_approved = bool(approval_times.get("date_s3_approved_raw"))
            
            # Inject approval timestamps into form_data for template placeholders
            if approval_times.get("holder_datetime"):
                updated_form_data["holder_datetime"] = approval_times["holder_datetime"]
            if is_s3_approved and approval_times.get("issuer_datetime"):
                updated_form_data["issuer_datetime"] = approval_times["issuer_datetime"]
        except Exception:
            pass  # Continue without stamp if error
    
    if progress_callback:
        progress_callback(10, "Downloading template...")
    
    # Generate main PDF
    template_bytes = _download_template_from_supabase()
    
    if progress_callback:
        progress_callback(30, "Generating PTW document...")
    
    doc_data = build_doc_data(updated_form_data)
    main_pdf = generate_ptw_pdf(template_bytes, doc_data, progress_callback=progress_callback)
    
    if progress_callback:
        progress_callback(50, "Fetching evidence files...")
    
    # Get evidence files
    isolation_files = _list_evidence_files(work_order_id, "isolation")
    toolbox_files = _list_evidence_files(work_order_id, "toolbox")
    
    if progress_callback:
        progress_callback(70, "Creating attachments page...")
    
    # Create attachments page
    attachments_pdf = _create_attachments_page(isolation_files, toolbox_files, work_order_id)
    
    if progress_callback:
        progress_callback(85, "Merging documents...")
    
    # Merge PDFs
    final_pdf = _merge_pdfs(main_pdf, attachments_pdf)
    
    # Apply APPROVED stamp if PTW is S3-approved
    if is_s3_approved and _HAS_S3_STAMP:
        if progress_callback:
            progress_callback(95, "Applying approval stamp...")
        try:
            final_pdf = add_floating_approval_stamp(
                final_pdf,
                approved_on=approval_times.get("issuer_datetime", ""),
            )
        except Exception:
            pass  # Continue without stamp if error
    
    if progress_callback:
        progress_callback(100, "Complete!")
    
    return final_pdf


# =============================================================================
# DATABASE HELPERS
# =============================================================================


def _fetch_ptw_for_s2(
    *,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """
    Fetch PTW requests for S2 review (only those with date_s1_created set).
    
    Returns PTWs that:
    - Have been submitted in S1 (date_s1_created is not null)
    - Are within the date range
    """
    sb = get_supabase_client(prefer_service_role=True)
    
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date + timedelta(days=1), datetime.min.time())
    
    # Fetch PTW requests
    q = sb.table(TABLE_PTW_REQUESTS).select(
        "ptw_id,permit_no,site_name,status,created_at,created_by,form_data"
    )
    
    q = q.gte("created_at", start_dt.isoformat())
    q = q.lt("created_at", end_dt.isoformat())
    q = q.order("created_at", desc=True)
    
    resp = q.execute()
    
    err = getattr(resp, "error", None)
    if err:
        raise RuntimeError(f"Failed to fetch PTW requests: {err}")
    
    data = getattr(resp, "data", None) or []
    df = pd.DataFrame(data)
    
    if df.empty:
        return df
    
    # Get work order IDs to check date_s1_created
    work_order_ids = df["permit_no"].unique().tolist()
    
    if work_order_ids:
        # Fetch work orders with date columns for status derivation
        wo_resp = (
            sb.table(TABLE_WORK_ORDERS)
            .select(
                "work_order_id,site_name,location,equipment,"
                "date_s1_created,date_s2_forwarded,date_s3_approved,"
                "date_s2_rejected,date_s3_rejected,isolation_requirement"
            )
            .in_("work_order_id", work_order_ids)
            .execute()
        )
        wo_data = getattr(wo_resp, "data", None) or []
        
        if wo_data:
            wo_df = pd.DataFrame(wo_data)
            
            # Filter to only those with date_s1_created (PTW actually submitted)
            wo_df = wo_df[wo_df["date_s1_created"].notna()]
            
            # Create lookup dicts
            status_lookup = {}
            s2_forwarded_lookup = {}
            isolation_lookup = {}
            location_lookup = {}
            
            for _, row in wo_df.iterrows():
                woid = row["work_order_id"]
                derived = derive_ptw_status(row.to_dict())
                # Map CLOSED -> APPROVED for UI display
                if derived == "CLOSED":
                    derived = "APPROVED"
                status_lookup[woid] = derived
                s2_forwarded_lookup[woid] = row.get("date_s2_forwarded")
                isolation_lookup[woid] = row.get("isolation_requirement")
                location_lookup[woid] = f"{row.get('location', '')}-{row.get('equipment', '')}"
            
            # Filter PTWs to only those with date_s1_created
            valid_wo_ids = set(wo_df["work_order_id"].tolist())
            df = df[df["permit_no"].isin(valid_wo_ids)]
            
            # Apply derived status
            df["status"] = df["permit_no"].map(status_lookup).fillna("OPEN")
            df["date_s2_forwarded"] = df["permit_no"].map(s2_forwarded_lookup)
            df["isolation_requirement"] = df["permit_no"].map(isolation_lookup)
            df["work_location"] = df["permit_no"].map(location_lookup)
    
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
        .update({
            "date_s2_forwarded": datetime.now().isoformat(sep=" ", timespec="seconds"),
            "isolation_requirement": isolation_requirement,
        })
        .eq("work_order_id", work_order_id)
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
        .eq("work_order_id", work_order_id)
        .execute()
    )
    
    err = getattr(resp, "error", None)
    if err:
        raise RuntimeError(f"Failed to revoke submission: {err}")


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

    # Mount progress UI early (BEFORE dependent filter queries) to avoid "reload then progress".
    early_prog_slot = st.empty()
    early_msg_slot = st.empty()
    
    # Initialize session state
    if "s2_wo_last_df" not in st.session_state:
        st.session_state["s2_wo_last_df"] = None
        st.session_state["s2_wo_last_meta"] = None
    if "s2_wo_last_kpis" not in st.session_state:
        st.session_state["s2_wo_last_kpis"] = None
    if "s2_wo_run_fetch" not in st.session_state:
        st.session_state["s2_wo_run_fetch"] = False
    
    site_options = ["(select)"] + sites
    
    # Get current selections for dependent filters
    ss_site = st.session_state.get("s2_wo_site")
    ss_start = st.session_state.get("s2_wo_start")
    ss_end = st.session_state.get("s2_wo_end")
    
    have_site = ss_site not in (None, "(select)", "")
    have_dates = isinstance(ss_start, date) and isinstance(ss_end, date)
    
    # If Submit was clicked, skip dependent filter queries so progress mounts immediately.
    if st.session_state.get("s2_wo_run_fetch"):
        locs, statuses_ui = [], []
    else:
        if have_site and have_dates:
            locs = _list_locations_from_work_orders(site_name=str(ss_site), start_date=ss_start, end_date=ss_end)
            statuses_ui = _list_statuses_from_work_orders(site_name=str(ss_site), start_date=ss_start, end_date=ss_end)
        else:
            locs = []
            statuses_ui = []
    
    loc_options = ["(all)"] + locs
    status_options = ["(all)"] + (statuses_ui if statuses_ui else UI_STATUSES)
    
    def _on_s2_wo_submit_click() -> None:
        st.session_state["s2_wo_run_fetch"] = True

    with st.form("s2_view_work_orders_filters", clear_on_submit=False):
        c1, c2, c3, c4, c5 = st.columns([2.0, 1.3, 1.3, 1.4, 1.4], vertical_alignment="bottom")
        with c1:
            site_name = st.selectbox("Site Name", options=site_options, index=0, key="s2_wo_site")
        with c2:
            start_date = st.date_input("Start Date", value=None, key="s2_wo_start")
        with c3:
            end_date = st.date_input("End Date", value=None, key="s2_wo_end")
        with c4:
            location = st.selectbox("Location", options=loc_options, index=0, key="s2_wo_location")
        with c5:
            status_ui = st.selectbox("Status", options=status_options, index=0, key="s2_wo_status")
        
        submitted = st.form_submit_button("Submit", on_click=_on_s2_wo_submit_click)
    
    if submitted or st.session_state.get("s2_wo_run_fetch"):
        st.session_state["s2_wo_run_fetch"] = False
        if not site_name or site_name == "(select)":
            early_prog_slot.empty()
            early_msg_slot.empty()
            st.error("Please select a Site Name.")
            return
        if start_date is None or end_date is None:
            early_prog_slot.empty()
            early_msg_slot.empty()
            st.error("Please select both Start Date and End Date.")
            return
        if start_date > end_date:
            early_prog_slot.empty()
            early_msg_slot.empty()
            st.error("Start Date must be on or before End Date.")
            return
        
        # Progress UX
        prog = early_prog_slot.progress(0, text="Safety First: Initializing...")
        early_msg_slot.caption("Safety First: Always verify permits and isolation before starting work.")
        _smooth_progress(prog, 0, 18, text="Validating filters...")
        _smooth_progress(prog, 18, 55, text="Fetching work orders...")
        
        loc_val = None if location in (None, "(all)") else location
        st_val = None if status_ui in (None, "(all)") else status_ui
        
        df = _fetch_work_orders(
            site_name=site_name,
            start_date=start_date,
            end_date=end_date,
            status_ui=st_val,
            location=loc_val,
        )
        
        _smooth_progress(prog, 55, 88, text="Preparing results...")
        _smooth_progress(prog, 88, 100, text="Results ready")
        early_prog_slot.empty()
        early_msg_slot.empty()
        
        # Persist results
        st.session_state["s2_wo_last_df"] = df
        st.session_state["s2_wo_last_meta"] = {
            "site_name": site_name,
            "start_date": start_date,
            "end_date": end_date,
            "status": st_val,
            "location": loc_val,
        }
        
        # Update KPIs
        if isinstance(df, pd.DataFrame) and not df.empty:
            total = int(len(df))
            c_rej = int((df["status"].astype("string").str.upper() == "REJECTED").sum())
            c_open = int((df["status"].astype("string").str.upper() == "OPEN").sum())
            c_wip = int((df["status"].astype("string").str.upper() == "WIP").sum())
            c_approved = int(df["status"].astype("string").str.upper().isin(["CLOSED", "APPROVED"]).sum())
        else:
            total = c_rej = c_open = c_wip = c_approved = 0
        
        st.session_state["s2_wo_last_kpis"] = {
            "total": total,
            "rejected": c_rej,
            "open": c_open,
            "wip": c_wip,
            "approved": c_approved,
        }
        
        if df.empty:
            st.info("No work orders found for the selected filters.")
            return
    
    # Render cached results
    df_last = st.session_state.get("s2_wo_last_df")
    if isinstance(df_last, pd.DataFrame) and not df_last.empty:
        df = df_last
        
        # KPI cards
        k = st.session_state.get("s2_wo_last_kpis") or {}
        total = int(k.get("total", 0) or 0)
        c_rej = int(k.get("rejected", 0) or 0)
        c_open = int(k.get("open", 0) or 0)
        c_wip = int(k.get("wip", 0) or 0)
        c_approved = int(k.get("approved", 0) or 0)

        st.markdown(
            f"""
            <div class="kpi-row">
              <div class="kpi-card"><div class="kpi-title">Work Orders</div><div class="kpi-value" style="color:#2563eb;">{total}</div></div>
              <div class="kpi-card"><div class="kpi-title">REJECTED</div><div class="kpi-value" style="color:#dc2626;">{c_rej}</div></div>
              <div class="kpi-card"><div class="kpi-title">OPEN</div><div class="kpi-value" style="color:#16a34a;">{c_open}</div></div>
              <div class="kpi-card"><div class="kpi-title">WIP</div><div class="kpi-value" style="color:#f97316;">{c_wip}</div></div>
              <div class="kpi-card"><div class="kpi-title">Approved</div><div class="kpi-value" style="color:#10b981;">{c_approved}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        st.divider()
        
        # Format date for display
        if "date_planned" in df.columns:
            df["date_planned"] = pd.to_datetime(df["date_planned"]).dt.strftime("%Y-%m-%d")
        
        # Table with styling
        styled = df.style.map(_highlight_status, subset=["status"]).set_table_styles(
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
    if "s2_ptw_run_fetch" not in st.session_state:
        st.session_state["s2_ptw_run_fetch"] = False

    # Date range filter + fetch (fragmented + queued) so progress mounts immediately.
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
            def _on_fetch() -> None:
                st.session_state["s2_ptw_run_fetch"] = True
            st.button("Fetch PTWs", type="primary", key="s2_fetch_ptw", on_click=_on_fetch)

        # Run fetch if requested (same click rerun), with progress mounted first
        if st.session_state.get("s2_ptw_run_fetch"):
            st.session_state["s2_ptw_run_fetch"] = False
            prog_slot = st.empty()
            prog = prog_slot.progress(0, text="Fetching submitted PTWs...")
            try:
                _smooth_progress(prog, 0, 20, text="Validating date range...")
                _smooth_progress(prog, 20, 70, text="Loading from database...")
                df = _fetch_ptw_for_s2(start_date=start_date, end_date=end_date)
                st.session_state["s2_ptw_view_df"] = df
                _smooth_progress(prog, 70, 100, text="PTWs ready")
            except Exception as e:
                st.error(f"Failed to fetch PTW requests: {e}")
                st.session_state["s2_ptw_view_df"] = pd.DataFrame()
            finally:
                prog_slot.empty()

    # Note: This function may already be running inside a fragment above.
    _filters_block()
    
    df = st.session_state.get("s2_ptw_view_df")
    
    if df is None:
        st.info("Select a date range and click 'Fetch PTWs' to load submitted permits.")
        return
    
    if df.empty:
        st.info("No submitted PTWs found for the selected date range.")
        return
    
    # Summary metrics (modern KPI cards)
    total = int(len(df))
    wip_count = int((df["status"].astype("string").str.upper() == "WIP").sum())
    approved_count = int(df["status"].astype("string").str.upper().isin(["CLOSED", "APPROVED"]).sum())
    rejected_count = int((df["status"].astype("string").str.upper() == "REJECTED").sum())

    st.markdown(
        f"""
        <div class="kpi-row">
          <div class="kpi-card"><div class="kpi-title">Total PTWs</div><div class="kpi-value" style="color:#2563eb;">{total}</div></div>
          <div class="kpi-card"><div class="kpi-title">WIP</div><div class="kpi-value" style="color:#f97316;">{wip_count}</div></div>
          <div class="kpi-card"><div class="kpi-title">Approved</div><div class="kpi-value" style="color:#10b981;">{approved_count}</div></div>
          <div class="kpi-card"><div class="kpi-title">REJECTED</div><div class="kpi-value" style="color:#dc2626;">{rejected_count}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.divider()
    
    # Render each PTW as an expander (keep active one open to prevent "reload collapse")
    for idx, row in df.iterrows():
        # Normalize IDs to string so UI state keys are stable across pandas/numpy dtypes.
        ptw_id = str(row.get("ptw_id", ""))
        work_order_id = str(row.get("permit_no", "") or "")
        site_name = row.get("site_name", "")
        status = row.get("status", "OPEN")
        created_at = row.get("created_at", "")
        form_data = row.get("form_data", {}) or {}
        date_s2_forwarded = row.get("date_s2_forwarded")
        work_location = row.get("work_location", "")
        
        # Format created_at
        if created_at:
            try:
                created_at = pd.to_datetime(created_at).strftime("%Y-%m-%d %H:%M")
            except Exception:
                pass
        
        # Check if already forwarded (disable editing)
        is_forwarded = date_s2_forwarded is not None and str(date_s2_forwarded).strip() != ""
        
        # Check if just submitted (show success state)
        # IMPORTANT: use work_order_id for the key (stable) to avoid dtype mismatches on reruns.
        # NOTE: We rely solely on the session flag, not on is_forwarded from the dataframe,
        # because the cached dataframe may be stale (not refreshed after submission).
        just_submitted_key = f"s2_wo_{work_order_id}_just_submitted"
        just_submitted = bool(st.session_state.get(just_submitted_key, False))
        
        # Status badge color
        status_color = {
            "OPEN": "green",
            "WIP": "orange",
            "APPROVED": "green",
            "CLOSED": "green",
            "REJECTED": "red",
        }.get(status, "gray")
        
        # Different label for submitted vs pending
        if just_submitted:
            expander_label = f"✅ **{work_order_id}** | {site_name} | SUBMITTED | {created_at}"
        else:
            expander_label = f"**{work_order_id}** | {site_name} | :{status_color}[{status}] | {created_at}"
        
        active = st.session_state.get("s2_active_ptw_id") == work_order_id
        with st.expander(expander_label, expanded=bool(just_submitted or active)):
            if just_submitted:
                # Show success message and download option
                _render_post_submit_view(
                    ptw_id=ptw_id,
                    work_order_id=work_order_id,
                    site_name=site_name,
                    form_data=form_data,
                    just_submitted_key=just_submitted_key,
                )
            else:
                _render_ptw_detail(
                    ptw_id=ptw_id,
                    work_order_id=work_order_id,
                    site_name=site_name,
                    work_location=work_location,
                    form_data=form_data,
                    is_forwarded=is_forwarded,
                    status=status,
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
        # Clear active PTW so all expanders collapse
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
        # Track active expander by work_order_id (stable across reruns)
        st.session_state["s2_active_ptw_id"] = work_order_id

    def _queue_autosave() -> None:
        _set_active()
        st.session_state[f"{key_prefix}autosave_pending"] = True

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

    # If autosave pending, show progress + update DB with latest widget values
    if st.session_state.get(f"{key_prefix}autosave_pending"):
        st.session_state[f"{key_prefix}autosave_pending"] = False
        prog_slot = st.empty()
        prog = prog_slot.progress(0, text="Saving changes...")
        try:
            _smooth_progress(prog, 0, 50, text="Updating PTW record...")
            updated = dict(form_data or {})
            updated["holder_name"] = st.session_state.get(f"{key_prefix}holder_name", updated.get("holder_name", ""))
            updated["isolation_required"] = st.session_state.get(
                f"{key_prefix}isolation_required", updated.get("isolation_required", "")
            )
            updated["toolbox_conducted"] = bool(
                st.session_state.get(f"{key_prefix}toolbox_conducted", updated.get("toolbox_conducted", False))
            )
            updated["s2_remarks"] = st.session_state.get(f"{key_prefix}s2_remarks", updated.get("s2_remarks", ""))
            _update_ptw_form_data(ptw_id, updated)
            _smooth_progress(prog, 50, 100, text="Saved")
        except Exception as e:
            st.error(f"Failed to save changes: {e}")
        finally:
            prog_slot.empty()
        # Continue rendering the form in the same run (prevents blank expander after autosave)

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
            updated_form["toolbox_conducted"] = bool(
                st.session_state.get(f"{key_prefix}toolbox_conducted", updated_form.get("toolbox_conducted", False))
            )
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
        toolbox_conducted = bool(
            st.session_state.get(f"{key_prefix}toolbox_conducted", (form_data or {}).get("toolbox_conducted", False))
        )
        s2_remarks = st.session_state.get(f"{key_prefix}s2_remarks", (form_data or {}).get("s2_remarks", ""))

        isolation_files_uploaded = st.session_state.get(f"{key_prefix}isolation_files", [])
        toolbox_files_uploaded = st.session_state.get(f"{key_prefix}toolbox_files", [])

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
    
    if is_forwarded or status in ("CLOSED", "APPROVED", "REJECTED"):
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
            if is_forwarded and status == "WIP":
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
                        _revoke_s2_submission(work_order_id)
                        _smooth_progress(prog, 60, 100, text="Done")
                        prog_slot.empty()
                        st.success("Submission revoked. You can now edit the PTW.")
                        # Clear the just_submitted flag so the editable form appears immediately
                        st.session_state[f"s2_wo_{work_order_id}_just_submitted"] = False
                        # Force a full re-fetch to get updated is_forwarded status
                        st.session_state.pop("s2_ptw_view_df", None)
                        st.session_state["s2_ptw_run_fetch"] = True
                        # Ensure this PTW stays expanded after refresh
                        st.session_state["s2_active_ptw_id"] = work_order_id
                        # Force rerun to show the editable form immediately
                        st.rerun()
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
    
    holder_name = st.text_input(
        "Permit Holder Name *",
        value=form_data.get("holder_name", ""),
        key=f"{key_prefix}holder_name",
        placeholder="Enter the name of the permit holder",
        on_change=_queue_autosave,
    )
    
    st.divider()
    
    # Section C: Isolation Requirement (MANDATORY)
    st.markdown("### C. Isolation Requirement (Required)")
    
    current_isolation = form_data.get("isolation_required", "")
    isolation_idx = 0 if current_isolation.upper() != "NO" else 1
    
    def _on_iso_change() -> None:
        _queue_autosave()
    isolation_required = st.radio(
        "Is Isolation Required? *",
        options=["YES", "NO"],
        index=isolation_idx,
        key=f"{key_prefix}isolation_required",
        horizontal=True,
        on_change=_on_iso_change,
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
    
    toolbox_conducted = st.checkbox(
        "Tool Box Talk Conducted *",
        value=form_data.get("toolbox_conducted", False),
        key=f"{key_prefix}toolbox_conducted",
        on_change=_queue_autosave,
    )
    
    # File upload for toolbox evidence (mandatory if checked)
    existing_toolbox_files = _list_evidence_files(work_order_id, "toolbox")
    toolbox_files_key = f"{key_prefix}toolbox_files"
    
    if toolbox_conducted:
        st.caption("Upload toolbox talk evidence (images or PDFs) - Required")
        
        # Show existing files
        if existing_toolbox_files:
            st.success(f"✓ {len(existing_toolbox_files)} toolbox file(s) already uploaded")
            with st.expander("View uploaded files"):
                for f in existing_toolbox_files:
                    st.write(f"  📎 {f['name']}")
        
        # New file upload
        toolbox_upload = st.file_uploader(
            "Upload Toolbox Talk Evidence",
            type=["pdf", "jpg", "jpeg", "png", "gif", "webp"],
            accept_multiple_files=True,
            key=f"{key_prefix}toolbox_upload",
        )
        
        # Store in session state
        st.session_state[toolbox_files_key] = toolbox_upload if toolbox_upload else []
    else:
        st.session_state[toolbox_files_key] = []
    
    st.divider()
    
    # Section E: Additional Remarks (Optional)
    st.markdown("### E. Supervisor Remarks (Optional)")
    
    s2_remarks = st.text_area(
        "Remarks / Notes",
        value=form_data.get("s2_remarks", ""),
        key=f"{key_prefix}s2_remarks",
        placeholder="Add any additional remarks or notes...",
        on_change=_queue_autosave,
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
    
    if toolbox_conducted:
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
        
        _update_work_order_s2_forwarded(work_order_id, isolation_required)
        
        # Step 4: Done
        progress.progress(100, text="Complete!")
        progress.empty()
        status_msg.empty()
        
        # Mark as just submitted - success view will render on the normal button-click rerun
        # Use work_order_id-based key (stable) so success view doesn't disappear on reruns.
        st.session_state[f"s2_wo_{work_order_id}_just_submitted"] = True
        st.session_state["s2_active_ptw_id"] = str(work_order_id)
        
        return updated_form
        
    except Exception as e:
        progress.empty()
        status_msg.empty()
        st.error(f"Failed to submit: {e}")
        st.exception(e)
        return None


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
    st.markdown("# S2 Portal")
    st.caption("PTW Review & Forwarding Stage")

    _apply_modern_tabs_css()
    
    # Initialize session state
    if "s2_ptw_view_df" not in st.session_state:
        st.session_state["s2_ptw_view_df"] = None
    
    # Top horizontal tabs
    tab1, tab2 = st.tabs(["View Work Order", "View Submitted PTW"])
    
    with tab1:
        _render_view_work_order_s2()
    
    with tab2:
        _render_view_submitted_ptw()
