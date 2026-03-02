from __future__ import annotations

import os
from datetime import datetime
from io import BytesIO
from typing import Any

from PIL import Image
from PyPDF2 import PdfReader, PdfWriter  # type: ignore
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from supabase_link import get_supabase_client
from S1 import _download_template_from_supabase, build_doc_data, generate_ptw_pdf
from ptw_approval_utils import get_ptw_approval_times

EVIDENCE_BUCKET = "ptw-evidence"


def _get_content_type(ext: str) -> str:
    content_types = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }
    return content_types.get(ext.lower(), "application/octet-stream")


def list_evidence_files(work_order_id: str, evidence_type: str) -> list[dict]:
    """
    List evidence files for a work order.

    Returns list of dicts with 'name' and 'path' keys.
    """
    sb = get_supabase_client(prefer_service_role=True)
    try:
        folder_path = f"{work_order_id}/{evidence_type}"
        resp = sb.storage.from_(EVIDENCE_BUCKET).list(folder_path)
        if not resp:
            return []

        files: list[dict] = []
        for item in resp:
            if isinstance(item, dict) and item.get("name"):
                file_path = f"{folder_path}/{item['name']}"
                files.append({"name": item["name"], "path": file_path})
        return files
    except Exception:
        return []


def download_evidence_file(file_path: str) -> bytes | None:
    sb = get_supabase_client(prefer_service_role=True)
    try:
        resp = sb.storage.from_(EVIDENCE_BUCKET).download(file_path)
        return resp
    except Exception:
        return None


def _create_attachments_page(
    isolation_files: list[dict],
    toolbox_files: list[dict],
    work_order_id: str,
) -> bytes | None:
    if not isolation_files and not toolbox_files:
        return None

    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    max_width = width - 120

    styles = getSampleStyleSheet()
    normal_style = styles["Normal"]

    def draw_wrapped_text(text: str, x: float, y: float, max_w: float) -> float:
        text_esc = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        p = Paragraph(text_esc, normal_style)
        _w, h = p.wrap(max_w, 100)
        p.drawOn(c, x, y - h)
        return y - h - 5

    y_position = height - 50

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_position, "EVIDENCE ATTACHMENTS")
    y_position -= 10

    c.setFont("Helvetica", 10)
    c.drawString(50, y_position, f"Work Order: {work_order_id}")
    y_position -= 5
    c.drawString(50, y_position, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    y_position -= 30

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

            if y_pos < 150:
                c.showPage()
                y_pos = height - 50

            ext = os.path.splitext(file_name)[1].lower()
            is_image = ext in [".jpg", ".jpeg", ".png", ".gif", ".webp"]

            if is_image:
                file_bytes = download_evidence_file(file_path)
                if file_bytes:
                    try:
                        img = Image.open(BytesIO(file_bytes))

                        max_w, max_h = 200, 150
                        img.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)

                        if img.mode in ("RGBA", "P"):
                            img = img.convert("RGB")

                        img_buffer = BytesIO()
                        img.save(img_buffer, format="JPEG", quality=85)
                        img_buffer.seek(0)

                        img_reader = ImageReader(img_buffer)
                        c.drawImage(
                            img_reader,
                            50,
                            y_pos - img.height,
                            width=img.width,
                            height=img.height,
                        )

                        c.setFont("Helvetica", 9)
                        y_pos = draw_wrapped_text(
                            f"{idx + 1}. {file_name}",
                            50,
                            y_pos - img.height - 15,
                            max_width,
                        )
                        y_pos -= 20
                    except Exception:
                        c.setFont("Helvetica", 10)
                        y_pos = draw_wrapped_text(
                            f"{idx + 1}. {file_name} (image)",
                            70,
                            y_pos,
                            max_width,
                        )
                        y_pos -= 15
                else:
                    c.setFont("Helvetica", 10)
                    y_pos = draw_wrapped_text(
                        f"{idx + 1}. {file_name} (image - unable to load)",
                        70,
                        y_pos,
                        max_width,
                    )
                    y_pos -= 15
            else:
                c.setFont("Helvetica", 10)
                y_pos = draw_wrapped_text(f"{idx + 1}. {file_name}", 70, y_pos, max_width)
                y_pos -= 15

        y_pos -= 10
        return y_pos

    y_position = add_section("ISOLATION EVIDENCE", isolation_files, y_position)
    y_position = add_section("TOOLBOX TALK EVIDENCE", toolbox_files, y_position)

    c.save()
    buffer.seek(0)
    return buffer.read()


def _merge_pdfs(main_pdf: bytes, attachments_pdf: bytes | None) -> bytes:
    if not attachments_pdf:
        return main_pdf

    try:
        writer = PdfWriter()

        main_reader = PdfReader(BytesIO(main_pdf))
        for page in main_reader.pages:
            writer.add_page(page)

        attachments_reader = PdfReader(BytesIO(attachments_pdf))
        for page in attachments_reader.pages:
            writer.add_page(page)

        output = BytesIO()
        writer.write(output)
        output.seek(0)
        return output.read()
    except Exception:
        return main_pdf


def generate_ptw_pdf_with_attachments(
    *,
    form_data: dict,
    work_order_id: str,
    progress_callback: Any | None = None,
) -> bytes:
    """
    Generate PTW PDF with evidence attachments on the last page.

    IMPORTANT: This function intentionally does NOT apply any approval stamp.
    Callers (S3, or S2 for already-approved PTWs) must stamp AFTER attachments merge.
    """
    updated_form_data = dict(form_data) if form_data else {}

    # Keep approval timestamps consistent with lifecycle engine (source of truth = work_orders)
    # - Always inject holder_datetime if available
    # - Inject issuer_datetime only if S3-approved
    try:
        approval_times = get_ptw_approval_times(work_order_id)
        if approval_times.get("holder_datetime"):
            updated_form_data["holder_datetime"] = approval_times["holder_datetime"]
        if approval_times.get("date_s3_approved_raw") and approval_times.get("issuer_datetime"):
            updated_form_data["issuer_datetime"] = approval_times["issuer_datetime"]
    except Exception:
        pass

    # Template placeholders expect "signature" fields; derive from names when missing.
    holder_name = updated_form_data.get("holder_name") or updated_form_data.get("permit_holder_name") or ""
    issuer_name = updated_form_data.get("issuer_name") or updated_form_data.get("permit_issuer_name") or ""
    if holder_name and not updated_form_data.get("holder_signature"):
        updated_form_data["holder_signature"] = holder_name
    if issuer_name and not updated_form_data.get("issuer_signature"):
        updated_form_data["issuer_signature"] = issuer_name

    if progress_callback:
        progress_callback(10, "Downloading template...")

    template_bytes = _download_template_from_supabase()

    if progress_callback:
        progress_callback(30, "Generating PTW document...")

    doc_data = build_doc_data(updated_form_data)
    main_pdf = generate_ptw_pdf(template_bytes, doc_data, progress_callback=progress_callback)

    if progress_callback:
        progress_callback(55, "Fetching evidence files...")

    isolation_files = list_evidence_files(work_order_id, "isolation")
    toolbox_files = list_evidence_files(work_order_id, "toolbox")

    if progress_callback:
        progress_callback(75, "Creating attachments page...")

    attachments_pdf = _create_attachments_page(isolation_files, toolbox_files, work_order_id)

    if progress_callback:
        progress_callback(90, "Merging documents...")

    final_pdf = _merge_pdfs(main_pdf, attachments_pdf)

    if progress_callback:
        progress_callback(100, "Complete")

    return final_pdf

