from __future__ import annotations

from io import BytesIO

import pandas as pd

from supabase_link import get_supabase_client
from S1 import TABLE_WORK_ORDERS

# PDF manipulation for approval stamp
try:
    from PyPDF2 import PdfReader, PdfWriter  # type: ignore
    from reportlab.pdfgen import canvas  # type: ignore
    from reportlab.lib.pagesizes import A4  # type: ignore
    from reportlab.lib.colors import Color  # type: ignore

    _HAS_PDF_LIBS = True
except Exception:
    _HAS_PDF_LIBS = False


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


def inject_approval_times_into_form_data(
    form_data: dict, work_order_id: str, *, is_approved: bool = False
) -> dict:
    """
    Inject holder_datetime and issuer_datetime into form_data from work_orders.

    - Always inject holder_datetime from S2 forwarding date
    - Inject issuer_datetime only if is_approved=True
    """
    approval_times = get_ptw_approval_times(work_order_id)
    updated = dict(form_data) if form_data else {}

    if approval_times.get("holder_datetime"):
        updated["holder_datetime"] = approval_times["holder_datetime"]
    if is_approved and approval_times.get("issuer_datetime"):
        updated["issuer_datetime"] = approval_times["issuer_datetime"]

    return updated


def add_floating_approval_stamp(pdf_bytes: bytes, *, approved_on: str) -> bytes:
    """
    Add a floating APPROVED stamp overlay to every page of a PDF.

    If PDF libraries aren't available, returns pdf_bytes unchanged.
    """
    if not _HAS_PDF_LIBS:
        return pdf_bytes

    try:
        stamp_buffer = BytesIO()
        c = canvas.Canvas(stamp_buffer, pagesize=A4)

        page_width, page_height = A4
        stamp_x = page_width - 200
        stamp_y = 120
        stamp_width = 160
        stamp_height = 60

        stamp_color = Color(0.8, 0.1, 0.1, alpha=0.85)
        c.setStrokeColor(stamp_color)
        c.setLineWidth(3)
        c.rect(stamp_x, stamp_y, stamp_width, stamp_height, stroke=1, fill=0)

        c.setLineWidth(1.5)
        c.rect(stamp_x + 4, stamp_y + 4, stamp_width - 8, stamp_height - 8, stroke=1, fill=0)

        c.setFillColor(stamp_color)
        c.setFont("Helvetica-Bold", 18)
        text_x = stamp_x + stamp_width / 2
        c.drawCentredString(text_x, stamp_y + 35, "APPROVED")

        c.setFont("Helvetica", 9)
        if approved_on:
            c.drawCentredString(text_x, stamp_y + 18, f"ON: {approved_on}")

        c.save()
        stamp_buffer.seek(0)

        original_pdf = PdfReader(BytesIO(pdf_bytes))
        stamp_pdf = PdfReader(stamp_buffer)

        output = PdfWriter()
        stamp_page = stamp_pdf.pages[0]
        for i in range(len(original_pdf.pages)):
            page = original_pdf.pages[i]
            page.merge_page(stamp_page)
            output.add_page(page)

        output_buffer = BytesIO()
        output.write(output_buffer)
        output_buffer.seek(0)
        return output_buffer.read()
    except Exception:
        return pdf_bytes

