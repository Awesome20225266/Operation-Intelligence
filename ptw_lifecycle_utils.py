from __future__ import annotations

from typing import Any

from supabase_link import get_supabase_client

TABLE_WORK_ORDERS = "work_orders"


def _get_all_work_order_ids_for_ptw(
    ptw_id: str, permit_no: str, form_data: dict | None
) -> list[str]:
    """
    Extract ALL linked work_order_ids for a PTW.

    Priority:
    1. form_data["work_order_ids"]
    2. permit_no split by "-"
    3. fallback to permit_no
    """
    _ = ptw_id  # signature reserved (debug/context)
    ids: list[str] = []

    if isinstance(form_data, dict):
        wo_list = form_data.get("work_order_ids")
        if isinstance(wo_list, list) and wo_list:
            ids = [str(x).strip() for x in wo_list if str(x).strip()]

    if not ids and permit_no:
        ids = [x.strip() for x in str(permit_no).split("-") if x.strip()]

    # fallback (keeps previous behavior if permit_no isn't delimited)
    if not ids and permit_no:
        ids = [str(permit_no).strip()]

    # de-dupe while keeping stable order
    seen: set[str] = set()
    out: list[str] = []
    for x in ids:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def _update_all_work_orders_lifecycle(work_order_ids: list[str], update_fields: dict[str, Any]) -> None:
    if not work_order_ids:
        return

    sb = get_supabase_client(prefer_service_role=True)

    resp = (
        sb.table(TABLE_WORK_ORDERS)
        .update(update_fields)
        .in_("work_order_id", [str(x).strip() for x in work_order_ids if str(x).strip()])
        .execute()
    )

    err = getattr(resp, "error", None)
    if err:
        raise RuntimeError(f"Atomic lifecycle update failed: {err}")

