from __future__ import annotations

from datetime import date
from typing import Any, Optional

import duckdb
import pandas as pd
import streamlit as st

from supabase_link import get_supabase_client
from aws_duckdb import get_duckdb_connection

try:
    # Optional (better UX). Falls back to st.dataframe if not installed.
    from st_aggrid import AgGrid  # type: ignore
    from st_aggrid.grid_options_builder import GridOptionsBuilder  # type: ignore
    from st_aggrid.shared import GridUpdateMode  # type: ignore

    _HAS_AGGRID = True
except Exception:
    AgGrid = None  # type: ignore
    GridOptionsBuilder = None  # type: ignore
    GridUpdateMode = None  # type: ignore
    _HAS_AGGRID = False


# -----------------------------
# Owner-configurable reasons list
# -----------------------------

REASONS: list[str] = sorted([
    "Disconnected String","SCB Fire","HT Panel Fire","Transformer Fire","Cable Failure","Clipping Loss","Deration Loss","Degrdation Loss","Shading Loss","Bypass Diode Failure","Bypass Diode Burnt","Soiling Loss","Passing Clouds","Reactive Power Loss","Fuse Failure","IGBT Failure","AC Breaker Issue","DC Breaker Issue","Module Damage","Temperature Loss","RISO Fault","MPPT Malfunction","Efficiency Loss","Ground Fault","Module Mismatch Loss","Array Misalignment","Tracker Failure","Inverter Fan Issue","Bifacial Factor Loss","Power Limitation","AC Current Imbalance","Vegetation Growth","Low Irradiation","Manual Trip/Stop","Trip","Overvoltage","Overcurrent","LVRT/HVRT","Land Undulation","DC Loading Pending","Inverter Card Issue","Inverter Software Fault",
    "Inverter Fire",
    "Connector Burn",
    "Grid Issue",
    "Curtailment","Unknown/No Issue","Capacitor Failure/Issue","Rainy Weather","Bad Weather","Late Wakeup","Theft","Cable Failure/Puncture","Rodent Issue","Heating Issue","Hotspot Formation","Outgoing Trip","Load Shifting","LT Panel Trip","Isolator Issue","Breaker Issue","Transposition Loss","Design Loss","IGBT Temperature Issue","Communication Issue","Bad Data","GFDI Protection Operated","Half String","Hardware Fault","Phase to Phase Fault","Phase to Ground Fault","Over Temperature Trip","Reactor Failure","Hardware Failure"
])


# -----------------------------
# DuckDB helpers (SYD source)
# -----------------------------


def _connect_ro(db_path: str) -> duckdb.DuckDBPyConnection:
    return get_duckdb_connection(db_local=db_path)


@st.cache_data(show_spinner=False)
def list_sites_from_syd(db_path: str) -> list[str]:
    con = _connect_ro(db_path)
    try:
        rows = con.execute("select distinct site_name from syd order by site_name").fetchall()
        return [r[0] for r in rows]
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def date_bounds_from_syd(db_path: str) -> tuple[Optional[date], Optional[date]]:
    con = _connect_ro(db_path)
    try:
        row = con.execute("select min(date) as dmin, max(date) as dmax from syd").fetchone()
        if not row:
            return None, None
        return row[0], row[1]
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def equipment_deviation_candidates(
    db_path: str,
    *,
    site_name: str,
    start_date: date,
    end_date: date,
    threshold: float,
) -> pd.DataFrame:
    """
    Returns equipment candidates for dropdown with their deviation (%):
    - Single day: deviation = syd_percent*100 for that date
    - Range: deviation = avg(syd_percent*100) over the range
    Only returns equipment with deviation < threshold.
    """
    con = _connect_ro(db_path)
    try:
        if start_date == end_date:
            return con.execute(
                """
                select
                  equipment_name,
                  syd_percent * 100.0 as syd_dev_pct
                from syd
                where site_name = ?
                  and date = ?
                  and (syd_percent * 100.0) < ?
                order by syd_dev_pct asc, equipment_name asc
                """,
                [site_name, start_date, float(threshold)],
            ).fetchdf()

        return con.execute(
            """
            select
              equipment_name,
              avg(syd_percent * 100.0) as syd_dev_pct
            from syd
            where site_name = ?
              and date between ? and ?
            group by 1
            having avg(syd_percent * 100.0) < ?
            order by syd_dev_pct asc, equipment_name asc
            """,
            [site_name, start_date, end_date, float(threshold)],
        ).fetchdf()
    finally:
        con.close()


# -----------------------------
# Supabase helpers (comments store)
# -----------------------------


def _to_iso(d: date) -> str:
    return d.isoformat()


@st.cache_data(show_spinner=False)
def fetch_comments(limit: int = 500) -> pd.DataFrame:
    sb = get_supabase_client(prefer_service_role=True)
    resp = sb.table("zelestra_comments").select("*").order("created_at", desc=True).limit(int(limit)).execute()
    rows = resp.data or []
    return pd.DataFrame(rows)


def fetch_comments_live(
    *,
    site_name: Optional[str] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    equipment_name: Optional[str] = None,
    limit: int = 500,
) -> pd.DataFrame:
    """
    Live fetch from Supabase (no caching) for viewing/filtering.

    Date filter uses overlap semantics:
      comment.start_date <= end_date AND comment.end_date >= start_date
    """
    sb = get_supabase_client(prefer_service_role=True)
    q = sb.table("zelestra_comments").select("*").order("created_at", desc=True).limit(int(limit))

    if site_name:
        q = q.eq("site_name", site_name)

    if start_date and end_date:
        q = q.lte("start_date", end_date.isoformat()).gte("end_date", start_date.isoformat())
    elif start_date and not end_date:
        q = q.gte("end_date", start_date.isoformat())
    elif end_date and not start_date:
        q = q.lte("start_date", end_date.isoformat())

    # Equipment filter: best-effort for array/json column
    if equipment_name:
        try:
            q = q.contains("equipment_names", [equipment_name])
        except Exception:
            # Fallback: no equipment filter if backend column type doesn't support contains
            pass

    resp = q.execute()
    rows = resp.data or []
    return pd.DataFrame(rows)


def _clear_comments_cache() -> None:
    fetch_comments.clear()


def insert_comment(payload: dict[str, Any]) -> None:
    sb = get_supabase_client(prefer_service_role=True)
    sb.table("zelestra_comments").insert(payload).execute()
    _clear_comments_cache()


def update_comment(comment_id: Any, payload: dict[str, Any]) -> None:
    sb = get_supabase_client(prefer_service_role=True)
    sb.table("zelestra_comments").update(payload).eq("id", comment_id).execute()
    _clear_comments_cache()


def _stringify_error(e: Exception) -> str:
    try:
        return str(e)
    except Exception:
        return repr(e)


def _needs_int_deviation(err: str) -> bool:
    """
    Some Supabase schemas define `deviation` as INTEGER. In that case, inserting
    a float like -10.0869 fails with 22P02.
    """
    s = err.lower()
    return ("22p02" in s and "integer" in s) or ("invalid input syntax for type integer" in s)


def _write_comment_with_fallback(*, comment_id: Any | None, payload: dict[str, Any], deviation_value: float) -> None:
    """
    Write to Supabase. If schema expects INTEGER deviation, retry with rounded int.
    """
    try:
        if comment_id is None:
            insert_comment(payload)
        else:
            update_comment(comment_id, payload)
        return
    except Exception as e:
        err = _stringify_error(e)
        if _needs_int_deviation(err):
            payload2 = dict(payload)
            payload2["deviation"] = int(round(float(deviation_value)))
            if comment_id is None:
                insert_comment(payload2)
            else:
                update_comment(comment_id, payload2)
            return
        raise


# -----------------------------
# UI
# -----------------------------


def _format_equipment_label(equipment_name: str, dev: float) -> str:
    # Keep the required "deviation_equipment" shape, but make it more readable
    return f"{dev:.2f}%_{equipment_name}"


def _reset_comment_form_state(*, default_site: Optional[str], dmin: date, dmax: date) -> None:
    """
    Clear all fields after add/update, and exit edit mode.
    """
    st.session_state["comments_edit_id"] = None
    st.session_state["comments_edit_row"] = None
    # IMPORTANT: do NOT directly mutate widget-backed keys after instantiation.
    # Set a pending-reset flag; it will be applied on the next rerun BEFORE widgets are created.
    st.session_state["ac_pending_reset"] = {
        "default_site": default_site,
        "dmin": dmin,
        "dmax": dmax,
    }


def _ensure_comment_form_state(*, dmin: date, dmax: date) -> None:
    """
    Initialize widget keys once so we can reliably clear them later.
    """
    st.session_state.setdefault("ac_site", None)
    st.session_state.setdefault("ac_threshold", -3.0)
    st.session_state.setdefault("ac_from", None)
    st.session_state.setdefault("ac_to", None)
    st.session_state.setdefault("ac_equipment_labels", [])
    st.session_state.setdefault("ac_reasons", [])
    st.session_state.setdefault("ac_remarks", "")
    st.session_state.setdefault("ac_pending_reset", None)


def _render_aggrid_table(df: pd.DataFrame, *, key: str, height: int = 380) -> None:
    """
    Render a filterable, hover-highlighted table.
    Falls back to st.dataframe if AG Grid isn't available.
    """
    if df is None or df.empty:
        st.info("No comments found.")
        return

    if not _HAS_AGGRID:
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)
        return

    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(sortable=True, filter=True, resizable=True, wrapText=True, autoHeight=True)
    gb.configure_grid_options(domLayout="normal")
    grid_options = gb.build()

    custom_css = {
        ".ag-row-hover": {"background-color": "rgba(255, 179, 0, 0.14) !important"},
        ".ag-theme-balham": {"font-family": "Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif"},
        ".ag-header-cell-label": {"font-weight": "700"},
    }

    AgGrid(  # type: ignore[misc]
        df,
        gridOptions=grid_options,
        theme="balham",
        height=height,
        fit_columns_on_grid_load=True,
        allow_unsafe_jscode=False,
        custom_css=custom_css,
        enable_enterprise_modules=False,
        update_mode=GridUpdateMode.NO_UPDATE,  # type: ignore[union-attr]
        key=key,
    )


def render(db_path: str) -> None:
    st.markdown("## Add Comments")
    st.caption("Add deviation-based comments for underperforming equipment to guide operations.")

    sites = list_sites_from_syd(db_path)
    if not sites:
        st.error("No sites found in `syd`. Load data first.")
        return

    dmin, dmax = date_bounds_from_syd(db_path)
    if not dmin or not dmax:
        st.error("No dates found in `syd`.")
        return

    st.session_state.setdefault("comments_edit_id", None)
    st.session_state.setdefault("comments_edit_row", None)

    edit_row = st.session_state.get("comments_edit_row") or {}
    is_edit_mode = bool(st.session_state.get("comments_edit_id"))

    _ensure_comment_form_state(dmin=dmin, dmax=dmax)

    # Apply pending reset BEFORE widgets are instantiated (fixes session_state modification error).
    pending = st.session_state.get("ac_pending_reset")
    if isinstance(pending, dict):
        st.session_state["ac_site"] = None
        st.session_state["ac_threshold"] = -3.0
        st.session_state["ac_from"] = None
        st.session_state["ac_to"] = None
        st.session_state["ac_equipment_labels"] = []
        st.session_state["ac_reasons"] = []
        st.session_state["ac_remarks"] = ""
        st.session_state["ac_pending_reset"] = None
        st.session_state["_ac_prefilled"] = False
        st.session_state["_ac_equipment_prefilled"] = False

    # If a row is loaded for editing, prefill widget state once (and keep it stable).
    if is_edit_mode and edit_row and not st.session_state.get("_ac_prefilled", False):
        try:
            st.session_state["ac_site"] = str(edit_row.get("site_name") or sites[0])
            st.session_state["ac_threshold"] = float(edit_row.get("threshold") or 2.0)
            st.session_state["ac_from"] = date.fromisoformat(str(edit_row.get("start_date"))) if edit_row.get("start_date") else dmin
            st.session_state["ac_to"] = date.fromisoformat(str(edit_row.get("end_date"))) if edit_row.get("end_date") else dmax
            st.session_state["ac_reasons"] = list(edit_row.get("reasons") or [])
            st.session_state["ac_remarks"] = str(edit_row.get("remarks") or "")
        except Exception:
            pass
        st.session_state["_ac_prefilled"] = True
    if not is_edit_mode:
        st.session_state["_ac_prefilled"] = False

    if is_edit_mode:
        st.info("Edit mode: update fields and click **Update**. (Ctrl+Enter will NOT submit.)")

    c1, c2, c3 = st.columns([2.4, 1.2, 2.4])
    with c1:
        site_name = st.selectbox("Site Name", options=["(select)", *sites], index=0 if st.session_state.get("ac_site") is None else (sites.index(st.session_state["ac_site"]) + 1 if st.session_state["ac_site"] in sites else 0), key="ac_site")
        if site_name == "(select)":
            site_name = None
    with c2:
        threshold = st.number_input(
            "Threshold (%)",
            min_value=-100.0,
            max_value=100.0,
            value=st.session_state.get("ac_threshold") if st.session_state.get("ac_threshold") is not None else -3.0,
            step=0.1,
            key="ac_threshold",
            help="Equipment with SYD deviation (%) below this value will appear in the equipment list.",
        )
    with c3:
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            start_date = st.date_input("From", value=st.session_state.get("ac_from"), min_value=dmin, max_value=dmax, key="ac_from")
        with dcol2:
            end_date = st.date_input("To", value=st.session_state.get("ac_to"), min_value=dmin, max_value=dmax, key="ac_to")

    if start_date is not None and end_date is not None and start_date > end_date:
        st.error("From date cannot be after To date.")

    # Equipment candidates based on site/date range/threshold (only if all required fields are filled)
    cand = pd.DataFrame()
    if site_name and start_date and end_date and threshold is not None:
        try:
            cand = equipment_deviation_candidates(
                db_path,
                site_name=str(site_name),
                start_date=start_date,
                end_date=end_date,
                threshold=float(threshold),
            )
        except Exception as e:
            st.error(f"Failed to compute equipment list from `syd`: {e}")
    elif not site_name or not start_date or not end_date or threshold is None:
        st.caption("Fill Site Name, Threshold, and Date Range to see equipment list.")

    label_to_equipment: dict[str, str] = {}
    label_to_dev: dict[str, float] = {}
    if not cand.empty:
        for _, r in cand.iterrows():
            eq = str(r["equipment_name"])
            dev = float(r["syd_dev_pct"])
            lbl = _format_equipment_label(eq, dev)
            label_to_equipment[lbl] = eq
            label_to_dev[lbl] = dev
    else:
        st.caption("No underperforming equipment found for this site/date/threshold. Try adjusting threshold or dates.")

    # If edit row has equipment_names, map them to current dropdown labels (best effort).
    if is_edit_mode and edit_row and not st.session_state.get("_ac_equipment_prefilled", False):
        try:
            desired = [str(x) for x in (edit_row.get("equipment_names") or [])]
            selected_labels: list[str] = []
            for eq in desired:
                matches = [l for l in label_to_equipment.keys() if l.endswith(f"_{eq}") or l.endswith(f"%_{eq}")]
                if matches:
                    selected_labels.append(matches[0])
            st.session_state["ac_equipment_labels"] = selected_labels
        except Exception:
            pass
        st.session_state["_ac_equipment_prefilled"] = True
    if not is_edit_mode:
        st.session_state["_ac_equipment_prefilled"] = False

    equipment_labels = st.multiselect(
        "Equipment Name (auto-generated from SYD)",
        options=list(label_to_equipment.keys()),
        key="ac_equipment_labels",
        help="Only equipment where deviation% < threshold are listed.",
    )

    reasons = st.multiselect("Reason(s)", options=REASONS, key="ac_reasons")
    remarks = st.text_area(
        "Remarks",
        key="ac_remarks",
        placeholder="Write a short operational note (what happened / what you observed / next steps)...",
        height=120,
    )

    # Friendly preview
    if equipment_labels:
        devs = [float(label_to_dev.get(lbl, 0.0)) for lbl in equipment_labels]
        avg_dev = float(sum(devs) / max(len(devs), 1))
        st.caption(f"Selected equipment: {len(equipment_labels)} | Average deviation: {avg_dev:.2f}%")

    b1, b3 = st.columns([1.6, 8.4])
    with b1:
        do_save = st.button("Update" if is_edit_mode else "Add Comment", type="primary", use_container_width=True)
    with b3:
        if is_edit_mode:
            st.caption("Update will modify the selected Supabase record. Use 'Clear edit mode' below to start a new comment.")
        else:
            st.caption("Add Comment saves a new Supabase record. Ctrl+Enter only adds a new line in remarks.")
    # No separate Clear button as requested (auto-clear after submit/update).

    if do_save:
        if start_date is None or end_date is None:
            st.error("Please select both From and To dates.")
            return
        if start_date > end_date:
            st.error("Fix date range first.")
            return
        if not equipment_labels:
            st.error("Select at least one equipment.")
            return

        equipment_names = [label_to_equipment[lbl] for lbl in equipment_labels]
        deviations = [float(label_to_dev.get(lbl, 0.0)) for lbl in equipment_labels]
        deviation_value = float(sum(deviations) / max(len(deviations), 1))

        payload: dict[str, Any] = {
            "site_name": str(site_name),
            "start_date": _to_iso(start_date),
            "end_date": _to_iso(end_date),
            "equipment_names": equipment_names,
            "reasons": reasons,
            "remarks": remarks,
            # Store deviation as numeric (Supabase should handle float, but ensure it's a number not string)
            "deviation": round(deviation_value, 6),  # Round to avoid precision issues
        }
        # Note: threshold is not stored in Supabase as per user requirement

        try:
            comment_id = st.session_state.get("comments_edit_id")
            _write_comment_with_fallback(comment_id=comment_id, payload=payload, deviation_value=deviation_value)
            if comment_id is None:
                st.success("Comment saved successfully")
            else:
                st.success("Comment updated successfully")
            # Clear all fields after add/update as requested
            _reset_comment_form_state(default_site=None, dmin=dmin, dmax=dmax)
            st.rerun()
        except Exception as e:
            st.error(f"Supabase write failed: {e}")

    # Edit picker (moved BELOW submit form as requested)
    st.write("")
    with st.expander("Edit existing comment", expanded=False):
        try:
            existing_for_edit = fetch_comments_live(limit=500)
        except Exception as e:
            st.error(f"Failed to fetch Supabase comments: {e}")
            existing_for_edit = pd.DataFrame()

        if existing_for_edit.empty:
            st.info("No comments found in Supabase.")
            return

        if "id" not in existing_for_edit.columns:
            st.warning("Supabase rows have no `id` column; update flow requires a primary key.")
            return

        def _equip_label(v: Any) -> str:
            if isinstance(v, list):
                if not v:
                    return ""
                return "+".join(str(x) for x in v[:3]) + ("…" if len(v) > 3 else "")
            return str(v or "")

        def _label_row(r: pd.Series) -> str:
            sid = str(r.get("site_name", "") or "")
            sd = str(r.get("start_date", "") or "")
            ed = str(r.get("end_date", "") or "")
            eq = _equip_label(r.get("equipment_names"))
            try:
                dv = float(r.get("deviation"))
                dv_s = f"{dv:.2f}%"
            except Exception:
                dv_s = str(r.get("deviation", "") or "")
            return f"{sid} | {sd} → {ed} | {eq}_{dv_s}"

        labels = [_label_row(existing_for_edit.iloc[i]) for i in range(len(existing_for_edit))]
        label_to_idx = {labels[i]: i for i in range(len(labels))}
        picked = st.selectbox("Select a comment to load into the form above", options=["(none)", *labels], index=0, key="ac_edit_pick")

        c_actions, c_note = st.columns([1.6, 8.4], vertical_alignment="center")
        with c_actions:
            if st.button("Load", use_container_width=True, disabled=(picked == "(none)"), key="ac_edit_load"):
                row = existing_for_edit.iloc[label_to_idx[picked]].to_dict()
                st.session_state["comments_edit_id"] = row.get("id")
                st.session_state["comments_edit_row"] = row
                st.rerun()
            if st.button("Clear edit mode", use_container_width=True, disabled=not bool(st.session_state.get("comments_edit_id")), key="ac_edit_clear"):
                _reset_comment_form_state(default_site=None, dmin=dmin, dmax=dmax)
                st.rerun()
        with c_note:
            st.caption("Pick a comment, click Load to prefill. Clear exits edit mode.")

    st.write("")
    with st.expander("Existing Comments (table)", expanded=False):
        # Filters (only affect this table)
        f1, f2, f3, f4 = st.columns([2.2, 2.0, 2.0, 3.8], vertical_alignment="bottom")
        with f1:
            f_site = st.selectbox("Filter: Site Name", options=["(all)", *sites], index=0, key="ac_exist_site")
            if f_site == "(all)":
                f_site = None
        with f2:
            f_from = st.date_input("Filter: From", value=None, min_value=dmin, max_value=dmax, key="ac_exist_from")
        with f3:
            f_to = st.date_input("Filter: To", value=None, min_value=dmin, max_value=dmax, key="ac_exist_to")

        # Fetch live (no cache)
        try:
            base = fetch_comments_live(site_name=f_site, start_date=f_from, end_date=f_to, limit=500)
        except Exception as e:
            st.error(f"Failed to fetch Supabase comments: {e}")
            base = pd.DataFrame()

        # Build equipment options from current result set
        eq_opts: list[str] = []
        if not base.empty and "equipment_names" in base.columns:
            vals: set[str] = set()
            for v in base["equipment_names"].tolist():
                if isinstance(v, list):
                    vals.update(str(x) for x in v if str(x).strip())
                elif v:
                    vals.add(str(v))
            eq_opts = sorted(vals)

        with f4:
            f_eq = st.selectbox("Filter: Equipment Name", options=["(all)", *eq_opts], index=0, key="ac_exist_eq")
            if f_eq == "(all)":
                f_eq = None

        if f_eq:
            # apply equipment filter via a second live query (no cache)
            try:
                base = fetch_comments_live(site_name=f_site, start_date=f_from, end_date=f_to, equipment_name=f_eq, limit=500)
            except Exception as e:
                st.error(f"Failed to fetch Supabase comments: {e}")
                base = pd.DataFrame()

        if base.empty:
            st.info("No comments found for the selected filters.")
            return

        show = base.copy()
        col_order = [
            "start_date",
            "end_date",
            "site_name",
            "deviation",
            "equipment_names",
            "reasons",
            "remarks",
        ]
        cols = [c for c in col_order if c in show.columns] + [c for c in show.columns if c not in col_order]
        show = show[cols]
        if "deviation" in show.columns:
            show["deviation"] = pd.to_numeric(show["deviation"], errors="coerce")
        if "equipment_names" in show.columns:
            show["equipment_names"] = show["equipment_names"].apply(
                lambda x: ", ".join(str(v) for v in x) if isinstance(x, list) else (str(x) if x is not None else "")
            )
        if "reasons" in show.columns:
            show["reasons"] = show["reasons"].apply(
                lambda x: ", ".join(str(v) for v in x) if isinstance(x, list) else (str(x) if x is not None else "")
            )
        _render_aggrid_table(show, key="ac_existing_comments", height=380)





