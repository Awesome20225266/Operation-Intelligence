from __future__ import annotations

from typing import Optional

# ---------------------------------
# Username-based access control
# ---------------------------------
# NOTE: This is UI/data-filter driven only (not authorization / not RLS).
# It is intentionally minimal and does not change DB schemas or business logic.

ADMIN_USERNAME = "admin"

RESTRICTED_USERS = {
    "aspl",
    "gspl_gum",
    "gspl",
    "tspl",
    "nspl",
    "rspl",
    "gspl_gap",
    "pspl",
    "esepl",
}

# ------------------------------------------------------------------
# Module access control (PTW portals)
# ------------------------------------------------------------------
# These rules are UI/routing enforcement only (no DB schema changes).
# They are intended to prevent manual session_state bypass in Streamlit.

S1_ONLY_USERS = {"labhchand"}
S2_ONLY_USERS = {"durgesh"}
S3_ONLY_USERS = {"richpal"}


def is_admin(username: Optional[str]) -> bool:
    return (username or "").strip().lower() == ADMIN_USERNAME


def allowed_modules_for_user(username: Optional[str]) -> list[str]:
    """
    Returns list of allowed PTW modules:
    - admin      → ["S1","S2","S3"]
    - labhchand  → ["S1"]
    - durgesh    → ["S2"]
    - richpal    → ["S3"]
    - otherwise  → []
    """
    u = (username or "").strip().lower()

    if is_admin(u):
        return ["S1", "S2", "S3"]
    if u in S1_ONLY_USERS:
        return ["S1"]
    if u in S2_ONLY_USERS:
        return ["S2"]
    if u in S3_ONLY_USERS:
        return ["S3"]
    return []


def is_restricted_user(username: Optional[str]) -> bool:
    return (username or "").strip().lower() in RESTRICTED_USERS


def allowed_sites_for_user(username: Optional[str]) -> list[str]:
    """
    - Admin: [] means "ALL sites allowed"
    - Restricted: [username] means "ONLY this site"
    - Unknown: [] (defaults to ALL, but caller may choose to treat as restricted if desired)
    """
    u = (username or "").strip().lower()
    if is_admin(u):
        return []
    if is_restricted_user(u):
        return [u]
    return []


