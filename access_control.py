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


def is_admin(username: Optional[str]) -> bool:
    return (username or "").strip().lower() == ADMIN_USERNAME


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


