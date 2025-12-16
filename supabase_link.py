from __future__ import annotations

import os
from typing import Optional

from supabase import Client, create_client


def _get_secret(name: str) -> Optional[str]:
    """
    Prefer Streamlit secrets when available; fallback to environment variables.
    This avoids hardcoding keys in the repo.
    """
    try:
        import streamlit as st  # type: ignore

        v = st.secrets.get(name)  # type: ignore[attr-defined]
        if v:
            return str(v)
    except Exception:
        pass
    v = os.getenv(name)
    return v if v else None


def get_supabase_client(*, prefer_service_role: bool = True) -> Client:
    """
    Initialize and return a Supabase client.

    Provide credentials via either:
    - Streamlit secrets: .streamlit/secrets.toml
    - Environment variables

    Required:
    - SUPABASE_URL
    - SUPABASE_ANON_KEY and/or SUPABASE_SERVICE_ROLE_KEY
    """
    url = _get_secret("SUPABASE_URL")
    if not url:
        raise ValueError("Missing SUPABASE_URL (set env var or .streamlit/secrets.toml)")

    anon_key = _get_secret("SUPABASE_ANON_KEY")
    service_key = _get_secret("SUPABASE_SERVICE_ROLE_KEY")

    key = service_key if (prefer_service_role and service_key) else anon_key
    if not key:
        raise ValueError("Missing SUPABASE key (set SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY)")

    return create_client(url, key)


if __name__ == "__main__":
    # Terminal connection check: prints 'successful' if Supabase is reachable
    # and the `zelestra_comments` table can be queried with your credentials.
    try:
        sb = get_supabase_client(prefer_service_role=True)
        sb.table("zelestra_comments").select("*").limit(1).execute()
        print("successful")
    except Exception as e:
        print(f"Connection failed: {e}")



