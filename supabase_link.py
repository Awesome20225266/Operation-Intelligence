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
        raise ValueError(
            "Missing SUPABASE_URL. Please configure it in:\n"
            "- Streamlit secrets: .streamlit/secrets.toml (add SUPABASE_URL = 'https://your-project.supabase.co')\n"
            "- Or environment variable: set SUPABASE_URL=..."
        )

    # Validate URL format
    url_str = str(url).strip()
    if not url_str.startswith(('http://', 'https://')):
        raise ValueError(
            f"Invalid SUPABASE_URL format: '{url_str}'. "
            "URL must start with http:// or https:// (e.g., https://your-project.supabase.co)"
        )

    anon_key = _get_secret("SUPABASE_ANON_KEY")
    service_key = _get_secret("SUPABASE_SERVICE_ROLE_KEY")

    key = service_key if (prefer_service_role and service_key) else anon_key
    if not key:
        raise ValueError(
            "Missing SUPABASE key. Please configure one of:\n"
            "- SUPABASE_SERVICE_ROLE_KEY (recommended for auth)\n"
            "- SUPABASE_ANON_KEY\n"
            "Set in .streamlit/secrets.toml or as environment variable."
        )

    try:
        return create_client(url_str, key)
    except Exception as e:
        error_msg = str(e)
        if "getaddrinfo failed" in error_msg or "11001" in error_msg:
            raise ConnectionError(
                f"Failed to connect to Supabase. DNS resolution error for URL: {url_str}\n\n"
                "Possible causes:\n"
                "1. Check your internet connection\n"
                "2. Verify SUPABASE_URL is correct (format: https://xxxxx.supabase.co)\n"
                "3. Check firewall/proxy settings blocking the connection\n"
                "4. If using a VPN, try disconnecting it\n\n"
                f"Original error: {error_msg}"
            ) from e
        elif "Invalid API key" in error_msg or "401" in error_msg:
            raise ValueError(
                f"Invalid Supabase API key. Please verify your SUPABASE_SERVICE_ROLE_KEY or SUPABASE_ANON_KEY.\n"
                f"Original error: {error_msg}"
            ) from e
        else:
            raise ConnectionError(
                f"Failed to connect to Supabase at {url_str}\n"
                f"Error: {error_msg}\n\n"
                "Please check:\n"
                "- SUPABASE_URL is correct and accessible\n"
                "- API keys are valid\n"
                "- Network connection is stable"
            ) from e


if __name__ == "__main__":
    # Terminal connection check: prints 'successful' if Supabase is reachable
    # and the `zelestra_comments` table can be queried with your credentials.
    try:
        sb = get_supabase_client(prefer_service_role=True)
        sb.table("zelestra_comments").select("*").limit(1).execute()
        print("successful")
    except Exception as e:
        print(f"Connection failed: {e}")



