from __future__ import annotations

import os
from typing import Optional

from supabase import Client, create_client


_SECRETS_DEBUG = str(os.getenv("ZELES_SECRETS_DEBUG", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
_CLI_SECRETS_CACHE: Optional[dict] = None
_CLI_SECRETS_SOURCE: Optional[str] = None


def _secrets_debug(msg: str) -> None:
    # Non-sensitive debug logging (presence + source only)
    if _SECRETS_DEBUG:
        print(f"[secrets] {msg}", flush=True)


def _load_cli_secrets() -> dict:
    """
    Best-effort loader for `.streamlit/secrets.toml` when running outside Streamlit.
    Never logs secret values.
    """
    global _CLI_SECRETS_CACHE, _CLI_SECRETS_SOURCE
    if _CLI_SECRETS_CACHE is not None:
        return _CLI_SECRETS_CACHE

    from pathlib import Path

    candidates = [
        # Prefer project-local secrets.toml (repo-migrated projects often run from different CWDs)
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]

    secrets: dict = {}
    source: Optional[str] = None
    for p in candidates:
        try:
            if p.exists() and p.is_file():
                try:
                    import tomllib as _toml  # py3.11+
                except Exception:  # pragma: no cover
                    import tomli as _toml  # type: ignore[import-not-found]

                txt = p.read_text(encoding="utf-8")
                secrets = dict(_toml.loads(txt) or {})
                source = str(p)
                break
        except Exception as e:
            _secrets_debug(f"secrets.toml read failed at {p}: {type(e).__name__}")
            continue

    _CLI_SECRETS_CACHE = secrets
    _CLI_SECRETS_SOURCE = source
    if source:
        _secrets_debug(f"loaded secrets.toml from {source}")
    else:
        _secrets_debug("no secrets.toml found in expected locations")
    return _CLI_SECRETS_CACHE


def _lookup_key(mapping: object, key: str) -> Optional[object]:
    if not isinstance(mapping, dict):
        return None
    if key in mapping:
        return mapping.get(key)
    for v in mapping.values():
        if isinstance(v, dict):
            found = _lookup_key(v, key)
            if found is not None:
                return found
    return None


def _get_secret(name: str) -> Optional[str]:
    """
    Prefer Streamlit secrets when available; fallback to environment variables.
    This avoids hardcoding keys in the repo.
    """
    # 1) Streamlit secrets (preferred)
    try:
        import streamlit as st  # type: ignore

        v = st.secrets.get(name)  # type: ignore[attr-defined]
        ok = bool(v and str(v).strip() != "")
        _secrets_debug(f"streamlit secrets: {name} => {'FOUND' if ok else 'NOT FOUND'}")
        if ok:
            return str(v)
    except Exception as e:
        _secrets_debug(f"streamlit secrets unavailable for {name}: {type(e).__name__}")

    # 1b) CLI compatibility: load `.streamlit/secrets.toml` directly if Streamlit isn't active
    try:
        cli_secrets = _load_cli_secrets()
        v_file = _lookup_key(cli_secrets, name)
        ok_file = bool(v_file is not None and str(v_file).strip() != "")
        _secrets_debug(
            f"secrets.toml ({_CLI_SECRETS_SOURCE or 'none'}): {name} => {'FOUND' if ok_file else 'NOT FOUND'}"
        )
        if ok_file:
            return str(v_file)
    except Exception as e:
        _secrets_debug(f"secrets.toml lookup failed for {name}: {type(e).__name__}")

    # 2) Environment variables
    v = os.getenv(name)
    ok_env = bool(v and v.strip() != "")
    _secrets_debug(f"env var: {name} => {'FOUND' if ok_env else 'NOT FOUND'}")
    return v if ok_env else None


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

    # UI/terminal hygiene: some Supabase Storage endpoints warn if the base URL
    # doesn't have a trailing slash. This does NOT change any business logic;
    # it only prevents noisy warnings in the terminal.
    if not url_str.endswith("/"):
        url_str = url_str + "/"

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



