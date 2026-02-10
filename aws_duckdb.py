from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import duckdb

try:
    import streamlit as st  # type: ignore

    _HAS_STREAMLIT = True
except Exception:
    st = None  # type: ignore
    _HAS_STREAMLIT = False


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

    candidates = [
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
    Read secrets from Streamlit if available; fallback to env vars.
    Never logs secret values.
    """
    # 1) Streamlit secrets (preferred)
    try:
        import streamlit as st  # type: ignore

        v = st.secrets.get(name)  # type: ignore[attr-defined]
        ok = v is not None and str(v).strip() != ""
        _secrets_debug(f"streamlit secrets: {name} => {'FOUND' if ok else 'NOT FOUND'}")
        if ok:
            return str(v)
    except Exception as e:
        _secrets_debug(f"streamlit secrets unavailable for {name}: {type(e).__name__}")

    # 1b) CLI compatibility: load `.streamlit/secrets.toml` directly if Streamlit isn't active
    try:
        cli_secrets = _load_cli_secrets()
        v_file = _lookup_key(cli_secrets, name)
        ok_file = v_file is not None and str(v_file).strip() != ""
        _secrets_debug(
            f"secrets.toml ({_CLI_SECRETS_SOURCE or 'none'}): {name} => {'FOUND' if ok_file else 'NOT FOUND'}"
        )
        if ok_file:
            return str(v_file)
    except Exception as e:
        _secrets_debug(f"secrets.toml lookup failed for {name}: {type(e).__name__}")

    # 2) Environment variables
    v2 = os.getenv(name)
    ok_env = bool(v2 and v2.strip() != "")
    _secrets_debug(f"env var: {name} => {'FOUND' if ok_env else 'NOT FOUND'}")
    return v2 if ok_env else None


def _get_s3_etag(
    *,
    s3_bucket: str,
    s3_key: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    aws_region: str,
) -> str:
    import boto3  # type: ignore

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )
    s3_response = s3.head_object(Bucket=s3_bucket, Key=s3_key)
    return str(s3_response.get("ETag", "")).strip().strip('"')


# Cache the ETag lookup to reduce S3 round-trips on every Streamlit rerun.
if _HAS_STREAMLIT:

    @st.cache_data(ttl=600, show_spinner=False)  # type: ignore[misc]
    def _get_s3_etag_cached(
        *,
        s3_bucket: str,
        s3_key: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_region: str,
    ) -> str:
        return _get_s3_etag(
            s3_bucket=s3_bucket,
            s3_key=s3_key,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region=aws_region,
        )

else:

    def _get_s3_etag_cached(  # type: ignore[no-redef]
        *,
        s3_bucket: str,
        s3_key: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_region: str,
    ) -> str:
        return _get_s3_etag(
            s3_bucket=s3_bucket,
            s3_key=s3_key,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_region=aws_region,
        )


def _ensure_db_local(db_local: str = "master.duckdb") -> str:
    """
    Ensure DuckDB file is present locally; downloads from S3 if missing or if S3 object changed (ETag-based).
    Uses a simple lockfile to avoid parallel downloads in the same environment.
    """
    local_path = Path(db_local)
    etag_path = local_path.with_suffix(local_path.suffix + ".etag")

    aws_access_key_id = _get_secret("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = _get_secret("AWS_SECRET_ACCESS_KEY")
    aws_region = _get_secret("AWS_REGION")
    s3_bucket = _get_secret("S3_BUCKET")
    s3_key = _get_secret("S3_KEY")

    missing = [k for k, v in {
        "AWS_ACCESS_KEY_ID": aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
        "AWS_REGION": aws_region,
        "S3_BUCKET": s3_bucket,
        "S3_KEY": s3_key,
    }.items() if not v]

    if missing:
        raise RuntimeError(
            "Missing AWS/S3 secrets required to download DuckDB from S3: "
            + ", ".join(missing)
            + ". Set them in `.streamlit/secrets.toml` (local) or Streamlit Cloud Secrets."
        )

    # Get current S3 ETag (before lock to avoid unnecessary lock acquisition)
    try:
        current_s3_etag = _get_s3_etag_cached(
            s3_bucket=str(s3_bucket),
            s3_key=str(s3_key),
            aws_access_key_id=str(aws_access_key_id),
            aws_secret_access_key=str(aws_secret_access_key),
            aws_region=str(aws_region),
        )
    except Exception as e:
        # If head_object fails, fall back to checking if local file exists
        if local_path.exists() and local_path.stat().st_size > 0:
            return str(local_path)
        raise RuntimeError(f"Failed to check S3 object freshness: {e}") from e

    # Fast path: local DB exists, .etag file exists, and ETags match
    if local_path.exists() and local_path.stat().st_size > 0 and etag_path.exists():
        try:
            stored_etag = etag_path.read_text().strip()
            if stored_etag == current_s3_etag:
                return str(local_path)
        except Exception:
            # If .etag file is corrupted/unreadable, treat as "needs refresh"
            pass

    # Lockfile (best-effort)
    lock_path = local_path.with_suffix(local_path.suffix + ".lock")
    # If a previous run crashed, the lockfile can be left behind forever.
    # Treat sufficiently old locks as stale and remove them.
    # Keep this relatively small: Streamlit reruns can happen frequently, and a crash can
    # leave a 0-byte lockfile behind. If a lock is older than this, we consider it stale.
    STALE_LOCK_SECONDS = 5 * 60  # 5 minutes
    start = time.time()
    while True:
        try:
            # atomic create
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            try:
                # best-effort metadata (helps debugging)
                os.write(fd, f"pid={os.getpid()}\ncreated_at={time.time()}\n".encode("utf-8"))
            except Exception:
                pass
            os.close(fd)
            break
        except FileExistsError:
            # Stale lock cleanup
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age > STALE_LOCK_SECONDS:
                    try:
                        os.remove(str(lock_path))
                    except Exception:
                        pass
                    continue
            except Exception:
                # If we can't stat it, just fall back to waiting
                pass
            # After lock wait, check again if another process downloaded it
            if local_path.exists() and local_path.stat().st_size > 0 and etag_path.exists():
                try:
                    stored_etag = etag_path.read_text().strip()
                    if stored_etag == current_s3_etag:
                        return str(local_path)
                except Exception:
                    pass
            if time.time() - start > 120:
                raise RuntimeError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(0.2)

    try:
        # Check again after acquiring lock (another process might have downloaded)
        if local_path.exists() and local_path.stat().st_size > 0 and etag_path.exists():
            try:
                stored_etag = etag_path.read_text().strip()
                if stored_etag == current_s3_etag:
                    return str(local_path)
            except Exception:
                pass

        import boto3  # type: ignore

        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )

        # Download fresh DuckDB from S3
        tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        s3.download_file(s3_bucket, s3_key, str(tmp_path))
        os.replace(str(tmp_path), str(local_path))

        # Store the ETag after successful download
        try:
            etag_path.write_text(current_s3_etag)
        except Exception:
            # Non-fatal: ETag storage failed, but DB download succeeded
            pass

        return str(local_path)
    finally:
        try:
            os.remove(str(lock_path))
        except Exception:
            pass


def get_duckdb_connection(db_local: str = "master.duckdb") -> duckdb.DuckDBPyConnection:
    """
    Always returns a READ-ONLY DuckDB connection to the locally cached DB.
    Downloads from S3 if the file is missing locally or if the S3 object has changed (ETag-based freshness check).
    """
    path = _ensure_db_local(db_local=db_local)
    return duckdb.connect(path, read_only=True)


