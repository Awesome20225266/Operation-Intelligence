from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import duckdb


def _get_secret(name: str) -> Optional[str]:
    """
    Read secrets from Streamlit if available; fallback to env vars.
    Never logs secret values.
    """
    try:
        import streamlit as st  # type: ignore

        v = st.secrets.get(name)  # type: ignore[attr-defined]
        if v is not None and str(v).strip() != "":
            return str(v)
    except Exception:
        pass
    v2 = os.getenv(name)
    return v2 if v2 and v2.strip() != "" else None


def _ensure_db_local(db_local: str = "master.duckdb") -> str:
    """
    Ensure DuckDB file is present locally; if missing, download from S3 once.
    Uses a simple lockfile to avoid parallel downloads in the same environment.
    """
    local_path = Path(db_local)
    if local_path.exists() and local_path.stat().st_size > 0:
        return str(local_path)

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

    # Lockfile (best-effort)
    lock_path = local_path.with_suffix(local_path.suffix + ".lock")
    start = time.time()
    while True:
        try:
            # atomic create
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            if local_path.exists() and local_path.stat().st_size > 0:
                return str(local_path)
            if time.time() - start > 120:
                raise RuntimeError(f"Timed out waiting for lock: {lock_path}")
            time.sleep(0.2)

    try:
        # Check again after acquiring lock
        if local_path.exists() and local_path.stat().st_size > 0:
            return str(local_path)

        import boto3  # type: ignore

        s3 = boto3.client(
            "s3",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region,
        )

        tmp_path = local_path.with_suffix(local_path.suffix + ".tmp")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
        s3.download_file(s3_bucket, s3_key, str(tmp_path))
        os.replace(str(tmp_path), str(local_path))
        return str(local_path)
    finally:
        try:
            os.remove(str(lock_path))
        except Exception:
            pass


def get_duckdb_connection(db_local: str = "master.duckdb") -> duckdb.DuckDBPyConnection:
    """
    Always returns a READ-ONLY DuckDB connection to the locally cached DB.
    Downloads from S3 only if the file is missing locally.
    """
    path = _ensure_db_local(db_local=db_local)
    return duckdb.connect(path, read_only=True)


