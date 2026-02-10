from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import boto3  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]
from boto3.s3.transfer import TransferConfig  # type: ignore[import-untyped]

import store_data_table_duckdb
import design_array_injestor
import apollo_injestor

# RE-Connect pipeline imports
sys.path.insert(0, str(Path(__file__).parent / "Web Scraping_RE_Connect"))
_reconnect_import_error = None
try:
    import reconnect_downloads  # type: ignore[import-untyped]
    import reconnect_inject  # type: ignore[import-untyped]
except ImportError as e:
    # Graceful degradation if RE-Connect modules not available
    reconnect_downloads = None  # type: ignore[assignment]
    reconnect_inject = None  # type: ignore[assignment]
    _reconnect_import_error = e


_SECRETS_DEBUG = str(os.getenv("ZELES_SECRETS_DEBUG", "")).strip().lower() in {"1", "true", "yes", "y", "on"}
_CLI_SECRETS_CACHE: Optional[dict] = None
_CLI_SECRETS_SOURCE: Optional[str] = None


def _secrets_debug(msg: str) -> None:
    # Non-sensitive debug logging (presence + source only)
    if _SECRETS_DEBUG:
        print(f"[secrets] {msg}", file=sys.stderr, flush=True)


def _load_cli_secrets() -> dict:
    """
    Best-effort loader for `.streamlit/secrets.toml` when running outside Streamlit.

    This is a compatibility shim for CLI runs so the secrets resolution order remains:
      1) Streamlit secrets (st.secrets, if available)
      2) `.streamlit/secrets.toml` (treated as Streamlit secrets source for CLI)
      3) Environment variables

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
    """
    Lookup a key in a possibly-nested TOML dict. Returns the first match found.
    """
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
    Never prints secret values.
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


def upload_duckdb_to_s3(*, db_path: Path, show_progress: bool = True) -> None:
    if not db_path.exists() or db_path.stat().st_size == 0:
        raise FileNotFoundError(f"DuckDB not found or empty: {db_path}")

    # Allow separate creds for uploading if your dashboard creds are intentionally read-only.
    aws_access_key_id = _get_secret("AWS_ACCESS_KEY_ID_UPLOAD") or _get_secret("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = _get_secret("AWS_SECRET_ACCESS_KEY_UPLOAD") or _get_secret("AWS_SECRET_ACCESS_KEY")
    aws_region = _get_secret("AWS_REGION")
    s3_bucket = _get_secret("S3_BUCKET")
    s3_key = _get_secret("S3_KEY")

    missing = [k for k, v in {
        "AWS_ACCESS_KEY_ID(_UPLOAD)": aws_access_key_id,
        "AWS_SECRET_ACCESS_KEY(_UPLOAD)": aws_secret_access_key,
        "AWS_REGION": aws_region,
        "S3_BUCKET": s3_bucket,
        "S3_KEY": s3_key,
    }.items() if not v]
    if missing:
        raise RuntimeError(
            "Missing AWS/S3 secrets required to upload DuckDB to S3: "
            + ", ".join(missing)
            + ". Set them in env vars or in `.streamlit/secrets.toml`."
        )

    s3 = boto3.client(
        "s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=aws_region,
    )

    # IMPORTANT:
    # boto3's upload_file() will use multipart uploads for larger files, which needs extra IAM actions.
    # To keep permissions simpler (only s3:PutObject), we force a single PUT here.
    try:
        file_size = int(db_path.stat().st_size)

        class _Progress:
            def __init__(self, total: int) -> None:
                self.total = max(1, int(total))
                self.seen = 0
                self._last_pct = -1

            def __call__(self, bytes_amount: int) -> None:
                self.seen += int(bytes_amount)
                pct = int((self.seen / self.total) * 100)
                if pct != self._last_pct:
                    self._last_pct = pct
                    mb_done = self.seen / (1024 * 1024)
                    mb_total = self.total / (1024 * 1024)
                    bar_len = 28
                    filled = int(bar_len * (pct / 100))
                    bar = ("#" * filled) + ("-" * (bar_len - filled))
                    print(f"\rUploading to S3: [{bar}] {pct:3d}% ({mb_done:,.1f} / {mb_total:,.1f} MB)", end="", flush=True)

            def finish(self) -> None:
                print("", flush=True)  # newline

        progress = _Progress(total=file_size) if show_progress else None

        # Force a single PUT (no multipart) while still enabling per-chunk callbacks.
        # Setting multipart_threshold > file size guarantees the TransferManager uses PutObject.
        # NOTE: multipart_chunksize is still validated even when multipart is not used.
        min_chunk = 8 * 1024 * 1024  # 8MB (>= S3 5MB minimum)
        chunk_size = max(min_chunk, file_size + 1)
        cfg = TransferConfig(
            multipart_threshold=file_size + 1,
            multipart_chunksize=chunk_size,
            max_concurrency=1,
            use_threads=False,
        )
        with db_path.open("rb") as f:
            s3.upload_fileobj(
                f,
                str(s3_bucket),
                str(s3_key),
                Callback=progress if progress is not None else None,
                Config=cfg,
            )
        if progress is not None:
            progress.finish()
    except ClientError as e:
        code = (e.response or {}).get("Error", {}).get("Code")
        if code in {"AccessDenied", "AllAccessDisabled"}:
            raise RuntimeError(
                "S3 upload failed due to missing WRITE permissions for this IAM user.\n"
                f"Target: s3://{s3_bucket}/{s3_key}\n\n"
                "Fix (AWS IAM): grant these permissions to the user/role used by master_run.py:\n"
                '- s3:PutObject on arn:aws:s3:::' + str(s3_bucket) + "/" + str(s3_key) + "\n"
                "- (optional but common) s3:ListBucket on arn:aws:s3:::" + str(s3_bucket) + "\n\n"
                "If you want to keep Streamlit read-only credentials, set separate upload creds:\n"
                "- AWS_ACCESS_KEY_ID_UPLOAD\n"
                "- AWS_SECRET_ACCESS_KEY_UPLOAD\n"
                "and keep the existing AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY for dashboard read-only access."
            ) from e
        raise


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="One-command pipeline: Excel -> DuckDB -> Design -> RE-Connect -> Apollo -> S3")
    parser.add_argument("--dgr-dir", default="DGR", help="Folder containing DGR_*.xlsx files (default: DGR)")
    parser.add_argument("--db", default="master.duckdb", help="Local DuckDB file path (default: master.duckdb)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for the loader")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars in the loader")
    parser.add_argument("--skip-reconnect", action="store_true", help="Skip RE-Connect download and ingestion steps")
    args = parser.parse_args(list(argv) if argv is not None else None)

    db_path = Path(args.db)

    # 1) Load Excel -> local DuckDB (uses existing loader as-is)
    print("=== Step 1/4: Updating local DuckDB from Excel ===")
    rc = store_data_table_duckdb.main(
        [
            "--dgr-dir",
            str(args.dgr_dir),
            "--db",
            str(db_path),
            *(["--verbose"] if args.verbose else []),
            *(["--no-progress"] if args.no_progress else []),
        ]
    )
    if rc != 0:
        print(f"Loader failed with exit code {rc}. Aborting pipeline.")
        return int(rc)

    # 1.5) Design Excel data ingestion
    print("\n=== Step 1.5/4: Injecting Design Excel Data ===")
    rc_design = design_array_injestor.main(
        [
            "--db",
            str(db_path),
            "--design-dir",
            "design_data",
        ]
    )
    if rc_design != 0:
        print(f"Design array ingestion failed with exit code {rc_design}. Aborting pipeline.")
        return int(rc_design)
    print("[OK] Design data ingestion completed")

    # 2) RE-Connect pipeline (download -> inject)
    if not args.skip_reconnect:
        if reconnect_downloads is None or reconnect_inject is None:
            print("=== Step 2/4: RE-Connect pipeline (SKIPPED - modules not available) ===")
            print(f"  Warning: RE-Connect modules could not be imported: {_reconnect_import_error}")
            print("  Continuing with remaining steps...")
        else:
            print("=== Step 2/4: RE-Connect pipeline ===")
            
            # 2.1) Download RE-Connect data
            print("\n--- 2.1: Downloading RE-Connect data ---")
            try:
                reconnect_downloads.main()
                print("[OK] RE-Connect download completed")
            except Exception as e:
                print(f"[ERROR] RE-Connect download failed: {e}")
                if args.verbose:
                    import traceback
                    traceback.print_exc()
                print("  Continuing with ingestion attempt...")
            
            # 2.2) Ingest RE-Connect data into DuckDB
            print("\n--- 2.2: Ingesting RE-Connect data into DuckDB ---")
            reconnect_inject_args = [
                "--db",
                str(db_path),
            ]
            if args.verbose:
                reconnect_inject_args.append("--verbose")
            
            rc_reconnect = reconnect_inject.main(reconnect_inject_args)
            if rc_reconnect != 0:
                print(f"[ERROR] RE-Connect ingestion failed with exit code {rc_reconnect}")
                print("  Continuing with S3 upload...")
            else:
                print("[OK] RE-Connect ingestion completed")
    else:
        print("=== Step 2/4: RE-Connect pipeline (SKIPPED by user request) ===")

    # 3) Apollo ingestion into DuckDB (MANDATORY pre-S3 step)
    print("\n=== Step 3/4: Injecting Apollo Raw Data into DuckDB ===")
    try:
        rc_apollo = apollo_injestor.main(
            [
                "--db",
                str(db_path),
                "--input-dir",
                "Apollo Raw Data",
                *(["--verbose"] if args.verbose else []),
                *(["--no-progress"] if args.no_progress else []),
            ]
        )
    except Exception as e:
        print(f"Apollo ingestion failed: {e}. Aborting pipeline (no S3 upload).")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    if rc_apollo != 0:
        print(f"Apollo ingestion failed with exit code {rc_apollo}. Aborting pipeline (no S3 upload).")
        return int(rc_apollo)
    print("[OK] Apollo ingestion completed")

    # 4) Upload DuckDB -> S3 (overwrites object at bucket/key)
    print("\n=== Step 4/4: Uploading DuckDB to AWS S3 ===")
    try:
        upload_duckdb_to_s3(db_path=db_path, show_progress=not bool(args.no_progress))
        print("[OK] Upload complete. Your deployed Streamlit app will use the latest DuckDB from S3.")
    except Exception as e:
        print(f"[ERROR] S3 upload failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




