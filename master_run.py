from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import boto3
from botocore.exceptions import ClientError

import store_data_table_duckdb

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


def _get_secret(name: str) -> Optional[str]:
    """
    Read secrets from Streamlit if available; fallback to env vars.
    Never prints secret values.
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


def upload_duckdb_to_s3(*, db_path: Path) -> None:
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
        with db_path.open("rb") as f:
            s3.put_object(Bucket=str(s3_bucket), Key=str(s3_key), Body=f)
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
    parser = argparse.ArgumentParser(description="One-command pipeline: Excel -> DuckDB -> RE-Connect -> S3")
    parser.add_argument("--dgr-dir", default="DGR", help="Folder containing DGR_*.xlsx files (default: DGR)")
    parser.add_argument("--db", default="master.duckdb", help="Local DuckDB file path (default: master.duckdb)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging for the loader")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars in the loader")
    parser.add_argument("--skip-reconnect", action="store_true", help="Skip RE-Connect download and ingestion steps")
    args = parser.parse_args(list(argv) if argv is not None else None)

    db_path = Path(args.db)

    # 1) Load Excel -> local DuckDB (uses existing loader as-is)
    print("=== Step 1/3: Updating local DuckDB from Excel ===")
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

    # 2) RE-Connect pipeline (download -> inject)
    if not args.skip_reconnect:
        if reconnect_downloads is None or reconnect_inject is None:
            print("=== Step 2/3: RE-Connect pipeline (SKIPPED - modules not available) ===")
            print(f"  Warning: RE-Connect modules could not be imported: {_reconnect_import_error}")
            print("  Continuing with remaining steps...")
        else:
            print("=== Step 2/3: RE-Connect pipeline ===")
            
            # 2.1) Download RE-Connect data
            print("\n--- 2.1: Downloading RE-Connect data ---")
            try:
                reconnect_downloads.main()
                print("✓ RE-Connect download completed")
            except Exception as e:
                print(f"✗ RE-Connect download failed: {e}")
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
                print(f"✗ RE-Connect ingestion failed with exit code {rc_reconnect}")
                print("  Continuing with S3 upload...")
            else:
                print("✓ RE-Connect ingestion completed")
    else:
        print("=== Step 2/3: RE-Connect pipeline (SKIPPED by user request) ===")

    # 3) Upload DuckDB -> S3 (overwrites object at bucket/key)
    print("\n=== Step 3/3: Uploading DuckDB to AWS S3 ===")
    try:
        upload_duckdb_to_s3(db_path=db_path)
        print("✓ Upload complete. Your deployed Streamlit app will use the latest DuckDB from S3.")
    except Exception as e:
        print(f"✗ S3 upload failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




