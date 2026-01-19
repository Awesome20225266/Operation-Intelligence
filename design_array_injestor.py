from __future__ import annotations

import argparse
import hashlib
from datetime import date, datetime
from pathlib import Path
from typing import Any, Optional

import duckdb
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


# -----------------------------
# Configuration
# -----------------------------

DEFAULT_DB_PATH = "master.duckdb"
DEFAULT_DESIGN_DIR = Path("design_data")

# NON-NEGOTIABLE constraint: only these tables may be created/updated
ALLOWED_TABLES = {"plant_details", "array_details"}


# -----------------------------
# Excel loading
# -----------------------------

def _read_excel_single_dataset(path: Path) -> pd.DataFrame:
    """Read a single-dataset workbook (row1 headers, remaining rows data)."""
    df = pd.read_excel(path, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _read_excel_union_all_sheets(path: Path) -> pd.DataFrame:
    """
    Read ALL sheets from an Excel workbook and union them into one dataframe.
    - No sheet is skipped under any circumstance
    - No extra columns (like sheet_name) are added
    - Assumes all sheets share the same schema
    """
    xls = pd.ExcelFile(path)
    sheet_names = list(xls.sheet_names)
    if not sheet_names:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    it = sheet_names
    if tqdm is not None:
        it = tqdm(sheet_names, desc=f"Reading sheets: {path.name}", unit="sheet")

    for sheet in it:
        df_sheet = pd.read_excel(xls, sheet_name=sheet, header=0)
        df_sheet.columns = [str(c).strip() for c in df_sheet.columns]
        frames.append(df_sheet)  # do not skip empties

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# -----------------------------
# Change detection (cell-level, order-independent)
# -----------------------------

def _canonical_cell(v: Any) -> str:
    """Canonical representation for safe equality checks across dtype differences."""
    if pd.isna(v):
        return "<NA>"

    if isinstance(v, (pd.Timestamp, datetime, date)):
        try:
            return pd.to_datetime(v).isoformat()
        except Exception:
            return str(v)

    if isinstance(v, bool):
        return "true" if v else "false"

    if isinstance(v, (int, float)):
        try:
            fv = float(v)
            if abs(fv - round(fv)) < 1e-9:
                return str(int(round(fv)))
            return format(fv, ".15g")
        except Exception:
            return str(v)

    return str(v)


def _row_hashes(df: pd.DataFrame, *, desc: str) -> pd.Series:
    """Compute a stable MD5 hash for each row after canonicalizing values."""
    cols = list(df.columns)

    canon = df.copy()
    col_it = cols
    if tqdm is not None:
        col_it = tqdm(cols, desc=f"Normalizing: {desc}", unit="col")
    for c in col_it:
        canon[c] = canon[c].map(_canonical_cell)

    def _hash_row(row: pd.Series) -> str:
        s = "\x1f".join(row.values.tolist())
        return hashlib.md5(s.encode("utf-8")).hexdigest()

    if tqdm is not None:
        tqdm.pandas(desc=f"Hashing rows: {desc}")  # type: ignore[attr-defined]
        return canon.progress_apply(_hash_row, axis=1)  # type: ignore[attr-defined]
    return canon.apply(_hash_row, axis=1)


def _table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    row = con.execute(
        "select 1 from information_schema.tables where table_schema='main' and table_name=? limit 1",
        [table_name],
    ).fetchone()
    return bool(row)


def _load_table_df(con: duckdb.DuckDBPyConnection, table_name: str) -> pd.DataFrame:
    return con.execute(f'SELECT * FROM "{table_name}"').df()


def _data_changed(excel_df: pd.DataFrame, db_df: pd.DataFrame, *, table_name: str) -> tuple[bool, str]:
    excel_cols = [str(c).strip() for c in excel_df.columns]
    db_cols = [str(c).strip() for c in db_df.columns]

    # Normalize column order (comparison should not fail just because DB column order differs)
    if set(excel_cols) != set(db_cols):
        return True, "Change detected (column mismatch)"

    # Reorder both frames to a deterministic order (Excel column order)
    excel_df = excel_df[excel_cols].copy()
    db_df = db_df[excel_cols].copy()

    excel_hashes = _row_hashes(excel_df, desc=f"{table_name} (excel)")
    db_hashes = _row_hashes(db_df, desc=f"{table_name} (db)")

    excel_counts = excel_hashes.value_counts(dropna=False).sort_index()
    db_counts = db_hashes.value_counts(dropna=False).sort_index()

    if not excel_counts.equals(db_counts):
        return True, "Change detected in row data"

    return False, "No change detected – table not updated"


# -----------------------------
# Safe write (isolated to allowed tables only)
# -----------------------------

def _write_table(con: duckdb.DuckDBPyConnection, *, table_name: str, df: pd.DataFrame, replace: bool) -> None:
    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Refusing to write to non-allowed table: {table_name}")

    view_name = "__design_df"
    con.register(view_name, df)
    try:
        con.execute("BEGIN TRANSACTION")
        if replace:
            con.execute(f'CREATE OR REPLACE TABLE "{table_name}" AS SELECT * FROM {view_name}')
        else:
            con.execute(f'CREATE TABLE "{table_name}" AS SELECT * FROM {view_name}')
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        try:
            con.unregister(view_name)
        except Exception:
            pass


def process_table(con: duckdb.DuckDBPyConnection, *, table_name: str, excel_df: pd.DataFrame) -> None:
    print(f"\nProcessing {table_name}...")

    if table_name not in ALLOWED_TABLES:
        raise ValueError(f"Table '{table_name}' is not allowed by constraints.")

    if not _table_exists(con, table_name):
        if tqdm is not None:
            p = tqdm(total=1, desc=f"Writing {table_name}", unit="step")
        else:
            p = None
        _write_table(con, table_name=table_name, df=excel_df, replace=False)
        if p is not None:
            p.update(1)
            p.close()
        print("Table created and data injected")
        return

    if tqdm is not None:
        p = tqdm(total=2, desc=f"Comparing {table_name}", unit="step")
    else:
        p = None
    db_df = _load_table_df(con, table_name)
    if p is not None:
        p.update(1)

    changed, remark = _data_changed(excel_df, db_df, table_name=table_name)
    if p is not None:
        p.update(1)
        p.close()

    if not changed:
        print(remark)
        return

    print(remark)
    if tqdm is not None:
        p2 = tqdm(total=1, desc=f"Updating {table_name}", unit="step")
    else:
        p2 = None
    _write_table(con, table_name=table_name, df=excel_df, replace=True)
    if p2 is not None:
        p2.update(1)
        p2.close()
    print("Data updated due to change detected")


# -----------------------------
# Entry point
# -----------------------------

def ingest_design_data(*, db_path: str, design_dir: Path) -> int:
    array_file = design_dir / "array_details.xlsx"
    plant_file = design_dir / "plant_details.xlsx"

    if not design_dir.exists():
        print(f"Error: Directory '{design_dir}' not found.")
        return 1

    if not plant_file.exists():
        print(f"Error: Missing required file: {plant_file}")
        return 1

    if not array_file.exists():
        print(f"Error: Missing required file: {array_file}")
        return 1

    # Load Excel (progress)
    if tqdm is not None:
        p = tqdm(total=2, desc="Loading design Excel files", unit="file")
    else:
        p = None
    plant_df = _read_excel_single_dataset(plant_file)
    if p is not None:
        p.update(1)
    array_df = _read_excel_union_all_sheets(array_file)
    if p is not None:
        p.update(1)
        p.close()

    con = duckdb.connect(db_path)
    try:
        process_table(con, table_name="plant_details", excel_df=plant_df)
        process_table(con, table_name="array_details", excel_df=array_df)
    finally:
        con.close()

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest design Excel files into DuckDB (safe + idempotent).")
    parser.add_argument("--db", default=DEFAULT_DB_PATH, help="DuckDB path (default: master.duckdb)")
    parser.add_argument("--design-dir", default=str(DEFAULT_DESIGN_DIR), help="Design data dir (default: design_data/)")
    args = parser.parse_args(argv)
    return ingest_design_data(db_path=str(args.db), design_dir=Path(args.design_dir))


if __name__ == "__main__":
    raise SystemExit(main())
