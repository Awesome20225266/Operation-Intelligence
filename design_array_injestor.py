from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path
from typing import Any, List

import duckdb
import pandas as pd

# -----------------------------
# Configuration
# -----------------------------
DB_PATH = "master.duckdb"
DESIGN_DATA_DIR = Path("design_data")

ARRAY_FILE = DESIGN_DATA_DIR / "array_details.xlsx"
PLANT_FILE = DESIGN_DATA_DIR / "plant_details.xlsx"

# -----------------------------
# Robust Helpers
# -----------------------------

def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the dataframe:
    1. Replaces all NA/None/Blank with 0.
    2. Coerces non-identity columns to numeric where possible, else 0.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    
    # Fill actual NA values
    df = df.fillna(0)
    
    # Clean strings and force "N/A" type strings to 0
    def _sanitize(val):
        if isinstance(val, str):
            v = val.strip().upper()
            if v in ["", "NA", "N/A", "NONE", "NULL", "-"]:
                return 0
        return val

    # Use apply with map instead of deprecated applymap
    for col in df.columns:
        df[col] = df[col].map(_sanitize)
        # Try to convert to numeric, if fails keep as is (will be 0 from sanitize)
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        except Exception:
            pass

    return df

def get_row_hash(row_dict: dict) -> str:
    """Creates a unique MD5 hash of the entire row for exact-match detection."""
    # Sort keys for consistent hashing
    sorted_items = sorted(row_dict.items())
    combined = "|".join(f"{k}:{v}" for k, v in sorted_items)
    return hashlib.md5(combined.encode("utf-8")).hexdigest()

def ingest_table(con: duckdb.DuckDBPyConnection, table_name: str, excel_path: Path, identity_cols: List[str]):
    """
    Core ingestion engine with Skip/Update/Insert logic.
    """
    if not excel_path.exists():
        print(f"Skipping {table_name}: File not found at {excel_path}")
        return

    print(f"\n>>> Processing {table_name} from {excel_path.name}...")

    # 1. Load All Sheets
    try:
        xls = pd.ExcelFile(excel_path)
        df_list = []
        for sheet in xls.sheet_names:
            temp_df = pd.read_excel(xls, sheet_name=sheet, header=0)
            if not temp_df.empty:
                df_list.append(temp_df)
        
        if not df_list:
            print(f"  Empty workbook found at {excel_path}")
            return
            
        df = pd.concat(df_list, ignore_index=True)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception as e:
        print(f"  CRITICAL ERROR reading Excel: {e}")
        import traceback
        traceback.print_exc()
        return

    # Validate identity columns
    missing = [c for c in identity_cols if c not in df.columns]
    if missing:
        print(f"  ERROR: Identity columns not found: {missing}")
        print(f"  Available columns: {list(df.columns)}")
        return

    # 2. Clean Data
    df = _clean_dataframe(df)

    # 3. Ensure Table Exists (using Excel schema)
    # Register temp view for schema creation
    con.register("_tmp_schema", df.head(0))
    try:
        con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} AS SELECT * FROM _tmp_schema WHERE 1=0")
    finally:
        try:
            con.unregister("_tmp_schema")
        except Exception:
            pass

    # 4. Fetch existing data for comparison
    try:
        existing_df = con.execute(f"SELECT * FROM {table_name}").df()
    except Exception:
        existing_df = pd.DataFrame()
    
    # Maps for high-speed lookup
    # full_hash -> True (Exact duplicate)
    # identity_tuple -> full_row_dict (Logical match for potential update)
    hash_lookup = {}
    id_lookup = {}

    if not existing_df.empty:
        existing_df = _clean_dataframe(existing_df)
        for _, row in existing_df.iterrows():
            r_dict = row.to_dict()
            h = get_row_hash(r_dict)
            hash_lookup[h] = True
            
            id_key = tuple(r_dict.get(c) for c in identity_cols)
            id_lookup[id_key] = r_dict

    # 5. Process Rows
    inserted, updated, skipped = 0, 0, 0

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        row_hash = get_row_hash(row_dict)
        id_key = tuple(row_dict.get(c) for c in identity_cols)
        
        # LOGIC A: Skip if exact match
        if row_hash in hash_lookup:
            id_display = ", ".join(str(v) for v in id_key)
            print(f"  SKIPPED (duplicate row): {id_display}")
            skipped += 1
            continue
        
        # LOGIC B: Update if ID exists but data changed
        if id_key in id_lookup:
            # Build WHERE clause for DELETE using proper escaping
            conditions = []
            for c in identity_cols:
                v = row_dict.get(c)
                if pd.isna(v):
                    conditions.append(f'"{c}" IS NULL')
                elif isinstance(v, (int, float)):
                    conditions.append(f'"{c}" = {v}')
                else:
                    escaped_val = str(v).replace("'", "''")
                    conditions.append(f'"{c}" = \'{escaped_val}\'')
            
            where_clause = " AND ".join(conditions)
            
            # Delete the old version
            con.execute(f"DELETE FROM {table_name} WHERE {where_clause}")
            
            # Insert the new version using VALUES with proper escaping
            cols = list(row_dict.keys())
            col_names = ", ".join([f'"{c}"' for c in cols])
            values = []
            for c in cols:
                v = row_dict.get(c)
                if pd.isna(v):
                    values.append("NULL")
                elif isinstance(v, (int, float)):
                    values.append(str(v))
                else:
                    escaped_val = str(v).replace("'", "''")
                    values.append(f"'{escaped_val}'")
            
            val_list = ", ".join(values)
            con.execute(f"INSERT INTO {table_name} ({col_names}) VALUES ({val_list})")
            
            id_display = ", ".join(str(v) for v in id_key)
            print(f"  UPDATED (row values changed): {id_display}")
            updated += 1
        
        # LOGIC C: Insert if brand new
        else:
            cols = list(row_dict.keys())
            col_names = ", ".join([f'"{c}"' for c in cols])
            values = []
            for c in cols:
                v = row_dict.get(c)
                if pd.isna(v):
                    values.append("NULL")
                elif isinstance(v, (int, float)):
                    values.append(str(v))
                else:
                    escaped_val = str(v).replace("'", "''")
                    values.append(f"'{escaped_val}'")
            
            val_list = ", ".join(values)
            con.execute(f"INSERT INTO {table_name} ({col_names}) VALUES ({val_list})")
            
            id_display = ", ".join(str(v) for v in id_key)
            print(f"  INSERTED (new row): {id_display}")
            inserted += 1

    print(f"--- Finished {table_name}: Inserted {inserted}, Updated {updated}, Skipped {skipped} ---")

# -----------------------------
# Entry Point
# -----------------------------

def main():
    if not DESIGN_DATA_DIR.exists():
        print(f"Error: Directory '{DESIGN_DATA_DIR}' not found. Please create it and add your Excel files.")
        sys.exit(1)
    
    # Connect to the DB (it will create master.duckdb if not present)
    con = duckdb.connect(DB_PATH)

    try:
        # Identity columns based on your Excel structures:
        # plant_details: 'category' + 'tags' uniquely defines a metric row
        ingest_table(
            con=con, 
            table_name="plant_details", 
            excel_path=PLANT_FILE, 
            identity_cols=["category", "tags"]
        )

        # array_details: 'tag_name' is the unique equipment identifier
        ingest_table(
            con=con, 
            table_name="array_details", 
            excel_path=ARRAY_FILE, 
            identity_cols=["tag_name"]
        )
    finally:
        con.close()

if __name__ == "__main__":
    main()
