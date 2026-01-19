from __future__ import annotations

import argparse
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import duckdb
import pandas as pd

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


LOGGER = logging.getLogger("apollo_injestor")


# -----------------------------
# Regex parsing (STRICT-ish, but robust)
# -----------------------------

# Example: "ASPL | ICR1-INV1-SCB01 - CB Current (A)"
RE_SITE_SPLIT = re.compile(r"^\s*([^|]+?)\s*\|\s*(.+?)\s*$")

# inv_stn_name rules:
# - ICR1 -> IS1 ; ICR20 -> IS20
# - ICR (no number) -> IS1
# - MCR -> IS2
RE_ICR_NUM = re.compile(r"\bICR\s*(\d+)\b", flags=re.IGNORECASE)
RE_ICR_BARE = re.compile(r"\bICR\b", flags=re.IGNORECASE)
RE_MCR = re.compile(r"\bMCR\b", flags=re.IGNORECASE)

# Special rule: "INV X.Y -> ISX"
RE_INV_DOT = re.compile(r"\bINV\s*(\d+)\s*\.\s*(\d+)\b", flags=re.IGNORECASE)

# inv_name rules:
# - INV1 -> INV1
# - INV 1.2 -> INV2 (take the part after dot)
RE_INV_ANY = re.compile(r"\bINV\s*(\d+)(?:\s*\.\s*(\d+))?\b", flags=re.IGNORECASE)

# SCB rules:
# - SCB01 -> SCB1
# - SCB10 -> SCB10
# - SCBML -> SCBML
RE_SCB = re.compile(r"\bSCB\s*([A-Za-z]+|\d+)\b", flags=re.IGNORECASE)


MANDATORY_BASE_COLS = ["date", "timestamp", "site_name", "inv_stn_name", "inv_name"]


@dataclass(frozen=True)
class ParsedHeader:
    site_name: str
    table_name: str
    inv_stn_name: str
    inv_name: str
    scb_col: str  # e.g. SCB1 / SCB10 / SCBML


def _sanitize_table_name(site_name: str) -> str:
    # Lowercase, keep alnum, replace others with underscore, collapse underscores.
    s = site_name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown_site"


def _iter_input_files(root: Path) -> list[Path]:
    files: list[Path] = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.name.startswith("~$"):
            continue
        # Apollo Raw Data may be exported as Excel OR CSV depending on source
        if p.suffix.lower() in {".xlsx", ".xls", ".xlsm", ".csv"}:
            files.append(p)
    return sorted(files)


def _resolve_input_dir(input_dir: Path) -> Path:
    """
    Resolve input_dir robustly on Windows where scripts are often executed from a different CWD.

    Resolution order:
    - If absolute: use as-is
    - Try relative to current working directory
    - Try relative to this script's directory (project root)
    - Fallback: case-insensitive directory name search under script directory (limited)
    """
    p = Path(input_dir).expanduser()
    if p.is_absolute():
        return p

    cwd_candidate = Path.cwd() / p
    if cwd_candidate.exists():
        return cwd_candidate

    script_root = Path(__file__).resolve().parent
    script_candidate = script_root / p
    if script_candidate.exists():
        return script_candidate

    # Fallback: locate folder by name under project root
    target_name = p.name.strip().lower()
    if target_name:
        # Limit search depth by pruning common huge dirs; and early-exit on first match.
        for d in script_root.rglob("*"):
            try:
                if not d.is_dir():
                    continue
                name = d.name.strip().lower()
                if name == target_name:
                    return d
            except Exception:
                continue

    # Default to CWD-relative path (for best error messaging)
    return cwd_candidate


def _excel_file(path: Path) -> pd.ExcelFile:
    """
    Prefer calamine (supports .xls + .xlsx). Fall back to pandas defaults if needed.
    """
    try:
        return pd.ExcelFile(path, engine="calamine")
    except Exception:
        return pd.ExcelFile(path)


def _read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    # Prefer calamine for broad Excel support; fall back to openpyxl for .xlsx.
    try:
        return pd.read_excel(path, sheet_name=sheet_name, engine="calamine")
    except Exception:
        return pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl" if path.suffix.lower() == ".xlsx" else None)


def _read_csv(path: Path) -> pd.DataFrame:
    # Be forgiving with encodings/export formats.
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except Exception:
            return pd.read_csv(path, encoding_errors="replace")


def _find_datetime_column(df: pd.DataFrame) -> Optional[str]:
    """
    Find a DATETIME column (case-insensitive), else attempt to find a column whose values
    look like YYYY-MM-DDTHH:MM.
    """
    if df is None or df.empty:
        return None

    cols = list(df.columns)
    # 1) direct name match
    for c in cols:
        if str(c).strip().lower() == "datetime":
            return c

    # 2) heuristic: a column with many values matching the pattern
    pattern = re.compile(r"^\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}\s*$")
    best_col = None
    best_score = 0
    sample_n = min(200, len(df))
    for c in cols[: min(30, len(cols))]:
        s = df[c].astype(str).head(sample_n)
        score = int(s.map(lambda x: bool(pattern.match(x))).sum())
        if score > best_score:
            best_score = score
            best_col = c
    if best_col is not None and best_score >= max(5, int(0.6 * min(200, len(df)))):
        return best_col
    return None


def _parse_datetime_series(s: pd.Series) -> pd.Series:
    # Expect ISO-like: YYYY-MM-DDTHH:MM
    return pd.to_datetime(s, errors="coerce", format="%Y-%m-%dT%H:%M")


def _format_date_timestamp(dt: pd.Series) -> tuple[pd.Series, pd.Series]:
    # date -> DD-MM-YYYY, timestamp -> HH:MM
    date_str = dt.dt.strftime("%d-%m-%Y")
    ts_str = dt.dt.strftime("%H:%M")
    return date_str, ts_str


def _parse_header(col_name: Any) -> Optional[ParsedHeader]:
    """
    Parse a measurement column header into site / IS / INV / SCB.
    If not confidently parseable, return None (skip).
    """
    raw = "" if col_name is None else str(col_name)
    raw = raw.strip()
    if raw == "":
        return None

    m_site = RE_SITE_SPLIT.match(raw)
    if not m_site:
        return None
    site_name = m_site.group(1).strip()
    rest = m_site.group(2).strip()
    if site_name == "" or rest == "":
        return None

    table_name = _sanitize_table_name(site_name)

    # SCB
    m_scb = RE_SCB.search(rest)
    if not m_scb:
        return None
    scb_raw = m_scb.group(1).strip()
    if scb_raw == "":
        return None
    if scb_raw.isdigit():
        scb_norm = str(int(scb_raw))  # strips leading zeros
        scb_col = f"SCB{scb_norm}"
    else:
        scb_col = f"SCB{scb_raw.upper()}"

    # inv_name (must exist)
    inv_name = ""
    m_inv = RE_INV_ANY.search(rest)
    if m_inv:
        a = m_inv.group(1)
        b = m_inv.group(2)
        if b is not None and str(b).strip() != "":
            inv_name = f"INV{int(b)}"
        else:
            inv_name = f"INV{int(a)}"
    if inv_name == "":
        return None

    # inv_stn_name
    inv_stn_name = ""
    m_icr_num = RE_ICR_NUM.search(rest)
    if m_icr_num:
        inv_stn_name = f"IS{int(m_icr_num.group(1))}"
    elif RE_MCR.search(rest):
        inv_stn_name = "IS2"
    elif RE_ICR_BARE.search(rest):
        inv_stn_name = "IS1"
    else:
        # Special rule: INV X.Y -> ISX
        m_dot = RE_INV_DOT.search(rest)
        if m_dot:
            inv_stn_name = f"IS{int(m_dot.group(1))}"

    if inv_stn_name == "":
        return None

    return ParsedHeader(
        site_name=site_name,
        table_name=table_name,
        inv_stn_name=inv_stn_name,
        inv_name=inv_name,
        scb_col=scb_col,
    )


def _build_rows_from_sheet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform one sheet into the common wide format.
    Output columns:
      date, timestamp, site_name, inv_stn_name, inv_name, SCB*
    """
    dt_col = _find_datetime_column(df)
    if dt_col is None:
        raise ValueError("DATETIME column not found")

    dt = _parse_datetime_series(df[dt_col])
    if dt.isna().all():
        raise ValueError("DATETIME column found but could not parse any timestamps")
    date_str, ts_str = _format_date_timestamp(dt)

    # Group measurement columns by (site, table, IS, INV)
    groups: dict[tuple[str, str, str, str], dict[str, pd.Series]] = {}
    scb_union: set[str] = set()
    skipped = 0

    for c in df.columns:
        if c == dt_col:
            continue
        parsed = _parse_header(c)
        if parsed is None:
            skipped += 1
            continue
        key = (parsed.site_name, parsed.table_name, parsed.inv_stn_name, parsed.inv_name)
        scb_union.add(parsed.scb_col)
        g = groups.setdefault(key, {})
        # If duplicate SCB columns appear, keep the first non-null-ish series.
        if parsed.scb_col not in g:
            g[parsed.scb_col] = pd.to_numeric(df[c], errors="coerce")

    if not groups:
        if skipped > 0:
            LOGGER.info("No parseable measurement columns found (skipped %d malformed headers).", skipped)
        return pd.DataFrame(columns=MANDATORY_BASE_COLS)

    out_frames: list[pd.DataFrame] = []
    for (site_name, _table_name, inv_stn_name, inv_name), scb_map in groups.items():
        base = pd.DataFrame(
            {
                "date": date_str,
                "timestamp": ts_str,
                "site_name": site_name,
                "inv_stn_name": inv_stn_name,
                "inv_name": inv_name,
            }
        )
        for scb_col, series in scb_map.items():
            base[scb_col] = series
        out_frames.append(base)

    out = pd.concat(out_frames, ignore_index=True)
    # Drop rows with null datetime-derived fields (unparseable timestamps)
    out = out.dropna(subset=["date", "timestamp"])
    return out


def _ensure_site_table(con: duckdb.DuckDBPyConnection, table: str, scb_cols: Iterable[str]) -> None:
    # Create base schema
    con.execute(
        f"""
        CREATE TABLE IF NOT EXISTS "{table}" (
            date TEXT,
            timestamp TEXT,
            site_name TEXT,
            inv_stn_name TEXT,
            inv_name TEXT
        )
        """
    )
    # Add SCB columns as needed (DOUBLE)
    for c in sorted(set(scb_cols)):
        if not c.upper().startswith("SCB"):
            continue
        con.execute(f'ALTER TABLE "{table}" ADD COLUMN IF NOT EXISTS "{c}" DOUBLE')


def _get_existing_dates(con: duckdb.DuckDBPyConnection, *, table: str) -> set[str]:
    """
    Check which dates already exist in a site table.
    Returns a set of date strings (DD-MM-YYYY format).
    """
    try:
        result = con.execute(f'SELECT DISTINCT date FROM "{table}"').fetchall()
        return {str(row[0]) for row in result if row[0] is not None}
    except Exception:
        # Table doesn't exist yet or empty - return empty set
        return set()


def _upsert_site_df(
    con: duckdb.DuckDBPyConnection,
    *,
    table: str,
    df: pd.DataFrame,
    existing_dates: Optional[set[str]] = None,
    overwrite: bool = False,
) -> int:
    if df is None or df.empty:
        return 0

    # Filter out dates that already exist (idempotent behavior), unless overwrite is requested
    if not overwrite:
        if existing_dates is None:
            existing_dates = _get_existing_dates(con, table=table)
        
        if existing_dates:
            df_filtered = df[~df["date"].isin(existing_dates)].copy()
            skipped_dates = sorted(set(df["date"].unique()) & existing_dates)
            if skipped_dates:
                LOGGER.info('Date data already exists in DB: "%s" (%s) (skipped)', table, ", ".join(skipped_dates))
        else:
            df_filtered = df.copy()
    else:
        df_filtered = df.copy()
    
    if df_filtered.empty:
        return 0

    scb_cols = [c for c in df_filtered.columns if str(c).upper().startswith("SCB")]
    _ensure_site_table(con, table, scb_cols)

    # Align df to table columns (base + sorted scbs)
    insert_cols = MANDATORY_BASE_COLS + sorted(scb_cols)
    for c in insert_cols:
        if c not in df_filtered.columns:
            df_filtered[c] = None
    df2 = df_filtered[insert_cols].copy()

    # Register + TEMP TABLE for set-based delete/insert
    con.register("_apollo_incoming_df", df2)
    con.execute('CREATE OR REPLACE TEMP TABLE _apollo_incoming AS SELECT * FROM _apollo_incoming_df')

    # If overwriting, delete existing rows for incoming dates first
    if overwrite:
        con.execute(
            f"""
            DELETE FROM "{table}" AS t
            USING _apollo_incoming AS s
            WHERE
                t.date = s.date
                AND t.timestamp = s.timestamp
                AND t.inv_stn_name = s.inv_stn_name
                AND t.inv_name = s.inv_name
            """
        )

    # Insert new data (existing dates already filtered out if not overwriting)
    col_list = ", ".join([f'"{c}"' for c in insert_cols])
    con.execute(f'INSERT INTO "{table}" ({col_list}) SELECT {col_list} FROM _apollo_incoming')
    inserted = int(con.execute("SELECT COUNT(*) FROM _apollo_incoming").fetchone()[0])
    con.unregister("_apollo_incoming_df")
    return inserted


def ingest_apollo(*, input_dir: Path, db_path: Path, verbose: bool = False, no_progress: bool = False, overwrite: bool = False) -> int:
    resolved_dir = _resolve_input_dir(input_dir)
    if resolved_dir != input_dir:
        LOGGER.info("Resolved input-dir: %s -> %s", str(input_dir), str(resolved_dir))
    files = _iter_input_files(resolved_dir)
    if not files:
        LOGGER.info("No input files (.xlsx/.xls/.xlsm/.csv) found under: %s", str(resolved_dir))
        return 0

    iterator = files
    if not no_progress and tqdm is not None:
        iterator = tqdm(files, desc="Apollo Excel files", unit="file")  # type: ignore[assignment]

    all_rows: list[pd.DataFrame] = []
    for path in iterator:
        suf = path.suffix.lower()
        if suf == ".csv":
            try:
                df_sheet = _read_csv(path)
            except Exception as e:
                LOGGER.warning("Skipping CSV (read failed): %s | %s", path.name, e)
                continue
            try:
                out = _build_rows_from_sheet(df_sheet)
            except Exception as e:
                LOGGER.warning("Skipping CSV (parse failed): %s | %s", path.name, e)
                continue
            if not out.empty:
                all_rows.append(out)
            continue

        # Excel variants
        try:
            xls = _excel_file(path)
        except Exception as e:
            LOGGER.warning("Skipping file (cannot open): %s | %s", path.name, e)
            continue

        for sheet in xls.sheet_names:
            try:
                df_sheet = _read_sheet(path, sheet)
            except Exception as e:
                LOGGER.warning("Skipping sheet (read failed): %s | %s | %s", path.name, sheet, e)
                continue

            try:
                out = _build_rows_from_sheet(df_sheet)
            except Exception as e:
                LOGGER.warning("Skipping sheet (parse failed): %s | %s | %s", path.name, sheet, e)
                continue

            if not out.empty:
                all_rows.append(out)

    if not all_rows:
        LOGGER.info("No output rows produced. Check input format and header parsing rules.")
        return 0

    df_all = pd.concat(all_rows, ignore_index=True)

    # Split by site table
    if "site_name" not in df_all.columns:
        LOGGER.error("Internal error: site_name column missing after parsing.")
        return 1

    con = duckdb.connect(str(db_path))
    try:
        total = 0
        for site_name, df_site in df_all.groupby("site_name", dropna=False):
            if site_name is None or str(site_name).strip() == "" or str(site_name).lower() == "nan":
                LOGGER.info("Skipping group with empty site_name")
                continue
            table = _sanitize_table_name(str(site_name))
            # Check existing dates once per site table for efficiency (unless overwriting)
            existing_dates = _get_existing_dates(con, table=table) if not overwrite else None
            inserted = _upsert_site_df(con, table=table, df=df_site, existing_dates=existing_dates, overwrite=overwrite)
            total += inserted
            if inserted > 0:
                LOGGER.info('Inserted into "%s": %d rows', table, inserted)
        LOGGER.info("Done. Total rows inserted: %d", total)
        return 0
    finally:
        con.close()


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Apollo time-series Excel -> DuckDB injestor (no CSV)")
    parser.add_argument("--input-dir", default="Apollo Raw Data", help="Root folder containing Apollo Raw Data Excel files")
    parser.add_argument("--db", default="master.duckdb", help="DuckDB file path (default: master.duckdb)")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars")
    parser.add_argument("--overwrite", action="store_true", help="Force re-insertion of data even if dates already exist in DB")
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    return ingest_apollo(
        input_dir=Path(args.input_dir),
        db_path=Path(args.db),
        verbose=bool(args.verbose),
        no_progress=bool(args.no_progress),
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    raise SystemExit(main())


