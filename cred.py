"""
cred.py
--------

SAFE, terminal-based utility to delete rows from DuckDB time-series tables
based on a user-specified date range.

Key safety properties:
- Only operates on tables that have a column named exactly "date".
- System/metadata schemas (information_schema, pg_catalog, etc.) are excluded.
- Supports selecting ALL eligible tables or a subset by index.
- Requires explicit, case-insensitive "Y" confirmation before any DELETE.
- Uses parameterized queries for the date range.

Usage (from terminal):

    python cred.py

The script assumes the DuckDB database file is named "master.duckdb"
and located in the current working directory. You can override this by
passing a path as the first argument:

    python cred.py path/to/master.duckdb
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date, datetime
from typing import List, Sequence

import duckdb


DEFAULT_DB_PATH = "master.duckdb"


def _quote_ident(ident: str) -> str:
    """Double-quote a DuckDB identifier and escape any internal double quotes."""
    return '"' + ident.replace('"', '""') + '"'


@dataclass(frozen=True)
class TableInfo:
    """Simple container for an eligible DuckDB table."""

    schema: str
    name: str
    date_column_type: str  # e.g. 'DATE', 'VARCHAR' (from information_schema.data_type)

    @property
    def qualified(self) -> str:
        """Return a safely double-quoted, fully-qualified table name."""
        return f"{_quote_ident(self.schema)}.{_quote_ident(self.name)}"

    @property
    def date_is_native(self) -> bool:
        """True if the date column is DuckDB DATE type; False if TEXT/VARCHAR (e.g. DD-MM-YYYY)."""
        return str(self.date_column_type).upper() == "DATE"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Safely delete rows from DuckDB tables that have a 'date' column, "
            "based on a user-provided date range."
        )
    )
    parser.add_argument(
        "db",
        nargs="?",
        default=DEFAULT_DB_PATH,
        help="Path to DuckDB database file (default: %(default)s)",
    )
    return parser.parse_args(argv)


def connect_db(path: str) -> duckdb.DuckDBPyConnection:
    """Connect to DuckDB and return a live connection."""
    try:
        con = duckdb.connect(path)
    except Exception as e:  # pragma: no cover - defensive
        print(f"[ERROR] Failed to open DuckDB database at '{path}': {e}", file=sys.stderr)
        raise SystemExit(1) from e
    return con


def discover_eligible_tables(con: duckdb.DuckDBPyConnection) -> List[TableInfo]:
    """
    Discover all user tables that contain a column exactly named 'date'.

    Constraints:
    - Exclude system/metadata schemas (information_schema, pg_catalog, temp, etc.).
    - Do NOT hardcode table names.
    """
    # 1) Find all tables that have a "date" column and their column data type.
    #    Limit to non-system schemas. DuckDB stores user tables in "main" by default.
    #    We need data_type to support both DATE columns and TEXT (DD-MM-YYYY) columns.
    sql = """
        SELECT DISTINCT
            c.table_schema,
            c.table_name,
            c.data_type
        FROM information_schema.columns AS c
        JOIN information_schema.tables AS t
          ON c.table_schema = t.table_schema
         AND c.table_name   = t.table_name
        WHERE c.column_name = 'date'
          AND t.table_type = 'BASE TABLE'
          AND c.table_schema NOT IN ('information_schema', 'pg_catalog', 'temp', 'system')
        ORDER BY c.table_schema, c.table_name;
    """
    try:
        rows = con.execute(sql).fetchall()
    except Exception as e:  # pragma: no cover - extremely unlikely
        print(f"[ERROR] Failed to inspect DuckDB schema: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    tables: List[TableInfo] = [
        TableInfo(schema=str(s), name=str(n), date_column_type=str(dt or ""))
        for (s, n, dt) in rows
    ]
    return tables


def prompt_table_selection(tables: Sequence[TableInfo]) -> List[TableInfo]:
    """
    Present a numbered menu of eligible tables and return the user's selection.

    Rules:
    - 0 → select ALL tables.
    - Comma-separated indices (e.g., "1,3,5") are allowed.
    - Keeps asking until a valid selection is made or user aborts with empty input.
    """
    if not tables:
        print("No eligible tables found (no tables with a 'date' column). Nothing to do.")
        raise SystemExit(0)

    print("\nEligible tables (only tables with a 'date' column are shown):")
    for idx, t in enumerate(tables, start=1):
        print(f"  {idx}. {t.schema}.{t.name}")
    print("  0. ALL tables listed above")

    while True:
        raw = input("\nEnter table numbers to affect (0 for ALL, or e.g. 1,3,5): ").strip()

        # Allow user to abort by just pressing Enter
        if raw == "":
            print("No selection made. Exiting without changes.")
            raise SystemExit(0)

        # ALL tables
        if raw == "0":
            return list(tables)

        parts = [p.strip() for p in raw.split(",") if p.strip()]
        indices: list[int] = []
        ok = True
        for p in parts:
            if not p.isdigit():
                print(f"Invalid entry '{p}'. Please enter numbers like '1,2,3' or '0' for ALL.")
                ok = False
                break
            i = int(p)
            if i < 1 or i > len(tables):
                print(f"Index out of range: {i}. Valid range is 1..{len(tables)} or 0 for ALL.")
                ok = False
                break
            indices.append(i)

        if not ok:
            continue

        # Deduplicate while preserving order
        seen: set[int] = set()
        selected: List[TableInfo] = []
        for i in indices:
            if i not in seen:
                seen.add(i)
                selected.append(tables[i - 1])

        if not selected:
            print("No valid tables selected. Please try again.")
            continue

        return selected


def _parse_date_input(raw: str) -> date | None:
    """
    Parse a date string; accept YYYY-MM-DD or DD-MM-YYYY.
    Returns None if neither format matches.
    """
    raw = raw.strip()
    if not raw:
        return None
    # Try YYYY-MM-DD first (ISO)
    try:
        return date.fromisoformat(raw)
    except ValueError:
        pass
    # Try DD-MM-YYYY (matches DB display habit)
    try:
        return datetime.strptime(raw, "%d-%m-%Y").date()
    except ValueError:
        pass
    return None


def prompt_date(prompt_text: str) -> date:
    """
    Prompt the user for a date (YYYY-MM-DD or DD-MM-YYYY) and return a datetime.date.

    Keeps asking until a valid date is entered or user aborts with empty input.
    """
    while True:
        raw = input(prompt_text).strip()
        if raw == "":
            print("No date entered. Exiting without changes.")
            raise SystemExit(0)
        d = _parse_date_input(raw)
        if d is not None:
            return d
        print("Invalid date format. Use YYYY-MM-DD (e.g. 2026-02-04) or DD-MM-YYYY (e.g. 04-02-2026).")


def show_data_range_hint(con: duckdb.DuckDBPyConnection, tables: Sequence[TableInfo]) -> None:
    """
    Query min/max date from selected tables and display as a hint before prompting for input.
    This helps prevent year mistakes (e.g. using 2025 when data is in 2026).
    """
    if not tables:
        return
    
    print("\n" + "=" * 60)
    print("Checking date range in selected tables...")
    print("=" * 60)
    
    all_mins: list[date] = []
    all_maxs: list[date] = []
    
    for t in tables:
        qualified = t.qualified
        try:
            if t.date_is_native:
                # Native DATE: direct MIN/MAX
                sql = f'SELECT MIN("date"), MAX("date") FROM {qualified}'
                row = con.execute(sql).fetchone()
            else:
                # TEXT (DD-MM-YYYY): parse then get min/max
                sql = f'SELECT MIN(strptime("date", \'%d-%m-%Y\')::DATE), MAX(strptime("date", \'%d-%m-%Y\')::DATE) FROM {qualified}'
                row = con.execute(sql).fetchone()
            
            if row and row[0] and row[1]:
                min_date = row[0] if isinstance(row[0], date) else date.fromisoformat(str(row[0]))
                max_date = row[1] if isinstance(row[1], date) else date.fromisoformat(str(row[1]))
                all_mins.append(min_date)
                all_maxs.append(max_date)
                print(f"  {t.schema}.{t.name}: {min_date.isoformat()} to {max_date.isoformat()}")
        except Exception as e:
            print(f"  {t.schema}.{t.name}: [Could not determine range: {e}]")
    
    if all_mins and all_maxs:
        overall_min = min(all_mins)
        overall_max = max(all_maxs)
        print(f"\n>>> Data in selected tables spans: {overall_min.isoformat()} to {overall_max.isoformat()}")
        print(">>> Use a deletion range WITHIN this span (e.g. if data is 2026, use 2026-XX-XX).")
    else:
        print("\n>>> Could not determine date range (tables may be empty or date parsing failed).")
    print("=" * 60)


def prompt_date_range() -> tuple[date, date]:
    """Prompt user for FROM/TO dates and ensure FROM <= TO."""
    print("\nEnter the date range for deletion (inclusive).")
    from_date = prompt_date("  FROM date (YYYY-MM-DD or DD-MM-YYYY): ")
    to_date = prompt_date("  TO   date (YYYY-MM-DD or DD-MM-YYYY): ")

    if from_date > to_date:
        print(f"[ERROR] FROM date ({from_date}) is after TO date ({to_date}). Exiting.")
        raise SystemExit(1)

    return from_date, to_date


def _where_clause_for_table(t: TableInfo) -> str:
    """Build the WHERE predicate for date range; same logic as DELETE."""
    if t.date_is_native:
        return '"date" >= ? AND "date" <= ?'
    return (
        "strptime(\"date\", '%d-%m-%Y')::DATE >= ? AND "
        "strptime(\"date\", '%d-%m-%Y')::DATE <= ?"
    )


def preview_row_counts(
    con: duckdb.DuckDBPyConnection,
    tables: Sequence[TableInfo],
    from_date: date,
    to_date: date,
) -> None:
    """
    For each table, run SELECT COUNT(*) with the same date-range predicate
    and print how many rows would be deleted. No data is modified.
    """
    params: list[date] = [from_date, to_date]
    print("\nPreview (rows that would be deleted per table):")
    total = 0
    for t in tables:
        qualified = t.qualified
        where = _where_clause_for_table(t)
        sql = f"SELECT COUNT(*) FROM {qualified} WHERE {where}"
        try:
            row = con.execute(sql, params).fetchone()
            count = int(row[0]) if row else 0
            total += count
            print(f"  - {t.schema}.{t.name}: {count} row(s)")
        except Exception as e:
            print(f"  - {t.schema}.{t.name}: [ERROR] {e}")
    print(f"  Total: {total} row(s)")
    if total == 0:
        print("  --> No rows in this date range. Possible reasons:")
        print("      (1) Wrong year (e.g. data is 2026 but you entered 2025).")
        print("      (2) Data was already deleted in a previous run.")
        print("      (3) These tables have no data for this range.")
        print("  Only type Y if you expect 0 deletions; otherwise cancel and fix the dates.")
    print()


def confirm_deletion(tables: Sequence[TableInfo], from_date: date, to_date: date) -> None:
    """
    Print a summary and require explicit confirmation before proceeding.

    Any response other than 'Y' (case-insensitive) aborts safely.
    """
    print("\n==============================================================")
    print(" PENDING DELETION SUMMARY (NO CHANGES MADE YET)")
    print("==============================================================")
    print(f"Date range (inclusive): {from_date.isoformat()}  →  {to_date.isoformat()}")
    print("Tables selected:")
    for t in tables:
        print(f"  - {t.schema}.{t.name}")
    print("\nWARNING: This operation will permanently DELETE rows in the")
    print("selected tables where:")
    print("  date >= FROM_DATE AND date <= TO_DATE")
    print("\nThis action CANNOT be undone. Make sure you have backups if needed.")

    answer = input("\nType 'Y' to proceed with deletion, or anything else to cancel: ").strip()
    if answer.lower() != "y":
        print("Confirmation not received. Exiting without making any changes.")
        raise SystemExit(0)


def delete_rows_for_date_range(
    con: duckdb.DuckDBPyConnection,
    tables: Sequence[TableInfo],
    from_date: date,
    to_date: date,
) -> None:
    """
    Execute the DELETE statements for each table and report row counts.
    Then run verification: count rows remaining in range (should be 0 for deleted data).
    """
    params: list[date] = [from_date, to_date]
    print("\nExecuting deletions...")
    print(f"  Date range: {from_date.isoformat()} to {to_date.isoformat()}")
    total_deleted = 0

    for t in tables:
        qualified = t.qualified
        where_clause = _where_clause_for_table(t)
        sql = f"""
            DELETE FROM {qualified}
            WHERE {where_clause}
            RETURNING 1;
        """
        try:
            rows = con.execute(sql, params).fetchall()
            deleted = len(rows)
            total_deleted += deleted
            print(f"  - {t.schema}.{t.name}: deleted {deleted} row(s)")
        except Exception as e:
            print(f"  - {t.schema}.{t.name}: [ERROR] {e}")

    print("\n--------------------------------------------------------------")
    print("Verification (rows remaining in date range after delete):")
    for t in tables:
        qualified = t.qualified
        where = _where_clause_for_table(t)
        try:
            row = con.execute(f"SELECT COUNT(*) FROM {qualified} WHERE {where}", params).fetchone()
            remaining = int(row[0]) if row else 0
            print(f"  - {t.schema}.{t.name}: {remaining} row(s) remaining")
        except Exception as e:
            print(f"  - {t.schema}.{t.name}: [ERROR] {e}")
    print("--------------------------------------------------------------")

    print("\n==============================================================")
    print(f"Deletion complete. Total rows deleted across all tables: {total_deleted}")
    if total_deleted == 0:
        print("(0 = no rows matched the date range. Use Preview above to confirm range/year.)")
    print("==============================================================")


def main(argv: Sequence[str] | None = None) -> int:
    """Main entry point for CLI usage."""
    args = parse_args(argv)
    db_path = args.db

    print("============================================")
    print(" DuckDB Date-Range Deletion Utility (SAFE) ")
    print("============================================")
    print(f"Database file: {db_path}\n")

    con = connect_db(db_path)
    try:
        tables = discover_eligible_tables(con)
        selected_tables = list(prompt_table_selection(tables))  # ensure list for multi-use
        show_data_range_hint(con, selected_tables)  # Show what dates exist in DB
        from_date, to_date = prompt_date_range()
        preview_row_counts(con, selected_tables, from_date, to_date)
        confirm_deletion(selected_tables, from_date, to_date)
        delete_rows_for_date_range(con, selected_tables, from_date, to_date)
    finally:
        try:
            con.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())

