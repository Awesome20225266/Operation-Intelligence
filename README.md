# Zelestra Energy Dashboard

This repo contains:
- `store_data_table_duckdb.py`: loads all `DGR/*.xlsx` into `master.duckdb`
- `dashboard.py`: Streamlit dashboard UI backed by `master.duckdb`

## Setup

```bash
python -m pip install -r requirements.txt
```

## 1) Load/refresh the DuckDB

```bash
python store_data_table_duckdb.py --dgr-dir DGR --db master.duckdb
```

## 2) Run the dashboard

```bash
streamlit run dashboard.py
```

## 3) Supabase (for Comments)

The **Comments** page stores/retrieves comments from Supabase table `zelestra_comments`.

Set credentials using **either** environment variables **or** Streamlit secrets.

### Option A: Streamlit secrets (recommended for local dev)

Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and fill values:

- `SUPABASE_URL`
- `SUPABASE_ANON_KEY` (optional if you use service role)
- `SUPABASE_SERVICE_ROLE_KEY` (optional if you use anon key)

### Option B: Environment variables

```bash
set SUPABASE_URL=...
set SUPABASE_ANON_KEY=...
set SUPABASE_SERVICE_ROLE_KEY=...
```

Optional: set the user email shown in the header:

```bash
set DASHBOARD_USER_EMAIL=you@company.com
```





