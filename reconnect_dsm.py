"""
Re Connect DSM Analysis Module

Self-contained DSM calculation engine (no external dependencies on dsm_dashboard.py)
"""

from __future__ import annotations

import json
import math
import statistics as stats
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from aws_duckdb import get_duckdb_connection

# =========================
# DSM Calculation Engine (Inlined from dsm_dashboard.py)
# =========================

@dataclass
class Band:
    direction: str        # "UI" | "OI"
    lower_pct: float      # inclusive lower bound
    upper_pct: float      # exclusive upper bound (use 999 for open-ended)
    rate_type: str        # "FLAT" | "PPA_FRAC" | "PPA_MULT" | "SCALED"
    rate_value: float     # flat ₹/kWh OR fraction/multiple 'a' in scaled
    rate_slope: float     # slope 'b' for scaled, else 0
    loss_zone: bool       # True → goes to OI_Loss (only used when direction="OI")

RATE_FLAT = "FLAT"
RATE_FRAC = "PPA_FRAC"
RATE_MULT = "PPA_MULT"
RATE_SCALED = "SCALED"

MODE_DEFAULT = "DEFAULT"
MODE_DYNAMIC = "DYNAMIC"

def safe_mode(values: List[float]) -> float:
    """MODE with sensible fallback (median) when multimodal/empty."""
    vals = [v for v in values if pd.notna(v)]
    if not vals:
        return 0.0
    try:
        # statistics.mode() requires exactly one most common value
        # In Python 3.8+, it raises StatisticsError if multiple modes exist
        return float(stats.mode(vals))
    except (ValueError, TypeError):
        # Also catch StatisticsError (Python 3.8+) which is raised when no unique mode
        # Note: StatisticsError is a subclass of ValueError, so it's caught here
        # Fallback to median if no unique mode or other error
        return float(np.median(vals))

def denominator_and_basis(avc: float, sch: float, mode: str, dyn_x: float) -> float:
    """Return denominator (also used as basis for energy) as per rule."""
    if mode == MODE_DYNAMIC:
        return (dyn_x * avc) + ((1.0 - dyn_x) * sch)
    return avc

def direction_from(actual: float, scheduled: float) -> str:
    if actual < scheduled:
        return "UI"
    elif actual > scheduled:
        return "OI"
    return "FLAT"

def slice_pct(abs_err: float, lower: float, upper: float) -> float:
    return max(0.0, min(abs_err, upper) - lower)

def kwh_from_slice(slice_pct_val: float, basis_mw: float) -> float:
    # 15-min block → 0.25 h; MW → kW × 1000; energy = P(kw)*h
    return (slice_pct_val / 100.0) * basis_mw * 0.25 * 1000.0

def band_rate(ppa: float, rate_type: str, rate_value: float, rate_slope: float, abs_err: float) -> float:
    if rate_type == RATE_FLAT:
        return rate_value
    if rate_type in (RATE_FRAC, RATE_MULT):
        return rate_value * ppa
    if rate_type == RATE_SCALED:
        return rate_value + rate_slope * abs_err
    return 0.0

def parse_bands_from_settings(settings_rows: List[Dict[str, Any]]) -> Tuple[List[Band], pd.DataFrame]:
    """Convert UI bands rows to Band models expected by the new engine.
    - Maps legacy rate_type values to new RATE_* constants
    - Defaults loss_zone to False if not provided
    - Converts flat_per_mwh to per-kWh
    """
    type_map = {
        "flat_per_kwh": RATE_FLAT,
        "ppa_fraction": RATE_FRAC,
        "ppa_multiple": RATE_MULT,
        "flat_per_mwh": RATE_FLAT,
        "scaled_excess": RATE_SCALED,
    }
    out: List[Band] = []
    for r in settings_rows or []:
        legacy_type = str(r.get("rate_type", "")).strip().lower()
        mapped_type = type_map.get(legacy_type, RATE_FLAT)
        raw_rate_value = float(r.get("rate_value", 0) or 0)
        # Convert MWh -> kWh if needed
        rate_value = (raw_rate_value / 1000.0) if legacy_type == "flat_per_mwh" else raw_rate_value
        out.append(Band(
            direction=str(r.get("direction", "")).strip().upper(),
            lower_pct=float(r.get("lower_pct", 0) or 0),
            upper_pct=float(r.get("upper_pct", 0) or 0),
            rate_type=mapped_type,
            rate_value=rate_value,
            rate_slope=float(r.get("excess_slope_per_pct", 0) or 0),
            loss_zone=bool(r.get("loss_zone", False)),
        ))
    # sort by direction then lower_pct
    out.sort(key=lambda b: (b.direction, b.lower_pct, b.upper_pct))
    bands_df = pd.DataFrame([b.__dict__ for b in out])
    return out, bands_df

def compute_slot_row(slot: Dict[str, Any], bands: List[Band], mode: str, dyn_x: float) -> Dict[str, Any]:
    """Return numeric metrics for a single 15-min slot using the new band engine."""
    avc = float(slot["AvC_MW"]) if pd.notna(slot.get("AvC_MW")) else 0.0
    sch = float(slot["Scheduled_MW"]) if pd.notna(slot.get("Scheduled_MW")) else 0.0
    act = float(slot["Actual_MW"]) if pd.notna(slot.get("Actual_MW")) else 0.0
    ppa = float(slot["PPA"]) if pd.notna(slot.get("PPA")) else 0.0

    denom = denominator_and_basis(avc, sch, mode, dyn_x)
    err_pct = 0.0 if denom == 0 else (act - sch) / denom * 100.0
    dirn = direction_from(act, sch)
    abs_err = abs(err_pct)

    ui_dev_kwh = 0.0
    oi_dev_kwh = 0.0
    ui_dsm = 0.0
    oi_dsm = 0.0
    oi_loss = 0.0

    for b in bands:
        if b.direction != dirn:
            continue
        sp = slice_pct(abs_err, b.lower_pct, b.upper_pct)
        if sp <= 0:
            continue
        kwh = kwh_from_slice(sp, denom)
        rate = band_rate(ppa, b.rate_type, b.rate_value, b.rate_slope, abs_err)
        amt = kwh * rate
        if dirn == "UI":
            ui_dev_kwh += kwh
            ui_dsm += amt
        elif dirn == "OI":
            oi_dev_kwh += kwh
            if b.loss_zone:
                oi_loss += amt
            else:
                oi_dsm += amt

    rev_act = act * 0.25 * 1000.0 * ppa
    rev_sch = sch * 0.25 * 1000.0 * ppa
    total_dsm = ui_dsm + oi_dsm
    revenue_loss = total_dsm + oi_loss

    reached = [b for b in bands if b.direction == dirn and abs_err >= b.lower_pct]
    band_level = ""
    if reached:
        top = max(reached, key=lambda x: x.upper_pct)
        lo = int(top.lower_pct)
        up = ("" if top.upper_pct >= 999 else int(top.upper_pct))
        band_level = f"{dirn} {lo}–{up}%" if up != "" else f"{dirn} >{lo}%"

    return {
        "error_pct": err_pct,
        "direction": dirn,
        "abs_err": abs_err,
        "band_level": band_level,
        "UI_Energy_deviation_bands": ui_dev_kwh,
        "OI_Energy_deviation_bands": oi_dev_kwh,
        "Revenue_as_per_generation": rev_act,
        "Scheduled_Revenue_as_per_generation": rev_sch,
        "UI_DSM": ui_dsm,
        "OI_DSM": oi_dsm,
        "OI_Loss": oi_loss,
        "Total_DSM": total_dsm,
        "Revenue_Loss": revenue_loss,
    }


# -----------------------------
# Data Loading
# -----------------------------

@st.cache_data(show_spinner=False)
def get_reconnect_plants(db_path: str) -> List[str]:
    """Get unique plant names from reconnect table."""
    con = get_duckdb_connection(db_local=db_path)
    try:
        result = con.execute("""
            SELECT DISTINCT plant_name 
            FROM reconnect 
            ORDER BY plant_name
        """).fetchall()
        return [r[0] for r in result] if result else []
    finally:
        con.close()


@st.cache_data(show_spinner=False)
def get_reconnect_date_range(db_path: str, plant_names: List[str]) -> Tuple[Optional[date], Optional[date]]:
    """Get min and max dates for selected plants."""
    if not plant_names:
        return None, None
    
    con = get_duckdb_connection(db_local=db_path)
    try:
        placeholders = ",".join(["?"] * len(plant_names))
        result = con.execute(f"""
            SELECT MIN(date) as min_date, MAX(date) as max_date
            FROM reconnect
            WHERE plant_name IN ({placeholders})
        """, plant_names).fetchone()
        
        if result and result[0] and result[1]:
            return result[0], result[1]
        return None, None
    finally:
        con.close()


def load_reconnect_data(
    db_path: str,
    plant_names: List[str],
    start_date: date,
    end_date: date,
    avc_mw: float,
    ppa: Optional[float] = None
) -> pd.DataFrame:
    """Load data from reconnect table and prepare for DSM calculation."""
    con = get_duckdb_connection(db_local=db_path)
    try:
        placeholders = ",".join(["?"] * len(plant_names))
        query = f"""
            SELECT 
                plant_name,
                date,
                time,
                block,
                forecast_da_mw,
                actual_mw,
                accepted_schedule_eod_mw,
                generated_schedule_mw
            FROM reconnect
            WHERE plant_name IN ({placeholders})
              AND date >= ?
              AND date <= ?
            ORDER BY plant_name, date, block
        """
        
        params = list(plant_names) + [start_date, end_date]
        df = con.execute(query, params).fetchdf()
        
        if df.empty:
            return pd.DataFrame()
        
        # Map to DSM engine expected columns
        df['region'] = 'NRPC'  # Default region, can be derived from plant_name if needed
        df['AvC_MW'] = avc_mw
        df['Scheduled_MW'] = df['accepted_schedule_eod_mw']
        df['Actual_MW'] = df['actual_mw']
        
        # Use user-provided PPA only if explicitly provided
        # Don't set default PPA here - let it be handled per-plant in compute_dsm_for_plant
        if ppa is not None:
            df['PPA'] = ppa
        # If ppa is None, don't set PPA column - it will be handled per-plant later
        
        # Derive from_time and to_time from block
        def block_to_time(block: int) -> time:
            """Convert block (1-96) to time."""
            hour = (block - 1) // 4
            minute = ((block - 1) % 4) * 15
            return time(hour, minute, 0)
        
        df['from_time'] = df['block'].apply(block_to_time)
        df['to_time'] = df['block'].apply(lambda b: block_to_time(min(96, b + 1)))
        df['time_block'] = df['block']
        
        return df
    finally:
        con.close()


# -----------------------------
# Virtual Data Correction Layer
# -----------------------------

def detect_missing_actual_blocks(df: pd.DataFrame) -> pd.Series:
    """
    Detect blocks where schedule exists but actual is missing/zero.
    
    Condition: Accepted_Schedule_EOD_MW > 0 AND Actual_MW <= 0
    """
    return (df['accepted_schedule_eod_mw'] > 0) & (df['Actual_MW'] <= 0)


def detect_missing_schedule_blocks(df: pd.DataFrame) -> pd.Series:
    """
    Detect blocks where actual exists but schedule is missing/zero.
    
    Condition: Actual_MW > 0 AND Accepted_Schedule_EOD_MW <= 0
    """
    return (df['Actual_MW'] > 0) & (df['accepted_schedule_eod_mw'] <= 0)


def detect_flatline_actual_blocks(df: pd.DataFrame, min_blocks: int = 2) -> pd.Series:
    """
    Detect blocks where actual values are flat (same value for consecutive blocks).
    
    Condition: Actual_MW[t] == Actual_MW[t+1] == ... for >= min_blocks continuous blocks
    """
    df_sorted = df.sort_values(['plant_name', 'date', 'block']).copy()
    flatline_mask = pd.Series(False, index=df_sorted.index)
    
    for plant_name in df_sorted['plant_name'].unique():
        plant_mask = df_sorted['plant_name'] == plant_name
        plant_df = df_sorted[plant_mask].copy()
        
        if len(plant_df) < min_blocks:
            continue
        
        # Group by date to check flatlines within same day
        for date_val in plant_df['date'].unique():
            day_mask = (plant_df['date'] == date_val)
            day_df = plant_df[day_mask].sort_values('block')
            
            if len(day_df) < min_blocks:
                continue
            
            actual_values = day_df['Actual_MW'].values
            block_indices = day_df.index.values
            
            # Check for consecutive equal values
            i = 0
            while i < len(actual_values) - (min_blocks - 1):
                # Check if next min_blocks values are all equal
                if pd.notna(actual_values[i]) and actual_values[i] > 0:
                    consecutive_equal = True
                    for j in range(1, min_blocks):
                        if i + j >= len(actual_values) or actual_values[i + j] != actual_values[i]:
                            consecutive_equal = False
                            break
                    
                    if consecutive_equal:
                        # Mark all consecutive blocks as flatline
                        for j in range(min_blocks):
                            if i + j < len(block_indices):
                                flatline_mask.loc[block_indices[i + j]] = True
                        i += min_blocks
                    else:
                        i += 1
                else:
                    i += 1
    
    return flatline_mask


def detect_anomalies(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Detect all types of anomalies in the dataframe.
    
    Returns dictionary with anomaly masks:
    - 'missing_actual': blocks with missing actual
    - 'missing_schedule': blocks with missing schedule
    - 'flatline_actual': blocks with flat-line actual values
    """
    return {
        'missing_actual': detect_missing_actual_blocks(df),
        'missing_schedule': detect_missing_schedule_blocks(df),
        'flatline_actual': detect_flatline_actual_blocks(df)
    }


def evaluate_excel_formula(
    formula: str,
    row: pd.Series,
    df: pd.DataFrame,
    row_idx: int,
    x_value: Optional[float] = None
) -> float:
    """
    Evaluate Excel-style formula with variable substitution.
    
    Allowed variables:
    - Actual_MW: Current block actual
    - Accepted_Schedule_EOD_MW: Current block schedule
    - Forecast_DA_MW: forecast_da_mw
    - Prev_Actual_MW: Previous block actual
    - Next_Actual_MW: Next block actual
    - AvC_MW: User provided AvC
    - X: User-defined percentage (must be provided as parameter)
    
    Also supports Excel functions: MIN, MAX, ABS, ROUND, etc.
    """
    import math
    import re
    
    # Get current row values
    actual_mw = float(row.get('Actual_MW', 0) or 0)
    schedule_mw = float(row.get('accepted_schedule_eod_mw', 0) or 0)
    forecast_da_mw = float(row.get('forecast_da_mw', 0) or 0)
    avc_mw = float(row.get('AvC_MW', 0) or 0)
    
    # Get previous and next actual values
    plant_name = row.get('plant_name')
    date_val = row.get('date')
    block = row.get('block', 0)
    
    prev_actual_mw = 0.0
    next_actual_mw = 0.0
    
    if plant_name and date_val is not None:
        plant_date_df = df[(df['plant_name'] == plant_name) & (df['date'] == date_val)].sort_values('block')
        block_indices = plant_date_df.index.tolist()
        
        if row_idx in block_indices:
            current_pos = block_indices.index(row_idx)
            if current_pos > 0:
                prev_idx = block_indices[current_pos - 1]
                prev_actual_mw = float(df.loc[prev_idx, 'Actual_MW'] if pd.notna(df.loc[prev_idx, 'Actual_MW']) else 0)
            if current_pos < len(block_indices) - 1:
                next_idx = block_indices[current_pos + 1]
                next_actual_mw = float(df.loc[next_idx, 'Actual_MW'] if pd.notna(df.loc[next_idx, 'Actual_MW']) else 0)
    
    # Variable substitution dictionary
    # Note: X must be provided separately as it's user-defined
    variables = {
        'Actual_MW': actual_mw,
        'Accepted_Schedule_EOD_MW': schedule_mw,
        'Forecast_DA_MW': forecast_da_mw,
        'Prev_Actual_MW': prev_actual_mw,
        'Next_Actual_MW': next_actual_mw,
        'AvC_MW': avc_mw,
    }
    
    # Replace variables in formula (case-insensitive)
    formula_lower = formula
    for var_name, var_value in variables.items():
        # Replace variable name (whole word, case-insensitive)
        pattern = re.compile(r'\b' + re.escape(var_name) + r'\b', re.IGNORECASE)
        formula_lower = pattern.sub(str(var_value), formula_lower)
    
    # Replace X% with X/100 (handle percentage notation) - but preserve X if x_value provided
    if x_value is not None:
        # Replace X% with (x_value/100) first
        formula_lower = re.sub(r'\bX\s*%', f'({x_value}/100)', formula_lower, flags=re.IGNORECASE)
        # Replace standalone X with x_value
        formula_lower = re.sub(r'\bX\b', str(x_value), formula_lower, flags=re.IGNORECASE)
    
    # Replace numeric percentages (e.g., 95% becomes (95/100))
    formula_lower = re.sub(r'(\d+(?:\.\d+)?)%', r'(\1/100)', formula_lower)
    
    # Replace Excel functions with Python equivalents
    formula_lower = re.sub(r'\bMIN\s*\(', 'min(', formula_lower, flags=re.IGNORECASE)
    formula_lower = re.sub(r'\bMAX\s*\(', 'max(', formula_lower, flags=re.IGNORECASE)
    formula_lower = re.sub(r'\bABS\s*\(', 'abs(', formula_lower, flags=re.IGNORECASE)
    formula_lower = re.sub(r'\bROUND\s*\(', 'round(', formula_lower, flags=re.IGNORECASE)
    formula_lower = re.sub(r'\bSQRT\s*\(', 'math.sqrt(', formula_lower, flags=re.IGNORECASE)
    formula_lower = re.sub(r'\bPOW\s*\(', 'math.pow(', formula_lower, flags=re.IGNORECASE)
    
    # Safe evaluation with math functions available
    try:
        result = eval(formula_lower, {"__builtins__": {}}, {"math": math, "min": min, "max": max, "abs": abs, "round": round})
        return float(result) if pd.notna(result) else 0.0
    except Exception as e:
        st.error(f"Formula evaluation error: {e}. Formula: {formula}")
        return 0.0


def apply_corrections(
    df_original: pd.DataFrame,
    corrections: Dict[str, Dict[str, Any]]  # {plant_name: {issue_type: {formula, x_value}}}
) -> pd.DataFrame:
    """
    Apply virtual corrections to dataframe based on anomaly masks and formulas.
    
    corrections structure:
    {
        'plant_name': {
            'missing_actual': {'formula': 'X% * Accepted_Schedule_EOD_MW', 'x_value': 95.0},
            'missing_schedule': {'formula': 'Actual_MW * 1.02', 'x_value': None},
            'flatline_actual': {'formula': '(Prev_Actual_MW + Next_Actual_MW) / 2', 'x_value': None}
        }
    }
    
    Returns dataframe with corrections applied and 'is_corrected' flag added.
    """
    df_virtual = df_original.copy()
    
    # Initialize correction flag
    df_virtual['is_corrected'] = False
    
    # Detect all anomalies
    anomalies = detect_anomalies(df_virtual)
    
    # Apply corrections plant by plant
    for plant_name, plant_corrections in corrections.items():
        plant_mask = df_virtual['plant_name'] == plant_name
        
        # Apply missing actual correction
        if 'missing_actual' in plant_corrections:
            missing_actual_mask = anomalies['missing_actual'] & plant_mask
            if missing_actual_mask.any():
                correction_config = plant_corrections['missing_actual']
                formula = correction_config.get('formula', '')
                x_value = correction_config.get('x_value')
                
                if formula:
                    for idx in df_virtual[missing_actual_mask].index:
                        row = df_virtual.loc[idx]
                        original_actual = df_virtual.loc[idx, 'Actual_MW']
                        corrected_value = evaluate_excel_formula(formula, row, df_virtual, idx, x_value)
                        df_virtual.loc[idx, 'Actual_MW'] = corrected_value
                        # Mark as corrected if value changed
                        if abs(corrected_value - original_actual) > 1e-6:
                            df_virtual.loc[idx, 'is_corrected'] = True
        
        # Apply missing schedule correction
        if 'missing_schedule' in plant_corrections:
            missing_schedule_mask = anomalies['missing_schedule'] & plant_mask
            if missing_schedule_mask.any():
                correction_config = plant_corrections['missing_schedule']
                formula = correction_config.get('formula', '')
                x_value = correction_config.get('x_value')
                
                if formula:
                    for idx in df_virtual[missing_schedule_mask].index:
                        row = df_virtual.loc[idx]
                        original_schedule = df_virtual.loc[idx, 'Scheduled_MW']
                        corrected_value = evaluate_excel_formula(formula, row, df_virtual, idx, x_value)
                        df_virtual.loc[idx, 'Scheduled_MW'] = corrected_value
                        # Mark as corrected if value changed
                        if abs(corrected_value - original_schedule) > 1e-6:
                            df_virtual.loc[idx, 'is_corrected'] = True
        
        # Apply flatline actual correction
        if 'flatline_actual' in plant_corrections:
            flatline_mask = anomalies['flatline_actual'] & plant_mask
            if flatline_mask.any():
                correction_config = plant_corrections['flatline_actual']
                formula = correction_config.get('formula', '')
                x_value = correction_config.get('x_value')
                
                if formula:
                    for idx in df_virtual[flatline_mask].index:
                        row = df_virtual.loc[idx]
                        original_actual = df_virtual.loc[idx, 'Actual_MW']
                        corrected_value = evaluate_excel_formula(formula, row, df_virtual, idx, x_value)
                        df_virtual.loc[idx, 'Actual_MW'] = corrected_value
                        # Mark as corrected if value changed
                        if abs(corrected_value - original_actual) > 1e-6:
                            df_virtual.loc[idx, 'is_corrected'] = True
    
    return df_virtual


def get_anomaly_summary(df: pd.DataFrame, plant_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get summary of anomalies detected in the dataframe.
    """
    if plant_name:
        df_filtered = df[df['plant_name'] == plant_name].copy()
    else:
        df_filtered = df.copy()
    
    anomalies = detect_anomalies(df_filtered)
    
    summary = {
        'total_blocks': len(df_filtered),
        'missing_actual_count': int(anomalies['missing_actual'].sum()),
        'missing_schedule_count': int(anomalies['missing_schedule'].sum()),
        'flatline_actual_count': int(anomalies['flatline_actual'].sum()),
        'missing_actual_pct': round(anomalies['missing_actual'].sum() / len(df_filtered) * 100, 2) if len(df_filtered) > 0 else 0.0,
        'missing_schedule_pct': round(anomalies['missing_schedule'].sum() / len(df_filtered) * 100, 2) if len(df_filtered) > 0 else 0.0,
        'flatline_actual_pct': round(anomalies['flatline_actual'].sum() / len(df_filtered) * 100, 2) if len(df_filtered) > 0 else 0.0,
    }
    
    return summary


def anomaly_badge(value: int) -> str:
    """Generate color-coded badge for anomaly count. GREEN if 0, RED if > 0."""
    if value == 0:
        return f"<span style='color:green;font-weight:bold'>✔ {value}</span>"
    else:
        return f"<span style='color:red;font-weight:bold'>✖ {value}</span>"


# -----------------------------
# Plant Defaults Configuration
# -----------------------------

PLANT_DEFAULTS = {
    'Achampet': {'avc_mw': 10.0, 'setting_type': 'Ghanpur'},
    'Renjal': {'avc_mw': 15.0, 'setting_type': 'Ghanpur'},
    'Ghanpur': {'avc_mw': 15.0, 'setting_type': 'Ghanpur'},
    'Gummadidala': {'avc_mw': 15.0, 'setting_type': 'Ghanpur'},
    'Thukkapur': {'avc_mw': 15.0, 'setting_type': 'Ghanpur'},
    'Karya': {'avc_mw': 20.0, 'setting_type': 'Nanj'},
    'Chincholi': {'avc_mw': 20.0, 'setting_type': 'Nanj'},
    'Padmajiwadi': {'avc_mw': 10.0, 'setting_type': 'Ghanpur'},
}

def get_plant_default(plant_name: str, field: str) -> Optional[Any]:
    """Get default value for a plant field (avc_mw or setting_type)."""
    return PLANT_DEFAULTS.get(plant_name, {}).get(field)


# -----------------------------
# DSM Settings Management
# -----------------------------

SETTINGS_FILE = Path("reconnect_dsm_settings.json")


def load_dsm_settings() -> Dict[str, Any]:
    """Load saved DSM settings from JSON file."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_dsm_settings(settings: Dict[str, Any]) -> None:
    """Save DSM settings to JSON file."""
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f, indent=2)


def get_default_bands() -> List[Dict[str, Any]]:
    """Get default DSM bands configuration."""
    return [
        {"direction": "UI", "lower_pct": 0.0, "upper_pct": 15.0, "tolerance_cut_pct": 15.0,
         "rate_type": "flat_per_kwh", "rate_value": 0.0, "excess_slope_per_pct": 0.0, "loss_zone": False, "label": "UI ≤15% (no penalty)"},
        {"direction": "UI", "lower_pct": 15.0, "upper_pct": 20.0, "tolerance_cut_pct": 15.0,
         "rate_type": "ppa_fraction", "rate_value": 0.10, "excess_slope_per_pct": 0.0, "loss_zone": False, "label": "UI 15–20% (10% of PPA)"},
        {"direction": "UI", "lower_pct": 20.0, "upper_pct": 1000.0, "tolerance_cut_pct": 20.0,
         "rate_type": "scaled_excess", "rate_value": 3.36, "excess_slope_per_pct": 0.08, "loss_zone": False, "label": "UI >20% (scaled)"},
        {"direction": "OI", "lower_pct": 0.0, "upper_pct": 15.0, "tolerance_cut_pct": 15.0,
         "rate_type": "flat_per_kwh", "rate_value": 0.0, "excess_slope_per_pct": 0.0, "loss_zone": False, "label": "OI ≤15% (no penalty)"},
        {"direction": "OI", "lower_pct": 15.0, "upper_pct": 20.0, "tolerance_cut_pct": 15.0,
         "rate_type": "ppa_fraction", "rate_value": 0.10, "excess_slope_per_pct": 0.0, "loss_zone": False, "label": "OI 15–20% (10% of PPA)"},
        {"direction": "OI", "lower_pct": 20.0, "upper_pct": 1000.0, "tolerance_cut_pct": 20.0,
         "rate_type": "scaled_excess", "rate_value": 3.36, "excess_slope_per_pct": 0.08, "loss_zone": True, "label": "OI >20% (scaled)"},
    ]


# -----------------------------
# DSM Calculation
# -----------------------------

def compute_dsm_for_plant(
    df: pd.DataFrame,
    plant_name: str,
    bands: List[Band],
    mode: str,
    dyn_x: float,
    ppa: Optional[float] = None
) -> Dict[str, Any]:
    """Compute DSM metrics for a single plant."""
    plant_df = df[df['plant_name'] == plant_name].copy()
    
    if plant_df.empty:
        return {
            'plant_name': plant_name,
            'plant_capacity': 0.0,
            'ppa': ppa or 0.0,
            'revenue_loss_pct': 0.0,
            'dsm_loss': 0.0,
            'revenue_loss_per_kwh': 0.0,
        }
    
    # Get AvC_MW (should be uniform, use mode)
    avc_values = plant_df['AvC_MW'].dropna().tolist()
    plant_capacity = safe_mode(avc_values) if avc_values else 0.0
    
    # Track if PPA was explicitly provided (before we potentially default it)
    ppa_was_overridden = ppa is not None
    
    # Get PPA (use provided or derive from data)
    if ppa is None:
        # Check if PPA column exists and has values
        if 'PPA' in plant_df.columns:
            ppa_values = plant_df['PPA'].dropna().tolist()
            if ppa_values:
                ppa = safe_mode(ppa_values)
                # PPA found in data, but wasn't explicitly overridden by user
                ppa_was_overridden = False
            else:
                # No PPA provided and no PPA in data - use default for calculations only
                # This is for internal calculations, but will show as N/A in results
                ppa = 4.0  # Default fallback for calculations when no PPA is specified
                ppa_was_overridden = False
        else:
            # PPA column doesn't exist - use default for calculations only
            ppa = 4.0  # Default fallback for calculations when no PPA is specified
            ppa_was_overridden = False
    
    # Compute DSM for each slot
    total_dsm_loss = 0.0
    total_revenue_as_per_generation = 0.0
    total_actual_energy_kwh = 0.0
    total_scheduled_energy_kwh = 0.0
    
    for _, row in plant_df.iterrows():
        slot = {
            'AvC_MW': row['AvC_MW'],
            'Scheduled_MW': row['Scheduled_MW'],
            'Actual_MW': row['Actual_MW'],
            'PPA': ppa,
        }
        
        result = compute_slot_row(slot, bands, mode, dyn_x)
        
        # Extract Revenue Loss (includes Total_DSM + OI_Loss)
        revenue_loss = result.get('Revenue_Loss', 0.0)
        total_dsm_loss += revenue_loss
        
        # Use Revenue_as_per_generation from compute_slot_row (correct method)
        revenue_as_per_gen = result.get('Revenue_as_per_generation', 0.0)
        total_revenue_as_per_generation += revenue_as_per_gen
        
        # Calculate energy for other purposes (MUs calculation, etc.)
        actual_mw = float(row['Actual_MW']) if pd.notna(row['Actual_MW']) else 0.0
        scheduled_mw = float(row['Scheduled_MW']) if pd.notna(row['Scheduled_MW']) else 0.0
        actual_energy_kwh = actual_mw * 0.25 * 1000.0  # 15-min block
        scheduled_energy_kwh = scheduled_mw * 0.25 * 1000.0  # 15-min block
        
        total_actual_energy_kwh += actual_energy_kwh
        total_scheduled_energy_kwh += scheduled_energy_kwh
    
    # Calculate revenue loss percentage - CORRECT FORMULA
    # Revenue Loss % = (Total DSM Revenue Loss / Sum of Revenue_as_per_generation) × 100
    revenue_loss_pct = (total_dsm_loss / total_revenue_as_per_generation * 100) if total_revenue_as_per_generation > 0 else 0.0
    
    # Revenue loss per kWh
    revenue_loss_per_kwh = (total_dsm_loss / total_actual_energy_kwh)*100 if total_actual_energy_kwh > 0 else 0.0
    
    # Calculate energy in MUs
    # Scheduled Energy (MUs) = sum(Scheduled_MW × 0.25 × 1000) / 1,000,000
    scheduled_energy_mus = total_scheduled_energy_kwh / 1_000_000.0
    
    # Actual Energy (MUs) = sum(Actual_MW × 0.25 × 1000) / 100,000 (non-standard as per user requirement)
    actual_energy_mus = total_actual_energy_kwh / 100_0000.0
    
    return {
        'plant_name': plant_name,
        'plant_capacity': round(plant_capacity, 2),
        'ppa': ppa,  # Keep as float for calculations
        'ppa_was_overridden': ppa_was_overridden,  # Track if PPA was explicitly provided by user
        'scheduled_energy_mus': round(scheduled_energy_mus, 4),
        'actual_energy_mus': round(actual_energy_mus, 4),
        'revenue_loss_pct': round(revenue_loss_pct, 2),
        'dsm_loss': round(total_dsm_loss, 2),
        'revenue_loss_per_kwh': round(revenue_loss_per_kwh, 4),
    }


def compute_detailed_block_data(
    df: pd.DataFrame,
    plant_name: str,
    bands: List[Band],
    mode: str,
    dyn_x: float,
    ppa: float,
    setting_name: str
) -> pd.DataFrame:
    """Compute detailed block-level DSM calculation data for a plant."""
    plant_df = df[df['plant_name'] == plant_name].copy()
    
    if plant_df.empty:
        return pd.DataFrame()
    
    detailed_rows = []
    
    for _, row in plant_df.iterrows():
        slot = {
            'AvC_MW': row['AvC_MW'],
            'Scheduled_MW': row['Scheduled_MW'],
            'Actual_MW': row['Actual_MW'],
            'PPA': ppa,
        }
        
        result = compute_slot_row(slot, bands, mode, dyn_x)
        
        # Calculate energy
        actual_energy_kwh = float(row['Actual_MW']) * 0.25 * 1000.0 if pd.notna(row['Actual_MW']) else 0.0
        scheduled_energy_kwh = float(row['Scheduled_MW']) * 0.25 * 1000.0 if pd.notna(row['Scheduled_MW']) else 0.0
        
        detailed_rows.append({
            'Region': 'NRPC',
            'Plant Name': plant_name,
            'Date': row['date'],
            'Time': row['time'],
            'Block': row['block'],
            'From Time': row['from_time'],
            'To Time': row['to_time'],
            'AvC_MW': row['AvC_MW'],
            'Scheduled_MW': row['Scheduled_MW'],
            'Actual_MW': row['Actual_MW'],
            'PPA': ppa,
            'Error %': round(result.get('error_pct', 0.0), 2),
            'Direction': result.get('direction', 'FLAT'),
            'Abs Error %': round(result.get('abs_err', 0.0), 2),
            'Band Level': result.get('band_level', ''),
            'UI Energy Deviation (kWh)': round(result.get('UI_Energy_deviation_bands', 0.0), 2),
            'OI Energy Deviation (kWh)': round(result.get('OI_Energy_deviation_bands', 0.0), 2),
            'Actual Energy (kWh)': round(actual_energy_kwh, 2),
            'Scheduled Energy (kWh)': round(scheduled_energy_kwh, 2),
            'Revenue as per Generation (₹)': round(result.get('Revenue_as_per_generation', 0.0), 2),
            'Scheduled Revenue (₹)': round(result.get('Scheduled_Revenue_as_per_generation', 0.0), 2),
            'UI DSM (₹)': round(result.get('UI_DSM', 0.0), 2),
            'OI DSM (₹)': result.get('OI_DSM', 0.0),
            'OI Loss (₹)': round(result.get('OI_Loss', 0.0), 2),
            'Total DSM (₹)': round(result.get('Total_DSM', 0.0), 2),
            'Revenue Loss (₹)': round(result.get('Revenue_Loss', 0.0), 2),
            'Custom Setting': setting_name,
        })
    
    return pd.DataFrame(detailed_rows)


def compute_dsm_results(
    df: pd.DataFrame,
    plant_configs: Dict[str, Dict[str, Any]]  # {plant_name: {bands, mode, dyn_x, ppa, setting_name}}
) -> pd.DataFrame:
    """Compute DSM results for all selected plants with individual configurations."""
    results = []
    
    for plant_name, config in plant_configs.items():
        bands = config['bands']
        mode = config['mode']
        dyn_x = config['dyn_x']
        ppa = config.get('ppa')
        
        result = compute_dsm_for_plant(df, plant_name, bands, mode, dyn_x, ppa)
        result['setting_name'] = config.get('setting_name') or 'Default'
        # Format PPA for display - show as None if not overridden
        if not result.get('ppa_was_overridden', False):
            result['ppa'] = None  # Will display as blank/N/A
        else:
            result['ppa'] = round(result['ppa'], 2)
        results.append(result)
    
    return pd.DataFrame(results)


# -----------------------------
# Streamlit UI
# -----------------------------

def render(db_path: str) -> None:
    """Render the Re Connect DSM Analysis page."""
    st.markdown("# 🔌 Re Connect DSM Analysis")
    
    # Initialize session state for results persistence
    if 'dsm_results_df' not in st.session_state:
        st.session_state['dsm_results_df'] = None
    if 'dsm_detailed_df' not in st.session_state:
        st.session_state['dsm_detailed_df'] = None
    if 'dsm_from_date' not in st.session_state:
        st.session_state['dsm_from_date'] = None
    if 'dsm_to_date' not in st.session_state:
        st.session_state['dsm_to_date'] = None
    
    # Load available plants
    plants = get_reconnect_plants(db_path)
    
    if not plants:
        st.warning("No plants found in reconnect table. Please run reconnect_inject.py first.")
        return
    
    # User Inputs
    col_select1, col_select2 = st.columns([3, 1])
    
    # Initialize session state for select all
    if 'select_all_plants' not in st.session_state:
        st.session_state['select_all_plants'] = False
    
    with col_select2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        if st.button("Select All", use_container_width=True, key="btn_select_all"):
            st.session_state['select_all_plants'] = True
            st.rerun()
        
        if st.button("Clear All", use_container_width=True, key="btn_clear_all"):
            st.session_state['select_all_plants'] = False
            st.rerun()
    
    with col_select1:
        # Determine default selection based on Select All state
        # Use session state key for persistence across tab switches
        if "dsm_selected_plants" not in st.session_state:
            st.session_state["dsm_selected_plants"] = plants if st.session_state['select_all_plants'] else []
        elif st.session_state['select_all_plants'] and not st.session_state["dsm_selected_plants"]:
            st.session_state["dsm_selected_plants"] = plants
        
        selected_plants = st.multiselect(
            "Site Name",
            options=plants,
            key="dsm_selected_plants",
            help="Select one or more plants for analysis"
        )
        
        # Update session state if user manually changes selection
        if selected_plants and set(selected_plants) == set(plants):
            st.session_state['select_all_plants'] = True
        elif not selected_plants:
            st.session_state['select_all_plants'] = False
    
    # Load DSM settings
    all_settings = load_dsm_settings()
    setting_names = list(all_settings.keys()) if all_settings else []
    
    # Ensure required default settings exist (Ghanpur, Nanj)
    required_settings = ['Ghanpur', 'Nanj', 'Default']
    for setting_name in required_settings:
        if setting_name not in all_settings:
            all_settings[setting_name] = {
                'bands': get_default_bands(),
                'mode': 'default',
                'dyn_x': 50.0
            }
    
    if all_settings:
        save_dsm_settings(all_settings)
        setting_names = list(all_settings.keys())
    
    if not setting_names:
        # Fallback: Create default setting
        default_name = "Default"
        all_settings[default_name] = {
            'bands': get_default_bands(),
            'mode': 'default',
            'dyn_x': 50.0
        }
        save_dsm_settings(all_settings)
        setting_names = [default_name]
    
    # Per-plant configuration if multiple plants selected
    if len(selected_plants) > 1:
        st.markdown("### ⚙️ Per-Plant Configuration")
        st.info("Configure AvC_MW, PPA, and Setting Type for each selected plant individually.")
        
        plant_configs = {}
        for plant_name in selected_plants:
            with st.expander(f"🔧 {plant_name}", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Auto-fill AvC_MW from plant defaults
                    default_avc = get_plant_default(plant_name, 'avc_mw')
                    avc_mw = st.number_input(
                        f"AvC_MW (MW)",
                        min_value=0.0,
                        value=default_avc if default_avc is not None else 0.0,
                        step=0.1,
                        key=f"avc_{plant_name}",
                        help=f"Available Capacity for {plant_name} (Auto-filled from defaults, editable)"
                    )
                
                with col2:
                    # PPA defaults to ON with value 5.7249
                    use_ppa_override = st.checkbox(
                        "Use PPA override",
                        value=True,  # Default to True
                        key=f"use_ppa_{plant_name}",
                        help="Check to override PPA, otherwise DSM engine will use default/derived PPA"
                    )
                    if use_ppa_override:
                        ppa = st.number_input(
                            f"PPA (₹/kWh)",
                            min_value=0.0,
                            value=5.7249,  # Default value
                            step=0.01,
                            key=f"ppa_{plant_name}",
                            help=f"PPA rate for {plant_name}"
                        )
                    else:
                        ppa = None
                        st.caption("PPA: Using default/derived from DSM engine")
                
                with col3:
                    # Add blank option at the beginning
                    setting_options = [""] + setting_names
                    # Auto-fill Setting Type from plant defaults
                    default_setting = get_plant_default(plant_name, 'setting_type')
                    default_index = 0  # Default to blank
                    if default_setting and default_setting in setting_names:
                        default_index = setting_names.index(default_setting) + 1  # +1 because of blank option
                    
                    setting_name = st.selectbox(
                        f"Setting Type",
                        options=setting_options,
                        index=default_index,
                        key=f"setting_{plant_name}",
                        help=f"DSM setting for {plant_name} (Auto-filled from defaults, editable)"
                    )
                    # Convert empty string to None
                    if setting_name == "":
                        setting_name = None
                
                plant_configs[plant_name] = {
                    'avc_mw': avc_mw,
                    'ppa': ppa if use_ppa_override else None,
                    'setting_name': setting_name
                }
    else:
        # Single plant or no selection - use uniform inputs
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Auto-fill AvC_MW from plant defaults (use first selected plant's default)
            default_avc = None
            if selected_plants:
                default_avc = get_plant_default(selected_plants[0], 'avc_mw')
            
            avc_mw = st.number_input(
                "AvC_MW (MW)",
                min_value=0.0,
                value=default_avc if default_avc is not None else 0.0,
                step=0.1,
                help="Available Capacity in MW (Auto-filled from defaults, editable)"
            )
        
        with col2:
            # PPA defaults to ON with value 5.7249
            use_ppa_override = st.checkbox(
                "Use PPA override", 
                value=True,  # Default to True
                help="Check to override PPA, otherwise DSM engine will use default/derived PPA"
            )
            if use_ppa_override:
                ppa = st.number_input(
                    "PPA (₹/kWh)",
                    min_value=0.0,
                    value=5.7249,  # Default value
                    step=0.01,
                    help="Power Purchase Agreement rate"
                )
            else:
                ppa = None
                st.caption("PPA: Using default/derived from DSM engine")
        
        with col3:
            # Add blank option at the beginning
            setting_options = [""] + setting_names
            # Auto-fill Setting Type from plant defaults (use first selected plant's default)
            default_setting = None
            default_index = 0  # Default to blank
            if selected_plants:
                default_setting = get_plant_default(selected_plants[0], 'setting_type')
                if default_setting and default_setting in setting_names:
                    default_index = setting_names.index(default_setting) + 1  # +1 because of blank option
            
            selected_setting_name = st.selectbox(
                "Setting Type",
                options=setting_options,
                index=default_index,
                help="Select a saved DSM setting configuration (Auto-filled from defaults, editable)"
            )
            # Convert empty string to None
            if selected_setting_name == "":
                selected_setting_name = None
        
        # Create config for single plant
        if selected_plants:
            plant_configs = {
                selected_plants[0]: {
                    'avc_mw': avc_mw,
                    'ppa': ppa if use_ppa_override else None,
                    'setting_name': selected_setting_name
                }
            }
        else:
            plant_configs = {}
    
    # Custom Settings Panel (Collapsible) - show for editing settings
    with st.expander("📋 Custom Setting", expanded=False):
        st.markdown("### Edit DSM Bands Configuration")
        
        # Select which setting to edit
        editing_setting_name = st.selectbox(
            "Select Setting to Edit",
            options=setting_names,
            index=0,
            help="Choose a setting to edit its bands configuration"
        )
        
        # Load selected setting
        current_setting = all_settings.get(editing_setting_name, {})
        bands_data = current_setting.get('bands', get_default_bands())
        mode = current_setting.get('mode', 'default')
        dyn_x = current_setting.get('dyn_x', 50.0)
        
        # Mode selection
        mode_col1, mode_col2 = st.columns(2)
        with mode_col1:
            mode = st.selectbox("Error % Mode", ["default", "dynamic"], index=0 if mode == "default" else 1)
        with mode_col2:
            if mode == "dynamic":
                dyn_x = st.number_input("Dynamic X (%)", min_value=0.0, max_value=100.0, value=dyn_x, step=1.0)
        
        # Bands editor with proper configuration
        st.markdown("**DSM Bands:** (Edit values, add/delete rows)")
        
        # Prepare bands dataframe
        bands_df = pd.DataFrame(bands_data)
        
        # Ensure all required columns exist
        required_cols = ['direction', 'lower_pct', 'upper_pct', 'rate_type', 'rate_value', 
                        'excess_slope_per_pct', 'loss_zone', 'label', 'tolerance_cut_pct']
        for col in required_cols:
            if col not in bands_df.columns:
                if col == 'tolerance_cut_pct':
                    bands_df[col] = bands_df['lower_pct']  # Default to lower_pct
                elif col == 'label':
                    # Auto-generate labels if missing
                    bands_df[col] = bands_df.apply(
                        lambda row: f"{row['direction']} {row['lower_pct']:.1f}–{row['upper_pct']:.1f}%" 
                        if row['upper_pct'] < 1000 else f"{row['direction']} >{row['lower_pct']:.1f}%",
                        axis=1
                    )
                else:
                    bands_df[col] = 0.0 if col != 'direction' and col != 'rate_type' else ('UI' if col == 'direction' else 'flat_per_kwh')
        
        # Configure column types for data editor
        column_config = {
            "direction": st.column_config.SelectboxColumn(
                "Direction",
                help="UI (Under Injection) or OI (Over Injection)",
                options=["UI", "OI"],
                required=True,
                width="small"
            ),
            "lower_pct": st.column_config.NumberColumn(
                "Lower %",
                help="Lower bound percentage (inclusive)",
                min_value=0.0,
                max_value=1000.0,
                step=0.1,
                format="%.1f",
                required=True
            ),
            "upper_pct": st.column_config.NumberColumn(
                "Upper %",
                help="Upper bound percentage (exclusive, use 1000 for open-ended)",
                min_value=0.0,
                max_value=1000.0,
                step=0.1,
                format="%.1f",
                required=True
            ),
            "rate_type": st.column_config.SelectboxColumn(
                "Rate Type",
                help="Type of rate calculation",
                options=["flat_per_kwh", "ppa_fraction", "ppa_multiple", "flat_per_mwh", "scaled_excess"],
                required=True,
                width="medium"
            ),
            "rate_value": st.column_config.NumberColumn(
                "Rate Value",
                help="Rate value (flat ₹/kWh, fraction, or base for scaled)",
                min_value=0.0,
                step=0.01,
                format="%.2f",
                required=True
            ),
            "excess_slope_per_pct": st.column_config.NumberColumn(
                "Excess Slope %",
                help="Slope per percentage point (for scaled_excess)",
                min_value=0.0,
                step=0.01,
                format="%.2f"
            ),
            "loss_zone": st.column_config.CheckboxColumn(
                "Loss Zone",
                help="If checked, goes to OI_Loss (only for OI direction)",
                default=False
            ),
            "label": st.column_config.TextColumn(
                "Label",
                help="Auto-generated from lower/upper pct",
                disabled=True
            ),
            "tolerance_cut_pct": st.column_config.NumberColumn(
                "Tolerance Cut %",
                help="Tolerance cutoff percentage",
                min_value=0.0,
                step=0.1,
                format="%.1f",
                default=0.0
            )
        }
        
        # Display bands editor
        edited_bands_df = st.data_editor(
            bands_df[required_cols],
            column_config=column_config,
            width='stretch',
            num_rows="dynamic",  # Allow adding/deleting rows
            hide_index=True,
            key=f"bands_editor_{editing_setting_name}"  # Unique key for each setting
        )
        
        # Auto-update labels based on lower_pct and upper_pct after editing
        if len(edited_bands_df) > 0:
            edited_bands_df = edited_bands_df.copy()
            # Regenerate labels from lower_pct and upper_pct
            edited_bands_df['label'] = edited_bands_df.apply(
                lambda row: (
                    f"{row['direction']} {row['lower_pct']:.1f}–{row['upper_pct']:.1f}%" 
                    if row['upper_pct'] < 1000 
                    else f"{row['direction']} >{row['lower_pct']:.1f}%"
                ),
                axis=1
            )
            
            st.caption("💡 Labels are auto-generated from Lower % and Upper % values. They update automatically when you save.")
        
        # Save setting
        new_setting_name = st.text_input("Save as new setting name", value="")
        col_save1, col_save2 = st.columns(2)
        with col_save1:
            if st.button("💾 Save Setting"):
                if new_setting_name:
                    all_settings[new_setting_name] = {
                        'bands': edited_bands_df.to_dict('records'),
                        'mode': mode,
                        'dyn_x': dyn_x
                    }
                    save_dsm_settings(all_settings)
                    st.success(f"Setting '{new_setting_name}' saved!")
                    st.rerun()
        with col_save2:
            if st.button("🔄 Update Current Setting"):
                all_settings[editing_setting_name] = {
                    'bands': edited_bands_df.to_dict('records'),
                    'mode': mode,
                    'dyn_x': dyn_x
                }
                save_dsm_settings(all_settings)
                st.success(f"Setting '{editing_setting_name}' updated!")
                st.rerun()
    
    # Date Range
    col5, col6 = st.columns(2)
    
    # Get date range once
    if selected_plants:
        min_date, max_date = get_reconnect_date_range(db_path, selected_plants)
    else:
        min_date, max_date = None, None
    
    with col5:
        # Default to today's date, but ensure it's within min/max range if available
        today = date.today()
        if min_date and max_date:
            # Ensure default value is within the allowed range
            default_from = min(max(today, min_date), max_date)
            from_date = st.date_input(
                "From Date",
                value=default_from,
                min_value=min_date,
                max_value=max_date,
                key="reconnect_from_date"
            )
        else:
            from_date = st.date_input(
                "From Date",
                value=today,
                key="reconnect_from_date"
            )
    
    with col6:
        # Default to today's date, but ensure it's within min/max range if available
        today = date.today()
        if min_date and max_date:
            # Ensure default value is within the allowed range
            default_to = min(max(today, min_date), max_date)
            to_date = st.date_input(
                "To Date",
                value=default_to,
                min_value=min_date,
                max_value=max_date,
                key="reconnect_to_date"
            )
        else:
            to_date = st.date_input(
                "To Date",
                value=today,
                key="reconnect_to_date"
            )
    
    # Data Quality Remarks Section (Auto-Generated)
    if selected_plants and from_date and to_date and from_date <= to_date:
        st.markdown("---")
        st.markdown("### 📊 Data Quality Remarks")
        st.caption("Automatically computed data health statistics for selected plants and date range")
        
        # Load data to check quality
        with st.spinner("Analyzing data quality..."):
            temp_df = load_reconnect_data(
                db_path,
                selected_plants,
                from_date,
                to_date,
                50.0,  # Default AvC for quality check
                None
            )
            
            if not temp_df.empty:
                # Get per-plant summaries for consistent reporting
                plant_summaries = {}
                for plant_name in selected_plants:
                    plant_summaries[plant_name] = get_anomaly_summary(temp_df, plant_name)
                
                # Show warnings per-plant to match table format
                warnings_shown = False
                
                # Check for missing actual data (per-plant)
                missing_actual_plants = []
                for plant_name, summary in plant_summaries.items():
                    if summary['missing_actual_count'] > 0:
                        missing_actual_plants.append(f"{plant_name}: {summary['missing_actual_count']} blocks ({summary['missing_actual_pct']}%)")
                
                if missing_actual_plants:
                    plant_list = ", ".join(missing_actual_plants)
                    st.warning(
                        f"🔴 **Missing Actual Data**: Actual power data is missing in the following plants during the selected date range. "
                        f"Schedule exists but actual generation telemetry is missing/zero.\n\n"
                        f"**{plant_list}**"
                    )
                    warnings_shown = True
                
                # Check for missing schedule data (per-plant)
                missing_schedule_plants = []
                for plant_name, summary in plant_summaries.items():
                    if summary['missing_schedule_count'] > 0:
                        missing_schedule_plants.append(f"{plant_name}: {summary['missing_schedule_count']} blocks ({summary['missing_schedule_pct']}%)")
                
                if missing_schedule_plants:
                    plant_list = ", ".join(missing_schedule_plants)
                    st.warning(
                        f"🔴 **Missing Schedule Data**: Schedule data is missing in the following plants during the selected date range. "
                        f"Plant is generating but schedule is not published/missing.\n\n"
                        f"**{plant_list}**"
                    )
                    warnings_shown = True
                
                # Check for flat-line data (per-plant)
                flatline_plants = []
                for plant_name, summary in plant_summaries.items():
                    if summary['flatline_actual_count'] > 0:
                        flatline_plants.append(f"{plant_name}: {summary['flatline_actual_count']} blocks ({summary['flatline_actual_pct']}%)")
                
                if flatline_plants:
                    plant_list = ", ".join(flatline_plants)
                    st.warning(
                        f"🟠 **Possible Bad/Flat-Line Data**: Actual MW repeated continuously across multiple time blocks in the following plants. "
                        f"This may indicate SCADA freeze, meter lock, or telemetry repetition.\n\n"
                        f"**{plant_list}**"
                    )
                    warnings_shown = True
                
                if not warnings_shown:
                    st.success("✅ **Data Quality Check**: No major anomalies detected in the selected date range.")
                
                # Show summary table (matches the warnings above)
                with st.expander("📋 Detailed Anomaly Summary by Plant", expanded=False):
                    summary_data = []
                    for plant_name in selected_plants:
                        summary = plant_summaries[plant_name]
                        summary_data.append({
                            'Plant': plant_name,
                            'Total Blocks': summary['total_blocks'],
                            'Missing Actual': f"{summary['missing_actual_count']} ({summary['missing_actual_pct']}%)",
                            'Missing Schedule': f"{summary['missing_schedule_count']} ({summary['missing_schedule_pct']}%)",
                            'Flat-Line Actual': f"{summary['flatline_actual_count']} ({summary['flatline_actual_pct']}%)"
                        })
                    
                    if summary_data:
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, hide_index=True, use_container_width=True)
            else:
                st.info("No data available for selected plants and date range.")
    
    # Virtual Data Correction Layer
    st.markdown("---")
    st.markdown("## 🔧 Virtual Data Correction Layer")
    st.info("""
    **What-If Simulator**: Detect data anomalies and apply Excel-style formulas to correct them virtually.
    Original data remains untouched. Corrections are applied in-memory before DSM computation.
    """)
    
    use_corrections = st.checkbox(
        "Enable Virtual Data Corrections",
        value=True,  # Default to True
        help="Enable anomaly detection and virtual corrections"
    )
    
    corrections_config = {}
    
    if use_corrections and selected_plants:
        # Detect anomalies for each plant
        with st.spinner("Detecting anomalies..."):
            # Load data temporarily to detect anomalies
            temp_df = load_reconnect_data(
                db_path,
                selected_plants,
                from_date,
                to_date,
                50.0,  # Default AvC for detection
                None
            )
            
            if not temp_df.empty:
                # Show anomaly summary with color-coded badges
                st.markdown("### 📊 Anomaly Summary")
                
                # Create columns for each plant
                num_cols = len(selected_plants)
                if num_cols > 0:
                    summary_cols = st.columns(num_cols)
                
                    for idx, plant_name in enumerate(selected_plants):
                        with summary_cols[idx]:
                            summary = get_anomaly_summary(temp_df, plant_name)
                            st.markdown(f"**{plant_name}**")
                            st.markdown(f"Total Blocks: {summary['total_blocks']}")
                            st.markdown(
                                f"Missing Actual<br>{anomaly_badge(summary['missing_actual_count'])} ({summary['missing_actual_pct']}%)",
                                unsafe_allow_html=True
                            )
                            st.markdown(
                                f"Missing Schedule<br>{anomaly_badge(summary['missing_schedule_count'])} ({summary['missing_schedule_pct']}%)",
                                unsafe_allow_html=True
                            )
                            st.markdown(
                                f"Flatline Actual<br>{anomaly_badge(summary['flatline_actual_count'])} ({summary['flatline_actual_pct']}%)",
                                unsafe_allow_html=True
                            )
                
                # Formula input for each plant
                st.markdown("### ✏️ Correction Formulas")
                st.markdown("""
                **Available Variables:**
                - `Actual_MW`: Current block actual
                - `Accepted_Schedule_EOD_MW`: Current block schedule  
                - `Forecast_DA_MW`: Forecast value
                - `Prev_Actual_MW`: Previous block actual
                - `Next_Actual_MW`: Next block actual
                - `AvC_MW`: Available capacity
                - `X`: User-defined percentage (use X% in formula)
                
                **Excel Functions:** MIN(), MAX(), ABS(), ROUND(), SQRT(), POW()
                
                **Examples:**
                - Fill missing actual: `95% * Accepted_Schedule_EOD_MW` or `0.95 * Accepted_Schedule_EOD_MW`
                - Smooth flat-line: `(Prev_Actual_MW + Next_Actual_MW) / 2`
                - Cap at AvC: `MIN(Actual_MW, AvC_MW)`
                """)
                
                for plant_name in selected_plants:
                    plant_summary = get_anomaly_summary(temp_df, plant_name)
                    
                    with st.expander(f"🔧 Corrections for {plant_name}", expanded=False):
                        corrections_config[plant_name] = {}
                        
                        # Missing Actual Correction
                        if plant_summary['missing_actual_count'] > 0:
                            st.markdown(f"**🔴 Missing Actual Blocks: {plant_summary['missing_actual_count']}**")
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                missing_actual_formula = st.text_input(
                                    "Formula for Missing Actual",
                                    value="100% * Accepted_Schedule_EOD_MW",
                                    key=f"formula_missing_actual_{plant_name}",
                                    help="Formula to compute Actual_MW when schedule exists but actual is missing"
                                )
                            with col2:
                                missing_actual_x = st.number_input(
                                    "X Value (%)",
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=100.0,  # Default to 100
                                    step=1.0,
                                    key=f"x_missing_actual_{plant_name}",
                                    help="Percentage value for X in formula (if using X%)"
                                )
                            
                            if missing_actual_formula:
                                corrections_config[plant_name]['missing_actual'] = {
                                    'formula': missing_actual_formula,
                                    'x_value': missing_actual_x
                                }
                        
                        # Missing Schedule Correction
                        if plant_summary['missing_schedule_count'] > 0:
                            st.markdown(f"**🔴 Missing Schedule Blocks: {plant_summary['missing_schedule_count']}**")
                            missing_schedule_formula = st.text_input(
                                "Formula for Missing Schedule",
                                value="Actual_MW * 1.02",
                                key=f"formula_missing_schedule_{plant_name}",
                                help="Formula to compute Scheduled_MW when actual exists but schedule is missing"
                            )
                            
                            if missing_schedule_formula:
                                corrections_config[plant_name]['missing_schedule'] = {
                                    'formula': missing_schedule_formula,
                                    'x_value': None
                                }
                        
                        # Flat-Line Actual Correction
                        if plant_summary['flatline_actual_count'] > 0:
                            st.markdown(f"**🟠 Flat-Line Actual Blocks: {plant_summary['flatline_actual_count']}**")
                            flatline_formula = st.text_input(
                                "Formula for Flat-Line Actual",
                                value="(Prev_Actual_MW + Next_Actual_MW) / 2",
                                key=f"formula_flatline_{plant_name}",
                                help="Formula to smooth flat-line actual values"
                            )
                            
                            if flatline_formula:
                                corrections_config[plant_name]['flatline_actual'] = {
                                    'formula': flatline_formula,
                                    'x_value': None
                                }
    
    # Upload/Download Template Section
    if use_corrections and selected_plants and from_date and to_date:
        st.markdown("---")
        st.markdown("### 📤 Upload/Download Correction Template")
        st.info("""
        **Workflow:**
        1. Download template CSV with current data
        2. Edit values offline in Excel/CSV
        3. Upload corrected file
        4. Run Analysis to use corrected values
        """)
        
        # Load data for template
        template_df = load_reconnect_data(
            db_path,
            selected_plants,
            from_date,
            to_date,
            50.0,  # Default AvC for template
            None
        )
        
        if not template_df.empty:
            # Prepare template with key columns
            # Ensure Scheduled_MW column exists (mapped from accepted_schedule_eod_mw in load_reconnect_data)
            if 'Scheduled_MW' not in template_df.columns and 'accepted_schedule_eod_mw' in template_df.columns:
                template_df['Scheduled_MW'] = template_df['accepted_schedule_eod_mw']
            
            template_cols = ["plant_name", "date", "block", "Actual_MW", "Scheduled_MW"]
            # Only include columns that exist
            available_cols = [col for col in template_cols if col in template_df.columns]
            template_data = template_df[available_cols].copy()
            
            # Download button
            csv_template = template_data.to_csv(index=False)
            st.download_button(
                "⬇ Download Correction Template",
                csv_template,
                file_name=f"virtual_correction_template_{from_date}_{to_date}.csv",
                mime="text/csv",
                help="Download CSV template with plant_name, date, block, Actual_MW, Scheduled_MW columns"
            )
            
            # Upload corrected file
            uploaded_file = st.file_uploader(
                "Upload Corrected Data (CSV)",
                type=["csv"],
                help="Upload CSV file with corrected Actual_MW and/or Scheduled_MW values"
            )
            
            if uploaded_file:
                try:
                    uploaded_df = pd.read_csv(uploaded_file)
                    
                    # Validate required columns
                    required_cols = ["plant_name", "date", "block"]
                    missing_cols = [col for col in required_cols if col not in uploaded_df.columns]
                    if missing_cols:
                        st.error(f"Missing required columns in uploaded file: {', '.join(missing_cols)}")
                    else:
                        # Parse date column if it's a string
                        if uploaded_df['date'].dtype == 'object':
                            uploaded_df['date'] = pd.to_datetime(uploaded_df['date']).dt.date
                        
                        # Ensure block is integer
                        uploaded_df['block'] = pd.to_numeric(uploaded_df['block'], errors='coerce').astype('Int64')
                        
                        # Ensure Actual_MW and Scheduled_MW are numeric (if present)
                        if 'Actual_MW' in uploaded_df.columns:
                            uploaded_df['Actual_MW'] = pd.to_numeric(uploaded_df['Actual_MW'], errors='coerce')
                        if 'Scheduled_MW' in uploaded_df.columns:
                            uploaded_df['Scheduled_MW'] = pd.to_numeric(uploaded_df['Scheduled_MW'], errors='coerce')
                        
                        # Store uploaded data in session state for use during analysis
                        st.session_state['uploaded_corrections'] = uploaded_df
                        st.success("✅ Uploaded data loaded successfully. Corrections will be applied when you run analysis.")
                        
                        # Show preview
                        with st.expander("📋 Preview Uploaded Data", expanded=False):
                            st.dataframe(uploaded_df.head(20), use_container_width=True)
                            st.caption(f"Total rows: {len(uploaded_df)}")
                except Exception as e:
                    st.error(f"Error reading uploaded file: {e}")
    
    # Visual Insight Section (Collapsed by default; lazy-load so it never blocks Run Analysis)
    if use_corrections and selected_plants and from_date and to_date:
        st.markdown("---")
        with st.expander("## 📈 Visual Insight", expanded=False):
            st.info(
                "Visual Insight is **lazy-loaded** to keep the app responsive.\n\n"
                "Click **Load Visual Insight** to fetch data and render charts."
            )

            if "load_visual_insight" not in st.session_state:
                st.session_state["load_visual_insight"] = False

            c_v1, c_v2 = st.columns([1, 1])
            with c_v1:
                if st.button("Load Visual Insight", type="secondary", use_container_width=True, key="btn_load_visual_insight"):
                    st.session_state["load_visual_insight"] = True
            with c_v2:
                if st.button("Reset Visual Insight", type="secondary", use_container_width=True, key="btn_reset_visual_insight"):
                    st.session_state["load_visual_insight"] = False
                    st.rerun()

            if not st.session_state.get("load_visual_insight", False):
                st.caption("Visual Insight is not loaded yet. This does **not** affect Run Analysis.")
            else:
                # Progress UI for visual insight
                viz_progress = st.progress(0, text="Loading visual insight…")
                try:
                    viz_progress.progress(10, text="Fetching data…")
                    viz_df = load_reconnect_data(
                        db_path,
                        selected_plants,
                        from_date,
                        to_date,
                        50.0,  # Default AvC for visualization
                        None
                    )

                    if viz_df.empty or len(selected_plants) == 0:
                        viz_progress.progress(100, text="Done")
                        st.info("No data available for visualization. Please ensure data exists for selected plants and date range.")
                    else:
                        viz_progress.progress(35, text="Applying virtual corrections…")
                        df_original = viz_df.copy()
                        df_virtual = df_original.copy()

                        # First apply formula-based corrections
                        if corrections_config:
                            df_virtual = apply_corrections(df_original, corrections_config)

                        # Then apply uploaded corrections (override formula-based if both exist)
                        if 'uploaded_corrections' in st.session_state and st.session_state['uploaded_corrections'] is not None:
                            uploaded_df = st.session_state['uploaded_corrections'].copy()

                            # Normalize date column types before merging (convert both to date objects)
                            if 'date' in df_virtual.columns:
                                if pd.api.types.is_datetime64_any_dtype(df_virtual['date']):
                                    df_virtual['date'] = df_virtual['date'].dt.date
                                else:
                                    df_virtual['date'] = pd.to_datetime(df_virtual['date']).dt.date

                            if 'date' in uploaded_df.columns:
                                if pd.api.types.is_datetime64_any_dtype(uploaded_df['date']):
                                    uploaded_df['date'] = uploaded_df['date'].dt.date
                                else:
                                    uploaded_df['date'] = pd.to_datetime(uploaded_df['date']).dt.date

                            # Ensure block is same type
                            df_virtual['block'] = df_virtual['block'].astype('int64')
                            uploaded_df['block'] = uploaded_df['block'].astype('int64')

                            # Merge uploaded corrections
                            df_virtual = df_virtual.merge(
                                uploaded_df[['plant_name', 'date', 'block', 'Actual_MW', 'Scheduled_MW']],
                                on=['plant_name', 'date', 'block'],
                                how='left',
                                suffixes=('', '_uploaded')
                            )

                            # Override values where uploaded data exists
                            mask_actual = df_virtual['Actual_MW_uploaded'].notna()
                            mask_schedule = df_virtual['Scheduled_MW_uploaded'].notna()

                            if mask_actual.any():
                                df_virtual.loc[mask_actual, 'Actual_MW'] = df_virtual.loc[mask_actual, 'Actual_MW_uploaded']
                                df_virtual.loc[mask_actual, 'is_corrected'] = True

                            if mask_schedule.any():
                                df_virtual.loc[mask_schedule, 'Scheduled_MW'] = df_virtual.loc[mask_schedule, 'Scheduled_MW_uploaded']
                                df_virtual.loc[mask_schedule, 'is_corrected'] = True

                            # Clean up temporary columns
                            df_virtual = df_virtual.drop(columns=[col for col in df_virtual.columns if col.endswith('_uploaded')])

                        # Add correction flag if not already present
                        if 'is_corrected' not in df_virtual.columns:
                            df_virtual['is_corrected'] = False

                        viz_progress.progress(55, text="Preparing chart…")

                        # Plant selector for visualization
                        selected_plant_viz = st.selectbox(
                            "Select Plant for Visual Insight",
                            selected_plants,
                            key="viz_plant_selector"
                        )

                        plot_df = df_virtual[df_virtual["plant_name"] == selected_plant_viz].copy()
                        plot_df = plot_df.sort_values(['date', 'block'])

                        if plot_df.empty:
                            viz_progress.progress(100, text="Done")
                            st.warning(f"No data available for {selected_plant_viz} in selected date range.")
                        else:
                            # Create block identifier for x-axis (date + block)
                            plot_df['block_id'] = plot_df.apply(
                                lambda row: f"{row['date']} B{row['block']}", axis=1
                            )

                            fig = go.Figure()

                            # Actual MW line
                            fig.add_trace(go.Scatter(
                                x=plot_df["block_id"],
                                y=plot_df["Actual_MW"],
                                mode="lines+markers",
                                name="Actual MW",
                                marker=dict(color="blue", size=6),
                                line=dict(width=2, color="blue")
                            ))

                            # Accepted Schedule line
                            fig.add_trace(go.Scatter(
                                x=plot_df["block_id"],
                                y=plot_df["Scheduled_MW"],
                                mode="lines+markers",
                                name="Accepted Schedule EOD MW",
                                marker=dict(color="orange", size=6, symbol="diamond"),
                                line=dict(dash="dash", width=2, color="orange")
                            ))

                            # Highlight corrected points
                            corrected = plot_df[plot_df["is_corrected"] == True]
                            if not corrected.empty:
                                fig.add_trace(go.Scatter(
                                    x=corrected["block_id"],
                                    y=corrected["Actual_MW"],
                                    mode="markers",
                                    name="Virtual Correction Applied",
                                    marker=dict(
                                        color="red",
                                        size=12,
                                        symbol="diamond",
                                        line=dict(width=2, color="darkred")
                                    ),
                                    hovertemplate="<b>Corrected Block</b><br>"
                                                  "Date: %{text}<br>"
                                                  "Actual: %{y:.2f} MW<extra></extra>",
                                    text=corrected["date"].astype(str)
                                ))

                            fig.update_layout(
                                title=f"Actual vs Schedule with Virtual Corrections – {selected_plant_viz}",
                                xaxis_title="Block",
                                yaxis_title="MW",
                                legend_title="Legend",
                                height=500,
                                hovermode="x unified",
                                xaxis=dict(
                                    tickangle=-45,
                                    tickmode='linear',
                                    tick0=0,
                                    dtick=max(1, len(plot_df) // 20)  # Show ~20 ticks
                                )
                            )

                            viz_progress.progress(85, text="Rendering chart…")
                            st.plotly_chart(fig, use_container_width=True)

                            # Show correction summary
                            num_corrected = int(plot_df["is_corrected"].sum())
                            if num_corrected > 0:
                                st.info(f"🔴 **{num_corrected} blocks** were virtually corrected for {selected_plant_viz}")
                            else:
                                st.success(f"✅ No virtual corrections applied for {selected_plant_viz}")

                            viz_progress.progress(100, text="Done")
                finally:
                    # keep the progress bar visible at 100% for user feedback
                    pass
    
    # Run Analysis Button + progress (so users see what's happening)
    run_analysis_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    run_progress_slot = st.empty()
    
    if run_analysis_clicked:
        if not selected_plants:
            st.error("Please select at least one plant.")
            st.session_state['dsm_results_df'] = None
            st.session_state['dsm_detailed_df'] = None
            return
        
        if from_date > to_date:
            st.error("From Date must be before or equal to To Date.")
            st.session_state['dsm_results_df'] = None
            st.session_state['dsm_detailed_df'] = None
            return
        
        if not plant_configs:
            st.error("Please configure plant settings.")
            st.session_state['dsm_results_df'] = None
            st.session_state['dsm_detailed_df'] = None
            return
        
        # Load data for all plants (with explicit progress)
        progress = run_progress_slot.progress(0, text="Starting analysis…")
        try:
            progress.progress(10, text="Preparing plant configurations…")
            # Prepare plant configurations with bands
            plant_configs_with_bands = {}
            detailed_data_list = []
            
            for plant_name, config in plant_configs.items():
                setting_name = config.get('setting_name')
                # If setting_name is None or empty, use default bands
                if setting_name:
                    current_setting = all_settings.get(setting_name, {})
                    bands_data = current_setting.get('bands', get_default_bands())
                    mode = current_setting.get('mode', 'default')
                    dyn_x = current_setting.get('dyn_x', 50.0)
                else:
                    # No setting selected - use defaults
                    bands_data = get_default_bands()
                    mode = 'default'
                    dyn_x = 50.0
                
                # Parse bands
                bands_list, _ = parse_bands_from_settings(bands_data)
                
                plant_configs_with_bands[plant_name] = {
                    'bands': bands_list,
                    'mode': mode,
                    'dyn_x': dyn_x,
                    'ppa': config.get('ppa'),
                    'setting_name': setting_name,
                    'avc_mw': config['avc_mw']
                }
            
            progress.progress(30, text="Loading RE-Connect data…")
            # Load data once for all plants
            df = load_reconnect_data(
                db_path,
                selected_plants,
                from_date,
                to_date,
                0.0,  # Will be overridden per plant
                None  # Will be overridden per plant
            )
            
            if df.empty:
                st.warning("No data found for selected plants and date range.")
                return
            
            progress.progress(45, text="Applying plant AvC/PPA settings…")
            # Update AvC_MW and PPA per plant in dataframe
            for plant_name, config in plant_configs_with_bands.items():
                plant_mask = df['plant_name'] == plant_name
                df.loc[plant_mask, 'AvC_MW'] = config['avc_mw']
                # Only set PPA if explicitly provided (user enabled override)
                if config.get('ppa') is not None:
                    df.loc[plant_mask, 'PPA'] = config['ppa']
                else:
                    # If PPA not overridden, remove/clear PPA for this plant
                    # This ensures compute_dsm_for_plant will derive it properly
                    if 'PPA' in df.columns:
                        df.loc[plant_mask, 'PPA'] = None
            
            # Apply virtual corrections if enabled
            if use_corrections:
                progress.progress(60, text="Applying virtual corrections…")
                # First apply formula-based corrections
                if corrections_config:
                    df = apply_corrections(df, corrections_config)
                    
                # Then apply uploaded corrections (override formula-based if both exist)
                if 'uploaded_corrections' in st.session_state and st.session_state['uploaded_corrections'] is not None:
                    uploaded_df = st.session_state['uploaded_corrections'].copy()
                        
                    # Normalize date column types before merging
                    # Convert both to date objects (not datetime)
                    if 'date' in df.columns:
                        if pd.api.types.is_datetime64_any_dtype(df['date']):
                            df['date'] = df['date'].dt.date
                        elif not isinstance(df['date'].iloc[0] if len(df) > 0 else None, date):
                            df['date'] = pd.to_datetime(df['date']).dt.date
                        
                    if 'date' in uploaded_df.columns:
                        if pd.api.types.is_datetime64_any_dtype(uploaded_df['date']):
                            uploaded_df['date'] = uploaded_df['date'].dt.date
                        elif uploaded_df['date'].dtype == 'object':
                            uploaded_df['date'] = pd.to_datetime(uploaded_df['date']).dt.date
                        
                    # Ensure block is same type
                    df['block'] = df['block'].astype('int64')
                    uploaded_df['block'] = uploaded_df['block'].astype('int64')
                        
                    # Merge uploaded corrections
                    df = df.merge(
                        uploaded_df[['plant_name', 'date', 'block', 'Actual_MW', 'Scheduled_MW']],
                        on=['plant_name', 'date', 'block'],
                        how='left',
                        suffixes=('', '_uploaded')
                    )
                        
                    # Override values where uploaded data exists
                    mask_actual = df['Actual_MW_uploaded'].notna()
                    mask_schedule = df['Scheduled_MW_uploaded'].notna()
                        
                    if mask_actual.any():
                        df.loc[mask_actual, 'Actual_MW'] = df.loc[mask_actual, 'Actual_MW_uploaded']
                        df.loc[mask_actual, 'is_corrected'] = True
                        
                    if mask_schedule.any():
                        df.loc[mask_schedule, 'Scheduled_MW'] = df.loc[mask_schedule, 'Scheduled_MW_uploaded']
                        df.loc[mask_schedule, 'is_corrected'] = True
                        
                    # Clean up temporary columns
                    df = df.drop(columns=[col for col in df.columns if col.endswith('_uploaded')])
                        
                    st.success("✅ Virtual corrections applied (formulas + uploaded data). Original data unchanged.")
                elif corrections_config:
                    st.success("✅ Virtual corrections applied. Original data unchanged.")
            
            progress.progress(75, text="Computing DSM summary results…")
            # Compute summary results (using df_virtual if corrections applied, otherwise df_original)
            results_df = compute_dsm_results(df, plant_configs_with_bands)
            
            # Add Region (same as Plant Name), Date Range, and Custom Setting columns
            results_df['Region'] = results_df['plant_name']  # Region = Plant Name
            results_df['Date Range'] = f"{from_date} to {to_date}"  # Date range column
            results_df['Custom Setting'] = results_df['setting_name']
            results_df = results_df.drop(columns=['setting_name'])
            
            # Rename columns to match required output
            results_df = results_df.rename(columns={
                'plant_name': 'Plant name',
                'plant_capacity': 'Plant Capacity',
                'ppa': 'PPA',
                'scheduled_energy_mus': 'Scheduled Energy (MUs)',
                'actual_energy_mus': 'Actual Energy (MUs)',
                'revenue_loss_pct': 'Revenue Loss (%)',
                'dsm_loss': 'DSM Loss',
                'revenue_loss_per_kwh': 'Revenue Loss (₹/kWh)'
            })
            
            # Replace None PPA with "N/A" or blank for display
            results_df['PPA'] = results_df['PPA'].apply(lambda x: 'N/A' if pd.isna(x) or x is None else x)
            
            # Reorder columns as specified
            output_columns = [
                'Region', 
                'Plant name', 
                'Date Range',
                'Plant Capacity', 
                'PPA',
                'Scheduled Energy (MUs)',
                'Actual Energy (MUs)',
                'Revenue Loss (%)', 
                'DSM Loss', 
                'Revenue Loss (₹/kWh)', 
                'Custom Setting'
            ]
            existing_columns = [col for col in output_columns if col in results_df.columns]
            results_df = results_df[existing_columns]
            
            # Store results in session state (persists after download)
            st.session_state['dsm_results_df'] = results_df
            st.session_state['dsm_detailed_df'] = None  # Will be set below
            st.session_state['dsm_from_date'] = from_date
            st.session_state['dsm_to_date'] = to_date
            
            progress.progress(90, text="Generating detailed block-level data…")
            # Generate detailed block-level data for CSV
            for plant_name, config in plant_configs_with_bands.items():
                plant_df = df[df['plant_name'] == plant_name].copy()
                if not plant_df.empty:
                    detailed_df = compute_detailed_block_data(
                        df,
                        plant_name,
                        config['bands'],
                        config['mode'],
                        config['dyn_x'],
                        config.get('ppa') or plant_df['PPA'].iloc[0] if 'PPA' in plant_df.columns else 4.0,
                        config['setting_name']
                    )
                    detailed_data_list.append(detailed_df)
            
            # Combine all detailed data
            if detailed_data_list:
                detailed_results_df = pd.concat(detailed_data_list, ignore_index=True)
                st.session_state['dsm_detailed_df'] = detailed_results_df
            progress.progress(100, text="Done")
        finally:
            # Keep the final progress visible; caller can rerun to clear
            pass
    
    # Display results from session state (persists after download button click)
    if st.session_state.get('dsm_results_df') is not None:
        st.markdown("## 📊 DSM Analysis Results")
        st.dataframe(
            st.session_state['dsm_results_df'],
            width='stretch',
            hide_index=True
        )
        
        # Download button for detailed data (doesn't cause page reload)
        if st.session_state.get('dsm_detailed_df') is not None:
            csv_detailed = st.session_state['dsm_detailed_df'].to_csv(index=False)
            st.download_button(
                label="📥 Download Detailed Block-Level Data as CSV",
                data=csv_detailed,
                file_name=f"reconnect_dsm_detailed_{st.session_state['dsm_from_date']}_{st.session_state['dsm_to_date']}.csv",
                mime="text/csv",
                help="Download complete calculation data for each time block",
                key="download_detailed_csv"  # Unique key to prevent rerun issues
            )

