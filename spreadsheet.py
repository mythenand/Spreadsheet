import io
import re
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

st.set_page_config(page_title="Laser Sheet Outlier Highlighter", layout="wide")

# ---------- Helpers ----------

def read_excel_any(file, sheet_name=None) -> pd.DataFrame:
    """Read Excel (xls/xlsx) without assuming header; return as raw DataFrame (no header)."""
    name = file.name.lower()
    engine = "openpyxl" if name.endswith(".xlsx") else "xlrd"
    file.seek(0)
    return pd.read_excel(file, sheet_name=sheet_name, header=None, engine=engine)

def list_sheets(file) -> List[str]:
    name = file.name.lower()
    engine = "openpyxl" if name.endswith(".xlsx") else "xlrd"
    file.seek(0)
    xl = pd.ExcelFile(file, engine=engine)
    sheets = xl.sheet_names
    file.seek(0)
    return sheets

def normalize_header(s: str) -> str:
    """Normalize header for robust matching."""
    if s is None:
        return ""
    s = str(s)
    s = s.strip().lower()
    s = re.sub(r"[\s_/()-]+", "", s)
    return s

def locate_max_corr_column(headers: List[str]) -> Optional[int]:
    """
    Try to locate the 'Maximum Corrosion (in)' column.
    We accept case/space/paren differences, and a few variants.
    Returns the 0-based column index within the headers list, or None.
    """
    norm_headers = [normalize_header(h) for h in headers]

    candidates = [
        "maximumcorrosionin",
        "maxcorrosionin",
        "maximumcorrosion",
        "maxcorrosion",
        "maximumcorrosioninch",
        "maximumcorrosioninches",
    ]
    for idx, nh in enumerate(norm_headers):
        if nh in candidates:
            return idx

    # Fallback: contains both "corrosion" and either "max" or "maximum"
    for idx, nh in enumerate(norm_headers):
        if "corrosion" in nh and ("max" in nh or "maximum" in nh):
            return idx

    return None

def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    mu = x.mean(skipna=True)
    sd = x.std(skipna=True, ddof=1)
    if pd.isna(mu) or pd.isna(sd) or sd == 0:
        return pd.Series([np.nan] * len(x), index=x.index)
    return (x - mu) / sd

def build_output_workbook(raw_df: pd.DataFrame,
                          sheet_name: str,
                          z: pd.Series,
                          outlier_mode: str,
                          z_threshold: float) -> bytes:
    """
    Create an .xlsx in memory:
      - Row1-2 (index 0-1) = project lines (bold)
      - Row3 (index 2) = headers (bold)
      - Row4+ (index >=3) = data
      - Highlight entire data row light-blue when outlier (based on z and mode)
    """
    # Prepare masks only over data rows (index >=3)
    n_rows, n_cols = raw_df.shape
    first_data_idx = 3
    last_data_idx = n_rows - 1

    if n_rows < 4:
        # Not enough rows to have data; still write whatever exists.
        first_data_idx = 3
        last_data_idx = 2  # no data rows

    # Decide outlier boolean per data row
    outlier_mask = pd.Series([False] * n_rows, index=raw_df.index)
    if z is not None and len(z) == n_rows:
        if outlier_mode == "High only (z > +K)":
            mask = (z > z_threshold)
        else:  # Two-sided
            mask = (z.abs() > z_threshold)
        # Only apply to data rows
        mask = mask & (raw_df.index >= first_data_idx)
        outlier_mask.loc[mask.index] = mask

    # Colors / fonts
    bold_font = Font(bold=True)
    fill_outlier = PatternFill(fill_type="solid", fgColor="ADD8E6")  # light blue

    wb = Workbook()
    ws = wb.active
    ws.title = sheet_name if sheet_name else "Sheet1"

    # Write all cells as values; preserve shapes
    for r in range(n_rows):
        for c in range(n_cols):
            val = raw_df.iat[r, c]
            ws.cell(row=r+1, column=c+1, value=val)

    # Bold first 3 rows (1-based rows 1..3)
    for r in [1, 2, 3]:
        if r <= ws.max_row:
            for c in range(1, n_cols + 1):
                ws.cell(row=r, column=c).font = bold_font

    # Highlight outlier data rows
    for r in range(first_data_idx, n_rows):
        if bool(outlier_mask.iloc[r]):
            for c in range(1, n_cols + 1):
                ws.cell(row=r+1, column=c).fill = fill_outlier

    # Auto-fit-ish column widths (simple heuristic)
    for c in range(1, n_cols + 1):
        col_letter = get_column_letter(c)
        # max length in this column (limit to avoid huge widths)
        texts = [str(raw_df.iat[r-1, c-1]) if r-1 < n_rows else "" for r in range(1, min(n_rows, 200)+1)]
        max_len = min(max((len(t) for t in texts if t is not None), default=0), 40)
        ws.column_dimensions[col_letter].width = max(10, max_len + 2)

    bio = io.BytesIO()
    wb.save(bio)
    bio.seek(0)
    return bio.getvalue()

# ---------- UI ----------

st.title("Laser Sheet — Outlier Highlighter")
st.write(
    "Upload a **single Excel file (XLS/XLSX)** where:\n"
    "- **Row 1–2**: project-level lines (kept and **bolded**)\n"
    "- **Row 3**: header row (**bolded**)\n"
    "- **Row 4+**: data rows (variable length)\n\n"
    "This app finds **“Maximum Corrosion (in)”**, normalizes it (z-score), and highlights outlier rows in **light blue**."
)

uploaded = st.file_uploader("Upload Excel (xls/xlsx)", type=["xls", "xlsx"])

if not uploaded:
    st.stop()

# Pick sheet
try:
    sheets = list_sheets(uploaded)
except Exception as e:
    st.error(f"Could not read workbook: {e}")
    st.stop()

sheet = st.selectbox("Sheet", sheets, index=0)

# Outlier settings
side = st.selectbox("Outlier mode", ["High only (z > +K)", "Two-sided (|z| > K)"], index=0)
k = st.slider("K (standard deviations)", 1.0, 3.0, 1.0, 0.1)

# Read raw sheet (no header)
try:
    df_raw = read_excel_any(uploaded, sheet_name=sheet)
except Exception as e:
    st.error(f"Failed reading sheet: {e}")
    st.stop()

n_rows, n_cols = df_raw.shape
if n_rows < 3:
    st.error("Sheet does not contain at least 3 rows (two project lines + header).")
    st.stop()

# Extract header row (#3, 0-based index 2)
headers = df_raw.iloc[2].tolist()
# Data starts at row index 3
data = df_raw.iloc[3:].copy()

# Trim trailing completely empty rows
if not data.empty:
    data = data.dropna(how="all")
else:
    st.warning("No data rows found beneath the header (row 3).")
    data = pd.DataFrame(columns=headers)

# Build a "dataframe with headers" for preview and locating the target column
if len(headers) != data.shape[1]:
    # align widths: if the sheet has extra blank columns, cap to headers length
    data = data.iloc[:, :len(headers)]

data.columns = headers

# Find target column
col_idx = locate_max_corr_column(headers)
if col_idx is None:
    st.error("Could not locate the 'Maximum Corrosion (in)' column (case/spacing tolerant). "
             "Please ensure the third row contains that header.")
    st.stop()

target_col_name = headers[col_idx]
st.info(f"Detected target column: **{target_col_name}**")

# Compute z-score on the entire column across all rows (including project/header rows)
# We'll construct a length-n vector with NaN for non-data rows so row indices match raw_df
z = pd.Series([np.nan] * len(df_raw), index=df_raw.index, dtype=float)
if not data.empty:
    # numeric series of the data region for the chosen column
    s_data = pd.to_numeric(data[target_col_name], errors="coerce")
    mu = s_data.mean(skipna=True)
    sd = s_data.std(skipna=True, ddof=1)
    if pd.isna(sd) or sd == 0:
        st.warning("Standard deviation is zero or undefined; outlier detection will mark no rows.")
        z_data = pd.Series([np.nan] * len(s_data), index=s_data.index)
    else:
        z_data = (s_data - mu) / sd

    # place into z with an offset of 3 rows (because data starts at row index 3)
    z.iloc[3:3+len(z_data)] = z_data.values

# Build outlier mask (over full raw indices, but only rows >= 3 will be highlighted)
if side == "High only (z > +K)":
    mask = (z > k)
else:
    mask = (z.abs() > k)

# Preview
preview = data.copy()
preview["__z__"] = z.iloc[3:3+len(data)].values
if side == "High only (z > +K)":
    preview["__OUTLIER__"] = preview["__z__"] > k
else:
    preview["__OUTLIER__"] = preview["__z__"].abs() > k

st.subheader("Preview (first 200 data rows)")
st.dataframe(preview.head(200), use_container_width=True)

n_out = int(preview["__OUTLIER__"].sum())
st.success(f"Outliers flagged: {n_out}")

# Build output workbook (.xlsx)
xlsx_bytes = build_output_workbook(
    raw_df=df_raw,
    sheet_name=sheet,
    z=z,
    outlier_mode=side,
    z_threshold=k
)

st.download_button(
    "Download highlighted Excel (.xlsx)",
    data=xlsx_bytes,
    file_name="Laser_Outliers_Highlighted.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
