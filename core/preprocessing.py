"""
Modul preprocessing dataset radiosonde.

Urutan pemrosesan wajib mengikuti langkah 1-8 sesuai spesifikasi:
  1. Parse numerik per sel secara generik
  2. Interpolasi per kolom jam (00.00 dan 12.00) secara terpisah
  3. Rata-ratakan kedua kolom jam setelah keduanya terisi penuh
  4. Hapus kolom jam asli
  5. Interpolasi kolom tunggal
  6. Opsional: hapus duplikat tanggal
  7. Sort kronologis
  8. Konversi kolom tanggal ke datetime64
"""

import numpy as np
import pandas as pd

from core.state import PAIRED_VARIABLES, SINGLE_COLUMNS


def to_numeric_safe(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.replace(",", "."))
        except ValueError:
            return np.nan
    return np.nan


def apply_numeric_conversion(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    result = df.copy()
    for col in columns:
        if col in result.columns:
            result[col] = result[col].apply(to_numeric_safe)
    return result


def interpolate_column(series: pd.Series) -> pd.Series:
    return series.interpolate(method="linear", limit_direction="both")


def _get_object_columns(df: pd.DataFrame) -> list:
    return [col for col in df.columns if df[col].dtype == object]


def _merge_paired_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    cols_to_drop = []

    for var in PAIRED_VARIABLES:
        col_00 = f"{var} 00.00"
        col_12 = f"{var} 12.00"

        if col_00 not in result.columns or col_12 not in result.columns:
            continue

        interp_00 = interpolate_column(result[col_00])
        interp_12 = interpolate_column(result[col_12])

        result[var] = (interp_00 + interp_12) / 2
        cols_to_drop.extend([col_00, col_12])

    result = result.drop(columns=cols_to_drop, errors="ignore")
    return result


def _interpolate_single_columns(df: pd.DataFrame, single_cols: list) -> pd.DataFrame:
    result = df.copy()
    for col in single_cols:
        if col in result.columns:
            result[col] = interpolate_column(result[col])
    return result


def build_clean_dataframe(
    df_raw: pd.DataFrame,
    col_tanggal: str,
    remove_duplicates: bool = False,
) -> pd.DataFrame:
    df = df_raw.copy()

    object_cols = _get_object_columns(df)
    numeric_cols = [c for c in object_cols if c != col_tanggal]
    df = apply_numeric_conversion(df, numeric_cols)

    df = _merge_paired_columns(df)

    present_singles = [c for c in SINGLE_COLUMNS if c in df.columns]
    df = _interpolate_single_columns(df, present_singles)

    if remove_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=[col_tanggal], keep="first")
        after = len(df)
        removed = before - after
    else:
        removed = 0

    df[col_tanggal] = pd.to_datetime(df[col_tanggal], errors="coerce")
    df = df.sort_values(col_tanggal).reset_index(drop=True)

    return df, removed


def compute_missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "Kolom": missing_counts.index,
        "Jumlah Missing": missing_counts.values,
        "Persentase (%)": missing_pct.values,
    })
    return summary[summary["Jumlah Missing"] > 0].reset_index(drop=True)


def count_duplicate_rows(df: pd.DataFrame, col_tanggal: str) -> int:
    return int(df.duplicated(subset=[col_tanggal]).sum())
