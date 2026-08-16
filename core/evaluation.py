"""
Modul evaluasi model: penghitungan metrik dan pembentukan tabel hasil.
"""

import numpy as np
import pandas as pd


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))

    nonzero_mask = y_true != 0
    if nonzero_mask.sum() > 0:
        mape = float(
            np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 10
        )
    else:
        mape = float("nan")

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}


def build_actual_vs_pred_df(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    return pd.DataFrame({
        "Tanggal": pd.to_datetime(dates).values,
        "Aktual (mm)": np.round(y_true, 4),
        "Prediksi (mm)": np.round(y_pred, 4),
        "Selisih (mm)": np.round(np.abs(y_true - y_pred), 4),
    })
