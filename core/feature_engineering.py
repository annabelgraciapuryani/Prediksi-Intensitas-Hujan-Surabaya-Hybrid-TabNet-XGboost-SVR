"""
Modul feature engineering: Pearson selection, TabNet importance,
CCF lag determination, dan pembentukan lag features.
"""

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import ccf

from core.state import (
    CCF_MAX_LAG,
    CH_LAG,
    DEFAULT_TABNET_PARAMS,
    FIXED_FEATURES_SVR,
    FIXED_FEATURES_XGB_EXTREME,
    MODEL_SVR,
    MODEL_XGB_EXTREME,
    PEARSON_THRESHOLD,
    TABNET_TOP_K,
)


def compute_pearson_selection(
    df_train: pd.DataFrame,
    features: list,
    target: str,
    threshold: float = PEARSON_THRESHOLD,
) -> tuple[pd.DataFrame, list]:
    results = []
    for feature in features:
        r = df_train[feature].corr(df_train[target], method="pearson")
        results.append({"Feature": feature, "Pearson_r": r, "Abs_r": abs(r)})

    pearson_df = pd.DataFrame(results).sort_values("Abs_r", ascending=False).reset_index(drop=True)
    selected = pearson_df.loc[pearson_df["Abs_r"] >= threshold, "Feature"].tolist()
    return pearson_df, selected


def train_tabnet_for_importance(
    df_train: pd.DataFrame,
    features_pearson: list,
    target: str,
    tabnet_params: dict = None,
) -> tuple[pd.DataFrame, list]:
    params = tabnet_params or DEFAULT_TABNET_PARAMS

    X = df_train[features_pearson].values.astype(float)
    y = df_train[target].values.astype(float).reshape(-1, 1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split_idx = int(len(X_scaled) * 0.9)
    X_tr, X_val = X_scaled[:split_idx], X_scaled[split_idx:]
    y_tr, y_val = y[:split_idx], y[split_idx:]

    model = TabNetRegressor(
        n_d=params["n_d"],
        n_a=params["n_a"],
        n_steps=params["n_steps"],
        gamma=params["gamma"],
        lambda_sparse=params["lambda_sparse"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=params["lr"]),
        seed=42,
        verbose=0,
    )
    model.fit(
        X_train=X_tr,
        y_train=y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        eval_name=["train", "valid"],
        eval_metric=["rmse"],
        max_epochs=params["max_epochs"],
        patience=params["patience"],
        batch_size=params["batch_size"],
        virtual_batch_size=params["virtual_batch_size"],
    )

    importance_df = pd.DataFrame({
        "Feature": features_pearson,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    top_k = min(TABNET_TOP_K, len(importance_df))
    selected = importance_df.head(top_k)["Feature"].tolist()
    return importance_df, selected


def compute_ccf_lags(
    df_train: pd.DataFrame,
    selected_features: list,
    target: str,
    max_lag: int = CCF_MAX_LAG,
) -> tuple[pd.DataFrame, dict]:
    lag_results = []
    for feature in selected_features:
        x = df_train[feature].values
        y = df_train[target].values

        valid_mask = np.isfinite(x) & np.isfinite(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]

        if len(x_valid) < max_lag + 2:
            lag_results.append({"Feature": feature, "Lag": 1, "CCF": np.nan, "Abs_CCF": np.nan})
            continue

        ccf_values = ccf(x_valid, y_valid, adjusted=False)
        candidate_ccf = ccf_values[1:max_lag + 1]
        best_idx = int(np.argmax(np.abs(candidate_ccf)))
        best_lag = best_idx + 1
        best_ccf_val = candidate_ccf[best_idx]

        lag_results.append({
            "Feature": feature,
            "Lag": best_lag,
            "CCF": best_ccf_val,
            "Abs_CCF": abs(best_ccf_val),
        })

    lag_df = pd.DataFrame(lag_results).sort_values("Abs_CCF", ascending=False).reset_index(drop=True)
    selected_lags = dict(zip(lag_df["Feature"], lag_df["Lag"]))
    return lag_df, selected_lags


def build_lag_features(
    df: pd.DataFrame,
    col_tanggal: str,
    target: str,
    selected_features: list,
    selected_lags: dict,
    ch_lag: int = CH_LAG,
) -> pd.DataFrame:
    df_model = df[[col_tanggal, target] + selected_features].copy()
    df_model = df_model.sort_values(col_tanggal).reset_index(drop=True)

    for feature, lag in selected_lags.items():
        df_model[f"{feature}_lag{lag}"] = df_model[feature].shift(lag)

    df_model[f"CH_lag{ch_lag}"] = df_model[target].shift(ch_lag)

    df_model["month"] = df_model[col_tanggal].dt.month
    df_model["month_sin"] = np.sin(2 * np.pi * df_model["month"] / 12)
    df_model["month_cos"] = np.cos(2 * np.pi * df_model["month"] / 12)

    lag_cols = [f"{f}_lag{l}" for f, l in selected_lags.items()]
    drop_na_cols = lag_cols + [f"CH_lag{ch_lag}"]
    df_model = df_model.dropna(subset=drop_na_cols).reset_index(drop=True)

    return df_model


def build_fixed_feature_frame(
    df_clean: pd.DataFrame,
    col_tanggal: str,
    target: str,
    model_name: str,
) -> pd.DataFrame:
    if model_name == MODEL_XGB_EXTREME:
        fixed_features = FIXED_FEATURES_XGB_EXTREME
    elif model_name == MODEL_SVR:
        fixed_features = FIXED_FEATURES_SVR
    else:
        raise ValueError(f"Model {model_name} tidak memiliki daftar fitur fixed.")

    base_cols_needed = _extract_base_columns(fixed_features)
    # Hindari duplikasi target di list kolom (jika CH_lag membuat CH masuk ke base_cols)
    if target in base_cols_needed:
        base_cols_needed.remove(target)
        
    missing = [c for c in base_cols_needed if c not in df_clean.columns]
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan di dataset: {missing}")

    needed_cols = [col_tanggal, target] + list(base_cols_needed)
    df_model = df_clean[needed_cols].copy()
    df_model = df_model.sort_values(col_tanggal).reset_index(drop=True)

    lag_map = _parse_lag_map(fixed_features)
    for base_col, lag in lag_map.items():
        if base_col == "CH":
            df_model[f"CH_lag{lag}"] = df_model[target].shift(lag)
        else:
            df_model[f"{base_col}_lag{lag}"] = df_model[base_col].shift(lag)

    df_model["month"] = df_model[col_tanggal].dt.month
    df_model["month_sin"] = np.sin(2 * np.pi * df_model["month"] / 12)
    df_model["month_cos"] = np.cos(2 * np.pi * df_model["month"] / 12)

    lag_col_names = [f"{b}_lag{l}" for b, l in lag_map.items()]
    df_model = df_model.dropna(subset=lag_col_names).reset_index(drop=True)

    return df_model


def _parse_lag_map(feature_list: list) -> dict:
    lag_map = {}
    for feat in feature_list:
        if "_lag" in feat:
            parts = feat.rsplit("_lag", 1)
            base = parts[0]
            lag = int(parts[1])
            if base not in lag_map or lag > lag_map[base]:
                lag_map[base] = lag
    return lag_map


def _extract_base_columns(feature_list: list) -> set:
    bases = set()
    for feat in feature_list:
        if "_lag" in feat:
            base = feat.rsplit("_lag", 1)[0]
            bases.add(base)
    return bases


def get_final_feature_list(
    df_model: pd.DataFrame,
    col_tanggal: str,
    target: str,
    selected_lags: dict,
    ch_lag: int = CH_LAG,
) -> list:
    lag_features = [f"{f}_lag{l}" for f, l in selected_lags.items()]
    xgb_features = lag_features + [f"CH_lag{ch_lag}", "month", "month_sin", "month_cos"]
    available = [f for f in xgb_features if f in df_model.columns]
    return available


def _parse_lag_map_from_features(feature_list: list) -> dict:
    lag_map = {}
    for feat in feature_list:
        if "_lag" in feat:
            parts = feat.rsplit("_lag", 1)
            if len(parts) == 2 and parts[1].isdigit():
                base = parts[0]
                lag = int(parts[1])
                if base not in lag_map or lag > lag_map[base]:
                    lag_map[base] = lag
    return lag_map


def _build_frame_from_lag_map(
    df_clean: pd.DataFrame,
    col_tanggal: str,
    target: str,
    lag_map: dict,
) -> pd.DataFrame:
    base_cols = set(lag_map.keys())
    base_cols.discard("CH")

    missing = [c for c in base_cols if c not in df_clean.columns]
    if missing:
        raise ValueError(f"Kolom berikut tidak ditemukan di dataset: {missing}")

    needed_cols = [col_tanggal, target] + list(base_cols)
    df_model = df_clean[needed_cols].copy()
    df_model = df_model.sort_values(col_tanggal).reset_index(drop=True)

    for base, lag in lag_map.items():
        if base == "CH":
            df_model[f"CH_lag{lag}"] = df_model[target].shift(lag)
        else:
            df_model[f"{base}_lag{lag}"] = df_model[base].shift(lag)

    df_model["month"] = df_model[col_tanggal].dt.month
    df_model["month_sin"] = np.sin(2 * np.pi * df_model["month"] / 12)
    df_model["month_cos"] = np.cos(2 * np.pi * df_model["month"] / 12)

    lag_col_names = [f"{b}_lag{l}" for b, l in lag_map.items()]
    df_model = df_model.dropna(subset=lag_col_names).reset_index(drop=True)

    return df_model


def _split_lag_feature(feature: str) -> tuple[str, int] | None:
    if "_lag" not in feature:
        return None
    parts = feature.rsplit("_lag", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return None


def _lookup_exogenous_value(
    exogenous: pd.DataFrame,
    base_col: str,
    ref_date: pd.Timestamp,
) -> float:
    if base_col not in exogenous.columns:
        return np.nan

    if ref_date in exogenous.index:
        val = exogenous.loc[ref_date, base_col]
        if isinstance(val, pd.Series):
            val = val.iloc[0]
        if pd.notna(val):
            return float(val)

    col_values = exogenous[base_col].dropna()
    if len(col_values) == 0:
        return np.nan
    if ref_date < col_values.index.min():
        return float(col_values.iloc[0])
    return float(col_values.iloc[-1])


def _predict_single_date(
    model,
    scaler,
    final_features: list,
    date: pd.Timestamp,
    exogenous: pd.DataFrame,
    ch_by_date: dict,
    col_target: str,
) -> float:
    month = date.month
    values = []

    for feature in final_features:
        if feature == "month":
            values.append(month)
        elif feature == "month_sin":
            values.append(np.sin(2 * np.pi * month / 12))
        elif feature == "month_cos":
            values.append(np.cos(2 * np.pi * month / 12))
        else:
            parsed = _split_lag_feature(feature)
            if parsed is None:
                values.append(np.nan)
                continue

            base_col, lag = parsed
            ref_date = date - pd.Timedelta(days=lag)

            if base_col == "CH" or base_col == col_target:
                value = ch_by_date.get(ref_date, np.nan)
            else:
                value = _lookup_exogenous_value(exogenous, base_col, ref_date)

            values.append(value)

    X = np.array([values], dtype=float)

    if np.isnan(X).any():
        raise ValueError(
            "Terdapat fitur yang tidak dapat dibentuk (nilai kosong) untuk rentang tanggal ini."
        )

    if scaler is not None:
        X = scaler.transform(X)

    prediction = float(model.predict(X)[0])
    return max(0.0, prediction)


def predict_date_range(
    model,
    scaler,
    df_clean: pd.DataFrame,
    col_tanggal: str,
    col_target: str,
    final_features: list,
    start_date,
    end_date,
    max_days: int = 7,
) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if end_ts < start_ts:
        raise ValueError("Tanggal akhir tidak boleh lebih awal dari tanggal awal.")

    if (end_ts - start_ts).days + 1 > max_days:
        raise ValueError(f"Rentang prediksi maksimal {max_days} hari.")

    work = df_clean.copy()
    work[col_tanggal] = pd.to_datetime(work[col_tanggal], errors="coerce")
    work = work.sort_values(col_tanggal).reset_index(drop=True)

    exogenous = work.drop_duplicates(subset=[col_tanggal], keep="last").set_index(col_tanggal)

    target_df = work[[col_tanggal, col_target]].dropna(subset=[col_target])
    ch_by_date = {
        ts: float(val)
        for ts, val in zip(target_df[col_tanggal], target_df[col_target])
    }

    if not ch_by_date:
        raise ValueError("Kolom target kosong. Tidak dapat membentuk fitur autoregresif.")

    last_known_date = max(ch_by_date.keys())

    forecast_start = last_known_date + pd.Timedelta(days=1)
    forecast_end = max(end_ts, start_ts)

    if forecast_start <= forecast_end:
        for date in pd.date_range(forecast_start, forecast_end, freq="D"):
            date = pd.Timestamp(date)
            ch_by_date[date] = _predict_single_date(
                model, scaler, final_features, date, exogenous, ch_by_date, col_target
            )

    rows = []
    for date in pd.date_range(start_ts, end_ts, freq="D"):
        date = pd.Timestamp(date)
        prediction = _predict_single_date(
            model, scaler, final_features, date, exogenous, ch_by_date, col_target
        )
        rows.append({"Tanggal": date, "Prediksi (mm)": round(prediction, 4)})

    return pd.DataFrame(rows)
