"""
Halaman Configuration — pemilihan model, mode eksekusi, dan pipeline training.
"""

import io
import traceback

import numpy as np
import pandas as pd
import streamlit as st

from core.evaluation import build_actual_vs_pred_df, compute_metrics
from core.feature_engineering import (
    _build_frame_from_lag_map,
    _parse_lag_map_from_features,
    build_fixed_feature_frame,
    build_lag_features,
    compute_ccf_lags,
    compute_pearson_selection,
    get_final_feature_list,
    train_tabnet_for_importance,
)
from core.models import (
    chronological_split,
    load_model_and_scaler,
    train_svr,
    train_xgboost,
)
from core.preprocessing import build_clean_dataframe
from core.state import (
    ALL_MODELS,
    CCF_MAX_LAG,
    CH_LAG,
    DEFAULT_SVR_PARAMS,
    DEFAULT_TABNET_PARAMS,
    DEFAULT_XGB_PARAMS,
    KEY_COLUMN_MAPPING,
    KEY_DF_CLEAN,
    KEY_DF_RAW,
    KEY_EXECUTION_MODE,
    KEY_SELECTED_MODEL,
    MODEL_SVR,
    MODEL_XGB,
    MODEL_XGB_EXTREME,
    MODE_LOAD,
    MODE_TRAIN,
    PEARSON_THRESHOLD,
    eval_results_key,
    feature_list_key,
    fe_info_key,
    model_key,
    scaler_key,
)


st.title("Configuration")


if KEY_DF_RAW not in st.session_state or KEY_COLUMN_MAPPING not in st.session_state:
    st.warning("Selesaikan pengaturan di halaman Data Input terlebih dahulu.")
    st.stop()

df_raw = st.session_state[KEY_DF_RAW]
col_map = st.session_state[KEY_COLUMN_MAPPING]
col_tanggal = col_map["col_tanggal"]
col_target = col_map["col_target"]
col_predictors = col_map["col_predictors"]

st.subheader("Pemilihan Model Pipeline")
selected_model = st.selectbox(
    "Model Pipeline",
    options=ALL_MODELS,
    key="cfg_model_select",
)
st.session_state[KEY_SELECTED_MODEL] = selected_model

st.subheader("Mode Eksekusi")
execution_mode = st.radio(
    "Mode",
    options=[MODE_TRAIN],
    key="cfg_exec_mode",
)
st.session_state[KEY_EXECUTION_MODE] = execution_mode

st.divider()

if execution_mode == MODE_LOAD:
    st.subheader("Upload File Model Tersimpan")

    uploaded_model = st.file_uploader(
        "File model (.pkl)",
        type=["pkl"],
        key="upload_model_file",
    )
    uploaded_scaler = st.file_uploader(
        "File scaler (.pkl)",
        type=["pkl"],
        key="upload_scaler_file",
        help="Wajib diisi untuk model SVR. Opsional untuk model XGBoost.",
    )

    st.subheader("Preprocessing")
    remove_duplicates = st.checkbox(
        "Hapus baris duplikat berdasarkan tanggal",
        value=False,
        key="cfg_remove_dup_load",
    )

    if st.button("Muat Model dan Jalankan Preprocessing", type="primary"):
        if uploaded_model is None:
            st.error("Upload file model (.pkl) terlebih dahulu.")
            st.stop()
        try:
            with st.spinner("Menjalankan preprocessing..."):
                df_clean, removed = build_clean_dataframe(df_raw, col_tanggal, remove_duplicates)
                st.session_state[KEY_DF_CLEAN] = df_clean

            if removed > 0:
                st.info(f"{removed} baris duplikat dihapus.")

            with st.spinner("Memuat model dari file..."):
                model, scaler = load_model_and_scaler(uploaded_model, uploaded_scaler)

            st.session_state[model_key(selected_model)] = model
            st.session_state[scaler_key(selected_model)] = scaler

            with st.spinner("Membangun feature frame..."):
                if selected_model in [MODEL_XGB_EXTREME, MODEL_SVR]:
                    df_model = build_fixed_feature_frame(df_clean, col_tanggal, col_target, selected_model)
                    if selected_model == MODEL_XGB_EXTREME:
                        from core.state import FIXED_FEATURES_XGB_EXTREME
                        final_features = FIXED_FEATURES_XGB_EXTREME
                    else:
                        from core.state import FIXED_FEATURES_SVR
                        final_features = FIXED_FEATURES_SVR
                else:
                    raw_feature_names = model.get_booster().feature_names
                    if not raw_feature_names:
                        st.error(
                            "Daftar fitur tidak ditemukan di dalam file model. "
                            "Pastikan model dilatih menggunakan DataFrame (bukan numpy array)."
                        )
                        st.stop()

                    final_features = list(raw_feature_names)
                    lag_map = _parse_lag_map_from_features(final_features)
                    df_model = _build_frame_from_lag_map(df_clean, col_tanggal, col_target, lag_map)

                    st.session_state[fe_info_key(selected_model)] = {
                        "pearson_df": None,
                        "importance_df": None,
                        "lag_df": None,
                        "selected_lags": lag_map,
                        "note": "Fitur dibaca langsung dari model tersimpan (get_booster().feature_names).",
                    }

            st.session_state[feature_list_key(selected_model)] = final_features

            with st.spinner("Menjalankan prediksi pada data test..."):
                _, test_df = chronological_split(df_model, col_tanggal)
                X_test = test_df[final_features].values
                y_test = test_df[col_target].values

                if scaler is not None:
                    X_test = scaler.transform(X_test)

                predictions = model.predict(X_test)
                metrics = compute_metrics(y_test, predictions)
                result_df = build_actual_vs_pred_df(test_df[col_tanggal], y_test, predictions)

                st.session_state[eval_results_key(selected_model)] = {
                    "metrics": metrics,
                    "result_df": result_df,
                    "test_df": test_df,
                    "model_metadata": {
                        "mode": MODE_LOAD,
                        "model_file": uploaded_model.name,
                        "n_train": len(df_model) - len(test_df),
                        "n_test": len(test_df),
                        "feature_list": final_features,
                    },
                }

            st.success("Model berhasil dimuat dan dievaluasi. Lihat hasil di halaman Analysis & Results.")

        except Exception:
            st.error("Terjadi kesalahan saat memuat model atau menjalankan pipeline.")
            st.error(traceback.format_exc())

else:
    st.subheader("Preprocessing")

    remove_duplicates = st.checkbox(
        "Hapus baris duplikat berdasarkan tanggal",
        value=False,
        key="cfg_remove_dup_train",
    )

    st.subheader("Optuna Hyperparameter Optimization")
    use_optuna = st.checkbox(
        "Aktifkan Optuna (pencarian hyperparameter otomatis)",
        value=False,
        key="cfg_use_optuna",
    )

    if use_optuna:
        n_trials = st.number_input(
            "Jumlah trial Optuna",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
        )
    else:
        n_trials = 50
        st.info("Optuna dinonaktifkan. Parameter default berikut akan digunakan:")

        if selected_model in [MODEL_XGB, MODEL_XGB_EXTREME]:
            st.json(DEFAULT_XGB_PARAMS)
        else:
            st.json(DEFAULT_SVR_PARAMS)

        with st.expander("Parameter TabNet (digunakan untuk Feature Importance)"):
            st.json(DEFAULT_TABNET_PARAMS)

    train_ratio = st.slider(
        "Rasio data training (%)",
        min_value=60,
        max_value=90,
        value=80,
        step=5,
        help="Sisa data digunakan sebagai data test (evaluasi). Split dilakukan secara kronologis.",
    )

    if st.button("Jalankan Pipeline Training", type="primary"):
        try:
            with st.spinner("Langkah 1/5 — Preprocessing data..."):
                df_clean, removed = build_clean_dataframe(df_raw, col_tanggal, remove_duplicates)
                st.session_state[KEY_DF_CLEAN] = df_clean

            if removed > 0:
                st.info(f"{removed} baris duplikat dihapus selama preprocessing.")

            available_features = [c for c in col_predictors if c in df_clean.columns]

            if selected_model in [MODEL_XGB_EXTREME, MODEL_SVR]:
                with st.spinner("Langkah 2/5 — Membangun feature frame dari daftar fitur fixed..."):
                    df_model = build_fixed_feature_frame(df_clean, col_tanggal, col_target, selected_model)
                    if selected_model == MODEL_XGB_EXTREME:
                        from core.state import FIXED_FEATURES_XGB_EXTREME
                        final_features = FIXED_FEATURES_XGB_EXTREME
                    else:
                        from core.state import FIXED_FEATURES_SVR
                        final_features = FIXED_FEATURES_SVR

                st.session_state[fe_info_key(selected_model)] = {
                    "pearson_df": None,
                    "importance_df": None,
                    "lag_df": None,
                    "selected_lags": None,
                    "note": f"Model {selected_model} menggunakan daftar fitur fixed dari Lampiran B.",
                }

            else:
                train_df_temp, _ = chronological_split(
                    df_clean[[col_tanggal, col_target] + available_features].dropna(),
                    col_tanggal,
                    train_ratio / 100,
                )

                with st.spinner("Langkah 2/5 — Pearson Correlation selection..."):
                    pearson_df, features_pearson = compute_pearson_selection(
                        train_df_temp, available_features, col_target
                    )
                    if not features_pearson:
                        st.error(
                            f"Tidak ada fitur yang lolos threshold Pearson ({PEARSON_THRESHOLD}). "
                            "Pastikan dataset memiliki kolom numerik yang berkorelasi dengan target."
                        )
                        st.stop()

                with st.spinner("Langkah 3/5 — TabNet Feature Importance (training sementara)..."):
                    importance_df, selected_features = train_tabnet_for_importance(
                        train_df_temp, features_pearson, col_target
                    )

                with st.spinner("Langkah 4/5 — CCF lag determination..."):
                    lag_df, selected_lags = compute_ccf_lags(
                        train_df_temp, selected_features, col_target, CCF_MAX_LAG
                    )

                df_model = build_lag_features(
                    df_clean, col_tanggal, col_target, selected_features, selected_lags, CH_LAG
                )
                final_features = get_final_feature_list(df_model, col_tanggal, col_target, selected_lags)

                st.session_state[fe_info_key(selected_model)] = {
                    "pearson_df": pearson_df,
                    "importance_df": importance_df,
                    "lag_df": lag_df,
                    "selected_lags": selected_lags,
                }

            st.session_state[feature_list_key(selected_model)] = final_features

            with st.spinner(f"Langkah 5/5 — Training model {selected_model}..."):
                train_df, test_df = chronological_split(df_model, col_tanggal, train_ratio / 100)

                optuna_callback = None
                if use_optuna:
                    prog_bar = st.progress(0, text="Memulai hyperparameter tuning (Optuna)...")
                    def _callback(study, trial):
                        current = trial.number + 1
                        val = study.best_value
                        prog_bar.progress(
                            current / n_trials,
                            text=f"Tuning {selected_model} (Optuna): Trial {current}/{n_trials} | Best RMSE: {val:.4f}"
                        )
                        if current == n_trials:
                            prog_bar.empty()
                    optuna_callback = _callback

                if selected_model in [MODEL_XGB, MODEL_XGB_EXTREME]:
                    model, scaler, best_params, X_test, y_test, predictions = train_xgboost(
                        train_df, test_df, final_features, col_target, col_tanggal,
                        use_optuna=use_optuna, n_trials=n_trials, optuna_callback=optuna_callback
                    )
                else:
                    model, scaler, best_params, X_test, y_test, predictions = train_svr(
                        train_df, test_df, final_features, col_target, col_tanggal,
                        use_optuna=use_optuna, n_trials=n_trials, optuna_callback=optuna_callback
                    )

            st.session_state[model_key(selected_model)] = model
            st.session_state[scaler_key(selected_model)] = scaler

            metrics = compute_metrics(y_test, predictions)
            result_df = build_actual_vs_pred_df(test_df[col_tanggal], y_test, predictions)

            st.session_state[eval_results_key(selected_model)] = {
                "metrics": metrics,
                "result_df": result_df,
                "test_df": test_df,
                "train_df": train_df,
                "model_metadata": {
                    "mode": MODE_TRAIN,
                    "n_train": len(train_df),
                    "n_test": len(test_df),
                    "feature_list": final_features,
                    "hyperparameters": best_params,
                    "use_optuna": use_optuna,
                },
            }

            st.success(
                f"Training selesai. "
                f"RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | "
                f"MAPE: {metrics['MAPE']:.2f}%. "
                "Lihat detail di halaman Analysis & Results."
            )

        except ValueError as e:
            st.error(f"Kesalahan konfigurasi data: {e}")
        except Exception:
            st.error("Terjadi kesalahan saat menjalankan pipeline.")
            st.error(traceback.format_exc())
