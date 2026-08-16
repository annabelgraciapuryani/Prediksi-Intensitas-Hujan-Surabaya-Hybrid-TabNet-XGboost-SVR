"""
Konstanta kunci session_state untuk seluruh aplikasi.
Gunakan konstanta ini di semua halaman agar tidak ada typo.
"""

KEY_DF_RAW = "df_raw"
KEY_DF_CLEAN = "df_clean"
KEY_COLUMN_MAPPING = "column_mapping"
KEY_SELECTED_MODEL = "selected_model"
KEY_EXECUTION_MODE = "execution_mode"

MODEL_XGB = "Hybrid TabNet-XGBoost"
MODEL_XGB_EXTREME = "Hybrid TabNet-XGBoost (Extreme)"
MODEL_SVR = "Hybrid TabNet-SVR"

ALL_MODELS = [MODEL_XGB, MODEL_XGB_EXTREME, MODEL_SVR]

MODE_LOAD = "Gunakan Model Tersimpan (.pkl)"
MODE_TRAIN = "Latih Ulang Model"

SINGLE_COLUMNS = ["CH", "Height", "Press0", "Temp0", "Humi0", "WS", "WD"]

PAIRED_VARIABLES = [
    "850", "700", "500", "SWEAT", "LCL", "CCL", "LFC",
    "LI", "KI", "CAPE", "SI", "TT", "KO", "TPW",
    "BOYDEN", "CIN", "MVV",
]

CLEAN_COLUMN_ORDER = [
    "Tanggal", "CH", "Height", "Press0", "Temp0", "Humi0",
    "WS", "WD", "850", "700", "500", "SWEAT", "LCL", "CCL",
    "LFC", "LI", "KI", "CAPE", "SI", "TT", "KO", "TPW",
    "BOYDEN", "CIN", "MVV",
]

PEARSON_THRESHOLD = 0.07
CCF_MAX_LAG = 7
TABNET_TOP_K = 15
CH_LAG = 1
MAX_PREDICT_RANGE_DAYS = 7

FIXED_FEATURES_XGB_EXTREME = [
    "Humi0_lag1", "TPW_lag1", "700_lag1", "LCL_lag1", "500_lag1",
    "KI_lag1", "850_lag1", "CCL_lag1", "CAPE_lag1", "Height_lag3",
    "LFC_lag1", "SWEAT_lag4", "Press0_lag7", "MVV_lag2", "Temp0_lag7",
    "CH_lag1", "month", "month_sin", "month_cos",
]

FIXED_FEATURES_SVR = [
    "Humi0_lag1", "TPW_lag1", "700_lag1", "LCL_lag1", "500_lag1",
    "KI_lag1", "850_lag4", "LI_lag1", "SI_lag1", "TT_lag3",
    "CAPE_lag3", "Height_lag1", "LFC_lag1", "Press0_lag7", "Temp0_lag7",
    "CH_lag1", "month", "month_sin", "month_cos",
]

DEFAULT_TABNET_PARAMS = {
    "n_d": 32,
    "n_a": 32,
    "n_steps": 7,
    "gamma": 1.8,
    "lambda_sparse": 1e-7,
    "lr": 0.005,
    "max_epochs": 100,
    "patience": 50,
    "batch_size": 64,
    "virtual_batch_size": 32,
}

DEFAULT_XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.0,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1,
}

DEFAULT_SVR_PARAMS = {
    "kernel": "rbf",
    "C": 100.0,
    "gamma": "scale",
    "epsilon": 0.1,
}


def scaler_key(model_name: str) -> str:
    return f"scaler_{model_name}"


def model_key(model_name: str) -> str:
    return f"model_{model_name}"


def feature_list_key(model_name: str) -> str:
    return f"feature_list_{model_name}"


def eval_results_key(model_name: str) -> str:
    return f"eval_results_{model_name}"


def fe_info_key(model_name: str) -> str:
    return f"fe_info_{model_name}"
