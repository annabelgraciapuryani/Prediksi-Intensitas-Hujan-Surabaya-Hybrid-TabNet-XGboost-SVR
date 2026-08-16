"""
Modul training dan loading model: TabNet+XGBoost, TabNet+XGBoost Extreme, TabNet+SVR.
Setiap model punya scaler sendiri dan tidak boleh berbagi scaler lintas model.
"""

import numpy as np
import optuna
import joblib
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from core.state import DEFAULT_SVR_PARAMS, DEFAULT_XGB_PARAMS

optuna.logging.set_verbosity(optuna.logging.WARNING)


def chronological_split(df_model, col_tanggal: str, train_ratio: float = 0.8):
    n = len(df_model)
    split_idx = int(n * train_ratio)
    train = df_model.iloc[:split_idx].copy()
    test = df_model.iloc[split_idx:].copy()
    return train, test


def _optuna_split_by_year(train_df, col_tanggal: str):
    years = sorted(train_df[col_tanggal].dt.year.unique())
    if len(years) < 2:
        split_idx = int(len(train_df) * 0.9)
        optuna_train = train_df.iloc[:split_idx]
        optuna_valid = train_df.iloc[split_idx:]
    else:
        cutoff_year = years[-1]
        optuna_train = train_df[train_df[col_tanggal].dt.year < cutoff_year]
        optuna_valid = train_df[train_df[col_tanggal].dt.year == cutoff_year]
    return optuna_train, optuna_valid


def run_optuna_xgb(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    optuna_callback=None,
) -> dict:
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0),
            "objective": "reg:squarederror",
            "random_state": 42,
            "n_jobs": -1,
        }
        model = XGBRegressor(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, pred)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    callbacks = [optuna_callback] if optuna_callback else None
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
    best = study.best_params.copy()
    best["objective"] = "reg:squarederror"
    best["random_state"] = 42
    best["n_jobs"] = -1
    return best


def run_optuna_svr(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = 50,
    optuna_callback=None,
) -> dict:
    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e1, log=True),
            "epsilon": trial.suggest_float("epsilon", 0.01, 1.0),
            "kernel": "rbf",
        }
        model = SVR(**params)
        model.fit(X_tr, y_tr)
        pred = model.predict(X_val)
        return float(np.sqrt(mean_squared_error(y_val, pred)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    callbacks = [optuna_callback] if optuna_callback else None
    study.optimize(objective, n_trials=n_trials, callbacks=callbacks)
    best = study.best_params.copy()
    best["kernel"] = "rbf"
    return best


def train_xgboost(
    train_df,
    test_df,
    feature_list: list,
    target: str,
    col_tanggal: str,
    use_optuna: bool = False,
    n_trials: int = 50,
    optuna_callback=None,
) -> tuple:
    X_train = train_df[feature_list].values
    y_train = train_df[target].values
    X_test = test_df[feature_list].values
    y_test = test_df[target].values

    if use_optuna:
        opt_train, opt_val = _optuna_split_by_year(train_df, col_tanggal)
        X_opt_tr = opt_train[feature_list].values
        y_opt_tr = opt_train[target].values
        X_opt_val = opt_val[feature_list].values
        y_opt_val = opt_val[target].values
        best_params = run_optuna_xgb(
            X_opt_tr, y_opt_tr, X_opt_val, y_opt_val, n_trials, optuna_callback
        )
    else:
        best_params = DEFAULT_XGB_PARAMS.copy()

    model = XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    return model, None, best_params, X_test, y_test, predictions


def train_svr(
    train_df,
    test_df,
    feature_list: list,
    target: str,
    col_tanggal: str,
    use_optuna: bool = False,
    n_trials: int = 50,
    optuna_callback=None,
) -> tuple:
    X_train = train_df[feature_list].values
    y_train = train_df[target].values
    X_test = test_df[feature_list].values
    y_test = test_df[target].values

    if use_optuna:
        opt_train, opt_val = _optuna_split_by_year(train_df, col_tanggal)
        X_opt_tr = opt_train[feature_list].values
        y_opt_tr = opt_train[target].values
        X_opt_val = opt_val[feature_list].values
        y_opt_val = opt_val[target].values

        scaler_opt = StandardScaler()
        X_opt_tr_sc = scaler_opt.fit_transform(X_opt_tr)
        X_opt_val_sc = scaler_opt.transform(X_opt_val)

        best_params = run_optuna_svr(X_opt_tr_sc, y_opt_tr, X_opt_val_sc, y_opt_val, n_trials, optuna_callback)
    else:
        best_params = DEFAULT_SVR_PARAMS.copy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = SVR(**best_params)
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)

    return model, scaler, best_params, X_test_scaled, y_test, predictions


def load_model_and_scaler(model_source, scaler_source=None) -> tuple:
    model = joblib.load(model_source)
    scaler = joblib.load(scaler_source) if scaler_source is not None else None
    return model, scaler


def save_model_and_scaler(model, scaler, model_path: str, scaler_path: str = None):
    joblib.dump(model, model_path)
    if scaler is not None and scaler_path:
        joblib.dump(scaler, scaler_path)
