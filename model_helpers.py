# -*- coding: utf-8 -*-
"""
Helper functions for model training, hyperparameter tuning, calibration and evaluation.
"""

import ast
import copy
import glob
import json
import os
import re
import warnings
import numpy as np
import itertools
import traceback
from collections import defaultdict

from scipy import special

from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold, ParameterGrid, RandomizedSearchCV, train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.frozen import FrozenEstimator
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier
import lightgbm as lgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from helper import log_failure


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_to_jsonable(v) for v in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _parse_best_params_cell(raw_value):
    if raw_value is None:
        return None
    if isinstance(raw_value, float) and np.isnan(raw_value):
        return None
    if isinstance(raw_value, dict):
        params = dict(raw_value)
        params.pop("model_n_iter", None)
        return params

    text = str(raw_value).strip()
    if not text or text in {"nan", "None"}:
        return None
    if text == "{}":
        return {}

    text = re.sub(r"['\"]model_n_iter['\"]\s*:\s*array\([^)]*\)\s*,?", "", text)
    text = re.sub(r",\s*}\s*$", "}", text)

    try:
        params = ast.literal_eval(text)
    except Exception:
        try:
            params = eval(
                text,
                {"__builtins__": {}},
                {
                    "array": np.array,
                    "np": np,
                    "int32": np.int32,
                    "int64": np.int64,
                    "float32": np.float32,
                    "float64": np.float64,
                },
            )
        except Exception:
            return None

    if not isinstance(params, dict):
        return None

    params.pop("model_n_iter", None)
    return params


def _infer_test_name_from_csv(csv_path):
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    stem = re.sub(r"_(-?\d+)$", "", stem)
    return stem


def _normalize_noise_level(value):
    if value is None:
        return "__NONE__"
    if isinstance(value, float) and np.isnan(value):
        return "__NONE__"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        return format(float(value), ".12g")
    return str(value)


def _top_k_unique_params(entries, k):
    scored = sorted(entries, key=lambda e: e.get("auc", float("-inf")), reverse=True)
    out = []
    seen = set()
    for item in scored:
        params = item.get("params")
        if params is None:
            continue
        canonical = json.dumps(_to_jsonable(params), sort_keys=True, default=str)
        if canonical in seen:
            continue
        seen.add(canonical)
        out.append(params)
        if len(out) >= k:
            break
    return out


def load_preset_best_params_repository(source_path, top_k_per_group=5, strict=False):
    """Load preset hyperparameters from a CSV file or directory of result CSVs.

    Returns a repository with lookup tiers:
    1) (test_name, model_name, noise_level)
    2) (test_name, model_name)
    3) (model_name)
    """
    repo = {
        "exact": {},
        "test_model": {},
        "model": {},
    }

    if not source_path:
        if strict:
            raise ValueError("source_path must be provided when strict=True")
        warnings.warn("No preset source path provided. Preset hyperparameters disabled.")
        return repo

    if os.path.isfile(source_path):
        csv_files = [source_path]
    elif os.path.isdir(source_path):
        csv_files = glob.glob(os.path.join(source_path, "**", "*.csv"), recursive=True)
    else:
        if strict:
            raise FileNotFoundError(f"Preset source path not found: {source_path}")
        warnings.warn(f"Preset source path not found: {source_path}. Preset hyperparameters disabled.")
        return repo

    if not csv_files:
        if strict:
            raise ValueError(f"No CSV files found in preset source path: {source_path}")
        warnings.warn(f"No CSV files found in preset source path: {source_path}")
        return repo

    exact_entries = defaultdict(list)
    test_model_entries = defaultdict(list)
    model_entries = defaultdict(list)
    malformed_files = []

    for csv_path in csv_files:
        # Aggregate exports do not follow the per-model hyperparameter result schema.
        # Ignore them even in strict mode.
        if os.path.basename(csv_path).lower() in {"m4.csv", "eicu.csv"}:
            continue

        try:
            import pandas as pd

            df = pd.read_csv(csv_path)
        except Exception:
            if strict:
                malformed_files.append((csv_path, "unable to read csv"))
            continue

        if "Model" not in df.columns:
            if strict:
                malformed_files.append((csv_path, "missing 'Model' column"))
            continue

        bp_col = "Best param" if "Best param" in df.columns else ("Best params" if "Best params" in df.columns else None)
        if bp_col is None:
            if strict:
                malformed_files.append((csv_path, "missing 'Best param'/'Best params' column"))
            continue

        test_name = _infer_test_name_from_csv(csv_path)
        has_noise = "Noise level" in df.columns
        has_auc = "AUC" in df.columns

        for _, row in df.iterrows():
            model_name = row.get("Model")
            if model_name is None or (isinstance(model_name, float) and np.isnan(model_name)):
                continue

            params = _parse_best_params_cell(row.get(bp_col))
            if params is None:
                continue

            noise_key = _normalize_noise_level(row.get("Noise level")) if has_noise else "__NONE__"
            auc_raw = row.get("AUC") if has_auc else float("-inf")
            if auc_raw is None or (isinstance(auc_raw, float) and np.isnan(auc_raw)):
                auc_value = float("-inf")
            else:
                try:
                    auc_value = float(auc_raw)
                except Exception:
                    auc_value = float("-inf")

            entry = {"params": params, "auc": auc_value}
            exact_entries[(test_name, str(model_name), noise_key)].append(entry)
            test_model_entries[(test_name, str(model_name))].append(entry)
            model_entries[str(model_name)].append(entry)

    if strict and malformed_files:
        details = "\n".join([f" - {p}: {msg}" for p, msg in malformed_files[:20]])
        raise ValueError(
            "Result file structure check failed for one or more CSV files:\n"
            f"{details}"
        )

    repo["exact"] = {}
    for key, entries in exact_entries.items():
        selected = _top_k_unique_params(entries, top_k_per_group)
        if selected:
            repo["exact"][key] = selected

    repo["test_model"] = {}
    for key, entries in test_model_entries.items():
        selected = _top_k_unique_params(entries, top_k_per_group)
        if selected:
            repo["test_model"][key] = selected

    repo["model"] = {}
    for key, entries in model_entries.items():
        selected = _top_k_unique_params(entries, top_k_per_group)
        if selected:
            repo["model"][key] = selected

    print(
        "Loaded preset hyperparameters: "
        f"exact={len(repo['exact'])}, test_model={len(repo['test_model'])}, model={len(repo['model'])}"
    )
    return repo


def select_preset_params(preset_repo, model_name, preset_context=None):
    if not preset_repo:
        return None

    model_name = str(model_name)
    preset_context = preset_context or {}
    test_name = preset_context.get("test_name")
    noise_key = _normalize_noise_level(preset_context.get("noise_level"))
    fold_idx = int(preset_context.get("fold_idx", 0))

    candidates = None
    if test_name is not None:
        candidates = preset_repo.get("exact", {}).get((str(test_name), model_name, noise_key))
        if not candidates:
            candidates = preset_repo.get("test_model", {}).get((str(test_name), model_name))

    if not candidates:
        candidates = preset_repo.get("model", {}).get(model_name)

    if not candidates:
        return None

    idx = fold_idx % len(candidates)
    return copy.deepcopy(candidates[idx])


def validate_preset_params_coverage(models_dict, preset_repo, strict=False):
    if not models_dict:
        return

    repo_models = set((preset_repo or {}).get("model", {}).keys())
    missing_models = [m for m in models_dict.keys() if m not in repo_models]
    if missing_models:
        msg = "Missing preset best params for model(s): " + ", ".join(missing_models)
        if strict:
            raise ValueError(msg)
        warnings.warn(msg)


def tune_model(
    model_name,
    model,
    X_train,
    y_train,
    param_grids,
    random_state=None,
    n_search_gs=15,
    njobs_gs=1,
    random_state_global=42,
    log_file=None,
    use_preset_best_params=False,
    strict_preset_best_params=False,
    preset_best_params_repo=None,
    preset_context=None,
):
    """
    Tune hyperparameters for a given model.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    model : estimator
        Model to tune
    X_train : array-like
        Training features
    y_train : array-like
        Training labels
    param_grids : dict
        Dictionary of parameter grids for each model
    random_state : int, optional
        Random state for CV
    n_search_gs : int
        Number of hyperparameter combinations to search
    njobs_gs : int
        Number of jobs for grid search
    random_state_global : int
        Global random state
    log_file : str, optional
        Log file path for failures
    
    Returns
    -------
    tuple
        (best_model, best_params)
    """
    if use_preset_best_params:
        params = select_preset_params(preset_best_params_repo, model_name, preset_context=preset_context)
        if params is None:
            msg = (
                f"No preset params found for model '{model_name}'"
                + (f" with context {preset_context}" if preset_context else "")
            )
            if strict_preset_best_params:
                raise ValueError(msg)
            warnings.warn(msg + ". Fitting model with default parameters.")
            model.fit(X_train, y_train)
            return model, {}

        try:
            model.set_params(**params)
        except Exception as e:
            if strict_preset_best_params:
                raise ValueError(
                    f"Could not apply preset params for {model_name}: {e}"
                ) from e
            warnings.warn(
                f"Could not apply preset params for {model_name}: {e}. "
                "Fitting model with default parameters."
            )
            model.fit(X_train, y_train)
            return model, {}

        model.fit(X_train, y_train)
        return model, params

    if model_name not in param_grids:
        model.fit(X_train, y_train)
        return model, {}

    cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)
    param_dist = param_grids[model_name]
    if model_name not in ['CatBoost', 'XGBoost', 'LightGBM']:
        # Prevent sklearn warning when requested n_iter exceeds the finite grid size.
        n_iter = n_search_gs
        try:
            n_iter = min(n_search_gs, len(ParameterGrid(param_dist)))
        except Exception:
            pass

        search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=n_iter, 
                                   scoring="roc_auc", cv=cv_inner, n_jobs=njobs_gs, 
                                   random_state=random_state, verbose=0)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        if model_name in ['Logistic', 'LASSO', 'Ridge']:
            search.best_params_['model_n_iter'] = best_model['lr'].n_iter_
        if model_name == 'Gradient Boosting':
            search.best_params_['model_n_iter'] = best_model['gb'].n_estimators_
        if model_name == 'MLP':
            search.best_params_['model_n_iter'] = best_model['mlp'].n_iter_

        print(f"Best params for {model_name}: {search.best_params_}")

        return best_model, search.best_params_
    else:
        keys, values = zip(*param_dist.items())
        param_grid = [dict(zip(keys, v)) for v in itertools.product(*values)]
        rng = np.random.default_rng(random_state)
        n_to_try = min(n_search_gs, len(param_grid))
        param_idx = rng.choice(len(param_grid), size=n_to_try, replace=False)
        best_auc = -np.inf
        best_model = None
        best_params = None
        for p in param_idx:
            params = param_grid[p]
            try:
                model_step, score = grid_step(model_name, X_train, y_train, cv_inner, params, 
                                             random_state_global=random_state_global)
            except Exception as e:
                error_msg = traceback.format_exc()
                if log_file:
                    log_failure(params, error_msg, logging_file=log_file)
                print(f"Error during tuning {model_name} with params {params}: {e}")
                continue
            if score > best_auc:
                best_auc = score
                best_params = params
                best_model = model_step
        if best_params is None:
            print(f"All hyperparameter combinations failed for {model_name}. Using default parameters.")
            model.fit(X_train, y_train)
            return model, {}
        print(f"Best params for {model_name}: {best_params}")
        
        X_train_refit, X_early_stop, y_train_refit, y_early_stop = train_test_split(
            X_train, y_train, test_size=0.1, stratify=y_train, random_state=random_state_global)

        if model_name == 'LightGBM':
            final_model = LGBMClassifier(**best_params)
            final_model.fit(X_train_refit, y_train_refit, eval_set=[(X_early_stop, y_early_stop)], 
                          callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
            best_params['model_n_iter'] = final_model.best_iteration_
        elif model_name == 'XGBoost':
            final_model = XGBClassifier(**best_params)
            final_model.fit(X_train_refit, y_train_refit, eval_set=[(X_early_stop, y_early_stop)], 
                          verbose=False)
            best_params['model_n_iter'] = final_model.best_iteration
        elif model_name == 'CatBoost':
            final_model = CatBoostClassifier(**best_params)
            final_model.fit(X_train_refit, y_train_refit, eval_set=[(X_early_stop, y_early_stop)], 
                          verbose=True)
            best_params['model_n_iter'] = final_model.get_best_iteration()
        else:
            final_model = best_model if best_model is not None else model
            final_model.fit(X_train, y_train)
        return final_model, best_params

 
def grid_step(model_name, X_tr, y_tr, cv_inner, param_grid, random_state_global=42):
    """
    Perform grid search step for a single parameter combination.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    X_tr : array-like
        Training features
    y_tr : array-like
        Training labels
    cv_inner : KFold
        Inner cross-validation splitter
    param_grid : dict
        Parameter grid for this step
    random_state_global : int
        Global random state
    
    Returns
    -------
    tuple
        (model_step, mean_auc)
    """
    aucs = []
    for train_idx, val_idx in cv_inner.split(X_tr):
        X_tr_fold, X_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_tr_fold, y_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
        X_tr_fold, X_early_stop, y_tr_fold, y_early_stop = train_test_split(
            X_tr_fold, y_tr_fold, test_size=0.1, stratify=y_tr_fold, random_state=random_state_global)    

        if model_name == 'LightGBM':
            dtrain = lgb.Dataset(X_tr_fold, y_tr_fold)
            dearly_stop = lgb.Dataset(X_early_stop, y_early_stop)
            model_step = lgb.train(
                params=param_grid,
                train_set=dtrain,
                valid_sets=[dearly_stop],
                callbacks=[lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=True)],
            )
            score = roc_auc_score(y_val, model_step.predict(X_val, num_iteration=model_step.best_iteration))
            aucs.append(score)           
        elif model_name == 'XGBoost':
            model_step = XGBClassifier()
            model_step.set_params(**param_grid)
            model_step.fit(X_tr_fold, y_tr_fold, eval_set=[(X_early_stop, y_early_stop)], verbose=False)
            score = roc_auc_score(y_val, model_step.predict_proba(X_val)[:, 1])
            aucs.append(score)
        elif model_name == 'CatBoost':
            model_step = CatBoostClassifier()
            model_step.set_params(**param_grid)
            model_step.fit(X_tr_fold, y_tr_fold, eval_set=[(X_early_stop, y_early_stop)], verbose=True)
            score = roc_auc_score(y_val, model_step.predict_proba(X_val)[:, 1])
            aucs.append(score)

    return model_step, np.mean(aucs)
    

def calibrate_model(model, X_val, y_val, calibration_size=0.1, random_state=None):
    """
    Calibrate a model using a holdout calibration set.
    
    Parameters
    ----------
    model : estimator
        Model to calibrate
    X_val : array-like
        Validation features
    y_val : array-like
        Validation labels
    calibration_size : float
        Fraction of data to use for calibration
    random_state : int, optional
        Random state
    
    Returns
    -------
    tuple
        (calibrated_model, X_val_updated, y_val_updated)
    """
    X_val, X_cal, y_val, y_cal = train_test_split(X_val, y_val, test_size=calibration_size, 
                                                   stratify=y_val, random_state=random_state)
    calibrated_clf = CalibratedClassifierCV(FrozenEstimator(model))
    calibrated_clf.fit(X_cal, y_cal)
    return calibrated_clf, X_val, y_val


def compute_calibration_metrics(y_true, y_pred, model_name=None, log_file=None):
    """
    Compute calibration intercept, slope, and calibration curve.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted probabilities
    model_name : str, optional
        Name of the model (for logging)
    log_file : str, optional
        Log file path for failures
    
    Returns
    -------
    tuple
        (intercept, slope, prob_true, prob_pred)
    """
    intercept, slope = None, None
    try:
        logits = special.logit(np.clip(y_pred, 1e-10, 1 - 1e-10))
        calib_model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=500)
        calib_model.fit(logits.reshape(-1, 1), y_true)
        intercept = calib_model.intercept_[0]
        slope = calib_model.coef_[0][0]
    except Exception:
        error_msg = traceback.format_exc()
        if model_name and log_file:
            log_failure({"model": model_name, "step": "calibration"}, error_msg, log_file)
    
    prob_true, prob_pred = calibration_curve(y_true, y_pred)
    return intercept, slope, prob_true, prob_pred
