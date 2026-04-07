# -*- coding: utf-8 -*-

import copy
import json
import os
import time
import warnings
from datetime import datetime
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import torch
from joblib import Memory

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  # Required to use IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier  # type: ignore[reportMissingImports]
from pytabkit import RealMLP_TD_Classifier, FTT_D_Classifier
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier

from helper import create_directory, predict_proba_batched, save_results, set_random_seed
from hpo_grid import PARAM_GRIDS
from m3_handling import (
    X,
    y,
    y_proxy_death_icu,
    y_proxy_death_overall,
    cont_features,
    cat_features,
    column_m3,
)
from missing_data_mechanism import add_missingness
from model_helpers import calibrate_model, compute_calibration_metrics, tune_model
from perturbation_function import add_measurement_noise, perturb_dic, shuffle_features
from run_helpers import run_cv_parallel_and_save


# If no CUDA GPU is available, allow CPU runs (slower) but do not abort.
if not torch.cuda.is_available():
    warnings.warn("GPU device not found. The script will run on CPU (may be much slower).")


now = datetime.now()
LOG_FILE = f"boosting_failures_multi_{now.strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE_CALIB = f"calibration_failures_multi_{now.strftime('%Y%m%d_%H%M%S')}.log"
DIR_NAME = f"res_multi_{now.strftime('%Y%m%d_%H%M%S')}"


NJOBS = 1
NJOBS_GS = 1
HPO = True
N_SEARCH_GS = 15
RANDOM_STATE = 42
N_COMPOSITES = 10
N_PERTURB_TYPES_PER_COMPOSITE = 3

print(f"N_JOBS: {NJOBS}")
print(f"N_COMPOSITES: {N_COMPOSITES}")
print(
    "Running models with hyperparameter optimization"
    if HPO
    else "Running models without hyperparameter optimization"
)

CACHE_DIR = os.path.join("results", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)
param_grids = PARAM_GRIDS if HPO else {}

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
splits = list(kf.split(X))


# MODELS
continuous_transformer_1 = Pipeline(
    steps=[("imputer", IterativeImputer(add_indicator=True, random_state=RANDOM_STATE)), ("scaler", StandardScaler())]
)
continuous_transformer_2 = Pipeline(
    steps=[("imputer", IterativeImputer(add_indicator=True, random_state=RANDOM_STATE))]
)
categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(add_indicator=True, strategy="most_frequent"))])

preprocessor_1 = ColumnTransformer(
    transformers=[
        ("cont", continuous_transformer_1, cont_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

preprocessor_2 = ColumnTransformer(
    transformers=[
        ("cont", continuous_transformer_2, cont_features),
        ("cat", categorical_transformer, cat_features),
    ]
)

models = {
    "Logistic": Pipeline([("preprocessor", preprocessor_1), ("lr", LogisticRegression(C=1e12))], memory=memory),
    "LASSO": Pipeline([("preprocessor", preprocessor_1), ("lr", LogisticRegression(l1_ratio=1, solver="liblinear"))],memory=memory,),
    "Ridge": Pipeline([("preprocessor", preprocessor_1), ("lr", LogisticRegression(l1_ratio=0))], memory=memory),
    "Random Forest": RandomForestClassifier(n_jobs=1),
    "Gradient Boosting": Pipeline([("preprocessor", preprocessor_2), ("gb", GradientBoostingClassifier())], memory=memory),
    "XGBoost": XGBClassifier(tree_method="hist", device="cuda"),
    "LightGBM": LGBMClassifier(device="gpu", gpu_use_dp=True), #Only works with GPU available
    "MLP": Pipeline([("preprocessor", preprocessor_1), ("mlp", MLPClassifier(max_iter=500))], memory=memory),
    "CatBoost": CatBoostClassifier(logging_level='Silent')
    #"TabPFN": TabPFNClassifier(),
    #"TabICL": TabICLClassifier(),
    #"RealMLP": Pipeline([("preprocessor", preprocessor_1), ("realMLP", RealMLP_TD_Classifier())], memory=memory),
    #"FTTransformer": Pipeline([("preprocessor", preprocessor_1), ("ftTransformer", FTT_D_Classifier())], memory=memory)
}


run_cv_parallel_and_save_cfg = partial(
    run_cv_parallel_and_save,
    models=models,
    splits=splits,
    n_jobs=NJOBS,
    n_folds=n_folds,
    save_results_fn=save_results,
)


def _sample_level(perturb_type, variant, rng):
    if perturb_type == "label_noise":
        if variant.startswith("proxy_"):
            if variant == "proxy_hospital_death":
                return "hospital_death"
            if variant == "proxy_icu_death":
                return "icu_death"
            return "overall_death"
        return float(rng.choice(np.linspace(0.1, 0.5, 5)))

    if perturb_type == "input_noise":
        return float(rng.choice(np.linspace(0.1, 0.5, 5)))

    if perturb_type == "missing_data":
        return float(rng.choice(np.linspace(0.1, 0.5, 5)))

    if perturb_type == "feature_permutation":
        return float(rng.choice(np.linspace(0.1, 0.7, 7)))

    if perturb_type == "class_imbalance":
        return float(rng.choice([0.25, 0.5, 0.75, 1.0]))

    if perturb_type == "training_data_regime":
        return float(rng.choice([0.25, 0.5, 0.75, 1.0]))

    raise ValueError(f"Unknown perturbation type: {perturb_type}")


def sample_composite_perturbations(n_composites=1, n_types=3, random_state=42):
    rng = np.random.default_rng(random_state)
    perturb_types = list(perturb_dic.keys())
    n_types = min(n_types, len(perturb_types))

    composites = []
    for _ in range(n_composites):
        chosen_types = list(rng.choice(perturb_types, size=n_types, replace=False))
        steps = []
        for perturb_type in chosen_types:
            variant = str(rng.choice(perturb_dic[perturb_type]))
            level = _sample_level(perturb_type, variant, rng)
            steps.append(
                {
                    "type": perturb_type,
                    "variant": variant,
                    "level": level,
                }
            )
        composites.append(steps)

    return composites


def save_composites_report(composites, output_path):
    report_payload = {
        "n_composites": len(composites),
        "composites": composites,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=2)


def _apply_label_noise(y_train, X_train, train_idx, step, fold_idx):
    variant = step["variant"]
    level = step["level"]
    seed = set_random_seed(f"multi_{variant}", level, fold_idx, base_seed=42)
    rng = np.random.default_rng(seed)

    y_noisy = y_train.copy()

    if variant == "random":
        probs = rng.random(len(y_noisy))
        y_noisy[probs < level] = 1 - y_noisy
    elif variant == "0to1":
        mask = y_noisy == 0
        probs = rng.random(mask.sum())
        idx_change = probs < level
        idxs = np.where(mask)[0][idx_change]
        y_noisy.iloc[idxs] = 1
    elif variant == "1to0":
        mask = y_noisy == 1
        probs = rng.random(mask.sum())
        idx_change = probs < level
        idxs = np.where(mask)[0][idx_change]
        y_noisy.iloc[idxs] = 0
    elif variant == "conditional":
        age_train_perc = X_train.age.rank(pct=True)
        swap_proba = age_train_perc * level
        swap_by_age = rng.binomial(1, p=swap_proba, size=len(age_train_perc))
        y_noisy = y_noisy * (1 - swap_by_age) + (1 - y_noisy) * swap_by_age
    elif variant == "proxy_hospital_death":
        y_noisy = y_noisy
    elif variant == "proxy_icu_death":
        # Use the current training index (may be downsampled by prior steps).
        y_noisy = y_proxy_death_icu.loc[y_train.index]
    elif variant == "proxy_overall_death":
        # Use the current training index (may be downsampled by prior steps).
        y_noisy = y_proxy_death_overall.loc[y_train.index]

    return y_noisy


def _apply_input_noise(X_train, X_val, step, fold_idx):
    variant = step["variant"]
    level = step["level"]

    if variant.startswith("train_val_"):
        which_set = "Train_Val"
        feature_part = variant[len("train_val_") :]
    elif variant.startswith("train_"):
        which_set = "Train"
        feature_part = variant[len("train_") :]
    elif variant.startswith("val_"):
        which_set = "Val"
        feature_part = variant[len("val_") :]
    else:
        raise ValueError(f"Unsupported input_noise variant: {variant}")

    if feature_part == "continuous":
        feature_type = "cont"
    elif feature_part == "categorical":
        feature_type = "cat"
    else:
        feature_type = "cont & cat"

    seed = set_random_seed(f"multi_input_{variant}", level, fold_idx, base_seed=42)
    rng = np.random.default_rng(seed)

    if which_set == "Train":
        X_train = add_measurement_noise(X_train, level, feature_type, rng=rng)
    elif which_set == "Val":
        X_val = add_measurement_noise(X_val, level, feature_type, rng=rng)
    else:
        X_train = add_measurement_noise(X_train, level, feature_type, rng=rng)
        X_val = add_measurement_noise(X_val, level, feature_type, rng=rng)

    return X_train, X_val


def _apply_missing_data(X_train, X_val, train_idx, val_idx, step, fold_idx):
    variant = step["variant"]
    level = step["level"]

    if variant.startswith("train_val_"):
        which_set = "Train_Val"
        mechanism_name = variant[len("train_val_") :]
    elif variant.startswith("train_"):
        which_set = "Train"
        mechanism_name = variant[len("train_") :]
    elif variant.startswith("val_"):
        which_set = "Val"
        mechanism_name = variant[len("val_") :]
    else:
        raise ValueError(f"Unsupported missing_data variant: {variant}")

    mechanism = mechanism_name.upper()

    X_train[cont_features] = X_train[cont_features].fillna(X_train[cont_features].mean())
    X_train[cat_features] = X_train[cat_features].fillna(X_train[cat_features].mode().loc[0])

    seed = set_random_seed(f"multi_missing_{variant}", level, fold_idx, base_seed=42)
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator()
    torch_gen.manual_seed(seed)

    if which_set == "Train":
        X_noisy = X_train.copy()
        mask = add_missingness(
            X_noisy,
            level,
            mechanism=mechanism,
            prop_cond_features=0.5,
            rng=rng,
            torch_gen=torch_gen,
        )
        X_train = X_noisy.mask(mask, np.nan)

    elif which_set == "Val":
        X_noisy = X_val.copy()
        X_noisy[cont_features] = X_noisy[cont_features].fillna(X_train[cont_features].mean())
        X_noisy[cat_features] = X_noisy[cat_features].fillna(X_train[cat_features].mode().loc[0])
        mask = add_missingness(
            X_noisy,
            level,
            mechanism=mechanism,
            prop_cond_features=0.5,
            rng=rng,
            torch_gen=torch_gen,
        )
        X_val = X_noisy.mask(mask, np.nan)

    else:
        # Preserve current row sets, since previous composite steps can alter train rows.
        current_train_idx = X_train.index
        current_val_idx = X_val.index
        X_noisy = pd.concat([X_train, X_val]).copy()
        X_noisy[cont_features] = X_noisy[cont_features].fillna(X_train[cont_features].mean())
        X_noisy[cat_features] = X_noisy[cat_features].fillna(X_train[cat_features].mode().loc[0])
        mask = add_missingness(
            X_noisy,
            level,
            mechanism=mechanism,
            prop_cond_features=0.5,
            rng=rng,
            torch_gen=torch_gen,
        )
        X_all = X_noisy.mask(mask, np.nan)
        X_train, X_val = X_all.loc[current_train_idx], X_all.loc[current_val_idx]

    return X_train, X_val


def _apply_feature_permutation(X_train, X_val, step, fold_idx):
    variant = step["variant"]
    level = step["level"]

    if variant == "train_feature_shuffle":
        which_set = "Train"
    elif variant == "val_feature_shuffle":
        which_set = "Val"
    else:
        which_set = "Train_Val"

    seed = set_random_seed(f"multi_perm_{variant}", level, fold_idx, base_seed=42)
    rng = np.random.default_rng(seed)

    if which_set == "Train":
        X_train, _ = shuffle_features(X_train, prop=level, rng=rng)
    elif which_set == "Val":
        X_val, _ = shuffle_features(X_val, prop=level, rng=rng)
    else:
        X_train, feat_to_shuffle = shuffle_features(X_train, prop=level, rng=rng)
        X_val, _ = shuffle_features(X_val, prop=level, feat_to_shuffle=feat_to_shuffle, rng=rng)

    return X_train, X_val


def _apply_class_imbalance(X_train, y_train, step: dict[str, Any], fold_idx):
    level = float(step["level"])
    seed = set_random_seed("multi_imbalance", level, fold_idx, base_seed=42)

    negative_mask = y_train == 0
    if negative_mask.sum() == 0:
        return X_train, y_train

    X_train_negative = X_train.loc[negative_mask].sample(frac=level, random_state=seed)
    y_train_negative = y_train.loc[X_train_negative.index]

    X_train_balanced = pd.concat([X_train_negative, X_train.loc[y_train == 1]])
    y_train_balanced = pd.concat([y_train_negative, y_train.loc[y_train == 1]])

    return X_train_balanced, y_train_balanced


def _apply_training_regime(X_train, y_train, step, fold_idx):
    level = step["level"]
    seed = set_random_seed("multi_training", level, fold_idx, base_seed=42)

    X_train_sampled = X_train.sample(frac=level, random_state=seed)
    y_train_sampled = y_train.loc[X_train_sampled.index]

    return X_train_sampled, y_train_sampled


def _apply_composite_perturbations(X_train, X_val, y_train, train_idx, val_idx, fold_idx, composite_steps):
    for step in composite_steps:
        perturb_type = step["type"]

        if perturb_type == "label_noise":
            y_train = _apply_label_noise(y_train, X_train, train_idx, step, fold_idx)
        elif perturb_type == "input_noise":
            X_train, X_val = _apply_input_noise(X_train, X_val, step, fold_idx)
        elif perturb_type == "missing_data":
            X_train, X_val = _apply_missing_data(X_train, X_val, train_idx, val_idx, step, fold_idx)
        elif perturb_type == "feature_permutation":
            X_train, X_val = _apply_feature_permutation(X_train, X_val, step, fold_idx)
        elif perturb_type == "class_imbalance":
            X_train, y_train = _apply_class_imbalance(X_train, y_train, step, fold_idx)
        elif perturb_type == "training_data_regime":
            X_train, y_train = _apply_training_regime(X_train, y_train, step, fold_idx)
        else:
            raise ValueError(f"Unsupported perturbation type in composite: {perturb_type}")

    return X_train, X_val, y_train


def composite_perturbation(
    model_name,
    model,
    composite_steps,
    train_idx,
    val_idx,
    fold_idx,
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
):
    if tune_model_fn is None:
        raise ValueError("tune_model_fn must be provided.")
    if calibrate_model_fn is None:
        raise ValueError("calibrate_model_fn must be provided.")
    if compute_calibration_metrics_fn is None:
        raise ValueError("compute_calibration_metrics_fn must be provided.")

    X_train, X_val = X.loc[train_idx].copy(), X.loc[val_idx].copy()
    y_train, y_val = y.loc[train_idx].copy(), y.loc[val_idx].copy()

    X_train, X_val, y_train = _apply_composite_perturbations(
        X_train,
        X_val,
        y_train,
        train_idx,
        val_idx,
        fold_idx,
        composite_steps,
    )

    seed = set_random_seed("composite", json.dumps(composite_steps, sort_keys=True), fold_idx, base_seed=42)

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params = tune_model_fn(model_name, model, X_train, y_train, random_state=seed)
    train_fit_time = time.time() - start

    tuned_model, X_val, y_val = calibrate_model_fn(
        tuned_model,
        X_val,
        y_val,
        calibration_size=0.1,
        random_state=42,
    )

    start = time.time()
    y_pred = predict_proba_batched(tuned_model, X_val)
    test_pred_time = time.time() - start

    intercept, slope, prob_true, prob_pred = compute_calibration_metrics_fn(y_val, y_pred, model_name)

    return (
        model_name,
        json.dumps(composite_steps, sort_keys=True),
        roc_auc_score(y_score=y_pred, y_true=y_val),
        brier_score_loss(y_val, y_pred),
        intercept,
        slope,
        prob_true,
        prob_pred,
        train_fit_time,
        test_pred_time,
        best_params,
    )


perturbation_task_kwargs = {
    "tune_model_fn": partial(
        tune_model,
        param_grids=param_grids,
        n_search_gs=N_SEARCH_GS,
        njobs_gs=NJOBS_GS,
        random_state_global=RANDOM_STATE,
        log_file=LOG_FILE,
    ),
    "calibrate_model_fn": calibrate_model,
    "compute_calibration_metrics_fn": partial(compute_calibration_metrics, log_file=LOG_FILE_CALIB),
}
composite_task = partial(composite_perturbation, **perturbation_task_kwargs)


directory_name = DIR_NAME
create_directory(directory_name)
directory = f"{directory_name}/multi_perturbation/"
create_directory(directory)

composites = sample_composite_perturbations(
    n_composites=N_COMPOSITES,
    n_types=N_PERTURB_TYPES_PER_COMPOSITE,
    random_state=RANDOM_STATE,
)

composites_report_path = os.path.join(directory, "composite_combinations.json")
save_composites_report(composites, composites_report_path)

print("Sampled composite perturbations:")
for i, comp in enumerate(composites, start=1):
    print(f"Composite {i}: {json.dumps(comp)}")
print(f"Saved composite report to {composites_report_path}")

run_cv_parallel_and_save_cfg(
    composite_task,
    composites,
    directory,
    "MULTI_PERTURBATION_COMPOSITE",
    column_m3,
)

print("MULTI-PERTURBATION BENCHMARK OVER")

