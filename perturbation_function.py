# -*- coding: utf-8 -*-

import copy
import time

import numpy as np
import pandas as pd
import torch

from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from helper import predict_proba_batched, set_random_seed
from m3_handling import (
    X,
    y,
    y_proxy_death_icu,
    y_proxy_death_overall,
    cont_features,
    cat_features,
    features,
    mimic_3,
)
from missing_data_mechanism import add_missingness


perturb_dic = {
    "label_noise": [
        "random",
        "0to1",
        "1to0",
        "conditional",
        "proxy_hospital_death",
        "proxy_icu_death",
        "proxy_overall_death",
    ],
    "input_noise": [
        "train_continuous",
        "train_categorical",
        "train_continuous_and_categorical",
        "val_continuous",
        "val_categorical",
        "val_continuous_and_categorical",
        "train_val_continuous",
        "train_val_categorical",
        "train_val_continuous_and_categorical",
    ],
    "missing_data": [
        "train_mcar",
        "train_mar",
        "train_mnar",
        "val_mcar",
        "val_mar",
        "val_mnar",
        "train_val_mcar",
        "train_val_mar",
        "train_val_mnar",
    ],
    "feature_permutation": [
        "train_feature_shuffle",
        "val_feature_shuffle",
        "train_val_feature_shuffle",
    ],
    "class_imbalance": [
        "negative_class_downsampling",
    ],
    "training_data_regime": [
        "reduced_training_size",
    ],
}


def _require_callbacks(tune_model_fn=None, calibrate_model_fn=None, compute_calibration_metrics_fn=None):
    if tune_model_fn is None:
        raise ValueError("tune_model_fn must be provided.")
    if calibrate_model_fn is None:
        raise ValueError("calibrate_model_fn must be provided.")
    if compute_calibration_metrics_fn is None:
        raise ValueError("compute_calibration_metrics_fn must be provided.")
    return tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn


def evaluate_standard_bundle(bundle, compute_calibration_metrics_fn=None):
    if compute_calibration_metrics_fn is None:
        raise ValueError("compute_calibration_metrics_fn must be provided.")

    model = bundle["model"]
    X_val = bundle["X_val"]
    y_val = bundle["y_val"]
    model_name = bundle["model_name"]

    start = time.time()
    y_pred = predict_proba_batched(model, X_val)
    end = time.time()
    test_pred_time = end - start

    intercept, slope, prob_true, prob_pred = compute_calibration_metrics_fn(y_val, y_pred, model_name)

    return (
        model_name,
        bundle["x_value"],
        roc_auc_score(y_score=y_pred, y_true=y_val),
        brier_score_loss(y_val, y_pred),
        intercept,
        slope,
        prob_true,
        prob_pred,
        bundle["train_fit_time"],
        test_pred_time,
        bundle["best_params"],
    )


def evaluate_subgroup_bundle(bundle, compute_calibration_metrics_fn=None):
    if compute_calibration_metrics_fn is None:
        raise ValueError("compute_calibration_metrics_fn must be provided.")

    model = bundle["model"]
    X_val = bundle["X_val"]
    y_val = bundle["y_val"]
    stratify_on = bundle["stratify_on"]
    model_name = bundle["model_name"]

    stratified_perf = []
    stratify_values = mimic_3.loc[X_val.index, stratify_on]

    for cat in mimic_3[stratify_on].dropna().unique():
        subgroup_mask = stratify_values == cat
        subgroup_index = stratify_values.index[subgroup_mask]
        X_val_strat = X_val.loc[subgroup_index]
        y_val_strat = y_val.loc[subgroup_index]

        if len(X_val_strat) == 0:
            continue

        start = time.time()
        y_pred = predict_proba_batched(model, X_val_strat)
        end = time.time()
        test_pred_time = end - start

        intercept, slope, prob_true, prob_pred = compute_calibration_metrics_fn(
            y_val_strat,
            y_pred,
            model_name,
        )

        stratified_perf.append(
            [
                model_name,
                stratify_on,
                cat,
                roc_auc_score(y_score=y_pred, y_true=y_val_strat),
                brier_score_loss(y_val_strat, y_pred),
                intercept,
                slope,
                prob_true,
                prob_pred,
                bundle["train_fit_time"],
                test_pred_time,
                bundle["best_params"],
            ]
        )

    return stratified_perf


def evaluate_temporal_bundle(bundle, compute_calibration_metrics_fn=None):
    if compute_calibration_metrics_fn is None:
        raise ValueError("compute_calibration_metrics_fn must be provided.")

    model = bundle["model"]
    X_test = bundle["X_test"]
    y_test = bundle["y_test"]
    stratify_on = bundle["stratify_on"]
    model_name = bundle["model_name"]
    cols = bundle["train_columns"]

    if stratify_on is not None:
        stratified_perf = []
        for cat in X_test[stratify_on].dropna().unique():
            subgroup_mask = X_test[stratify_on] == cat
            subgroup_index = X_test.index[subgroup_mask]
            X_eval = X_test.loc[subgroup_index, cols]
            y_eval = y_test.loc[subgroup_index]

            if len(X_eval) == 0:
                continue

            start = time.time()
            y_pred = predict_proba_batched(model, X_eval)
            end = time.time()
            test_pred_time = end - start

            intercept, slope, prob_true, prob_pred = compute_calibration_metrics_fn(
                y_eval,
                y_pred,
                model_name,
            )

            stratified_perf.append(
                [
                    model_name,
                    stratify_on,
                    cat,
                    roc_auc_score(y_score=y_pred, y_true=y_eval),
                    brier_score_loss(y_eval, y_pred),
                    intercept,
                    slope,
                    prob_true,
                    prob_pred,
                    bundle["train_fit_time"],
                    test_pred_time,
                    bundle["best_params"],
                ]
            )

        return stratified_perf

    start = time.time()
    y_pred = predict_proba_batched(model, X_test[cols])
    end = time.time()
    test_pred_time = end - start

    intercept, slope, prob_true, prob_pred = compute_calibration_metrics_fn(y_test, y_pred, model_name)

    return (
        model_name,
        roc_auc_score(y_score=y_pred, y_true=y_test),
        brier_score_loss(y_test, y_pred),
        intercept,
        slope,
        prob_true,
        prob_pred,
        bundle["train_fit_time"],
        test_pred_time,
    )


def label_noise(
    model_name,
    model,
    noise_level,
    train_idx,
    val_idx,
    fold_idx,
    noise_type="random",
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
):
    tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn = _require_callbacks(
        tune_model_fn,
        calibrate_model_fn,
        compute_calibration_metrics_fn,
    )

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]
    y_proxy_death_icu_train = y_proxy_death_icu.loc[train_idx]
    y_proxy_death_overall_train = y_proxy_death_overall.loc[train_idx]

    y_train_noisy = y_train.copy()
    seed = set_random_seed(noise_type, noise_level, fold_idx, base_seed=42)
    rng = np.random.default_rng(seed)

    if noise_type == "random":
        probs = rng.random(len(y_train_noisy))
        idx_change_outcome = probs < noise_level
        y_train_noisy[idx_change_outcome] = 1 - y_train_noisy
    elif noise_type == "0to1":
        mask = y_train_noisy == 0
        probs = rng.random(mask.sum())
        idx_change = probs < noise_level
        idxs = np.where(mask)[0][idx_change]
        y_train_noisy.iloc[idxs] = 1
    elif noise_type == "1to0":
        mask = y_train_noisy == 1
        probs = rng.random(mask.sum())
        idx_change = probs < noise_level
        idxs = np.where(mask)[0][idx_change]
        y_train_noisy.iloc[idxs] = 0
    elif noise_type == "conditional":
        age_train_perc = X_train.age.rank(pct=True)
        swap_proba = age_train_perc * noise_level
        swap_by_age = rng.binomial(1, p=swap_proba, size=len(age_train_perc))
        y_train_noisy = y_train_noisy * (1 - swap_by_age) + (1 - y_train_noisy) * (swap_by_age)
    elif noise_type == "proxy":
        if noise_level == "hospital_death":
            y_train_noisy = y_train_noisy
        elif noise_level == "icu_death":
            y_train_noisy = y_proxy_death_icu_train
        elif noise_level == "overall_death":
            y_train_noisy = y_proxy_death_overall_train

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params = tune_model_fn(
        model_name,
        model,
        X_train,
        y_train_noisy,
        random_state=seed,
        preset_context={
            "test_name": preset_test_name,
            "noise_level": noise_level,
            "fold_idx": fold_idx,
        },
    )
    end = time.time()
    train_fit_time = end - start

    tuned_model, X_val, y_val = calibrate_model_fn(
        tuned_model,
        X_val,
        y_val,
        calibration_size=0.1,
        random_state=42,
    )

    if return_trained:
        return {
            "model": tuned_model,
            "X_val": X_val,
            "y_val": y_val,
            "model_name": model_name,
            "x_value": noise_level,
            "train_fit_time": train_fit_time,
            "best_params": best_params,
        }

    start = time.time()
    y_pred = predict_proba_batched(tuned_model, X_val)
    end = time.time()
    test_pred_time = end - start

    intercept, slope, prob_true, prob_pred = compute_calibration_metrics_fn(y_val, y_pred, model_name)

    del tuned_model

    return (
        model_name,
        noise_level,
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


def add_measurement_noise(X_frame, noise_level, feature_type="cont & cat", rng=None):
    X_noisy = X_frame.copy()
    size = X_noisy.shape
    if rng is None:
        rng = np.random.default_rng()

    if "cont" in feature_type:
        for feature in cont_features:
            std_j = X_noisy[feature].std()
            noise_j = rng.normal(loc=0.0, scale=std_j * (noise_level * 2), size=size[0])
            X_noisy[feature] = X_noisy[feature] + noise_j

    if "cat" in feature_type:
        for feature in cat_features:
            mask = rng.random(len(X_frame)) < noise_level

            value_counts = X_noisy[feature].value_counts(normalize=True)
            categories = value_counts.index.values
            probabilities = value_counts.values

            replacements = rng.choice(categories, size=mask.sum(), p=probabilities)
            X_noisy.loc[mask, feature] = replacements

    return X_noisy


def input_noise(
    model_name,
    model,
    noise_level,
    train_idx,
    val_idx,
    fold_idx,
    which_set="Train",
    feature_type="cont & cat",
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
):
    tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn = _require_callbacks(
        tune_model_fn,
        calibrate_model_fn,
        compute_calibration_metrics_fn,
    )

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

    seed = set_random_seed(which_set + feature_type, noise_level, fold_idx, base_seed=42)
    rng = np.random.default_rng(seed)

    if which_set == "Train":
        X_train = add_measurement_noise(X_train, noise_level, feature_type, rng=rng)
    if which_set == "Val":
        X_val = add_measurement_noise(X_val, noise_level, feature_type, rng=rng)
    if which_set == "Train_Val":
        X_train = add_measurement_noise(X_train, noise_level, feature_type, rng=rng)
        X_val = add_measurement_noise(X_val, noise_level, feature_type, rng=rng)

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params = tune_model_fn(
        model_name,
        model,
        X_train,
        y_train,
        random_state=seed,
        preset_context={
            "test_name": preset_test_name,
            "noise_level": noise_level,
            "fold_idx": fold_idx,
        },
    )
    end = time.time()
    train_fit_time = end - start

    tuned_model, X_val, y_val = calibrate_model_fn(
        tuned_model,
        X_val,
        y_val,
        calibration_size=0.1,
        random_state=42,
    )

    if return_trained:
        return {
            "model": tuned_model,
            "X_val": X_val,
            "y_val": y_val,
            "model_name": model_name,
            "x_value": noise_level,
            "train_fit_time": train_fit_time,
            "best_params": best_params,
        }

    start = time.time()
    y_pred = predict_proba_batched(tuned_model, X_val)
    end = time.time()
    test_pred_time = end - start

    intercept, slope, prob_true, prob_pred = compute_calibration_metrics_fn(y_val, y_pred, model_name)

    del tuned_model

    return (
        model_name,
        noise_level,
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


def imbalance_data(
    model_name,
    model,
    imbalance_ratio,
    train_idx,
    val_idx,
    fold_idx,
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
):
    tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn = _require_callbacks(
        tune_model_fn,
        calibrate_model_fn,
        compute_calibration_metrics_fn,
    )

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

    seed = set_random_seed("imbalance", imbalance_ratio, fold_idx, base_seed=42)

    X_train_negative = X_train.loc[y_train == 0].sample(frac=imbalance_ratio, random_state=seed)
    y_train_negative = y_train.loc[X_train_negative.index]
    X_train_balanced = pd.concat([X_train_negative, X_train.loc[y_train == 1]])
    y_train_balanced = pd.concat([y_train_negative, y_train.loc[y_train == 1]])

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params = tune_model_fn(
        model_name,
        model,
        X_train_balanced,
        y_train_balanced,
        random_state=seed,
        preset_context={
            "test_name": preset_test_name,
            "noise_level": imbalance_ratio,
            "fold_idx": fold_idx,
        },
    )
    end = time.time()
    train_fit_time = end - start

    tuned_model, X_val, y_val = calibrate_model_fn(
        tuned_model,
        X_val,
        y_val,
        calibration_size=0.1,
        random_state=42,
    )

    if return_trained:
        return {
            "model": tuned_model,
            "X_val": X_val,
            "y_val": y_val,
            "model_name": model_name,
            "x_value": imbalance_ratio,
            "train_fit_time": train_fit_time,
            "best_params": best_params,
        }

    start = time.time()
    y_pred = predict_proba_batched(tuned_model, X_val)
    end = time.time()
    test_pred_time = end - start

    intercept, slope, prob_true, prob_pred = compute_calibration_metrics_fn(y_val, y_pred, model_name)

    del tuned_model

    return (
        model_name,
        imbalance_ratio,
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


def training_data_regime(
    model_name,
    model,
    training_size,
    train_idx,
    val_idx,
    fold_idx,
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
):
    tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn = _require_callbacks(
        tune_model_fn,
        calibrate_model_fn,
        compute_calibration_metrics_fn,
    )

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

    seed = set_random_seed("training", training_size, fold_idx, base_seed=42)

    X_train_sampled = X_train.sample(frac=training_size, random_state=seed)
    y_train_sampled = y_train.loc[X_train_sampled.index]

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params = tune_model_fn(
        model_name,
        model,
        X_train_sampled,
        y_train_sampled,
        random_state=seed,
        preset_context={
            "test_name": preset_test_name,
            "noise_level": training_size,
            "fold_idx": fold_idx,
        },
    )
    end = time.time()
    train_fit_time = end - start

    tuned_model, X_val, y_val = calibrate_model_fn(
        tuned_model,
        X_val,
        y_val,
        calibration_size=0.1,
        random_state=42,
    )

    if return_trained:
        return {
            "model": tuned_model,
            "X_val": X_val,
            "y_val": y_val,
            "model_name": model_name,
            "x_value": training_size,
            "train_fit_time": train_fit_time,
            "best_params": best_params,
        }

    start = time.time()
    y_pred = predict_proba_batched(tuned_model, X_val)
    end = time.time()
    test_pred_time = end - start

    intercept, slope, prob_true, prob_pred = compute_calibration_metrics_fn(y_val, y_pred, model_name)

    del tuned_model

    return (
        model_name,
        training_size,
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


def shuffle_features(X_frame, prop=0.5, feat_to_shuffle=None, rng=None):
    X_noisy = X_frame.copy()
    size = X_noisy.shape

    if rng is None:
        rng = np.random.default_rng()

    n_feat_to_shuffle = int(size[1] * prop)
    if feat_to_shuffle is None:
        feat_to_shuffle = list(rng.choice(X_noisy.columns, size=n_feat_to_shuffle, replace=False))

    for col in feat_to_shuffle:
        X_noisy[col] = rng.permutation(X_noisy[col].values)

    return X_noisy, feat_to_shuffle


def permutation_features(
    model_name,
    model,
    noise_level,
    train_idx,
    val_idx,
    fold_idx,
    which_set="Train",
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
):
    tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn = _require_callbacks(
        tune_model_fn,
        calibrate_model_fn,
        compute_calibration_metrics_fn,
    )

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

    seed = set_random_seed(which_set, noise_level, fold_idx, base_seed=42)
    rng = np.random.default_rng(seed)

    if which_set == "Train":
        X_train, _ = shuffle_features(X_train, noise_level, rng=rng)
    if which_set == "Val":
        X_val, _ = shuffle_features(X_val, noise_level, rng=rng)
    if which_set == "Train_Val":
        X_train, feat_to_shuffle = shuffle_features(X_train, noise_level, rng=rng)
        X_val, _ = shuffle_features(X_val, noise_level, feat_to_shuffle=feat_to_shuffle, rng=rng)

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params = tune_model_fn(
        model_name,
        model,
        X_train,
        y_train,
        random_state=seed,
        preset_context={
            "test_name": preset_test_name,
            "noise_level": noise_level,
            "fold_idx": fold_idx,
        },
    )
    end = time.time()
    train_fit_time = end - start

    tuned_model, X_val, y_val = calibrate_model_fn(
        tuned_model,
        X_val,
        y_val,
        calibration_size=0.1,
        random_state=42,
    )

    if return_trained:
        return {
            "model": tuned_model,
            "X_val": X_val,
            "y_val": y_val,
            "model_name": model_name,
            "x_value": noise_level,
            "train_fit_time": train_fit_time,
            "best_params": best_params,
        }

    start = time.time()
    y_pred = predict_proba_batched(tuned_model, X_val)
    end = time.time()
    test_pred_time = end - start

    intercept, slope, prob_true, prob_pred = compute_calibration_metrics_fn(y_val, y_pred, model_name)

    del tuned_model

    return (
        model_name,
        noise_level,
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


def missing_data(
    model_name,
    model,
    noise_level,
    train_idx,
    val_idx,
    fold_idx,
    which_set="Train",
    mechanism="MNAR",
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
):
    tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn = _require_callbacks(
        tune_model_fn,
        calibrate_model_fn,
        compute_calibration_metrics_fn,
    )

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

    X_train[cont_features] = X_train[cont_features].fillna(X_train[cont_features].mean())
    X_train[cat_features] = X_train[cat_features].fillna(X_train[cat_features].mode().loc[0])

    if which_set == "Train":
        X_noisy = X_train.copy()
        seed = set_random_seed(which_set + mechanism, noise_level, fold_idx, base_seed=42)
        rng = np.random.default_rng(seed)
        torch_gen = torch.Generator()
        torch_gen.manual_seed(seed)
        mask = add_missingness(
            X_noisy,
            noise_level,
            mechanism=mechanism,
            prop_cond_features=0.5,
            rng=rng,
            torch_gen=torch_gen,
        )
        X_train = X_noisy.mask(mask, np.nan)

    if which_set == "Val":
        X_noisy = X_val.copy()
        X_noisy[cont_features] = X_noisy[cont_features].fillna(X_train[cont_features].mean())
        X_noisy[cat_features] = X_noisy[cat_features].fillna(X_train[cat_features].mode().loc[0])
        seed = set_random_seed(which_set + mechanism, noise_level, fold_idx, base_seed=42)
        rng = np.random.default_rng(seed)
        torch_gen = torch.Generator()
        torch_gen.manual_seed(seed)
        mask = add_missingness(
            X_noisy,
            noise_level,
            mechanism=mechanism,
            prop_cond_features=0.5,
            rng=rng,
            torch_gen=torch_gen,
        )
        X_val = X_noisy.mask(mask, np.nan)

    if which_set == "Train_Val":
        X_noisy = pd.concat([X_train, X_val]).copy()
        X_noisy[cont_features] = X_noisy[cont_features].fillna(X_train[cont_features].mean())
        X_noisy[cat_features] = X_noisy[cat_features].fillna(X_train[cat_features].mode().loc[0])
        seed = set_random_seed(which_set + mechanism, noise_level, fold_idx, base_seed=42)
        rng = np.random.default_rng(seed)
        torch_gen = torch.Generator()
        torch_gen.manual_seed(seed)
        mask = add_missingness(
            X_noisy,
            noise_level,
            mechanism=mechanism,
            prop_cond_features=0.5,
            rng=rng,
            torch_gen=torch_gen,
        )
        X_all = X_noisy.mask(mask, np.nan)
        X_train, X_val = X_all.loc[train_idx], X_all.loc[val_idx]

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params = tune_model_fn(
        model_name,
        model,
        X_train,
        y_train,
        random_state=seed,
        preset_context={
            "test_name": preset_test_name,
            "noise_level": noise_level,
            "fold_idx": fold_idx,
        },
    )
    end = time.time()
    train_fit_time = end - start

    tuned_model, X_val, y_val = calibrate_model_fn(
        tuned_model,
        X_val,
        y_val,
        calibration_size=0.1,
        random_state=42,
    )

    if return_trained:
        return {
            "model": tuned_model,
            "X_val": X_val,
            "y_val": y_val,
            "model_name": model_name,
            "x_value": noise_level,
            "train_fit_time": train_fit_time,
            "best_params": best_params,
        }

    start = time.time()
    y_pred = predict_proba_batched(tuned_model, X_val)
    end = time.time()
    test_pred_time = end - start

    intercept, slope, prob_true, prob_pred = compute_calibration_metrics_fn(y_val, y_pred, model_name)

    del tuned_model

    return (
        model_name,
        noise_level,
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


def subgroup_analysis(
    model_name,
    model,
    train_idx,
    val_idx,
    stratify_on="gender",
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    calibrate_model_fn=None,
    compute_calibration_metrics_fn=None,
):
    tune_model_fn, calibrate_model_fn, compute_calibration_metrics_fn = _require_callbacks(
        tune_model_fn,
        calibrate_model_fn,
        compute_calibration_metrics_fn,
    )

    X_train, X_val = mimic_3.loc[train_idx], mimic_3.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

    seed = set_random_seed("subgroup", stratify_on, 0, base_seed=42)

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params = tune_model_fn(
        model_name,
        model,
        X_train[features],
        y_train,
        random_state=seed,
        preset_context={
            "test_name": preset_test_name,
            "noise_level": None,
            "fold_idx": 0,
        },
    )
    end = time.time()
    train_fit_time = end - start

    tuned_model, X_val, y_val = calibrate_model_fn(
        tuned_model,
        X_val[features],
        y_val,
        calibration_size=0.1,
        random_state=42,
    )

    if return_trained:
        return {
            "model": tuned_model,
            "X_val": X_val,
            "y_val": y_val,
            "stratify_on": stratify_on,
            "model_name": model_name,
            "train_fit_time": train_fit_time,
            "best_params": best_params,
        }

    start = time.time()
    y_pred_full = predict_proba_batched(tuned_model, X_val)
    end = time.time()
    test_pred_time = end - start

    stratified_perf = []
    stratify_values = mimic_3.loc[X_val.index, stratify_on]

    for cat in mimic_3[stratify_on].dropna().unique():
        subgroup_mask = stratify_values == cat
        subgroup_index = stratify_values.index[subgroup_mask]
        y_val_strat = y_val.loc[subgroup_index]
        y_pred = y_pred_full[subgroup_mask.to_numpy()]
        intercept, slope, prob_true, prob_pred = compute_calibration_metrics_fn(
            y_val_strat,
            y_pred,
            model_name,
        )

        stratified_perf.append(
            [
                model_name,
                stratify_on,
                cat,
                roc_auc_score(y_score=y_pred, y_true=y_val_strat),
                brier_score_loss(y_val_strat, y_pred),
                intercept,
                slope,
                prob_true,
                prob_pred,
                train_fit_time,
                test_pred_time,
                best_params,
            ]
        )

    del tuned_model

    return stratified_perf


def train_evaluate(
    model_name,
    model,
    X_train,
    y_train,
    df_test,
    stratify_on="gender",
    preset_test_name=None,
    return_trained=False,
    tune_model_fn=None,
    compute_calibration_metrics_fn=None,
):
    if tune_model_fn is None:
        raise ValueError("tune_model_fn must be provided.")
    if compute_calibration_metrics_fn is None:
        raise ValueError("compute_calibration_metrics_fn must be provided.")
    assert tune_model_fn is not None
    assert compute_calibration_metrics_fn is not None

    seed = set_random_seed("m4", stratify_on, 0, base_seed=42)

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params = tune_model_fn(
        model_name,
        model,
        X_train,
        y_train,
        random_state=seed,
        preset_context={
            "test_name": preset_test_name,
            "noise_level": None,
            "fold_idx": 0,
        },
    )
    end = time.time()
    train_fit_time = end - start

    X_test, X_cal, y_test, y_cal = train_test_split(
        df_test,
        df_test["hospital_mortality"],
        test_size=0.1,
        stratify=df_test["hospital_mortality"],
        random_state=42,
    )
    calibrated_clf = CalibratedClassifierCV(FrozenEstimator(tuned_model))
    calibrated_clf.fit(X_cal[X_train.columns], y_cal)
    tuned_model = calibrated_clf

    if return_trained:
        return {
            "model": tuned_model,
            "X_test": X_test,
            "y_test": y_test,
            "stratify_on": stratify_on,
            "model_name": model_name,
            "train_columns": X_train.columns,
            "train_fit_time": train_fit_time,
            "best_params": best_params,
        }

    start = time.time()
    y_pred_full = predict_proba_batched(tuned_model, X_test[X_train.columns])
    end = time.time()
    test_pred_time = end - start

    stratified_perf = []
    if stratify_on is not None:
        for cat in X_test[stratify_on].dropna().unique():
            subgroup_mask = X_test[stratify_on] == cat
            subgroup_index = X_test.index[subgroup_mask]
            y_eval_strat = y_test.loc[subgroup_index]
            y_pred = y_pred_full[subgroup_mask.to_numpy()]

            intercept, slope, prob_true, prob_pred = compute_calibration_metrics_fn(
                y_eval_strat,
                y_pred,
                model_name,
            )

            stratified_perf.append(
                [
                    model_name,
                    stratify_on,
                    cat,
                    roc_auc_score(y_score=y_pred, y_true=y_eval_strat),
                    brier_score_loss(y_eval_strat, y_pred),
                    intercept,
                    slope,
                    prob_true,
                    prob_pred,
                    train_fit_time,
                    test_pred_time,
                    best_params,
                ]
            )

        return stratified_perf

    intercept, slope, prob_true, prob_pred = compute_calibration_metrics_fn(y_test, y_pred_full, model_name)

    del tuned_model, calibrated_clf

    return (
        model_name,
        roc_auc_score(y_score=y_pred_full, y_true=y_test),
        brier_score_loss(y_test, y_pred_full),
        intercept,
        slope,
        prob_true,
        prob_pred,
        train_fit_time,
        test_pred_time,
    )
