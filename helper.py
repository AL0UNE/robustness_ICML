import json
import os
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import special


def stable_hash(s: str) -> int:
    """Return a stable hash of the given string."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def set_random_seed(test_name, noise_level, fold_idx, base_seed=42):
    """Return a deterministic integer seed for the given test/noise/fold.

    NOTE: this function does not mutate global RNG state. Callers should use
    the returned seed to construct local numpy and torch generators.
    """
    combined = f"{test_name}_{noise_level}_{fold_idx}_{base_seed}"
    seed = abs(stable_hash(combined)) % (2**32)
    return int(seed)


def create_directory(directory_name):
    os.makedirs(directory_name, exist_ok=True)


def process_df(df, n_folds=5):
    df_true_prob = (
        df["Prob true"].explode().groupby(level=0).apply(lambda x: pd.Series(x.values)).unstack()
    )
    df_true_prob.columns = ["Prob_true_fold_" + str(i + 1) for i in range(n_folds)]
    df_pred_prob = (
        df["Prob pred"].explode().groupby(level=0).apply(lambda x: pd.Series(x.values)).unstack()
    )
    df_pred_prob.columns = ["Prob_pred_fold_" + str(i + 1) for i in range(n_folds)]
    df = df.drop(["Prob true", "Prob pred"], axis=1)
    df = pd.concat([df, df_true_prob, df_pred_prob], axis=1)
    return df


def save_results(
    results,
    directory,
    n_folds,
    columns=[
        "Model",
        "Noise level",
        "AUC",
        "Brier score",
        "Intercept",
        "Slope",
        "Prob true",
        "Prob pred",
        "Train fit time",
        "Test pred time",
        "Best param",
    ],
    test_name="MEASUREMENT NOISE",
    n_tr=-1,
):
    df_results = pd.DataFrame(results, columns=columns)
    df_results = process_df(df_results, n_folds)
    df_results.to_csv(os.path.join(directory, f"{test_name}_{n_tr}.csv"), index=False)
    print(f"Saved {test_name} to {directory}")
    print("\n ======= \n")
    print(test_name, " OVER")


def make_json_safe(params):
    safe = {}
    for k, v in params.items():
        if isinstance(v, (np.integer, np.floating)):
            safe[k] = v.item()
        else:
            safe[k] = v
    return safe


def log_failure(params, error_msg, logging_file):
    """Append a failure message to a log file."""
    params = make_json_safe(params)
    with open(logging_file, "a") as f:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "params": params,
            "error": error_msg.strip().split("\n")[-1],
        }
        f.write(json.dumps(log_entry) + "\n")


def predict_proba_batched(model, X, batch_size: int = 100000):
    """
    Work around the CUDA 65 535-block limit in TabPFN's SDPA kernel
    by splitting any large matrix into manageable chunks.
    Returns the class-1 probabilities concatenated in order.
    """
    if hasattr(model, "predict_proba") and len(X) <= batch_size:
        return model.predict_proba(X)[:, 1]

    # Best-effort fallback for estimators without predict_proba.
    if not hasattr(model, "predict_proba"):
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            return special.expit(scores)
        raise TypeError(
            f"Model of type {type(model).__name__} does not support predict_proba or decision_function"
        )

    out = []
    for start in range(0, len(X), batch_size):
        X_batch = X.iloc[start : start + batch_size] if hasattr(X, "iloc") else X[start : start + batch_size]
        out.append(model.predict_proba(X_batch)[:, 1])
    return np.concatenate(out)
