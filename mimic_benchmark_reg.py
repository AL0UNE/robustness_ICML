# -*- coding: utf-8 -*-


import os
import hashlib
import warnings
import time
import json
import numpy as np
import pandas as pd
import torch

from scipy import optimize, special

import itertools
import traceback
from datetime import datetime
import gc

from joblib import Parallel, delayed, Memory

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold,RandomizedSearchCV, train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.frozen import FrozenEstimator
from sklearn.base import clone
import copy

from sklearn.linear_model import LinearRegression, Lasso, Ridge

from sklearn.neural_network import MLPRegressor

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from tabpfn import TabPFNRegressor



from hpo_grid_reg import PARAM_GRIDS

#from tabpfn import TabPFNClassifier
#from tabicl import TabICLClassifier


# If no CUDA GPU is available, allow CPU runs (slower) but do not abort.
if not torch.cuda.is_available():
    warnings.warn(
        "GPU device not found. The script will run on CPU (may be much slower)."
    )

#warnings.filterwarnings("ignore") 

#Datetime
now = datetime.now()


LOG_FILE = f"boosting_failures_{now.strftime('%Y%m%d_%H%M%S')}.log"
DIR_NAME = f"res_{now.strftime('%Y%m%d_%H%M%S')}"
LOG_FILE_CALIB = f"calibration_failures_{now.strftime('%Y%m%d_%H%M%S')}.log"


NJOBS = -1
NJOBS_GS = 1
N_TRAINING_SAMPLE = -1 ## No need to change it since we do not have to run the xp for 10k

HPO = True
N_SEARCH_GS = 15


print(f'N_JOBS: {NJOBS}')   
print(f'N_TRAINING_SAMPLE: {N_TRAINING_SAMPLE}')
print("Running models with hyperparameter optimization") if HPO else print("Running models without hyperparameter optimization")

RANDOM_STATE = 42

CACHE_DIR = os.path.join("results", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)


def stable_hash(s: str) -> int:
    """Return a stable hash of the given string."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16)


def set_random_seed(test_name, noise_level, fold_idx, base_seed=RANDOM_STATE):
    """Return a deterministic integer seed for the given test/noise/fold.

    NOTE: this function no longer mutates global RNG state. Callers should use
    the returned seed to construct local numpy and torch generators (for
    example `rng = np.random.default_rng(seed)` and
    `torch_gen = torch.Generator(); torch_gen.manual_seed(seed)`). This
    prevents interference with scikit-learn's randomized procedures.
    """
    combined = f"{test_name}_{noise_level}_{fold_idx}_{base_seed}"
    seed = abs(stable_hash(combined)) % (2**32)
    return int(seed)


cont_features = [
    "age",
    "heartrate_max",
    "heartrate_min",
    "sysbp_max",
    "sysbp_min",
    "tempc_max",
    "tempc_min",
    "urineoutput",
    "bun_min",
    "bun_max",
    "wbc_min",
    "wbc_max",
    "potassium_min",
    "potassium_max",
    "sodium_min",
    "sodium_max",
    "bicarbonate_min",
    "bicarbonate_max",
    "mingcs",
    'pao2fio2_vent_min',
    'bilirubin_min', 
    'bilirubin_max',
]


param_grids = PARAM_GRIDS if HPO else {}
#print(param_grids)

mimic_3 = pd.read_csv("m3.csv")

cat_features = ["aids", "hem", "mets", "admissiontype"]

features = cont_features + cat_features
mimic_3['icu_los'] = (pd.DatetimeIndex(mimic_3['outtime']) - pd.DatetimeIndex(mimic_3['intime']))
mimic_3['icu_los'] = mimic_3['icu_los'] = mimic_3['icu_los'].dt.total_seconds()/3600/24 ## compute length of icu stay in days

#outcome = ["hospital_mortality"]


X = mimic_3[features]
y = mimic_3["icu_los"] ## length of ICU stay in days, regression task


n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
splits = list(kf.split(X))

def create_directory(directory_name):
    os.makedirs(directory_name, exist_ok=True)


def save_results(
    results,
    directory,
    n_folds,
    columns=["Model", "Noise level", "MAE", "RMSE", "R2","Train fit time", "Test pred time", "Best param"],
    test_name="MEASUREMENT NOISE",
    n_tr = N_TRAINING_SAMPLE
):
    df_results = pd.DataFrame(results, columns=columns)
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

def log_failure(params, error_msg, logging_file=LOG_FILE):
    """Append a failure message to a log file."""
    params = make_json_safe(params)
    with open(logging_file, "a") as f:
        log_entry = {"timestamp": datetime.now().isoformat(), "params": params, "error": error_msg.strip().split("\n")[-1]}
        f.write(json.dumps(log_entry) + "\n")
    

def predict_proba_batched(model, X, batch_size: int = 100000):
    """
    Work around the CUDA 65 535-block limit in TabPFN’s SDPA kernel
    by splitting any large matrix into manageable chunks.
    """
    if len(X) <= batch_size:
        return model.predict(X)
    out = []
    for start in range(0, len(X), batch_size):
        X_batch = X.iloc[start:start + batch_size] if hasattr(X, "iloc") else X[start:start + batch_size]
        out.append(model.predict(X_batch))
    return np.concatenate(out)

def compute_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


##################### MISSING DATA MECHANISMS #############################
## Code directly taken from https://raw.githubusercontent.com/BorisMuzellec/MissingDataOT/master/utils.py
##### Missing At Random 

def MAR_mask(X, p, p_obs, rng=None, torch_gen=None):
    """
    Missing at random mechanism with a logistic masking model. First, a subset of variables with *no* missing values is
    randomly selected. The remaining variables have missing values according to a logistic model with random weights,
    re-scaled so as to attain the desired proportion of missing values on those variables.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated. If a numpy array is provided,
        it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    p_obs : float
        Proportion of variables with *no* missing values that will be used for the logistic masking model.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    mask = torch.zeros(n, d).bool() if to_torch else np.zeros((n, d)).astype(bool)

    if rng is None:
        rng = np.random.default_rng()
    if torch_gen is None:
        torch_gen = torch.Generator()

    d_obs = max(int(p_obs * d), 1)  ## number of variables that will have no missing values (at least one variable)
    d_na = d - d_obs  ## number of variables that will have missing values

    ### Sample variables that will all be observed, and those with missing values:
    idxs_obs = rng.choice(d, size=d_obs, replace=False)
    idxs_nas = np.array([i for i in range(d) if i not in idxs_obs])

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, idxs_obs, idxs_nas, torch_gen=torch_gen)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)

    ps = torch.sigmoid(X[:, idxs_obs].mm(coeffs) + intercepts)

    ber = torch.rand(n, d_na, generator=torch_gen)
    mask[:, idxs_nas] = ber < ps

    return mask

##### Missing not at random ######
def MNAR_self_mask_logistic(X, p, rng=None, torch_gen=None):
    """
    Missing not at random mechanism with a logistic self-masking model. Variables have missing values probabilities
    given by a logistic model, taking the same variable as input (hence, missingness is independent from one variable
    to another). The intercepts are selected to attain the desired missing rate.

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data for which missing values will be simulated.
        If a numpy array is provided, it will be converted to a pytorch tensor.

    p : float
        Proportion of missing values to generate for variables which will have missing values.

    Returns
    -------
    mask : torch.BoolTensor or np.ndarray (depending on type of X)
        Mask of generated missing values (True if the value is missing).

    """

    n, d = X.shape

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    to_torch = torch.is_tensor(X)  ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)

    if rng is None:
        rng = np.random.default_rng()
    if torch_gen is None:
        torch_gen = torch.Generator()

    ### Pick coefficients so that W^Tx has unit variance (avoids shrinking)
    coeffs = pick_coeffs(X, self_mask=True, torch_gen=torch_gen)
    ### Pick the intercepts to have a desired amount of missing values
    intercepts = fit_intercepts(X, coeffs, p, self_mask=True)

    ps = torch.sigmoid(X * coeffs + intercepts)

    ber = torch.rand(n, d, generator=torch_gen)
    mask = ber < ps

    return mask


def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False, torch_gen=None):
    n, d = X.shape
    if torch_gen is None:
        torch_gen = torch.Generator()
    if self_mask:
        coeffs = torch.randn(d, generator=torch_gen)
        Wx = X * coeffs
        coeffs /= torch.std(Wx, 0)
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = torch.randn(d_obs, d_na, generator=torch_gen)
        Wx = X[:, idxs_obs].mm(coeffs)
        coeffs /= torch.std(Wx, 0, keepdim=True)
    return coeffs


def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        d = len(coeffs)
        intercepts = torch.zeros(d)
        for j in range(d):
            def f(x):
                return torch.sigmoid(X * coeffs[j] + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = torch.zeros(d_na)
        for j in range(d_na):
            def f(x):
                return torch.sigmoid(X.mv(coeffs[:, j]) + x).mean().item() - p
            intercepts[j] = optimize.bisect(f, -50, 50)
    return intercepts


def tune_model(model_name, model, X_train, y_train, random_state=None):
    if model_name not in param_grids:
        model.fit(X_train, y_train)
        return model, {}

    cv_inner = KFold(n_splits=3, shuffle=True, random_state=random_state)
    param_dist = param_grids[model_name]
    if model_name not in ['CatBoost', 'XGBoost', 'LightGBM']:
        search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=N_SEARCH_GS, scoring="neg_mean_squared_error", cv=cv_inner, n_jobs=NJOBS_GS, random_state=random_state, verbose=0)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        if model_name in ['LASSO', 'Ridge']:
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
        n_to_try = min(N_SEARCH_GS, len(param_grid))
        param_idx = rng.choice(len(param_grid), size=n_to_try, replace=False)
        best_score = np.inf
        best_model = None
        best_params = None
        for p in param_idx:
            params = param_grid[p]
            try:
                model_step, score = grid_step(model_name, X_train, y_train, cv_inner, params)
            except Exception as e:
                error_msg = traceback.format_exc()
                log_failure(params, error_msg)
                print(f"Error during tuning {model_name} with params {params}: {e}")
                continue
            if score < best_score:
                best_score = score
                best_params = params
                best_model = model_step
        if best_params is None:
            print(f"All hyperparameter combinations failed for {model_name}. Using default parameters.")
            model.fit(X_train, y_train)
            return model, {}
        print(f"Best params for {model_name}: {best_params}")
        
        X_train_refit, X_early_stop, y_train_refit, y_early_stop = train_test_split(
            X_train, y_train, test_size=0.1, random_state=RANDOM_STATE)

#        print(f"Best params for {model_name}: {best_params}, Best iteration: {best_model.best_iteration}")
        if model_name =='LightGBM':
            final_model = LGBMRegressor(**best_params)
            final_model.fit(X_train_refit, y_train_refit, eval_set=[(X_early_stop, y_early_stop)], callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
            best_params['model_n_iter'] = final_model.best_iteration_
        elif model_name == 'XGBoost':
            final_model = XGBRegressor(**best_params)
            final_model.fit(X_train_refit, y_train_refit, eval_set=[(X_early_stop, y_early_stop)], verbose=False)
            best_params['model_n_iter'] = final_model.best_iteration
        elif model_name == 'CatBoost':
            final_model = CatBoostRegressor(**best_params)
            final_model.fit(X_train_refit, y_train_refit, eval_set=[(X_early_stop, y_early_stop)], verbose=True)
            best_params['model_n_iter'] = final_model.get_best_iteration()
        else:
            final_model = best_model
            final_model.fit(X_train, y_train)
        return final_model, best_params

 
def grid_step(model_name, X_tr, y_tr, cv_inner, param_grid):
    scores = []
    for train_idx, val_idx in cv_inner.split(X_tr):
        X_tr_fold, X_val = X_tr.iloc[train_idx], X_tr.iloc[val_idx]
        y_tr_fold, y_val = y_tr.iloc[train_idx], y_tr.iloc[val_idx]
        X_tr_fold, X_early_stop, y_tr_fold, y_early_stop = train_test_split(X_tr_fold, y_tr_fold, test_size=0.1, random_state=RANDOM_STATE)    

        if model_name == 'LightGBM':
            dtrain = lgb.Dataset(X_tr_fold, y_tr_fold)
            dearly_stop = lgb.Dataset(X_early_stop, y_early_stop)
            model_step = lgb.train(
                params=param_grid,
                train_set=dtrain,
                valid_sets=[dearly_stop],
                callbacks=[lgb.early_stopping(stopping_rounds=50, first_metric_only=True, verbose=True)],
            )
            score = np.sqrt(mean_squared_error(y_val, model_step.predict(X_val)))
            scores.append(score)           
        elif model_name == 'XGBoost':
            model_step = XGBRegressor()
            model_step.set_params(**param_grid)
            model_step.fit(X_tr_fold, y_tr_fold, eval_set=[(X_early_stop, y_early_stop)], verbose=False)
            score = np.sqrt(mean_squared_error(y_val, model_step.predict(X_val)))
            scores.append(score)
        elif model_name == 'CatBoost':
            model_step = CatBoostRegressor()
            model_step.set_params(**param_grid)
            model_step.fit(X_tr_fold, y_tr_fold, eval_set=[(X_early_stop, y_early_stop)], verbose=True)
            score = np.sqrt(mean_squared_error(y_val, model_step.predict(X_val)))
            scores.append(score)
    return model_step, np.mean(scores)
    

continuous_transformer_1 = Pipeline(steps=[("imputer", IterativeImputer(add_indicator=True, random_state=RANDOM_STATE)), ("scaler", StandardScaler())])

continuous_transformer_2 = Pipeline(steps=[("imputer", IterativeImputer(add_indicator=True, random_state=RANDOM_STATE))])

categorical_transformer = Pipeline(steps=[("imputer", SimpleImputer(add_indicator=True, strategy="most_frequent"))])

preprocessor_1 = ColumnTransformer(transformers=[
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
    # Linear method
    #"Linear": Pipeline([("preprocessor", preprocessor_1), ("lr", LinearRegression())], memory=memory),
    #"LASSO": Pipeline([("preprocessor", preprocessor_1),("lr", Lasso()),], memory=memory),
    #"Ridge": Pipeline([("preprocessor", preprocessor_1),("lr", Ridge()),], memory=memory),

    # Tree-based methods
    #"Random Forest": RandomForestRegressor(n_jobs=1),
    ## Boosting    
    #"Gradient Boosting": Pipeline([("preprocessor", preprocessor_2), ("gb", GradientBoostingRegressor())], memory=memory),
    "XGBoost": XGBRegressor(tree_method='hist',device='cuda'),
    "LightGBM": LGBMRegressor(device='gpu',gpu_use_dp=True),
    #"CatBoost": CatBoostRegressor(),

    # Deep learning
    #"MLP": Pipeline([("preprocessor", preprocessor_1), ("mlp", MLPRegressor(max_iter=500))], memory=memory),

    ## Transformers
    "TabPFN": TabPFNClassifier(),
    #"TabICL": TabICLClassifier(),
}

models_name = list(models.keys())

n_models = len(models_name)

directory_name = "regression" 
create_directory(directory_name)

## Robustness tests
"""### Label noise
"""
random_label_noise_levels = np.linspace(0, 1, 11)


def label_noise(
    model_name, model, noise_level, train_idx, val_idx, fold_idx, noise_type="random"
):

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

    
    y_train_noisy = y_train.copy()
    seed = set_random_seed(noise_type, noise_level, fold_idx, base_seed=RANDOM_STATE)
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator()
    torch_gen.manual_seed(seed)

    scale = np.std(y_train) if np.std(y_train) > 0 else 1.0
    base_amp = noise_level * 0.5 * scale
    z = rng.normal(0.0, 1.0, size=len(y_train))

    if noise_type == "random":
        # use local rng
        eps = (z ** 2) * base_amp  # always >= 0
        y_train_noisy = y_train_noisy + eps
    elif noise_type == "conditional":
        age_train_perc = X_train.age.rank(pct=True)
        eps = (z ** 2) * (base_amp * age_train_perc)  # always >= 0
        y_train_noisy = y_train_noisy + eps


    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params = tune_model(model_name, model, X_train, y_train_noisy, random_state=seed)
    end = time.time()
    train_fit_time = end - start

    start = time.time()
    y_pred = predict_proba_batched(tuned_model, X_val)
    end = time.time()
    test_pred_time = end - start

    mae, rmse, r2 = compute_regression_metrics(y_val, y_pred)

    del tuned_model
    gc.collect()

    return (
        model_name,
        noise_level,
        mae,
        rmse,
        r2,
        train_fit_time,
        test_pred_time,
        best_params
    )


# Safeguard
directory = f"{directory_name}/label_noise/"
create_directory(directory)


results_random_label_noise = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(label_noise)(
        model_name, model, noise_level, train_idx, val_idx, fold_idx, noise_type="random"
    )
    for model_name, model in models.items()
    for noise_level in random_label_noise_levels
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(
    results_random_label_noise, directory, n_folds, test_name="RANDOM_LABEL_NOISE"
)

results_label_noise_age = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(label_noise)(
        model_name, model, noise_level, train_idx, val_idx, fold_idx, noise_type="conditional"
    )
    for model_name, model in models.items()
    for noise_level in random_label_noise_levels
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(results_label_noise_age, directory, n_folds, test_name="AGE_LABEL_NOISE")

print("LABEL NOISE OVER")

"""### Measurement noise

"""
input_noise_level = np.linspace(0, 1, 11)

def add_measurement_noise(X, noise_level, feature_type="cont & cat", rng=None):
    """Add noise to continuous and/or categorical features.

    Parameters
    - rng: optional numpy.random.Generator for deterministic noise. If None,
      falls back to np.random.default_rng() (non-deterministic).
    """
    X_noisy = X.copy()
    size = X_noisy.shape
    if rng is None:
        rng = np.random.default_rng()

    if "cont" in feature_type:
        for j in cont_features:
            std_j = X_noisy[j].std()
            noise_j = rng.normal(loc=0.0, scale=std_j * (noise_level * 2), size=size[0])
            X_noisy[j] = X_noisy[j] + noise_j

    if "cat" in feature_type:
        for j in cat_features:
            mask = rng.random(len(X)) < noise_level

            value_counts = X_noisy[j].value_counts(normalize=True)
            categories = value_counts.index.values
            probabilities = value_counts.values

            replacements = rng.choice(categories, size=mask.sum(), p=probabilities)
            X_noisy.loc[mask, j] = replacements

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
):

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]


    # create local RNGs seeded deterministically for this (which_set, noise_level, fold)
    seed = set_random_seed(which_set + feature_type, noise_level, fold_idx, base_seed=RANDOM_STATE)
    rng = np.random.default_rng(seed)
    torch_gen = torch.Generator()
    torch_gen.manual_seed(seed)

    if which_set == "Train":
        X_train = add_measurement_noise(X_train, noise_level, feature_type, rng=rng)
    if which_set == "Val":
        X_val = add_measurement_noise(X_val, noise_level, feature_type, rng=rng)
    if which_set == "Train_Val":
        # use same rng to produce paired noise across train and val
        X_train = add_measurement_noise(X_train, noise_level, feature_type, rng=rng)
        X_val = add_measurement_noise(X_val, noise_level, feature_type, rng=rng)

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params = tune_model(model_name, model, X_train, y_train, random_state=seed)
    end = time.time()
    train_fit_time = end - start

    start = time.time()
    y_pred = predict_proba_batched(tuned_model, X_val)
    end = time.time()
    test_pred_time = end - start

    mae, rmse, r2 = compute_regression_metrics(y_val, y_pred)


    del tuned_model
    gc.collect()

    return (
        model_name,
        noise_level,
        mae,
        rmse,
        r2,
        train_fit_time,
        test_pred_time,
        best_params
    )


directory = f"{directory_name}/input_noise/"
create_directory(directory)

results_input_noise_train = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(model_name, model, noise_level, train_idx, val_idx, fold_idx)
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(
    results_input_noise_train, directory, n_folds, test_name="INPUT_NOISE_TRAIN"
)


results_input_noise_val = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name, model, noise_level, train_idx, val_idx, fold_idx, which_set="Val"
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(results_input_noise_val, directory, n_folds, test_name="INPUT_NOISE_VAL")

results_input_noise_all = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name, model, noise_level, train_idx, val_idx, fold_idx, which_set="Train_Val"
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(
    results_input_noise_all, directory, n_folds, test_name="INPUT_NOISE_ALL"
    )

results_input_noise_train_cont = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name, model, noise_level, train_idx, val_idx, fold_idx, feature_type="cont"
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(
    results_input_noise_train_cont, directory, n_folds, test_name="INPUT_NOISE_TRAIN_CONT")

results_input_noise_val_cont = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name, model, noise_level, train_idx, val_idx, fold_idx, which_set="Val", feature_type="cont",
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(
    results_input_noise_val_cont, directory, n_folds, test_name="INPUT_NOISE_VAL_CONT"
)

results_input_noise_all_cont = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name, model, noise_level, train_idx, val_idx, fold_idx, which_set="Train_Val", feature_type="cont",
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(
    results_input_noise_all_cont, directory, n_folds, test_name="INPUT_NOISE_ALL_CONT"
)

results_input_noise_train_cat = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name, model, noise_level, train_idx, val_idx, fold_idx, feature_type="cat"
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(
    results_input_noise_train_cat, directory, n_folds, test_name="INPUT_NOISE_TRAIN_CAT"
)

results_input_noise_val_cat = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name, model, noise_level, train_idx, val_idx, fold_idx, which_set="Val", feature_type="cat",
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(
    results_input_noise_val_cat, directory, n_folds, test_name="INPUT_NOISE_VAL_CAT"
)

results_input_noise_all_cat = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(input_noise)(
        model_name, model, noise_level, train_idx, val_idx, fold_idx, which_set="Train_Val", feature_type="cat",
    )
    for model_name, model in models.items()
    for noise_level in input_noise_level
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(
    results_input_noise_all_cat, directory, n_folds, test_name="INPUT_NOISE_ALL_CAT"
)


print("MEASUREMENT NOISE OVER")



"""### Training data size

"""
training_data_size = [0.05, 0.1, 0.25, 0.5, 0.8, 1]

def training_data_regime(model_name, model, training_size, train_idx, val_idx, fold_idx):
    
    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

    seed = set_random_seed("training", training_size, fold_idx, base_seed=RANDOM_STATE)

    X_train_sampled = X_train.sample(frac=training_size, random_state=seed)
    y_train_sampled = y_train.loc[X_train_sampled.index]
    
    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params = tune_model(model_name, model, X_train_sampled, y_train_sampled, random_state=seed)
    end = time.time()
    train_fit_time = end - start

    ## calibration here

    start = time.time()
    y_pred = predict_proba_batched(tuned_model, X_val)
    end = time.time()
    test_pred_time = end - start

    mae, rmse, r2 = compute_regression_metrics(y_val, y_pred)


    del tuned_model
    gc.collect()

    return (
        model_name,
        training_size,
        mae,
        rmse,
        r2,
        train_fit_time,
        test_pred_time,
        best_params
    )

results_training_size = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(training_data_regime)(model_name, model, ratio, train_idx, val_idx, fold_idx)
    for model_name, model in models.items()
    for ratio in training_data_size
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

directory = f"{directory_name}/training_size/"
create_directory(directory)

save_results(results_training_size, directory, n_folds, test_name="TRAINING_SIZE")

print("Training data size OVER")

### Feature shuffling

shuffle_ratio = np.linspace(0, 1, 11, endpoint=True)


directory = f"{directory_name}/feature_shuffle/"
create_directory(directory)


def shuffle_features(X, prop=0.5, feat_to_shuffle=None, rng=None):
    X_noisy = X.copy()
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
    model_name, model, noise_level, train_idx, val_idx, fold_idx, which_set="Train"
):

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]


    seed = set_random_seed(which_set, noise_level, fold_idx, base_seed=RANDOM_STATE)
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
    tuned_model, best_params = tune_model(model_name, model, X_train, y_train, random_state=seed)
    end = time.time()
    train_fit_time = end - start    

    start = time.time()
    y_pred = predict_proba_batched(tuned_model, X_val)
    end = time.time()
    test_pred_time = end - start

    mae, rmse, r2 = compute_regression_metrics(y_val, y_pred)

    del tuned_model
    gc.collect()

    return (
        model_name,
        noise_level,
        mae,
        rmse,   
        r2,
        train_fit_time,
        test_pred_time,
        best_params
    )


results_shuffled_data_train = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(permutation_features)(
        model_name, model, ratio, train_idx, val_idx, fold_idx, which_set="Train"
    )
    for model_name, model in models.items()
    for ratio in shuffle_ratio
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(
    results_shuffled_data_train, directory, n_folds, test_name="SHUFFLED_TRAIN_DATA"
)

results_shuffled_data_val = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(permutation_features)(
        model_name, model, ratio, train_idx, val_idx, fold_idx, which_set="Val"
    )
    for model_name, model in models.items()
    for ratio in shuffle_ratio
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(
    results_shuffled_data_val, directory, n_folds, test_name="SHUFFLED_VAL_DATA"
)

results_shuffled_data_all = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(permutation_features)(
        model_name, model, ratio, train_idx, val_idx, fold_idx, which_set="Train_Val"
    )
    for model_name, model in models.items()
    for ratio in shuffle_ratio
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(
    results_shuffled_data_all, directory, n_folds, test_name="SHUFFLED_ALL_DATA"
)

print("SHUFFLE NOISE OVER")

"""### Missing data

"""

directory = f"{directory_name}/missing_data/"
create_directory(directory)

missing_ratio = np.linspace(0.1, 1, 9, endpoint=False)

def add_missingness(X_noisy, noise_level, mechanism="MCAR", prop_cond_features=0.5):
    size = X_noisy.shape
    mask = None

    if mechanism == "MCAR":
        mask = np.random.rand(*size) < noise_level

    elif mechanism == "MAR":
        X_float = X_noisy.astype(np.float32).values
        # can't create rng here because we need caller's seed to be used; caller
        # should call MAR_mask directly with rng/torch_gen when determinism is
        # required. For backwards compatibility, fall back to MAR_mask default.
        mask = MAR_mask(X_float, p=noise_level, p_obs=prop_cond_features)

    elif mechanism == "MNAR":
        X_float = X_noisy.astype(np.float32).values
        mask = MNAR_self_mask_logistic(X_float, noise_level)

    if mask is None: 
        print("DEBUG : EMPTY MASK") 
    return mask


def missing_data(
    model_name,
    model,
    noise_level,
    train_idx,
    val_idx,
    fold_idx,
    which_set="Train",
    mechanism="MNAR",
):

    X_train, X_val = X.loc[train_idx], X.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

    ## imputation is required to create the MAR and MNAR masks
    X_train[cont_features] = X_train[cont_features].fillna(X_train[cont_features].mean())
    X_train[cat_features] = X_train[cat_features].fillna(X_train[cat_features].mode().loc[0])

    if which_set == "Train":
        X_noisy = X_train.copy()
        seed = set_random_seed(which_set+mechanism, noise_level, fold_idx, base_seed=RANDOM_STATE)
        rng = np.random.default_rng(seed)
        torch_gen = torch.Generator()
        torch_gen.manual_seed(seed)
        size = X_noisy.shape
        mask = (
            MAR_mask(X_noisy.astype(np.float32).values, p=noise_level, p_obs=0.5, rng=rng, torch_gen=torch_gen)
            if mechanism == "MAR"
            else (MNAR_self_mask_logistic(X_noisy.astype(np.float32).values, noise_level, rng=rng, torch_gen=torch_gen) if mechanism == "MNAR" else (rng.random(size) < noise_level))
        )
        X_train = X_noisy.mask(mask, np.nan)

    if which_set == "Val":
        X_noisy = X_val.copy()
        X_noisy[cont_features] = X_noisy[cont_features].fillna(X_train[cont_features].mean()) ## imputation is required to create the MAR and MNAR masks
        X_noisy[cat_features] = X_noisy[cat_features].fillna(X_train[cat_features].mode().loc[0])
        seed = set_random_seed(which_set+mechanism, noise_level, fold_idx, base_seed=RANDOM_STATE)
        rng = np.random.default_rng(seed)
        torch_gen = torch.Generator()
        torch_gen.manual_seed(seed)
        size = X_noisy.shape
        mask = (
            MAR_mask(X_noisy.astype(np.float32).values, p=noise_level, p_obs=0.5, rng=rng, torch_gen=torch_gen)
            if mechanism == "MAR"
            else (MNAR_self_mask_logistic(X_noisy.astype(np.float32).values, noise_level, rng=rng, torch_gen=torch_gen) if mechanism == "MNAR" else (rng.random(size) < noise_level))
        )
        X_val = X_noisy.mask(mask, np.nan)

    if which_set == "Train_Val":
        X_noisy = pd.concat([X_train, X_val]).copy()
        X_noisy[cont_features] = X_noisy[cont_features].fillna(X_train[cont_features].mean()) 
        X_noisy[cat_features] = X_noisy[cat_features].fillna(X_train[cat_features].mode().loc[0])
        seed = set_random_seed(which_set+mechanism, noise_level, fold_idx, base_seed=RANDOM_STATE)
        rng = np.random.default_rng(seed)
        torch_gen = torch.Generator()
        torch_gen.manual_seed(seed)
        size = X_noisy.shape
        mask = (
            MAR_mask(X_noisy.astype(np.float32).values, p=noise_level, p_obs=0.5, rng=rng, torch_gen=torch_gen)
            if mechanism == "MAR"
            else (MNAR_self_mask_logistic(X_noisy.astype(np.float32).values, noise_level, rng=rng, torch_gen=torch_gen) if mechanism == "MNAR" else (rng.random(size) < noise_level))
        )
        X_all = X_noisy.mask(mask, np.nan)
        X_train, X_val = X_all.loc[train_idx], X_all.loc[val_idx] 

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    #print(f"Debug {model_name}, {model}, {noise_level}")
    tuned_model, best_params = tune_model(model_name, model, X_train, y_train, random_state=seed)
    end = time.time()
    train_fit_time = end - start

    start = time.time()
    y_pred = predict_proba_batched(tuned_model, X_val)
    end = time.time()
    test_pred_time = end - start


    mae, rmse, r2 = compute_regression_metrics(y_val, y_pred)


    del tuned_model
    gc.collect()

    return (
        model_name,
        noise_level,
        mae,
        rmse,
        r2,
        train_fit_time,
        test_pred_time,
        best_params
    )

results_MCAR_train = Parallel(n_jobs=NJOBS, verbose=1)(
    delayed(missing_data)(
        model_name, model, ratio, train_idx, val_idx, fold_idx, which_set="Train", mechanism="MCAR"
    )
    for model_name, model in models.items()
    for ratio in missing_ratio
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(results_MCAR_train, directory, n_folds, test_name="MCAR_TRAIN")

results_MCAR_val = Parallel(n_jobs=NJOBS, verbose=1)(
    delayed(missing_data)(model_name, model, ratio, train_idx, val_idx, fold_idx, which_set="Val", mechanism="MCAR")
    for model_name, model in models.items()
    for ratio in missing_ratio
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(results_MCAR_val, directory, n_folds, test_name="MCAR_VAL")

results_MCAR_all = Parallel(n_jobs=NJOBS, verbose=1)(
    delayed(missing_data)(
        model_name, model, ratio, train_idx, val_idx, fold_idx, which_set="Train_Val", mechanism="MCAR"
    )
    for model_name, model in models.items()
    for ratio in missing_ratio
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(results_MCAR_all, directory, n_folds, test_name="MCAR_ALL")


results_MAR_train = Parallel(n_jobs=NJOBS, verbose=1)(
    delayed(missing_data)(
        model_name, model, ratio, train_idx, val_idx, fold_idx, which_set="Train", mechanism="MAR"
    )
    for model_name, model in models.items()
    for ratio in missing_ratio
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(results_MAR_train, directory, n_folds, test_name="MAR_TRAIN")

results_MAR_val = Parallel(n_jobs=NJOBS, verbose=1)(
    delayed(missing_data)(model_name, model, ratio, train_idx, val_idx, fold_idx, which_set="Val", mechanism="MAR")
    for model_name, model in models.items()
    for ratio in missing_ratio
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(results_MAR_val, directory, n_folds, test_name="MAR_VAL")

results_MAR_all = Parallel(n_jobs=NJOBS, verbose=1)(
    delayed(missing_data)(
        model_name, model, ratio, train_idx, val_idx, fold_idx, which_set="Train_Val", mechanism="MAR"
        )
    for model_name, model in models.items()
    for ratio in missing_ratio
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(results_MAR_all, directory, n_folds, test_name="MAR_ALL")

results_MNAR_train = Parallel(n_jobs=NJOBS, verbose=1)(
    delayed(missing_data)(
        model_name, model, ratio, train_idx, val_idx, fold_idx, which_set="Train"
    )
    for model_name, model in models.items()
    for ratio in missing_ratio
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(results_MNAR_train, directory, n_folds, test_name="MNAR_TRAIN")

results_MNAR_val = Parallel(n_jobs=NJOBS, verbose=1)(
    delayed(missing_data)(model_name, model, ratio, train_idx, val_idx, fold_idx, which_set="Val")
    for model_name, model in models.items()
    for ratio in missing_ratio
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(results_MNAR_val, directory, n_folds, test_name="MNAR_VAL")

results_MNAR_all = Parallel(n_jobs=NJOBS, verbose=1)(
    delayed(missing_data)(
        model_name, model, ratio, train_idx, val_idx, fold_idx, which_set="Train_Val",
        )
    for model_name, model in models.items()
    for ratio in missing_ratio
    for fold_idx, (train_idx, val_idx) in enumerate(splits)
)

save_results(results_MNAR_all, directory, n_folds, test_name="MNAR_ALL")


print("MISSING DATA OVER")

"""
### Subgroup analysis
"""
#Here we consider different types of subgroup analysis: both within MIMIC-III and across MIMIC-IV. <br> For each analysis we consider either stratifying on one of the included feature (e.g., age) in the model or on a external variable (e.g., gender, icu unit).  

directory = f"{directory_name}/subgroups/"
create_directory(directory)

#### MIMIC-III

#### By gender


def subgroup_analysis(model_name, model, train_idx, val_idx, stratify_on="gender"):

    X_train, X_val = mimic_3.loc[train_idx], mimic_3.loc[val_idx]
    y_train, y_val = y.loc[train_idx], y.loc[val_idx]

    seed = set_random_seed("subgroup", stratify_on, 0, base_seed=RANDOM_STATE)

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params = tune_model(model_name, model, X_train[features], y_train, random_state=seed)
    #model.fit(X_train, y_train)
    end = time.time()
    train_fit_time = end - start
    
    stratified_perf = []
    X_val[stratify_on] = mimic_3.loc[X_val.index, stratify_on]

    for cat in mimic_3[stratify_on].dropna().unique():
        
        X_val_strat = X_val.loc[X_val[stratify_on] == cat][features]
        y_val_strat = y_val.loc[X_val_strat.index]

        start = time.time()
        y_pred = predict_proba_batched(tuned_model, X_val_strat)
        end = time.time()
        test_pred_time = end - start
        mae, rmse, r2 = compute_regression_metrics(y_val_strat, y_pred)

        stratified_perf.append(
            [
                model_name,
                stratify_on,
                cat,
                mae,
                rmse,
                r2,
                train_fit_time,
                test_pred_time,
                best_params
            ]
        )

    del tuned_model
    gc.collect()

    return stratified_perf


results_subgroup_m3_gender = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(subgroup_analysis)(model_name, model, train_idx, val_idx, variable)
    for model_name, model in models.items()
    for variable in ["gender"]
    for train_idx, val_idx in splits
)
results_subgroup_m3_gender = [x for xs in results_subgroup_m3_gender for x in xs]

save_results(
    results_subgroup_m3_gender,
    directory,
    n_folds,
    columns=[
        "Model",
        "Variable",
        "Category",
        "MAE",
        "RMSE",
        "R2",
        "Train fit time",
        "Test pred time",
        "Best params"
    ],
    test_name="SUBGROUP_GENDER",
)


results_subgroup_m3_agegroup = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(subgroup_analysis)(model_name, model, train_idx, val_idx, variable)
    for model_name, model in models.items()
    for variable in ["age_group"]
    for train_idx, val_idx in splits
)
results_subgroup_m3_agegroup = [x for xs in results_subgroup_m3_agegroup for x in xs]

save_results(
    results_subgroup_m3_agegroup,
    directory,
    n_folds,
    columns=[
        "Model",
        "Variable",
        "Category",
        "MAE",
        "RMSE",
        "R2",
        "Train fit time",
        "Test pred time",
        "Best params"
    ],
    test_name="SUBGROUP_AGEGROUP",
)

results_subgroup_m3_icu_unit = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(subgroup_analysis)(model_name, model, train_idx, val_idx, variable)
    for model_name, model in models.items()
    for variable in ["ICU_unit"]
    for train_idx, val_idx in splits
)
results_subgroup_m3_icu_unit = [x for xs in results_subgroup_m3_icu_unit for x in xs]

save_results(
    results_subgroup_m3_icu_unit,
    directory,
    n_folds,
    columns=[
        "Model",
        "Variable",
        "Category",
        "MAE",
        "RMSE",
        "R2",
        "Train fit time",
        "Test pred time",
        "Best params"
    ],
    test_name="SUBGROUP_ICU_UNIT",
)


print("SUBGROUP NOISE OVER")

"""### Temporal validation"""

directory = f"{directory_name}/m4/"
create_directory(directory)

mimic_4 = pd.read_csv("m4.csv")

mimic_4 = mimic_4[mimic_4[features].isna().mean(axis=1)<=0.5] ## excluding patients with more than half of the features missing.

mimic_4['icu_los'] = (pd.DatetimeIndex(mimic_4['outtime']) - pd.DatetimeIndex(mimic_4['intime']))
mimic_4['icu_los'] = mimic_4['icu_los'].dt.total_seconds()/3600/24 ## compute length of icu stay in hours

def train_evaluate(model_name, model, X_train, y_train, df_test, stratify_on="gender"):


    seed = set_random_seed("m4", stratify_on, 0, base_seed=RANDOM_STATE)

    try:
        model = clone(model)
    except Exception:
        model = copy.deepcopy(model)

    start = time.time()
    tuned_model, best_params = tune_model(model_name, model, X_train, y_train, random_state=seed)
    #model.fit(X_train, y_train)
    end = time.time()
    train_fit_time = end - start


    start = time.time()
    y_pred = predict_proba_batched(tuned_model, df_test[X_train.columns])

    end = time.time()
    test_pred_time = end - start

    stratified_perf = []
    if stratify_on is not None:
        for cat in df_test[stratify_on].dropna().unique():
            X_eval_strat = df_test.loc[df_test[stratify_on] == cat][X_train.columns]
            y_eval_strat = df_test.loc[df_test[stratify_on] == cat]["icu_los"]

            start = time.time()
            y_pred = predict_proba_batched(tuned_model, X_eval_strat)
            end = time.time()
            test_pred_time = end - start

            mae, rmse, r2 = compute_regression_metrics(y_eval_strat, y_pred)

            stratified_perf.append(
                [
                    model_name,
                    stratify_on,
                    cat,
                    mae,
                    rmse,
                    r2,
                    train_fit_time,
                    test_pred_time,
                    best_params
                ]
            )

        return stratified_perf
    
    mae, rmse, r2 = compute_regression_metrics(df_test["icu_los"], y_pred)

    del tuned_model
    gc.collect()

    return (
        model_name,
        mae,
        rmse,
        r2,
        train_fit_time,
        test_pred_time,
    )

results_m4_gender = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(train_evaluate)(model_name, model, X, y, mimic_4, stratify_on=variable)
    for model_name, model in models.items()
    for variable in ["gender"]
)

results_m4_gender = [x for xs in results_m4_gender for x in xs]

save_results(
    results_m4_gender,
    directory,
    n_folds,
    columns=[
        "Model",
        "Variable",
        "Category",
        "MAE",
        "RMSE",
        "R2",
        "Train fit time",
        "Test pred time",
        "Best params"
    ],
    test_name="MIMIC_4_GENDER",
)

results_m4_age = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(train_evaluate)(model_name, model, X, y, mimic_4, stratify_on=variable)
    for model_name, model in models.items()
    for variable in ["age_group"]
)

results_m4_age = [x for xs in results_m4_age for x in xs]

save_results(
    results_m4_age,
    directory,
    n_folds,
    columns=[
        "Model",
        "Variable",
        "Category",
        "MAE",
        "RMSE",
        "R2",
        "Train fit time",
        "Test pred time",
        "Best params"
    ],
    test_name="MIMIC_4_AGE",
)

results_m4_icu_unit = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(train_evaluate)(model_name, model, X, y, mimic_4, stratify_on=variable)
    for model_name, model in models.items()
    for variable in ["ICU_unit"]
)

results_m4_icu_unit = [x for xs in results_m4_icu_unit for x in xs]

save_results(
    results_m4_icu_unit,
    directory,
    n_folds,
    columns=[
        "Model",
        "Variable",
        "Category",
        "MAE",
        "RMSE",
        "R2",
        "Train fit time",
        "Test pred time",
        "Best params"
    ],
    test_name="MIMIC_4_ICU_UNIT",
)

results_m4_year = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(train_evaluate)(model_name, model, X, y, mimic_4, stratify_on=variable)
    for model_name, model in models.items()
    for variable in ["anchor_year_group"]
)

results_m4_year = [x for xs in results_m4_year for x in xs]

save_results(
    results_m4_year,
    directory,
    n_folds,
    columns=[
        "Model",
        "Variable",
        "Category",
        "MAE",
        "RMSE",
        "R2",
        "Train fit time",
        "Test pred time",
        "Best params"
    ],
    test_name="MIMIC_4_YEAR",
)

results_m4 = Parallel(n_jobs=NJOBS, verbose=0)(
    delayed(train_evaluate)(model_name, model, X, y, mimic_4, stratify_on=None)
    for model_name, model in models.items()
)

results_m4_df = pd.DataFrame(results_m4)
results_m4_df.to_csv(directory + "m4.csv")


print("M4 EVALUATION OVER")

