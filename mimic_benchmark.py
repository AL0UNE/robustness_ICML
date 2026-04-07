# -*- coding: utf-8 -*-


import os
import warnings
from datetime import datetime
from functools import partial

import numpy as np
import pandas as pd
import torch
from joblib import Memory

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer #Required to use IterativeImputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from pytabkit import RealMLP_TD_Classifier, FTT_D_Classifier
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier


from helper import (
    create_directory,
    save_results,
)
from m3_handling import (
    cont_features,
    cat_features,
    features,
    X,
    y,
    column_m3,
    column_m4,
    external_stratifications,
)
from model_helpers import (
    tune_model,
    calibrate_model,
    compute_calibration_metrics,
    load_preset_best_params_repository,
    validate_preset_params_coverage,
)



from hpo_grid import PARAM_GRIDS
from run_helpers import (
    run_cv_parallel_and_save,
    run_subgroup_parallel_and_save,
    run_temporal_parallel,
    run_temporal_parallel_and_save,
)
from perturbation_function import (
    evaluate_standard_bundle,
    evaluate_subgroup_bundle,
    evaluate_temporal_bundle,
    label_noise,
    input_noise,
    imbalance_data,
    training_data_regime,
    permutation_features,
    missing_data,
    subgroup_analysis,
    train_evaluate,
)

#from tabpfn import TabPFNClassifier
#from tabicl import TabICLClassifier


# If no CUDA GPU is available, allow CPU runs (slower) but do not abort.
if not torch.cuda.is_available():
    warnings.warn(
        "GPU device not found. The script will run on CPU (may be much slower)."
    )

#warnings.filterwarnings("ignore") 

#Datetime
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
now = datetime.now()
timestamp = now.strftime('%Y%m%d_%H%M%S')


DIR_NAME_DEFAULT = f"res_{timestamp}"
OUTPUT_BASE_DIR = os.getenv("BENCHMARK_OUTPUT_BASE_DIR", "")
DIR_NAME = os.getenv("BENCHMARK_RESULTS_DIR_NAME", DIR_NAME_DEFAULT)
directory_name = os.path.join(OUTPUT_BASE_DIR, DIR_NAME) if OUTPUT_BASE_DIR else DIR_NAME
create_directory(directory_name)


LOG_FILE = os.path.join(directory_name, f"boosting_failures_{timestamp}.log")
LOG_FILE_CALIB = os.path.join(directory_name, f"calibration_failures_{timestamp}.log")


NJOBS = -1
NJOBS_TRAIN = NJOBS
NJOBS_PREDICT = 1
NJOBS_GS = 1
N_TRAINING_SAMPLE = -1 ## No need to change it since we do not have to run the xp for 10k
SPLIT_TRAIN_PREDICT = True

HPO = False 
N_SEARCH_GS = 15

USE_PRESET_BEST_PARAMS = True 
STRICT_PRESET_BEST_PARAMS = True 
PRESET_RESULTS_PATH = os.getenv("PRESET_RESULTS_PATH", os.path.join(ROOT_DIR, "results_hpo_cpu"))
PRESET_TOP_K = 5


print(f'N_JOBS: {NJOBS}')   
print(f'N_JOBS_TRAIN: {NJOBS_TRAIN}')
print(f'N_JOBS_PREDICT: {NJOBS_PREDICT}')
print(f'N_TRAINING_SAMPLE: {N_TRAINING_SAMPLE}')
print(f'USE_PRESET_BEST_PARAMS: {USE_PRESET_BEST_PARAMS}')
if USE_PRESET_BEST_PARAMS:
    print(f'PRESET_RESULTS_PATH: {PRESET_RESULTS_PATH}')
    print(f'PRESET_TOP_K: {PRESET_TOP_K}')
print("Running models with hyperparameter optimization") if HPO else print("Running models without hyperparameter optimization")

RANDOM_STATE = 42

CACHE_DIR = os.getenv("BENCHMARK_CACHE_DIR", os.path.join("results", "cache"))
os.makedirs(CACHE_DIR, exist_ok=True)
memory = Memory(CACHE_DIR, verbose=0)
param_grids = PARAM_GRIDS if HPO else {}

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
splits = list(kf.split(X))




# MODELS

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
    "Logistic": Pipeline([("preprocessor", preprocessor_1), ("lr", LogisticRegression(C=1e12))], memory=memory),
    #"LASSO": Pipeline([("preprocessor", preprocessor_1),("lr", LogisticRegression(l1_ratio=1, solver="liblinear")),], memory=memory),
    #"Ridge": Pipeline([("preprocessor", preprocessor_1),("lr", LogisticRegression(l1_ratio=0)),], memory=memory),

    # Tree-based methods
    #"Random Forest": RandomForestClassifier(n_jobs=1),
    ## Boosting    
    #"Gradient Boosting": Pipeline([("preprocessor", preprocessor_2), ("gb", GradientBoostingClassifier())], memory=memory),
    #"XGBoost": XGBClassifier(tree_method='hist',device='cuda'),
    #"LightGBM": LGBMClassifier(device='gpu',gpu_use_dp=True),
    #"CatBoost": CatBoostClassifier(logging_level='Silent'),
    # Deep learning
    #"MLP": Pipeline([("preprocessor", preprocessor_1), ("mlp", MLPClassifier(max_iter=500))], memory=memory),
    ## Transformers
    #"TabPFN": TabPFNClassifier(),
    #"TabICL": TabICLClassifier(),
    #"RealMLP": Pipeline([("preprocessor", preprocessor_1), ("realMLP", RealMLP_TD_Classifier())], memory=memory),
    #"FTTransformer": Pipeline([("preprocessor", preprocessor_1), ("ftTransformer", FTT_D_Classifier())], memory=memory),
}

models_name = list(models.keys())

n_models = len(models_name)

PRESET_BEST_PARAMS_REPO = (
    load_preset_best_params_repository(
        PRESET_RESULTS_PATH,
        top_k_per_group=PRESET_TOP_K,
        strict=STRICT_PRESET_BEST_PARAMS,
    )
    if USE_PRESET_BEST_PARAMS
    else {}
)

if USE_PRESET_BEST_PARAMS:
    validate_preset_params_coverage(
        models,
        PRESET_BEST_PARAMS_REPO,
        strict=STRICT_PRESET_BEST_PARAMS,
    )

run_cv_parallel_and_save_cfg = partial(
    run_cv_parallel_and_save,
    models=models,
    splits=splits,
    n_jobs=NJOBS,
    n_folds=n_folds,
    save_results_fn=save_results,
    split_train_predict=SPLIT_TRAIN_PREDICT,
    train_n_jobs=NJOBS_TRAIN,
    predict_n_jobs=NJOBS_PREDICT,
    evaluate_bundle_fn=evaluate_standard_bundle,
    evaluate_bundle_kwargs={
        "compute_calibration_metrics_fn": partial(compute_calibration_metrics, log_file=LOG_FILE_CALIB),
    },
)
run_subgroup_parallel_and_save_cfg = partial(
    run_subgroup_parallel_and_save,
    models=models,
    splits=splits,
    n_jobs=NJOBS,
    n_folds=n_folds,
    save_results_fn=save_results,
    split_train_predict=SPLIT_TRAIN_PREDICT,
    train_n_jobs=NJOBS_TRAIN,
    predict_n_jobs=NJOBS_PREDICT,
    evaluate_bundle_fn=evaluate_subgroup_bundle,
    evaluate_bundle_kwargs={
        "compute_calibration_metrics_fn": partial(compute_calibration_metrics, log_file=LOG_FILE_CALIB),
    },
)
run_temporal_parallel_cfg = partial(
    run_temporal_parallel,
    models=models,
    n_jobs=NJOBS,
    split_train_predict=SPLIT_TRAIN_PREDICT,
    train_n_jobs=NJOBS_TRAIN,
    predict_n_jobs=NJOBS_PREDICT,
    evaluate_bundle_fn=evaluate_temporal_bundle,
    evaluate_bundle_kwargs={
        "compute_calibration_metrics_fn": partial(compute_calibration_metrics, log_file=LOG_FILE_CALIB),
    },
)
run_temporal_parallel_and_save_cfg = partial(
    run_temporal_parallel_and_save,
    models=models,
    n_jobs=NJOBS,
    n_folds=n_folds,
    save_results_fn=save_results,
    split_train_predict=SPLIT_TRAIN_PREDICT,
    train_n_jobs=NJOBS_TRAIN,
    predict_n_jobs=NJOBS_PREDICT,
    evaluate_bundle_fn=evaluate_temporal_bundle,
    evaluate_bundle_kwargs={
        "compute_calibration_metrics_fn": partial(compute_calibration_metrics, log_file=LOG_FILE_CALIB),
    },
)

perturbation_task_kwargs = {
    "tune_model_fn": partial(
        tune_model,
        param_grids=param_grids,
        n_search_gs=N_SEARCH_GS,
        njobs_gs=NJOBS_GS,
        random_state_global=RANDOM_STATE,
        log_file=LOG_FILE,
        use_preset_best_params=USE_PRESET_BEST_PARAMS,
        strict_preset_best_params=STRICT_PRESET_BEST_PARAMS,
        preset_best_params_repo=PRESET_BEST_PARAMS_REPO,
    ),
    "calibrate_model_fn": calibrate_model,
    "compute_calibration_metrics_fn": partial(compute_calibration_metrics, log_file=LOG_FILE_CALIB),
}
label_noise_task = partial(label_noise, **perturbation_task_kwargs)
input_noise_task = partial(input_noise, **perturbation_task_kwargs)
imbalance_data_task = partial(imbalance_data, **perturbation_task_kwargs)
training_data_regime_task = partial(training_data_regime, **perturbation_task_kwargs)
permutation_features_task = partial(permutation_features, **perturbation_task_kwargs)
missing_data_task = partial(missing_data, **perturbation_task_kwargs)
subgroup_analysis_task = partial(subgroup_analysis, **perturbation_task_kwargs)
train_evaluate_task = partial(
    train_evaluate,
    tune_model_fn=partial(
        tune_model,
        param_grids=param_grids,
        n_search_gs=N_SEARCH_GS,
        njobs_gs=NJOBS_GS,
        random_state_global=RANDOM_STATE,
        log_file=LOG_FILE,
        use_preset_best_params=USE_PRESET_BEST_PARAMS,
        strict_preset_best_params=STRICT_PRESET_BEST_PARAMS,
        preset_best_params_repo=PRESET_BEST_PARAMS_REPO,
    ),
    compute_calibration_metrics_fn=partial(compute_calibration_metrics, log_file=LOG_FILE_CALIB),
)

## Robustness tests
### Label noise
random_label_noise_levels = np.linspace(0, 1, 11)
targeted_label_noise_levels = np.linspace(0, 1, 10, endpoint=False)
directory = f"{directory_name}/label_noise/"
create_directory(directory)

run_cv_parallel_and_save_cfg(label_noise_task, random_label_noise_levels, directory, "RANDOM_LABEL_NOISE", column_m3, task_kwargs={"noise_type": "random"})
"""
run_cv_parallel_and_save_cfg(label_noise_task, targeted_label_noise_levels, directory, "01_LABEL_NOISE", column_m3, task_kwargs={"noise_type": "0to1"})
run_cv_parallel_and_save_cfg(label_noise_task, targeted_label_noise_levels, directory, "10_LABEL_NOISE", column_m3, task_kwargs={"noise_type": "1to0"})
run_cv_parallel_and_save_cfg(label_noise_task, random_label_noise_levels, directory, "AGE_LABEL_NOISE", column_m3, task_kwargs={"noise_type": "conditional"})
run_cv_parallel_and_save_cfg(label_noise_task, ["hospital_death", "icu_death", "overall_death"], directory, "PROXY_LABEL_NOISE", column_m3, task_kwargs={"noise_type": "proxy"})

print("LABEL NOISE OVER")

### Measurement noise
input_noise_level = np.linspace(0, 1, 11)

directory = f"{directory_name}/input_noise/"
create_directory(directory)

run_cv_parallel_and_save_cfg(input_noise_task, input_noise_level, directory, "INPUT_NOISE_TRAIN", column_m3)
run_cv_parallel_and_save_cfg(input_noise_task, input_noise_level, directory, "INPUT_NOISE_VAL", column_m3, task_kwargs={"which_set": "Val"})
run_cv_parallel_and_save_cfg(input_noise_task, input_noise_level, directory, "INPUT_NOISE_ALL", column_m3, task_kwargs={"which_set": "Train_Val"})
run_cv_parallel_and_save_cfg(input_noise_task, input_noise_level, directory, "INPUT_NOISE_TRAIN_CONT", column_m3, task_kwargs={"feature_type": "cont"})
run_cv_parallel_and_save_cfg(input_noise_task, input_noise_level, directory, "INPUT_NOISE_VAL_CONT", column_m3, task_kwargs={"which_set": "Val", "feature_type": "cont"})
run_cv_parallel_and_save_cfg(input_noise_task, input_noise_level, directory, "INPUT_NOISE_ALL_CONT", column_m3, task_kwargs={"which_set": "Train_Val", "feature_type": "cont"})
run_cv_parallel_and_save_cfg(input_noise_task, input_noise_level, directory, "INPUT_NOISE_TRAIN_CAT", column_m3, task_kwargs={"feature_type": "cat"})
run_cv_parallel_and_save_cfg(input_noise_task, input_noise_level, directory, "INPUT_NOISE_VAL_CAT", column_m3, task_kwargs={"which_set": "Val", "feature_type": "cat"})
run_cv_parallel_and_save_cfg(input_noise_task, input_noise_level, directory, "INPUT_NOISE_ALL_CAT", column_m3, task_kwargs={"which_set": "Train_Val", "feature_type": "cat"})

print("MEASUREMENT NOISE OVER")

### Imbalanced data

imbalance_ratio = np.linspace(1, 0, 10, endpoint=False)

directory = f"{directory_name}/imbalance_data/"
create_directory(directory)

run_cv_parallel_and_save_cfg(imbalance_data_task, imbalance_ratio, directory, "IMBALANCED_DATA", column_m3)

print("IMBALANCE NOISE OVER")

### Training data size

training_data_size = [0.05, 0.1, 0.25, 0.5, 0.8, 1]

directory = f"{directory_name}/training_size/"
create_directory(directory)

run_cv_parallel_and_save_cfg(training_data_regime_task, training_data_size, directory, "TRAINING_SIZE", column_m3)

print("Training data size OVER")

### Feature shuffling

shuffle_ratio = np.linspace(0, 1, 11, endpoint=True)

directory = f"{directory_name}/feature_shuffle/"
create_directory(directory)

run_cv_parallel_and_save_cfg(permutation_features_task, shuffle_ratio, directory, "SHUFFLED_TRAIN_DATA", column_m3, task_kwargs={"which_set": "Train"})
run_cv_parallel_and_save_cfg(permutation_features_task, shuffle_ratio, directory, "SHUFFLED_VAL_DATA", column_m3, task_kwargs={"which_set": "Val"})
run_cv_parallel_and_save_cfg(permutation_features_task, shuffle_ratio, directory, "SHUFFLED_ALL_DATA", column_m3, task_kwargs={"which_set": "Train_Val"})

print("SHUFFLE NOISE OVER")

### Missing data

missing_ratio = np.linspace(0.1, 1, 9, endpoint=False)

directory = f"{directory_name}/missing_data/"
create_directory(directory)

run_cv_parallel_and_save_cfg(missing_data_task, missing_ratio, directory, "MCAR_TRAIN", column_m3, task_kwargs={"which_set": "Train", "mechanism": "MCAR"}, verbose=1)
run_cv_parallel_and_save_cfg(missing_data_task, missing_ratio, directory, "MCAR_VAL", column_m3, task_kwargs={"which_set": "Val", "mechanism": "MCAR"}, verbose=1)
run_cv_parallel_and_save_cfg(missing_data_task, missing_ratio, directory, "MCAR_ALL", column_m3, task_kwargs={"which_set": "Train_Val", "mechanism": "MCAR"}, verbose=1)
run_cv_parallel_and_save_cfg(missing_data_task, missing_ratio, directory, "MAR_TRAIN", column_m3, task_kwargs={"which_set": "Train", "mechanism": "MAR"}, verbose=1)
run_cv_parallel_and_save_cfg(missing_data_task, missing_ratio, directory, "MAR_VAL", column_m3, task_kwargs={"which_set": "Val", "mechanism": "MAR"}, verbose=1)
run_cv_parallel_and_save_cfg(missing_data_task, missing_ratio, directory, "MAR_ALL", column_m3, task_kwargs={"which_set": "Train_Val", "mechanism": "MAR"}, verbose=1)
run_cv_parallel_and_save_cfg(missing_data_task, missing_ratio, directory, "MNAR_TRAIN", column_m3, task_kwargs={"which_set": "Train"}, verbose=1)
run_cv_parallel_and_save_cfg(missing_data_task, missing_ratio, directory, "MNAR_VAL", column_m3, task_kwargs={"which_set": "Val"}, verbose=1)
run_cv_parallel_and_save_cfg(missing_data_task, missing_ratio, directory, "MNAR_ALL", column_m3, task_kwargs={"which_set": "Train_Val"}, verbose=1)

print("MISSING DATA OVER")

### Subgroup analysis
'''
Here we consider different types of subgroup analysis: both within MIMIC-III and across MIMIC-IV. 
<br> For each analysis we consider either stratifying on one of the included feature (e.g., age) in
the model or on a external variable (e.g., gender, icu unit).  
'''

directory = f"{directory_name}/subgroups/"
create_directory(directory)

#### MIMIC-III 

run_subgroup_parallel_and_save_cfg(subgroup_analysis_task, "gender", directory, "SUBGROUP_GENDER", column_m4)
run_subgroup_parallel_and_save_cfg(subgroup_analysis_task, "age_group", directory, "SUBGROUP_AGEGROUP", column_m4)
run_subgroup_parallel_and_save_cfg(subgroup_analysis_task, "ICU_unit", directory, "SUBGROUP_ICU_UNIT", column_m4)

print("SUBGROUP NOISE OVER")

### Temporal validation

mimic_4 = pd.read_csv(os.path.join(ROOT_DIR, "m4.csv"))

directory = f"{directory_name}/m4/"
create_directory(directory)


mimic_4 = mimic_4[mimic_4[features].isna().mean(axis=1)<=0.5] ## excluding patients with more than half of the features missing.
run_temporal_parallel_and_save_cfg(train_evaluate_task, X, y, mimic_4, directory, "MIMIC_4_GENDER", column_m4, stratify_on="gender")
run_temporal_parallel_and_save_cfg(train_evaluate_task, X, y, mimic_4, directory, "MIMIC_4_AGE", column_m4, stratify_on="age_group")
run_temporal_parallel_and_save_cfg(train_evaluate_task, X, y, mimic_4, directory, "MIMIC_4_ICU_UNIT", column_m4, stratify_on="ICU_unit")
run_temporal_parallel_and_save_cfg(train_evaluate_task, X, y, mimic_4, directory, "MIMIC_4_YEAR", column_m4, stratify_on="anchor_year_group")
results_m4 = run_temporal_parallel_cfg(
    train_evaluate_task,
    X,
    y,
    mimic_4,
    stratify_on=None,
    benchmark_test_name="MIMIC_4_OVERALL",
)

results_m4_df = pd.DataFrame(results_m4)
results_m4_df.to_csv(directory + "m4.csv")

print("M4 EVALUATION OVER")

### External validation

directory = f"{directory_name}/eICU/"
create_directory(directory)

eICU = pd.read_csv(os.path.join(ROOT_DIR, "eICU.csv"))

for stratify_on, test_name in external_stratifications:
    run_temporal_parallel_and_save_cfg(train_evaluate_task, X, y, eICU, directory, test_name, column_m4, stratify_on=stratify_on)
results_eICU = run_temporal_parallel_cfg(
    train_evaluate_task,
    X,
    y,
    eICU,
    stratify_on=None,
    benchmark_test_name="eICU",
)
results_eICU_df = pd.DataFrame(results_eICU)
results_eICU_df.to_csv(directory + "eICU.csv")

print("External validation OVER")

#"""