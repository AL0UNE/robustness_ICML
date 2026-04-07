import os

import pandas as pd


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
    "pao2fio2_vent_min",
    "bilirubin_min",
    "bilirubin_max",
]

cat_features = ["aids", "hem", "mets", "admissiontype"]
features = cont_features + cat_features
outcome = ["hospital_mortality"]

required_columns = list(
    dict.fromkeys(
        features
        + [
            "hospital_mortality",
            "death_icu",
            "death_overall",
            "gender",
            "age_group",
            "ICU_unit",
        ]
    )
)

M3_CSV_PATH = os.getenv("M3_CSV_PATH", "m3.csv")

mimic_3 = pd.read_csv(M3_CSV_PATH, usecols=required_columns)

X = mimic_3[features]
y = mimic_3["hospital_mortality"]

y_proxy_death_icu = mimic_3["death_icu"]
y_proxy_death_overall = mimic_3["death_overall"]

column_m3 = [
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
]

column_m4 = [
    "Model",
    "Variable",
    "Category",
    "AUC",
    "Brier score",
    "Intercept",
    "Slope",
    "Prob true",
    "Prob pred",
    "Train fit time",
    "Test pred time",
    "Best params",
]

external_stratifications = [
    ("gender", "eICU_GENDER"),
    ("age_group", "MIMIC_eICU_AGE"),
    ("ICU_unit", "eICU_ICU_UNIT"),
    ("region", "eICU_region"),
]

