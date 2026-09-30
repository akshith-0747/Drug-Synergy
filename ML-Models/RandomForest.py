
# ============================================================
# RANDOM FOREST - DRUG SYNERGY PREDICTION
# DRS-BASED MACHINE LEARNING MODEL
# ============================================================

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    KFold,
    cross_val_score
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "drs_ml_ready_dataset.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "..",
    "backend",
    "models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 2. SETTINGS
# ============================================================

TARGET_COLUMNS = [
    "ZIP",
    "Bliss",
    "Loewe",
    "HSA"
]

ACCURACY_TOLERANCE = 5

RANDOM_STATE = 42

TEST_SIZE = 0.20

N_SPLITS = 5


# ============================================================
# 3. RANDOM FOREST PARAMETER SEARCH
# ============================================================

PARAMETER_SETS = [

    {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },

    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },

    {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },

    {
        "n_estimators": 300,
        "max_depth": 30,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },

    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },

    {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt"
    },

    {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    },

    {
        "n_estimators": 500,
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt"
    }
]


# ============================================================
# 4. HEADER
# ============================================================

print("\n============================================")
print("DRUG SYNERGY PREDICTION")
print("RANDOM FOREST MODEL")
print("5-FOLD CROSS-VALIDATION")
print("============================================")


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\nLoading DRS ML-ready dataset...")

df = pd.read_csv(
    DATA_PATH
)

print("Dataset loaded successfully!")

print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# 6. TARGET COLUMNS
# ============================================================

print("\nTarget columns:")

print(
    TARGET_COLUMNS
)


# ============================================================
# 7. DATA CHECK
# ============================================================

print("\n============================================")
print("DATA CHECK")
print("============================================")


missing_values = (
    df.isnull().sum().sum()
)

duplicate_rows = (
    df.duplicated().sum()
)


print(
    "Missing values:",
    missing_values
)

print(
    "Duplicate rows:",
    duplicate_rows
)


if missing_values > 0:

    print(
        "\nRemoving rows containing missing values..."
    )

    df = df.dropna()


print(
    "\nFinal dataset shape:",
    df.shape
)


# ============================================================
# 8. SELECT DRS FEATURES
# ============================================================

print("\n============================================")
print("DRS FEATURE INFORMATION")
print("============================================")


drug1_features = [

    column

    for column in df.columns

    if column.startswith(
        "Drug1_DRS_PC"
    )

]


drug2_features = [

    column

    for column in df.columns

    if column.startswith(
        "Drug2_DRS_PC"
    )

]


feature_columns = (

    drug1_features

    +

    drug2_features

)


if len(feature_columns) == 0:

    raise ValueError(
        "No DRS feature columns were found."
    )


print(
    "Drug 1 DRS features:",
    len(drug1_features)
)

print(
    "Drug 2 DRS features:",
    len(drug2_features)
)

print(
    "Total DRS features:",
    len(feature_columns)
)


print("\nFeatures:")

print(
    feature_columns
)


# ============================================================
# 9. FEATURES AND TARGETS
# ============================================================

X = df[
    feature_columns
]

y = df[
    TARGET_COLUMNS
]


print(
    "\nNumber of samples:",
    X.shape[0]
)

print(
    "Number of features:",
    X.shape[1]
)


# ============================================================
# 10. TRAIN-TEST SPLIT
# ============================================================

print("\n============================================")
print("TRAIN-TEST SPLIT")
print("============================================")


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE

)


print(
    "Training samples:",
    X_train.shape[0]
)

print(
    "Testing samples :",
    X_test.shape[0]
)


# ============================================================
# 11. 5-FOLD CROSS-VALIDATION
# ============================================================

cv = KFold(

    n_splits=N_SPLITS,

    shuffle=True,

    random_state=RANDOM_STATE

)


# ============================================================
# 12. RANDOM FOREST MODEL SELECTION
# ============================================================

print("\n============================================")
print("RANDOM FOREST MODEL SELECTION")
print("============================================")


print(
    "Testing",
    len(PARAMETER_SETS),
    "parameter configurations."
)


cv_results = []

best_models = {}


for target in TARGET_COLUMNS:

    print("\n============================================")

    print(
        "TARGET:",
        target
    )

    print("============================================")


    best_cv_r2 = -np.inf

    best_parameters = None

    best_model = None


    target_train = y_train[
        target
    ]


    for config_number, params in enumerate(

        PARAMETER_SETS,

        start=1

    ):


        print("\n--------------------------------------------")

        print(
            "Configuration:",
            config_number
        )

        print("--------------------------------------------")


        print(
            "Parameters:",
            params
        )


        model = RandomForestRegressor(

            n_estimators=params[
                "n_estimators"
            ],

            max_depth=params[
                "max_depth"
            ],

            min_samples_split=params[
                "min_samples_split"
            ],

            min_samples_leaf=params[
                "min_samples_leaf"
            ],

            max_features=params[
                "max_features"
            ],

            random_state=RANDOM_STATE,

            n_jobs=-1

        )


        cv_scores = cross_val_score(

            model,

            X_train,

            target_train,

            cv=cv,

            scoring="r2",

            n_jobs=-1

        )


        mean_cv_r2 = np.mean(
            cv_scores
        )


        std_cv_r2 = np.std(
            cv_scores
        )


        print(
            "CV R² scores:",
            np.round(
                cv_scores,
                4
            )
        )


        print(
            f"Mean CV R²: "
            f"{mean_cv_r2:.4f}"
        )


        print(
            f"CV R² Std : "
            f"{std_cv_r2:.4f}"
        )


        cv_results.append({

            "Target":
                target,

            "Configuration":
                config_number,

            "N_Estimators":
                params[
                    "n_estimators"
                ],

            "Max_Depth":
                params[
                    "max_depth"
                ],

            "Min_Samples_Split":
                params[
                    "min_samples_split"
                ],

            "Min_Samples_Leaf":
                params[
                    "min_samples_leaf"
                ],

            "Max_Features":
                params[
                    "max_features"
                ],

            "Mean_CV_R2":
                mean_cv_r2,

            "CV_R2_STD":
                std_cv_r2

        })


        if mean_cv_r2 > best_cv_r2:

            best_cv_r2 = (
                mean_cv_r2
            )

            best_parameters = params

            best_model = model


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    print("\n============================================")

    print(
        "BEST RANDOM FOREST CONFIGURATION"
    )

    print("============================================")


    print(
        "Target:",
        target
    )


    print(
        "Best parameters:",
        best_parameters
    )


    print(
        f"Best Mean CV R²: "
        f"{best_cv_r2:.4f}"
    )


    best_models[target] = {

        "model":
            best_model,

        "parameters":
            best_parameters,

        "mean_cv_r2":
            best_cv_r2

    }


# ============================================================
# 13. TRAIN FINAL RANDOM FOREST MODELS
# ============================================================

print("\n============================================")
print("TRAINING FINAL RANDOM FOREST MODELS")
print("============================================")


final_models = {}


for target in TARGET_COLUMNS:

    print(
        "\nTraining final model for",
        target,
        "..."
    )


    model = best_models[target][
        "model"
    ]


    model.fit(

        X_train,

        y_train[target]

    )


    final_models[target] = model


    print(
        target,
        "completed."
    )


# ============================================================
# 14. FINAL TEST SET EVALUATION
# ============================================================

print("\n============================================")
print("FINAL TEST SET EVALUATION")
print("============================================")


results = []


all_predictions = {}


for target in TARGET_COLUMNS:

    model = final_models[
        target
    ]


    predictions = model.predict(
        X_test
    )


    actual_values = (
        y_test[target].values
    )


    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = mean_absolute_error(

        actual_values,

        predictions

    )


    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    mse = mean_squared_error(

        actual_values,

        predictions

    )


    # --------------------------------------------------------
    # RMSE
    # --------------------------------------------------------

    rmse = np.sqrt(
        mse
    )


    # --------------------------------------------------------
    # R²
    # --------------------------------------------------------

    r2 = r2_score(

        actual_values,

        predictions

    )


    # --------------------------------------------------------
    # ACCURACY WITHIN ±5
    # --------------------------------------------------------

    absolute_errors = np.abs(

        actual_values -

        predictions

    )


    correct_predictions = (

        absolute_errors

        <=

        ACCURACY_TOLERANCE

    )


    accuracy = (

        np.mean(
            correct_predictions
        )

        *

        100

    )


    # --------------------------------------------------------
    # STORE PREDICTIONS
    # --------------------------------------------------------

    all_predictions[
        target
    ] = predictions


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    results.append({

        "Target":
            target,

        "Mean_CV_R2":
            best_models[target][
                "mean_cv_r2"
            ],

        "N_Estimators":
            best_models[target][
                "parameters"
            ]["n_estimators"],

        "Max_Depth":
            best_models[target][
                "parameters"
            ]["max_depth"],

        "Min_Samples_Split":
            best_models[target][
                "parameters"
            ]["min_samples_split"],

        "Min_Samples_Leaf":
            best_models[target][
                "parameters"
            ]["min_samples_leaf"],

        "MAE":
            mae,

        "MSE":
            mse,

        "RMSE":
            rmse,

        "R2":
            r2,

        "Accuracy_Within_5":
            accuracy

    })


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n--------------------------------------------")

    print(
        target
    )

    print("--------------------------------------------")


    print(
        f"Mean CV R² : "
        f"{best_models[target]['mean_cv_r2']:.4f}"
    )


    print(
        "Best parameters:",
        best_models[target][
            "parameters"
        ]
    )


    print(
        f"Test MAE   : "
        f"{mae:.4f}"
    )


    print(
        f"Test MSE   : "
        f"{mse:.4f}"
    )


    print(
        f"Test RMSE  : "
        f"{rmse:.4f}"
    )


    print(
        f"Test R²    : "
        f"{r2:.4f}"
    )


    print(
        f"Accuracy within ±5 : "
        f"{accuracy:.2f}%"
    )


# ============================================================
# 15. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 16. FINAL RESULTS TABLE
# ============================================================

print("\n============================================")
print("FINAL RANDOM FOREST EVALUATION RESULTS")
print("============================================")


display_df = results_df.copy()


display_df[
    "Mean_CV_R2"
] = display_df[
    "Mean_CV_R2"
].round(4)


display_df[
    "MAE"
] = display_df[
    "MAE"
].round(4)


display_df[
    "MSE"
] = display_df[
    "MSE"
].round(4)


display_df[
    "RMSE"
] = display_df[
    "RMSE"
].round(4)


display_df[
    "R2"
] = display_df[
    "R2"
].round(4)


display_df[
    "Accuracy_Within_5"
] = display_df[
    "Accuracy_Within_5"
].round(2)


print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# 17. OVERALL ACCURACY
# ============================================================

print("\n============================================")
print("OVERALL RANDOM FOREST PERFORMANCE")
print("============================================")


overall_accuracy = (

    results_df[
        "Accuracy_Within_5"
    ]

    .mean()

)


overall_r2 = (

    results_df[
        "R2"
    ]

    .mean()

)


overall_mae = (

    results_df[
        "MAE"
    ]

    .mean()

)


overall_rmse = (

    results_df[
        "RMSE"
    ]

    .mean()

)


print(
    f"Overall Accuracy within ±5 : "
    f"{overall_accuracy:.2f}%"
)


print(
    f"Overall Mean R²             : "
    f"{overall_r2:.4f}"
)


print(
    f"Overall Mean MAE            : "
    f"{overall_mae:.4f}"
)


print(
    f"Overall Mean RMSE           : "
    f"{overall_rmse:.4f}"
)


# ============================================================
# 18. SAVE EVALUATION RESULTS
# ============================================================

results_path = os.path.join(

    MODEL_DIR,

    "randomforest_evaluation_results.csv"

)


results_df.to_csv(

    results_path,

    index=False

)


print(
    "\nEvaluation results saved:"
)

print(
    results_path
)


# ============================================================
# 19. SAVE CROSS-VALIDATION RESULTS
# ============================================================

cv_results_df = pd.DataFrame(
    cv_results
)


cv_results_path = os.path.join(

    MODEL_DIR,

    "randomforest_cv_results.csv"

)


cv_results_df.to_csv(

    cv_results_path,

    index=False

)


print(
    "\nCross-validation results saved:"
)

print(
    cv_results_path
)


# ============================================================
# 20. SAVE RANDOM FOREST MODELS
# ============================================================

print("\n============================================")
print("SAVING RANDOM FOREST MODELS")
print("============================================")


for target in TARGET_COLUMNS:

    model_path = os.path.join(

        MODEL_DIR,

        f"{target}_randomforest_model.pkl"

    )


    joblib.dump(

        final_models[target],

        model_path

    )


    print(
        f"{target} model saved:"
    )

    print(
        model_path
    )


# ============================================================
# 21. SAVE BEST PARAMETERS
# ============================================================

best_parameters = []


for target in TARGET_COLUMNS:

    params = best_models[target][
        "parameters"
    ]


    best_parameters.append({

        "Target":
            target,

        "N_Estimators":
            params[
                "n_estimators"
            ],

        "Max_Depth":
            params[
                "max_depth"
            ],

        "Min_Samples_Split":
            params[
                "min_samples_split"
            ],

        "Min_Samples_Leaf":
            params[
                "min_samples_leaf"
            ],

        "Max_Features":
            params[
                "max_features"
            ],

        "Mean_CV_R2":
            best_models[target][
                "mean_cv_r2"
            ]

    })


best_parameters_df = pd.DataFrame(

    best_parameters

)


parameters_path = os.path.join(

    MODEL_DIR,

    "randomforest_best_parameters.csv"

)


best_parameters_df.to_csv(

    parameters_path,

    index=False

)


print(
    "\nBest parameters saved:"
)

print(
    parameters_path
)


# ============================================================
# 22. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = y_test.copy()


for target in TARGET_COLUMNS:

    prediction_df[
        f"{target}_Predicted"
    ] = all_predictions[
        target
    ]


prediction_path = os.path.join(

    MODEL_DIR,

    "randomforest_test_predictions.csv"

)


prediction_df.to_csv(

    prediction_path,

    index=False

)


print(
    "\nTest predictions saved:"
)

print(
    prediction_path
)


# ============================================================
# 23. FINAL SUMMARY
# ============================================================

print("\n============================================")
print("FINAL RANDOM FOREST MODEL SUMMARY")
print("============================================")


for _, row in results_df.iterrows():

    print(
        "\n",
        row["Target"]
    )

    print(
        "--------------------------------------------"
    )


    print(
        f"Mean CV R² : "
        f"{row['Mean_CV_R2']:.4f}"
    )


    print(
        f"Test MAE   : "
        f"{row['MAE']:.4f}"
    )


    print(
        f"Test MSE   : "
        f"{row['MSE']:.4f}"
    )


    print(
        f"Test RMSE  : "
        f"{row['RMSE']:.4f}"
    )


    print(
        f"Test R²    : "
        f"{row['R2']:.4f}"
    )


    print(
        f"Accuracy ±5 : "
        f"{row['Accuracy_Within_5']:.2f}%"
    )


print("\n--------------------------------------------")


print(
    f"OVERALL ACCURACY ±5 : "
    f"{overall_accuracy:.2f}%"
)


print(
    f"OVERALL MEAN R²     : "
    f"{overall_r2:.4f}"
)


print(
    f"OVERALL MEAN MAE    : "
    f"{overall_mae:.4f}"
)


print(
    f"OVERALL MEAN RMSE   : "
    f"{overall_rmse:.4f}"
)


print("--------------------------------------------")


# ============================================================
# 24. COMPLETION
# ============================================================

print("\n============================================")
print("RANDOM FOREST TRAINING COMPLETED")
print("============================================")


print("\nOutputs created:")

print(
    "- ZIP_randomforest_model.pkl"
)

print(
    "- Bliss_randomforest_model.pkl"
)

print(
    "- Loewe_randomforest_model.pkl"
)

print(
    "- HSA_randomforest_model.pkl"
)

print(
    "- randomforest_evaluation_results.csv"
)

print(
    "- randomforest_cv_results.csv"
)

print(
    "- randomforest_best_parameters.csv"
)

print(
    "- randomforest_test_predictions.csv"
)


print("\nMetrics calculated:")

print("- Mean CV R²")

print("- MAE")

print("- MSE")

print("- RMSE")

print("- Test R²")

print("- Accuracy within ±5")

print("- Overall Accuracy")


print("\n============================================")
print("DONE")
print("============================================")
