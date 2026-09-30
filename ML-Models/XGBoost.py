import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
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

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "drs_evaluation_results.csv"
)


# ============================================================
# 2. START
# ============================================================

print("=" * 60)
print("DRUG SYNERGY PREDICTION")
print("DRS XGBOOST MODEL EVALUATION")
print("=" * 60)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading DRS ML-ready dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nDRS dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 4. TARGET COLUMNS
# ============================================================

target_columns = [
    "ZIP",
    "Bliss",
    "Loewe",
    "HSA"
]

print("\nTarget columns:")
print(target_columns)


# ============================================================
# 5. DATA CHECK
# ============================================================

print("\n" + "=" * 60)
print("DATA CHECK")
print("=" * 60)

missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()

print("Missing values:", missing_values)
print("Duplicate rows:", duplicate_rows)

if missing_values > 0:

    print("\nRemoving rows containing missing values...")

    df = df.dropna()

if duplicate_rows > 0:

    print("\nRemoving duplicate rows...")

    df = df.drop_duplicates()

print("\nFinal dataset shape:", df.shape)


# ============================================================
# 6. IDENTIFY DRS FEATURES
# ============================================================

print("\n" + "=" * 60)
print("DRS FEATURE INFORMATION")
print("=" * 60)


drs_feature_columns = [
    column
    for column in df.columns
    if column.startswith("Drug1_DRS_PC")
    or column.startswith("Drug2_DRS_PC")
]


print(
    "Number of DRS features:",
    len(drs_feature_columns)
)

print("\nDRS features:")

for column in drs_feature_columns:
    print(column)


if len(drs_feature_columns) == 0:

    raise ValueError(
        "\nNo DRS feature columns were found."
    )


# ============================================================
# 7. FEATURES AND TARGETS
# ============================================================

X = df[drs_feature_columns]

y = df[target_columns]


print("\nNumber of samples:", X.shape[0])
print("Number of DRS features:", X.shape[1])


# ============================================================
# 8. LOAD SAVED FEATURE COLUMNS
# ============================================================

print("\n" + "=" * 60)
print("LOADING DRS FEATURE COLUMNS")
print("=" * 60)


feature_columns_path = os.path.join(
    MODEL_DIR,
    "drs_feature_columns.pkl"
)


if not os.path.exists(feature_columns_path):

    raise FileNotFoundError(
        f"\nDRS feature columns file not found:\n"
        f"{feature_columns_path}"
    )


saved_feature_columns = joblib.load(
    feature_columns_path
)


print(
    "Saved DRS feature columns loaded."
)

print(
    "Number of saved features:",
    len(saved_feature_columns)
)


# ============================================================
# 9. CHECK FEATURE ORDER
# ============================================================

print("\n" + "=" * 60)
print("CHECKING FEATURE ORDER")
print("=" * 60)


missing_features = [
    column
    for column in saved_feature_columns
    if column not in X.columns
]


if len(missing_features) > 0:

    print("\nERROR: Missing DRS features:")

    for column in missing_features:
        print("-", column)

    raise ValueError(
        "Dataset does not contain all saved DRS features."
    )


X = X[
    saved_feature_columns
]


print(
    "Feature order is correct."
)


# ============================================================
# 10. TRAIN-TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("TRAIN-TEST SPLIT")
print("=" * 60)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(
    "Training samples:",
    X_train.shape[0]
)

print(
    "Testing samples:",
    X_test.shape[0]
)


# ============================================================
# 11. DRS XGBOOST MODEL FILES
# ============================================================

model_files = {

    "ZIP":
        "drs_zip_best_model.pkl",

    "Bliss":
        "drs_bliss_best_model.pkl",

    "Loewe":
        "drs_loewe_best_model.pkl",

    "HSA":
        "drs_hsa_best_model.pkl"
}


# ============================================================
# 12. MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("DRS XGBOOST MODEL EVALUATION")
print("=" * 60)


evaluation_results = []


for target in target_columns:

    print("\n")
    print("=" * 60)
    print(f"EVALUATING DRS {target}")
    print("=" * 60)


    # --------------------------------------------------------
    # MODEL PATH
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        model_files[target]
    )


    if not os.path.exists(model_path):

        print(
            "\nERROR: Model not found:"
        )

        print(model_path)

        raise FileNotFoundError(
            model_path
        )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading DRS XGBoost model...")

    model = joblib.load(
        model_path
    )

    print(
        "Model loaded successfully."
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    print(
        "Generating predictions..."
    )

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # ACTUAL VALUES
    # --------------------------------------------------------

    actual_values = y_test[
        target
    ].values


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
    # R2
    # --------------------------------------------------------

    r2 = r2_score(
        actual_values,
        predictions
    )


    # --------------------------------------------------------
    # ACCURACY WITHIN ±5 UNITS
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # This is NOT ±5%.
    #
    # Example:
    #
    # Actual = 10
    # Predicted = 14
    #
    # Absolute error = 4
    #
    # Since 4 <= 5,
    # prediction is considered correct.
    #
    # --------------------------------------------------------

    absolute_error = np.abs(
        actual_values - predictions
    )

    correct_predictions = (
        absolute_error <= 5
    )

    accuracy_within_5 = (
        np.mean(
            correct_predictions
        ) * 100
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    evaluation_results.append({

        "Model":
            "DRS XGBoost",

        "Target":
            target,

        "MAE":
            mae,

        "MSE":
            mse,

        "RMSE":
            rmse,

        "R2":
            r2,

        "Accuracy_Within_5":
            accuracy_within_5
    })


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n" + "-" * 50)
    print(f"{target} RESULTS")
    print("-" * 50)

    print(
        f"MAE                 : {mae:.4f}"
    )

    print(
        f"MSE                 : {mse:.4f}"
    )

    print(
        f"RMSE                : {rmse:.4f}"
    )

    print(
        f"R²                  : {r2:.4f}"
    )

    print(
        f"Accuracy within ±5  : "
        f"{accuracy_within_5:.2f}%"
    )


# ============================================================
# 13. RESULTS DATAFRAME
# ============================================================

evaluation_results_df = pd.DataFrame(
    evaluation_results
)


# ============================================================
# 14. FINAL RESULTS
# ============================================================

print("\n\n" + "=" * 80)
print("FINAL DRS XGBOOST EVALUATION RESULTS")
print("=" * 80)

print(
    evaluation_results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 15. OVERALL AVERAGES
# ============================================================

average_r2 = evaluation_results_df[
    "R2"
].mean()

average_mae = evaluation_results_df[
    "MAE"
].mean()

average_mse = evaluation_results_df[
    "MSE"
].mean()

average_rmse = evaluation_results_df[
    "RMSE"
].mean()

average_accuracy = evaluation_results_df[
    "Accuracy_Within_5"
].mean()


# ============================================================
# 16. OVERALL SUMMARY
# ============================================================

print("\n\n" + "=" * 80)
print("OVERALL DRS XGBOOST PERFORMANCE")
print("=" * 80)

print(
    f"Average R²                 : "
    f"{average_r2:.4f}"
)

print(
    f"Average MAE                : "
    f"{average_mae:.4f}"
)

print(
    f"Average MSE                : "
    f"{average_mse:.4f}"
)

print(
    f"Average RMSE               : "
    f"{average_rmse:.4f}"
)

print(
    f"Average Accuracy within ±5 : "
    f"{average_accuracy:.2f}%"
)


# ============================================================
# 17. SAVE EVALUATION RESULTS
# ============================================================

evaluation_results_df.to_csv(
    RESULTS_PATH,
    index=False
)


print("\nEvaluation results saved:")
print(RESULTS_PATH)


# ============================================================
# 18. FINAL COMPLETION
# ============================================================

print("\n" + "=" * 80)
print("DRS XGBOOST MODEL EVALUATION COMPLETED")
print("=" * 80)

print("\nModels evaluated:")

for target in target_columns:

    print(
        f"- {model_files[target]}"
    )

print("\nMetrics calculated:")

print("EVALUATION COMPLETE")
