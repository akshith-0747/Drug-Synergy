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
    "ml_ready_dataset_clean.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "..",
    "backend",
    "models"
)


# ============================================================
# 2. START
# ============================================================

print("============================================")
print("DRUG SYNERGY PREDICTION")
print("XGBOOST MODEL EVALUATION")
print("============================================")


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\nLoading cleaned ML-ready dataset...")

df = pd.read_csv(
    DATA_PATH
)

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
        "\nRemoving rows containing "
        "missing values..."
    )

    df = df.dropna()


print(
    "Final dataset shape:",
    df.shape
)


# ============================================================
# 6. FEATURES AND TARGETS
# ============================================================

X = df.drop(
    columns=target_columns
)

y = df[
    target_columns
]


print("\n============================================")
print("FEATURE INFORMATION")
print("============================================")

print(
    "Number of samples:",
    X.shape[0]
)

print(
    "Number of features:",
    X.shape[1]
)


# ============================================================
# 7. LOAD FEATURE COLUMNS
# ============================================================

print("\n============================================")
print("LOADING FEATURE COLUMNS")
print("============================================")

feature_columns_path = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)


if not os.path.exists(
    feature_columns_path
):

    print(
        "\nERROR: feature_columns.pkl not found."
    )

    print(
        feature_columns_path
    )

    raise FileNotFoundError(
        feature_columns_path
    )


feature_columns = joblib.load(
    feature_columns_path
)


print(
    "Saved feature columns loaded."
)

print(
    "Number of saved features:",
    len(feature_columns)
)


# ============================================================
# 8. CHECK FEATURE ORDER
# ============================================================

print("\n============================================")
print("CHECKING FEATURE ORDER")
print("============================================")


if list(X.columns) != list(
    feature_columns
):

    print(
        "Feature order does not match."
    )

    print(
        "Reordering features..."
    )

    X = X[
        feature_columns
    ]

else:

    print(
        "Feature order is correct."
    )


# ============================================================
# 9. TRAIN-TEST SPLIT
# ============================================================

print("\n============================================")
print("TRAIN-TEST SPLIT")
print("============================================")


X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
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
# 10. XGBOOST MODEL FILES
# ============================================================

model_files = {

    "ZIP":
        "ZIP_best_model.pkl",

    "Bliss":
        "Bliss_best_model.pkl",

    "Loewe":
        "Loewe_best_model.pkl",

    "HSA":
        "HSA_best_model.pkl"
}


# ============================================================
# 11. MODEL EVALUATION
# ============================================================

print("\n============================================")
print("XGBOOST MODEL EVALUATION")
print("============================================")


evaluation_results = []


for target in target_columns:

    print("\n")
    print("============================================")
    print(f"EVALUATING {target}")
    print("============================================")


    # --------------------------------------------------------
    # MODEL PATH
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        model_files[target]
    )


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not os.path.exists(
        model_path
    ):

        print(
            "\nERROR: Model not found:"
        )

        print(
            model_path
        )

        continue


    # --------------------------------------------------------
    # LOAD XGBOOST MODEL
    # --------------------------------------------------------

    print(
        "Loading XGBoost model..."
    )


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
    # MAE
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test[target],
        predictions
    )


    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    mse = mean_squared_error(
        y_test[target],
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
        y_test[target],
        predictions
    )


    # --------------------------------------------------------
    # ACCURACY WITHIN ±5 UNITS
    # --------------------------------------------------------

    actual_values = (
        y_test[target].values
    )


    absolute_error = np.abs(
        actual_values - predictions
    )


    correct_predictions = (
        absolute_error <= 5
    )


    accuracy = (
        np.mean(
            correct_predictions
        ) * 100
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    evaluation_results.append({

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
            accuracy
    })


    # ========================================================
    # PRINT TARGET RESULTS
    # ========================================================

    print("\n--------------------------------------------")
    print(f"{target} RESULTS")
    print("--------------------------------------------")

    print(
        f"MAE             : {mae:.4f}"
    )

    print(
        f"MSE             : {mse:.4f}"
    )

    print(
        f"RMSE            : {rmse:.4f}"
    )

    print(
        f"R²              : {r2:.4f}"
    )

    print(
        f"Accuracy ±5     : {accuracy:.2f}%"
    )


# ============================================================
# 12. FINAL RESULTS
# ============================================================

print("\n")
print("============================================================")
print("FINAL XGBOOST EVALUATION RESULTS")
print("============================================================")


evaluation_results_df = pd.DataFrame(
    evaluation_results
)


if not evaluation_results_df.empty:

    print(
        evaluation_results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

else:

    print(
        "No model results available."
    )


# ============================================================
# 13. DETAILED FINAL SUMMARY
# ============================================================

print("\n")
print("============================================================")
print("FINAL SUMMARY")
print("============================================================")


for _, row in evaluation_results_df.iterrows():

    print(
        f"\n{row['Target']}"
    )

    print(
        "--------------------------------------------"
    )

    print(
        f"R²          : "
        f"{row['R2']:.4f}"
    )

    print(
        f"MAE         : "
        f"{row['MAE']:.4f}"
    )

    print(
        f"MSE         : "
        f"{row['MSE']:.4f}"
    )

    print(
        f"RMSE        : "
        f"{row['RMSE']:.4f}"
    )

    print(
        f"Accuracy ±5 : "
        f"{row['Accuracy_Within_5']:.2f}%"
    )


# ============================================================
# 14. COMPLETION
# ============================================================

print("\n")
print("============================================================")
print("XGBOOST MODEL EVALUATION COMPLETED")
print("============================================================")

print("\nModels evaluated:")

for target in target_columns:

    print(
        f"- {target}_best_model.pkl"
    )


print("\nMetrics calculated:")

print("- R²")
print("- MAE")
print("- MSE")
print("- RMSE")
print("- Accuracy within ±5 units")


print("\nNo CSV file was created.")

print("No graphs were created.")

print("No graph folder was created.")

print("\n============================================================")
print("EVALUATION COMPLETE")
print("============================================================")
