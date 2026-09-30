
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# 1. PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

os.makedirs(MODEL_DIR, exist_ok=True)


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

# LASSO regularization values to test
ALPHAS = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0
]


# ============================================================
# 3. HEADER
# ============================================================

print("\n============================================")
print("DRUG SYNERGY PREDICTION")
print("LASSO MODEL TRAINING")
print("5-FOLD CROSS-VALIDATION")
print("============================================")


# ============================================================
# 4. LOAD DATASET
# ============================================================

print("\nLoading DRS ML-ready dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 5. TARGET COLUMNS
# ============================================================

print("\nTarget columns:")
print(TARGET_COLUMNS)


# ============================================================
# 6. DATA CHECK
# ============================================================

print("\n============================================")
print("DATA CHECK")
print("============================================")

missing_values = df.isnull().sum().sum()
duplicate_rows = df.duplicated().sum()

print("Missing values:", missing_values)
print("Duplicate rows:", duplicate_rows)

if missing_values > 0:
    print("\nRemoving rows containing missing values...")
    df = df.dropna()

print("\nFinal dataset shape:", df.shape)


# ============================================================
# 7. FEATURES AND TARGETS
# ============================================================

X = df.drop(columns=TARGET_COLUMNS)

y = df[TARGET_COLUMNS]


print("\n============================================")
print("FEATURE INFORMATION")
print("============================================")

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

print("\nFeatures:")
print(X.columns.tolist())


# ============================================================
# 8. TRAIN-TEST SPLIT
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

print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])


# ============================================================
# 9. 5-FOLD CROSS-VALIDATION
# ============================================================

cv = KFold(
    n_splits=N_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


# ============================================================
# 10. LASSO MODEL SELECTION
# ============================================================

print("\n============================================")
print("LASSO MODEL SELECTION")
print("============================================")

print("Testing", len(ALPHAS), "alpha values for each target.")


cv_results = []

best_models = {}


for target in TARGET_COLUMNS:

    print("\n============================================")
    print("TARGET:", target)
    print("============================================")

    best_alpha = None
    best_cv_r2 = -np.inf
    best_pipeline = None

    target_train = y_train[target]

    for alpha in ALPHAS:

        print("\nTesting alpha =", alpha)

        pipeline = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "lasso",
                Lasso(
                    alpha=alpha,
                    max_iter=100000,
                    random_state=RANDOM_STATE
                )
            )
        ])

        cv_scores = cross_val_score(
            pipeline,
            X_train,
            target_train,
            cv=cv,
            scoring="r2",
            n_jobs=-1
        )

        mean_cv_r2 = np.mean(cv_scores)

        std_cv_r2 = np.std(cv_scores)

        print(
            "CV R² scores:",
            np.round(cv_scores, 4)
        )

        print(
            f"Mean CV R²: {mean_cv_r2:.4f}"
        )

        print(
            f"CV R² Std : {std_cv_r2:.4f}"
        )

        cv_results.append({
            "Target": target,
            "Alpha": alpha,
            "Mean_CV_R2": mean_cv_r2,
            "CV_R2_STD": std_cv_r2
        })

        if mean_cv_r2 > best_cv_r2:

            best_cv_r2 = mean_cv_r2

            best_alpha = alpha

            best_pipeline = pipeline


    print("\nBEST CONFIGURATION")
    print("--------------------------------------------")

    print("Best alpha:", best_alpha)

    print(
        f"Best Mean CV R²: {best_cv_r2:.4f}"
    )

    best_models[target] = {
        "model": best_pipeline,
        "alpha": best_alpha,
        "mean_cv_r2": best_cv_r2
    }


# ============================================================
# 11. TRAIN FINAL LASSO MODELS
# ============================================================

print("\n============================================")
print("TRAINING FINAL LASSO MODELS")
print("============================================")


final_models = {}

evaluation_results = {}


for target in TARGET_COLUMNS:

    print("\nTraining final model for", target, "...")

    model = best_models[target]["model"]

    model.fit(
        X_train,
        y_train[target]
    )

    final_models[target] = model

    print(target, "completed.")


# ============================================================
# 12. FINAL TEST SET EVALUATION
# ============================================================

print("\n============================================")
print("FINAL TEST SET EVALUATION")
print("============================================")


all_predictions = {}

all_actual_values = {}

all_residuals = []

results = []


for target in TARGET_COLUMNS:

    model = final_models[target]

    predictions = model.predict(X_test)

    actual_values = y_test[target].values


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

    rmse = np.sqrt(mse)


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
        actual_values - predictions
    )

    correct_predictions = (
        absolute_errors <= ACCURACY_TOLERANCE
    )

    accuracy = (
        np.mean(correct_predictions) * 100
    )


    # --------------------------------------------------------
    # RESIDUALS
    # --------------------------------------------------------

    residuals = (
        actual_values - predictions
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    all_predictions[target] = predictions

    all_actual_values[target] = actual_values

    all_residuals.append(residuals)


    results.append({
        "Target": target,
        "Mean_CV_R2":
            best_models[target]["mean_cv_r2"],
        "Alpha":
            best_models[target]["alpha"],
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Accuracy_Within_5": accuracy
    })


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print("\n--------------------------------------------")
    print(target)
    print("--------------------------------------------")

    print(
        f"Mean CV R² : "
        f"{best_models[target]['mean_cv_r2']:.4f}"
    )

    print(
        f"Best Alpha : "
        f"{best_models[target]['alpha']}"
    )

    print(
        f"Test MAE   : {mae:.4f}"
    )

    print(
        f"Test MSE   : {mse:.4f}"
    )

    print(
        f"Test RMSE  : {rmse:.4f}"
    )

    print(
        f"Test R²    : {r2:.4f}"
    )

    print(
        f"Accuracy within ±5 : "
        f"{accuracy:.2f}%"
    )


# ============================================================
# 13. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(results)


# ============================================================
# 14. FINAL RESULTS TABLE
# ============================================================

print("\n============================================")
print("FINAL LASSO EVALUATION RESULTS")
print("============================================")

display_df = results_df.copy()

display_df["Mean_CV_R2"] = (
    display_df["Mean_CV_R2"].round(4)
)

display_df["MAE"] = (
    display_df["MAE"].round(4)
)

display_df["MSE"] = (
    display_df["MSE"].round(4)
)

display_df["RMSE"] = (
    display_df["RMSE"].round(4)
)

display_df["R2"] = (
    display_df["R2"].round(4)
)

display_df["Accuracy_Within_5"] = (
    display_df["Accuracy_Within_5"].round(2)
)

print(
    display_df.to_string(index=False)
)


# ============================================================
# 15. OVERALL ACCURACY
# ============================================================

print("\n============================================")
print("OVERALL ACCURACY")
print("============================================")

overall_accuracy = (
    results_df["Accuracy_Within_5"].mean()
)

overall_r2 = (
    results_df["R2"].mean()
)

overall_mae = (
    results_df["MAE"].mean()
)

overall_rmse = (
    results_df["RMSE"].mean()
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
# 16. SAVE EVALUATION RESULTS
# ============================================================

results_path = os.path.join(
    MODEL_DIR,
    "lasso_evaluation_results.csv"
)

results_df.to_csv(
    results_path,
    index=False
)

print("\nEvaluation results saved:")
print(results_path)


# ============================================================
# 17. SAVE CROSS-VALIDATION RESULTS
# ============================================================

cv_results_df = pd.DataFrame(cv_results)

cv_results_path = os.path.join(
    MODEL_DIR,
    "lasso_cv_results.csv"
)

cv_results_df.to_csv(
    cv_results_path,
    index=False
)

print("\nCross-validation results saved:")
print(cv_results_path)


# ============================================================
# 18. SAVE LASSO MODELS
# ============================================================

print("\n============================================")
print("SAVING LASSO MODELS")
print("============================================")


for target in TARGET_COLUMNS:

    model_path = os.path.join(
        MODEL_DIR,
        f"{target}_lasso_model.pkl"
    )

    joblib.dump(
        final_models[target],
        model_path
    )

    print(
        f"{target} model saved:"
    )

    print(model_path)


# ============================================================
# 19. SAVE BEST PARAMETERS
# ============================================================

best_parameters = []

for target in TARGET_COLUMNS:

    best_parameters.append({
        "Target": target,
        "Best_Alpha":
            best_models[target]["alpha"],
        "Mean_CV_R2":
            best_models[target]["mean_cv_r2"]
    })


best_parameters_df = pd.DataFrame(
    best_parameters
)

parameters_path = os.path.join(
    MODEL_DIR,
    "lasso_best_parameters.csv"
)

best_parameters_df.to_csv(
    parameters_path,
    index=False
)

print("\nBest parameters saved:")
print(parameters_path)


# ============================================================
# 20. FINAL SUMMARY
# ============================================================

print("\n============================================")
print("FINAL LASSO MODEL SUMMARY")
print("============================================")


for _, row in results_df.iterrows():

    print("\n", row["Target"])
    print("--------------------------------------------")

    print(
        f"Mean CV R² : "
        f"{row['Mean_CV_R2']:.4f}"
    )

    print(
        f"Best Alpha : "
        f"{row['Alpha']}"
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
print("--------------------------------------------")


# ============================================================
# 21. COMPLETION
# ============================================================

print("\n============================================")
print("LASSO TRAINING COMPLETED")
print("============================================")

print("\nModels created:")

for target in TARGET_COLUMNS:

    print(
        f"- {target}_lasso_model.pkl"
    )

print("\nMetrics calculated:")

