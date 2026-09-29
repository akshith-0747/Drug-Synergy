import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_val_score
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
    "ml_ready_dataset_clean.csv"
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
# 2. START
# ============================================================

print("============================================")
print("DRUG SYNERGY PREDICTION")
print("RANDOM FOREST MODEL TRAINING")
print("5-FOLD CROSS-VALIDATION")
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

print(
    "Missing values:",
    df.isnull().sum().sum()
)

print(
    "Duplicate rows:",
    df.duplicated().sum()
)


if df.isnull().sum().sum() > 0:

    print(
        "\nRemoving rows containing missing values..."
    )

    df = df.dropna()


print(
    "\nFinal dataset shape:",
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

print("\nFeatures:")
print(X.columns.tolist())


# ============================================================
# 7. TRAIN-TEST SPLIT
# ============================================================

print("\n============================================")
print("TRAIN-TEST SPLIT")
print("============================================")

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
    "Testing samples :",
    X_test.shape[0]
)


# ============================================================
# 8. CROSS-VALIDATION
# ============================================================

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# ============================================================
# 9. RANDOM FOREST CONFIGURATIONS
# ============================================================

parameter_sets = [

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
        "max_depth": 20,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "max_features": "sqrt"
    },

    {
        "n_estimators": 500,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": 0.8
    }
]


# ============================================================
# 10. STORAGE
# ============================================================

cv_results = []

selected_parameters = {}

final_models = {}


# ============================================================
# 11. CROSS-VALIDATED MODEL SELECTION
# ============================================================

print("\n============================================")
print("5-FOLD CROSS-VALIDATION")
print("============================================")


for target in target_columns:

    print("\n============================================")
    print(f"TARGET: {target}")
    print("============================================")

    best_cv_r2 = float("-inf")
    best_params = None


    for config_number, params in enumerate(
        parameter_sets,
        start=1
    ):

        print(
            f"\nConfiguration "
            f"{config_number}/{len(parameter_sets)}"
        )

        print(
            f"n_estimators={params['n_estimators']}, "
            f"max_depth={params['max_depth']}, "
            f"min_samples_split={params['min_samples_split']}, "
            f"min_samples_leaf={params['min_samples_leaf']}, "
            f"max_features={params['max_features']}"
        )


        model = RandomForestRegressor(

            n_estimators=params["n_estimators"],

            max_depth=params["max_depth"],

            min_samples_split=params[
                "min_samples_split"
            ],

            min_samples_leaf=params[
                "min_samples_leaf"
            ],

            max_features=params[
                "max_features"
            ],

            random_state=42,

            n_jobs=-1
        )


        # ----------------------------------------------------
        # 5-FOLD CROSS-VALIDATION
        # ----------------------------------------------------

        cv_scores = cross_val_score(

            model,

            X_train,

            y_train[target],

            cv=cv,

            scoring="r2",

            n_jobs=1
        )


        mean_cv_r2 = cv_scores.mean()

        std_cv_r2 = cv_scores.std()


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
            f"CV R² Std: "
            f"{std_cv_r2:.4f}"
        )


        cv_results.append({

            "Target":
                target,

            "Configuration":
                config_number,

            "n_estimators":
                params["n_estimators"],

            "max_depth":
                params["max_depth"],

            "min_samples_split":
                params["min_samples_split"],

            "min_samples_leaf":
                params["min_samples_leaf"],

            "max_features":
                params["max_features"],

            "Mean_CV_R2":
                mean_cv_r2,

            "Std_CV_R2":
                std_cv_r2
        })


        # ----------------------------------------------------
        # SELECT BEST CONFIGURATION
        # ----------------------------------------------------

        if mean_cv_r2 > best_cv_r2:

            best_cv_r2 = mean_cv_r2

            best_params = params


    # ========================================================
    # BEST CONFIGURATION
    # ========================================================

    selected_parameters[target] = best_params


    print("\nBEST CONFIGURATION")
    print("--------------------------------------------")

    print(
        f"Mean CV R²: "
        f"{best_cv_r2:.4f}"
    )

    print(
        f"n_estimators: "
        f"{best_params['n_estimators']}"
    )

    print(
        f"max_depth: "
        f"{best_params['max_depth']}"
    )

    print(
        f"min_samples_split: "
        f"{best_params['min_samples_split']}"
    )

    print(
        f"min_samples_leaf: "
        f"{best_params['min_samples_leaf']}"
    )

    print(
        f"max_features: "
        f"{best_params['max_features']}"
    )


# ============================================================
# 12. TRAIN FINAL MODELS
# ============================================================

print("\n============================================")
print("TRAINING FINAL RANDOM FOREST MODELS")
print("============================================")


final_results = []


for target in target_columns:

    params = selected_parameters[target]


    print(
        f"\nTraining final model for {target}..."
    )


    model = RandomForestRegressor(

        n_estimators=params["n_estimators"],

        max_depth=params["max_depth"],

        min_samples_split=params[
            "min_samples_split"
        ],

        min_samples_leaf=params[
            "min_samples_leaf"
        ],

        max_features=params[
            "max_features"
        ],

        random_state=42,

        n_jobs=-1
    )


    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.fit(
        X_train,
        y_train[target]
    )


    # --------------------------------------------------------
    # TEST PREDICTION
    # --------------------------------------------------------

    predictions = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    mae = mean_absolute_error(
        y_test[target],
        predictions
    )

    mse = mean_squared_error(
        y_test[target],
        predictions
    )

    rmse = np.sqrt(
        mse
    )

    r2 = r2_score(
        y_test[target],
        predictions
    )


    # --------------------------------------------------------
    # ACCURACY WITHIN ±5
    # --------------------------------------------------------

    accuracy = (
        np.mean(
            np.abs(
                y_test[target].values
                - predictions
            ) <= 5
        )
        * 100
    )


    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    final_models[target] = model


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    final_results.append({

        "Target":
            target,

        "Mean_CV_R2":
            max(
                result["Mean_CV_R2"]
                for result in cv_results
                if result["Target"] == target
            ),

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


    print(
        f"{target} completed."
    )


# ============================================================
# 13. FINAL RESULTS
# ============================================================

final_results_df = pd.DataFrame(
    final_results
)


print("\n============================================")
print("FINAL RANDOM FOREST RESULTS")
print("============================================")

print(
    final_results_df.to_string(
        index=False
    )
)


# ============================================================
# 14. SAVE MODELS
# ============================================================

print("\n============================================")
print("SAVING RANDOM FOREST MODELS")
print("============================================")


for target, model in final_models.items():

    model_path = os.path.join(
        MODEL_DIR,
        f"RF_{target}_best_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"{target} model saved:"
    )

    print(
        model_path
    )


# ============================================================
# 15. SAVE FEATURE COLUMNS
# ============================================================

feature_columns = X.columns.tolist()


feature_path = os.path.join(
    MODEL_DIR,
    "rf_feature_columns.pkl"
)


joblib.dump(
    feature_columns,
    feature_path
)


print("\nFeature columns saved:")
print(feature_path)


# ============================================================
# 16. SAVE CV RESULTS
# ============================================================

cv_results_df = pd.DataFrame(
    cv_results
)


cv_results_path = os.path.join(
    MODEL_DIR,
    "random_forest_cv_results.csv"
)


cv_results_df.to_csv(
    cv_results_path,
    index=False
)


print("\nCross-validation results saved:")
print(cv_results_path)


# ============================================================
# 17. SAVE FINAL RESULTS
# ============================================================

final_results_path = os.path.join(
    MODEL_DIR,
    "random_forest_results.csv"
)


final_results_df.to_csv(
    final_results_path,
    index=False
)


print("\nFinal Random Forest results saved:")
print(final_results_path)


# ============================================================
# 18. SAVE SELECTED PARAMETERS
# ============================================================

parameter_rows = []


for target, params in selected_parameters.items():

    parameter_rows.append({

        "Target":
            target,

        "n_estimators":
            params["n_estimators"],

        "max_depth":
            params["max_depth"],

        "min_samples_split":
            params["min_samples_split"],

        "min_samples_leaf":
            params["min_samples_leaf"],

        "max_features":
            params["max_features"]
    })


parameter_df = pd.DataFrame(
    parameter_rows
)


parameter_path = os.path.join(
    MODEL_DIR,
    "selected_random_forest_parameters.csv"
)


parameter_df.to_csv(
    parameter_path,
    index=False
)


print("\nSelected parameters saved:")
print(parameter_path)


# ============================================================
# 19. FINAL SUMMARY
# ============================================================

print("\n============================================")
print("RANDOM FOREST MODEL SUMMARY")
print("============================================")


for _, row in final_results_df.iterrows():

    print(
        f"\n{row['Target']}"
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
        f"Accuracy ±5: "
        f"{row['Accuracy_Within_5']:.2f}%"
    )


# ============================================================
# 20. COMPLETION
# ============================================================


print("RANDOM FOREST TRAINING COMPLETED")


print("\nModels created:")

for target in target_columns:

    print(
        f"- RF_{target}_best_model.pkl"
    )

print("\nMetrics calculated:")

print("- R²")
print("- MAE")
print("- MSE")
print("- RMSE")
print("- Accuracy within ±5 units")

p
