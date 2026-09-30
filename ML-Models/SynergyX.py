
import os
import random
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)
from tensorflow.keras.optimizers import Adam


# ============================================================
# 1. REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# 2. PATHS
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
# 3. SETTINGS
# ============================================================

TARGET_COLUMNS = [
    "ZIP",
    "Bliss",
    "Loewe",
    "HSA"
]

ACCURACY_TOLERANCE = 5

TEST_SIZE = 0.20

VALIDATION_SIZE = 0.20

EPOCHS = 100

BATCH_SIZE = 64

LEARNING_RATE = 0.001


# ============================================================
# 4. HEADER
# ============================================================

print("\n============================================")
print("DRUG SYNERGY PREDICTION")
print("SYNERGYX DEEP LEARNING MODEL")
print("============================================")


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\nLoading DRS ML-ready dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 6. TARGET COLUMNS
# ============================================================

print("\nTarget columns:")
print(TARGET_COLUMNS)


# ============================================================
# 7. DATA CHECK
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
# 8. SELECT DRS FEATURES
# ============================================================

print("\n============================================")
print("DRS FEATURE INFORMATION")
print("============================================")


drug1_features = [
    column
    for column in df.columns
    if column.startswith("Drug1_DRS_PC")
]


drug2_features = [
    column
    for column in df.columns
    if column.startswith("Drug2_DRS_PC")
]


feature_columns = (
    drug1_features +
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

print(feature_columns)


# ============================================================
# 9. FEATURES AND TARGETS
# ============================================================

X = df[feature_columns]

y = df[TARGET_COLUMNS]


print("\nNumber of samples:", X.shape[0])

print("Number of features:", X.shape[1])


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

    random_state=SEED
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
# 11. FEATURE SCALING
# ============================================================

print("\n============================================")
print("FEATURE SCALING")
print("============================================")


scaler = StandardScaler()


X_train_scaled = scaler.fit_transform(
    X_train
)


X_test_scaled = scaler.transform(
    X_test
)


print("Feature scaling completed.")


# ============================================================
# 12. BUILD SYNERGYX MODEL
# ============================================================

print("\n============================================")
print("BUILDING SYNERGYX MODEL")
print("============================================")


input_dimension = X_train_scaled.shape[1]


inputs = Input(
    shape=(input_dimension,),
    name="DRS_Input"
)


# ------------------------------------------------------------
# DENSE BLOCK 1
# ------------------------------------------------------------

x = Dense(
    256,
    activation="relu",
    name="Dense_256"
)(inputs)

x = BatchNormalization(
    name="BatchNorm_1"
)(x)

x = Dropout(
    0.30,
    name="Dropout_1"
)(x)


# ------------------------------------------------------------
# DENSE BLOCK 2
# ------------------------------------------------------------

x = Dense(
    128,
    activation="relu",
    name="Dense_128"
)(x)

x = BatchNormalization(
    name="BatchNorm_2"
)(x)

x = Dropout(
    0.25,
    name="Dropout_2"
)(x)


# ------------------------------------------------------------
# DENSE BLOCK 3
# ------------------------------------------------------------

x = Dense(
    64,
    activation="relu",
    name="Dense_64"
)(x)

x = Dropout(
    0.20,
    name="Dropout_3"
)(x)


# ------------------------------------------------------------
# OUTPUT LAYER
# ------------------------------------------------------------

outputs = Dense(
    4,
    activation="linear",
    name="Synergy_Output"
)(x)


model = Model(
    inputs=inputs,
    outputs=outputs,
    name="SynergyX"
)


# ============================================================
# 13. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="mse",

    metrics=[
        "mae"
    ]
)


print("\nSynergyX architecture:")

model.summary()


# ============================================================
# 14. CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=12,

    restore_best_weights=True
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=5,

    min_lr=0.000001
)


# ============================================================
# 15. TRAIN MODEL
# ============================================================

print("\n============================================")
print("TRAINING SYNERGYX")
print("============================================")


history = model.fit(

    X_train_scaled,

    y_train.values,

    validation_split=VALIDATION_SIZE,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


print("\nSynergyX training completed.")


# ============================================================
# 16. PREDICTION
# ============================================================

print("\n============================================")
print("GENERATING PREDICTIONS")
print("============================================")


predictions = model.predict(
    X_test_scaled,
    verbose=0
)


print("Predictions generated successfully.")


# ============================================================
# 17. EVALUATION
# ============================================================

print("\n============================================")
print("FINAL TEST SET EVALUATION")
print("============================================")


results = []


for index, target in enumerate(TARGET_COLUMNS):

    actual_values = (
        y_test[target].values
    )

    predicted_values = (
        predictions[:, index]
    )


    # --------------------------------------------------------
    # MAE
    # --------------------------------------------------------

    mae = mean_absolute_error(
        actual_values,
        predicted_values
    )


    # --------------------------------------------------------
    # MSE
    # --------------------------------------------------------

    mse = mean_squared_error(
        actual_values,
        predicted_values
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
        predicted_values
    )


    # --------------------------------------------------------
    # ACCURACY WITHIN ±5
    # --------------------------------------------------------

    absolute_errors = np.abs(
        actual_values -
        predicted_values
    )


    correct_predictions = (
        absolute_errors <=
        ACCURACY_TOLERANCE
    )


    accuracy = (
        np.mean(
            correct_predictions
        ) * 100
    )


    # --------------------------------------------------------
    # STORE
    # --------------------------------------------------------

    results.append({

        "Target": target,

        "MAE": mae,

        "MSE": mse,

        "RMSE": rmse,

        "R2": r2,

        "Accuracy_Within_5":
            accuracy

    })


    # --------------------------------------------------------
    # PRINT
    # --------------------------------------------------------

    print("\n--------------------------------------------")

    print(target)

    print("--------------------------------------------")

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
# 18. RESULTS DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    results
)


# ============================================================
# 19. FINAL RESULTS TABLE
# ============================================================

print("\n============================================")
print("FINAL SYNERGYX RESULTS")
print("============================================")


display_df = results_df.copy()


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
    display_df[
        "Accuracy_Within_5"
    ].round(2)
)


print(
    display_df.to_string(
        index=False
    )
)


# ============================================================
# 20. OVERALL ACCURACY
# ============================================================

print("\n============================================")
print("OVERALL SYNERGYX PERFORMANCE")
print("============================================")


overall_accuracy = (
    results_df[
        "Accuracy_Within_5"
    ].mean()
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
# 21. SAVE MODEL
# ============================================================

print("\n============================================")
print("SAVING SYNERGYX MODEL")
print("============================================")


model_path = os.path.join(
    MODEL_DIR,
    "SynergyX_model.keras"
)


model.save(
    model_path
)


print(
    "SynergyX model saved:"
)

print(model_path)


# ============================================================
# 22. SAVE SCALER
# ============================================================

scaler_path = os.path.join(
    MODEL_DIR,
    "SynergyX_scaler.pkl"
)


joblib.dump(
    scaler,
    scaler_path
)


print(
    "\nScaler saved:"
)

print(scaler_path)


# ============================================================
# 23. SAVE FEATURE COLUMNS
# ============================================================

feature_path = os.path.join(
    MODEL_DIR,
    "SynergyX_feature_columns.pkl"
)


joblib.dump(
    feature_columns,
    feature_path
)


print(
    "\nFeature columns saved:"
)

print(feature_path)


# ============================================================
# 24. SAVE EVALUATION RESULTS
# ============================================================

results_path = os.path.join(
    MODEL_DIR,
    "SynergyX_evaluation_results.csv"
)


results_df.to_csv(
    results_path,
    index=False
)


print(
    "\nEvaluation results saved:"
)

print(results_path)


# ============================================================
# 25. SAVE TEST PREDICTIONS
# ============================================================

prediction_df = y_test.copy()


for index, target in enumerate(
    TARGET_COLUMNS
):

    prediction_df[
        f"{target}_Predicted"
    ] = predictions[:, index]


prediction_path = os.path.join(
    MODEL_DIR,
    "SynergyX_test_predictions.csv"
)


prediction_df.to_csv(
    prediction_path,
    index=False
)


print(
    "\nTest predictions saved:"
)

print(prediction_path)


# ============================================================
# 26. SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(
    history.history
)


history_path = os.path.join(
    MODEL_DIR,
    "SynergyX_training_history.csv"
)


history_df.to_csv(
    history_path,
    index=False
)


print(
    "\nTraining history saved:"
)

print(history_path)


# ============================================================
# 27. FINAL SUMMARY
# ============================================================

print("\n============================================")
print("FINAL SYNERGYX MODEL SUMMARY")
print("============================================")


for _, row in results_df.iterrows():

    print("\n", row["Target"])

    print("--------------------------------------------")

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

print("SYNERGYX TRAINING COMPLETED")



print("\nMetrics calculated:")
