
"""Local API for the DRS drug-combination prediction project."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FRONTEND = ROOT / "Frontend"
STATIC = FRONTEND / "static"

PROCESSED = (
    ROOT
    / "ML"
    / "data"
    / "processed"
    / "matched_drugcomb.csv"
)

RESULTS = ROOT / "ML" / "results"


# ============================================================
# GEMINI AI CONFIGURATION
# ============================================================

load_dotenv(ROOT / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Current lightweight Gemini model
GEMINI_MODEL = "gemini-3.5-flash-lite"

gemini_client = (
    genai.Client(api_key=GEMINI_API_KEY)
    if GEMINI_API_KEY
    else None
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="DRS Drug Combination Predictor",
    version="1.1",
)

# CSS and JS are inside Frontend/static/
app.mount(
    "/static",
    StaticFiles(directory=str(STATIC)),
    name="static",
)


# ============================================================
# TRAINING STATE
# ============================================================

_state_lock = threading.Lock()

_training_state = {
    "status": "idle",
    "message": "No training run yet.",
}


# ============================================================
# REQUEST MODELS
# ============================================================

class TrainRequest(BaseModel):

    feature_sets: list[str] = Field(
        default_factory=lambda: ["structure", "ds", "drs"]
    )

    max_rows: int = Field(
        default=2000,
        ge=100,
        le=1000000,
    )

    folds: int = Field(
        default=3,
        ge=2,
        le=5,
    )

    repeats: int = Field(
        default=1,
        ge=1,
        le=10,
    )


class PredictRequest(BaseModel):

    drug_a: str
    drug_b: str
    cell_line: str

    feature_set: str = "auto"

    model: str | None = None


# ============================================================
# AI CHAT REQUEST MODELS
# ============================================================

class ChatMessage(BaseModel):

    role: str
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):

    message: str = Field(
        min_length=1,
        max_length=4000,
    )

    history: list[ChatMessage] = Field(
        default_factory=list,
    )


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/")
def home():

    return FileResponse(FRONTEND / "index.html")


# ============================================================
# STATUS
# ============================================================

@app.get("/api/status")
def status():

    model_dir = ROOT / "ML" / "results" / "models"

    models = []

    if model_dir.exists():

        for pattern in ("*__*.joblib", "*__*.pkl"):

            for path in model_dir.glob(pattern):

                if not path.stem.startswith("feature_context"):

                    models.append(path.stem)

    models = sorted(set(models))

    return {
        "project": "Drug Resistance Signature drug-combination prediction",
        "data_prepared": PROCESSED.exists(),
        "prepared_file": str(PROCESSED)
        if PROCESSED.exists()
        else None,
        "trained_models": models,
        "training": dict(_training_state),
        "ai_available": gemini_client is not None,
    }


# ============================================================
# OPTIONS
# ============================================================

@app.get("/api/options")
def options():

    if not PROCESSED.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "Prepared DrugComb data is missing. "
                "Run ML/data_preparation/prepare_data.py first."
            ),
        )

    try:

        frame = pd.read_csv(
            PROCESSED,
            usecols=[
                "drug_row",
                "drug_col",
                "cell_line_name",
            ],
            dtype=str,
        )

        drugs = sorted(
            set(frame["drug_row"].dropna())
            | set(frame["drug_col"].dropna())
        )

        cells = sorted(
            frame["cell_line_name"]
            .dropna()
            .unique()
            .tolist()
        )

        return {
            "drugs": drugs,
            "cell_lines": cells,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Could not read prepared data options: {exc}",
        ) from exc


# ============================================================
# METRICS
# ============================================================

@app.get("/api/metrics")
def metrics():

    file = RESULTS / "model_comparison.csv"

    if not file.exists():

        return {
            "rows": [],
            "message": (
                "Train models to create the comparison table."
            ),
        }

    try:

        table = pd.read_csv(file).astype(object)

        table = table.where(
            pd.notna(table),
            None,
        )

        return {
            "rows": table.to_dict("records")
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Could not read model metrics: {exc}",
        ) from exc


# ============================================================
# TRAINING
# ============================================================

def _run_training(settings: TrainRequest):

    from ML.training.train import train

    with _state_lock:

        _training_state.update(
            status="running",
            message="Model evaluation is running.",
        )

    try:

        report = train(
            SimpleNamespace(
                feature_sets=",".join(
                    settings.feature_sets
                ),
                models=(
                    "lasso,"
                    "random_forest,"
                    "adaboost"
                ),
                max_rows=settings.max_rows,
                folds=settings.folds,
                repeats=settings.repeats,
                jobs=-1,
            )
        )

        with _state_lock:

            _training_state.update(
                status="complete",
                message=(
                    "Training finished; comparison "
                    "metrics and models were saved."
                ),
                rows=report.get("rows_loaded"),
            )

    except Exception as exc:

        with _state_lock:

            _training_state.update(
                status="error",
                message=str(exc),
            )


@app.post("/api/train", status_code=202)
def start_training(request: TrainRequest):

    with _state_lock:

        if _training_state.get("status") in ("running", "queued"):

            raise HTTPException(
                status_code=409,
                detail="A training run is already active.",
            )

        _training_state.update(
            status="queued",
            message="Training is starting.",
        )

    thread = threading.Thread(
        target=_run_training,
        args=(request,),
        daemon=True,
    )

    thread.start()

    return {
        "status": "queued",
        "message": (
            "Training started. "
            "Refresh the status panel to see when it finishes."
        ),
    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/api/predict")
def prediction(request: PredictRequest):

    from ML.prediction.predict import predict

    try:

        result = predict(
            request.drug_a,
            request.drug_b,
            request.cell_line,
            request.feature_set,
            request.model,
        )

        return result

    except (
        ValueError,
        FileNotFoundError,
    ) as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


# ============================================================
# AI GUIDE - GEMINI CHAT
# ============================================================

@app.post("/api/chat")
def ai_chat(request: ChatRequest):

    if gemini_client is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini API key is missing. "
                "Check your project root .env file."
            ),
        )

    if not request.message.strip():

        raise HTTPException(
            status_code=400,
            detail="Please enter a message.",
        )

    system_prompt = """
    You are the AI Guide for a B.Tech project called
    Drug Resistance Signature (DRS) based Drug
    Combination Synergy Prediction.

    YOUR ROLE:
    Help students, researchers and visitors understand
    the project and its functionality.

    PROJECT INFORMATION:

    1. PROJECT OBJECTIVE

    The project predicts the synergy score of a combination
    of two anticancer drugs in a particular cancer cell line.

    2. FEATURE SETS

    STRUCTURE:
    Represents chemical structure information of drugs.
    The project uses molecular fingerprints where available.

    DS (DRUG SIGNATURES):
    Represents changes in gene expression caused by drugs.
    The project uses LINCS drug perturbation signatures.

    DRS (DRUG RESISTANCE SIGNATURES):
    Represents gene-expression patterns associated with
    drug resistance and sensitivity.
    These signatures are derived using drug sensitivity
    and gene-expression information.

    3. MACHINE LEARNING MODELS

    The project considers:
    - LASSO
    - Random Forest
    - AdaBoost
    - XGBoost
    - SynergyX

    Do not claim that every model is currently implemented
    or trained unless verified by the user.

    4. DATA

    The project uses DrugComb combination-response data,
    chemical information, LINCS signatures and gene-expression
    and drug-sensitivity information.

    5. EVALUATION

    Models are evaluated using cross-validation and
    regression metrics such as:
    - R-squared
    - RMSE
    - MAE

    6. BACKEND

    The backend uses FastAPI.
    The machine learning pipeline is implemented in Python.
    The frontend communicates with the backend through APIs.

    RESPONSE RULES:

    - Use simple, clear English.
    - Explain difficult concepts step by step.
    - Give examples whenever useful.
    - Use headings and bullet points where appropriate.
    - Keep answers relevant to drug synergy prediction.
    - Answer general questions briefly when appropriate.
    - Never invent experimental results, model accuracy,
      dataset statistics or prediction outputs.
    - Do not claim to execute an actual ML prediction.
      Actual predictions are performed by the prediction API.
    - Clearly distinguish project information from general
      scientific explanations.
    """

    try:

        contents = []

        # Include recent conversation history.
        for item in request.history[-10:]:

            role = item.role.strip().lower()

            # Gemini uses "user" and "model" roles.
            if role == "assistant":
                role = "model"

            if role not in ("user", "model"):
                continue

            if not item.content.strip():
                continue

            contents.append(
                types.Content(
                    role=role,
                    parts=[
                        types.Part.from_text(
                            text=item.content
                        )
                    ],
                )
            )

        # Add current user question.
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=request.message.strip()
                    )
                ],
            )
        )

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.5,
                max_output_tokens=1000,
            ),
        )

        answer = response.text

        if not answer:

            raise HTTPException(
                status_code=502,
                detail="Gemini returned an empty response.",
            )

        return {
            "answer": answer,
            "model": GEMINI_MODEL,
            "status": "success",
        }

    except HTTPException:
        raise

    except Exception as exc:

        print("Gemini API error:", str(exc))

        raise HTTPException(
            status_code=502,
            detail=(
                "The AI Guide could not generate a response. "
                "Check your API key, model availability "
                "and internet connection."
            ),
        ) from exc


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )

