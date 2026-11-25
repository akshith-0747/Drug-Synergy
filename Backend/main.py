
"""Local API for the DRS drug-combination prediction project."""

from __future__ import annotations

import os
import hashlib
import hmac
import re
import secrets
import sqlite3
import sys
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
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
    / "matched_drugcomb_v1_5_ssum.csv"
)
PROCESSED_V15_S_SUM = (
    ROOT
    / "ML"
    / "data"
    / "processed"
    / "matched_drugcomb_v1_5_ssum.csv"
)

RESULTS = ROOT / "ML" / "results"
APP_DB = ROOT / "Backend" / "data" / "synergia_app.sqlite3"
FREE_PREDICTION_LIMIT = 2
VISITOR_COOKIE = "synergia_visitor"
SESSION_COOKIE = "synergia_session"
PASSWORD_ITERATIONS = 310_000


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


class SignupRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    username: str = Field(min_length=3, max_length=32)
    password: str = Field(min_length=8, max_length=256)


class LoginRequest(BaseModel):
    identifier: str = Field(min_length=1, max_length=254)
    password: str = Field(min_length=1, max_length=256)


class ReviewRequest(BaseModel):
    prediction_id: str = Field(min_length=1, max_length=64)
    rating: int = Field(ge=1, le=5)
    outcome: str = Field(pattern="^(not_checked|agrees|differs)$")
    comment: str = Field(default="", max_length=1600)


# ============================================================
# LOCAL ACCOUNTS, FREE-PREDICTION METER, AND REVIEWS
# ============================================================

def _db_connect() -> sqlite3.Connection:
    APP_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(APP_DB, timeout=20)
    connection.row_factory = sqlite3.Row
    return connection


def _initialize_app_db() -> None:
    with _db_connect() as connection:
        connection.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sessions (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                expires_at INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS anonymous_usage (
                visitor_hash TEXT PRIMARY KEY,
                predictions_used INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id TEXT PRIMARY KEY,
                user_id INTEGER REFERENCES users(id),
                visitor_hash TEXT,
                drug_a TEXT NOT NULL,
                drug_b TEXT NOT NULL,
                cell_line TEXT NOT NULL,
                predicted_score REAL,
                feature_set TEXT,
                model TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT NOT NULL REFERENCES predictions(id) ON DELETE CASCADE,
                user_id INTEGER REFERENCES users(id),
                visitor_hash TEXT,
                rating INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                outcome TEXT NOT NULL,
                comment TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS predictions_user_date
                ON predictions(user_id, created_at DESC);
            CREATE INDEX IF NOT EXISTS predictions_visitor_date
                ON predictions(visitor_hash, created_at DESC);
            CREATE INDEX IF NOT EXISTS reviews_user_date
                ON reviews(user_id, created_at DESC);
            """
        )


_initialize_app_db()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _cookie_secure() -> bool:
    return os.getenv("COOKIE_SECURE", "").strip().lower() in {"1", "true", "yes"}


def _get_visitor_hash(request: Request, response: Response) -> str:
    token = request.cookies.get(VISITOR_COOKIE)
    if not token or len(token) > 200:
        token = secrets.token_urlsafe(32)
        response.set_cookie(
            VISITOR_COOKIE,
            token,
            max_age=60 * 60 * 24 * 365,
            httponly=True,
            secure=_cookie_secure(),
            samesite="lax",
            path="/",
        )
    return _token_hash(token)


def _current_user(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if not token or len(token) > 200:
        return None
    with _db_connect() as connection:
        return connection.execute(
            """SELECT users.id, users.email, users.username
               FROM sessions JOIN users ON users.id = sessions.user_id
               WHERE sessions.token_hash = ? AND sessions.expires_at > ?""",
            (_token_hash(token), int(datetime.now(timezone.utc).timestamp())),
        ).fetchone()


def _set_session_cookie(response: Response, user_id: int) -> None:
    token = secrets.token_urlsafe(40)
    expires_at = int((datetime.now(timezone.utc) + timedelta(days=14)).timestamp())
    with _db_connect() as connection:
        connection.execute(
            "INSERT INTO sessions(token_hash, user_id, expires_at) VALUES (?, ?, ?)",
            (_token_hash(token), user_id, expires_at),
        )
    response.set_cookie(
        SESSION_COOKIE,
        token,
        max_age=60 * 60 * 24 * 14,
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
        path="/",
    )


def _password_digest(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return f"{PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def _password_matches(password: str, encoded: str) -> bool:
    try:
        iterations_text, salt_text, expected = encoded.split("$", 2)
        actual = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_text), int(iterations_text)
        ).hex()
        return hmac.compare_digest(actual, expected)
    except (ValueError, TypeError):
        return False


def _attach_guest_records(connection: sqlite3.Connection, user_id: int, visitor_hash: str) -> None:
    connection.execute(
        "UPDATE predictions SET user_id = ?, visitor_hash = NULL WHERE visitor_hash = ? AND user_id IS NULL",
        (user_id, visitor_hash),
    )
    connection.execute(
        "UPDATE reviews SET user_id = ?, visitor_hash = NULL WHERE visitor_hash = ? AND user_id IS NULL",
        (user_id, visitor_hash),
    )


def _reserve_free_prediction(visitor_hash: str) -> int:
    with _db_connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT OR IGNORE INTO anonymous_usage(visitor_hash, predictions_used) VALUES (?, 0)",
            (visitor_hash,),
        )
        row = connection.execute(
            "SELECT predictions_used FROM anonymous_usage WHERE visitor_hash = ?",
            (visitor_hash,),
        ).fetchone()
        used = int(row["predictions_used"])
        if used >= FREE_PREDICTION_LIMIT:
            raise HTTPException(
                status_code=401,
                detail="Your 2 free predictions are used. Log in or create an account to continue.",
            )
        used += 1
        connection.execute(
            "UPDATE anonymous_usage SET predictions_used = ? WHERE visitor_hash = ?",
            (used, visitor_hash),
        )
        return used


def _release_free_prediction(visitor_hash: str) -> None:
    with _db_connect() as connection:
        connection.execute(
            "UPDATE anonymous_usage SET predictions_used = MAX(predictions_used - 1, 0) WHERE visitor_hash = ?",
            (visitor_hash,),
        )


def _prediction_score(result: dict) -> float | None:
    for key in ("predicted_s_score", "s_score", "prediction", "score"):
        try:
            value = float(result.get(key))
        except (TypeError, ValueError):
            continue
        if value == value and abs(value) != float("inf"):
            return value
    return None


@app.get("/api/auth/state")
def auth_state(request: Request, response: Response):
    user = _current_user(request)
    if user:
        with _db_connect() as connection:
            total = connection.execute(
                "SELECT COUNT(*) AS n FROM predictions WHERE user_id = ?", (user["id"],)
            ).fetchone()["n"]
        return {
            "authenticated": True,
            "username": user["username"],
            "predictions_used": int(total),
            "free_predictions_left": None,
        }

    visitor_hash = _get_visitor_hash(request, response)
    with _db_connect() as connection:
        row = connection.execute(
            "SELECT predictions_used FROM anonymous_usage WHERE visitor_hash = ?",
            (visitor_hash,),
        ).fetchone()
    used = int(row["predictions_used"]) if row else 0
    return {
        "authenticated": False,
        "username": None,
        "predictions_used": used,
        "free_predictions_left": max(FREE_PREDICTION_LIMIT - used, 0),
    }


@app.post("/api/auth/signup")
def signup(request: Request, response: Response, payload: SignupRequest):
    email = payload.email.strip().lower()
    username = payload.username.strip()
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if not re.fullmatch(r"[A-Za-z0-9_.-]{3,32}", username):
        raise HTTPException(
            status_code=400,
            detail="Username must be 3–32 characters using letters, numbers, dots, dashes, or underscores.",
        )

    try:
        with _db_connect() as connection:
            cursor = connection.execute(
                "INSERT INTO users(email, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
                (email, username, _password_digest(payload.password), datetime.now(timezone.utc).isoformat()),
            )
            user_id = int(cursor.lastrowid)
            _attach_guest_records(connection, user_id, _get_visitor_hash(request, response))
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="That email or username already has an account.") from exc

    _set_session_cookie(response, user_id)
    return {"authenticated": True, "username": username}


@app.post("/api/auth/login")
def login(request: Request, response: Response, payload: LoginRequest):
    identifier = payload.identifier.strip()
    with _db_connect() as connection:
        user = connection.execute(
            "SELECT id, email, username, password_hash FROM users WHERE email = ? COLLATE NOCASE OR username = ? COLLATE NOCASE",
            (identifier, identifier),
        ).fetchone()
    if not user or not _password_matches(payload.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="The email/username or password is incorrect.")

    visitor_hash = _get_visitor_hash(request, response)
    with _db_connect() as connection:
        _attach_guest_records(connection, int(user["id"]), visitor_hash)
    _set_session_cookie(response, int(user["id"]))
    return {"authenticated": True, "username": user["username"]}


@app.post("/api/auth/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        with _db_connect() as connection:
            connection.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="lax", secure=_cookie_secure())
    return {"authenticated": False}


@app.get("/api/reviews")
def list_reviews(request: Request, response: Response):
    user = _current_user(request)
    owner_column = "user_id" if user else "visitor_hash"
    owner_value = int(user["id"]) if user else _get_visitor_hash(request, response)
    with _db_connect() as connection:
        predictions = connection.execute(
            f"""SELECT id, drug_a, drug_b, cell_line, predicted_score, feature_set, model, created_at
                FROM predictions WHERE {owner_column} = ? ORDER BY created_at DESC LIMIT 100""",
            (owner_value,),
        ).fetchall()
        reviews = connection.execute(
            f"""SELECT reviews.id, reviews.prediction_id, reviews.rating, reviews.outcome,
                       reviews.comment, reviews.created_at, predictions.drug_a, predictions.drug_b,
                       predictions.cell_line, predictions.predicted_score
                FROM reviews JOIN predictions ON predictions.id = reviews.prediction_id
                WHERE reviews.{owner_column} = ? ORDER BY reviews.created_at DESC LIMIT 100""",
            (owner_value,),
        ).fetchall()
    return {"predictions": [dict(row) for row in predictions], "reviews": [dict(row) for row in reviews]}


@app.post("/api/reviews")
def submit_review(request: Request, response: Response, payload: ReviewRequest):
    user = _current_user(request)
    visitor_hash = None if user else _get_visitor_hash(request, response)
    owner_column = "user_id" if user else "visitor_hash"
    owner_value = int(user["id"]) if user else visitor_hash
    with _db_connect() as connection:
        prediction_row = connection.execute(
            f"SELECT id FROM predictions WHERE id = ? AND {owner_column} = ?",
            (payload.prediction_id, owner_value),
        ).fetchone()
        if not prediction_row:
            raise HTTPException(status_code=404, detail="Choose one of your own saved predictions to review.")
        connection.execute(
            """INSERT INTO reviews(prediction_id, user_id, visitor_hash, rating, outcome, comment, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                payload.prediction_id,
                int(user["id"]) if user else None,
                visitor_hash,
                payload.rating,
                payload.outcome,
                payload.comment.strip(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return {"saved": True, "message": "Thanks—your review has been saved."}


# ============================================================
# TRAINING STATE
# ============================================================

_state_lock = threading.Lock()

_training_state = {
    "status": "idle",
    "message": "No training run yet.",
}
_latest_metrics_file: Path | None = None


# ============================================================
# REQUEST MODELS
# ============================================================

class TrainRequest(BaseModel):

    feature_sets: list[str] = Field(
        default_factory=lambda: ["structure", "ds", "drs"]
    )

    models: list[str] = Field(
        default_factory=lambda: [
            "lasso", "random_forest", "adaboost", "xgboost",
            "xgboost_pairwise", "synergyx"
        ]
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

    synergyx_epochs: int = Field(
        default=40,
        ge=10,
        le=120,
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
    from ML.prediction.predict import available_models
    from ML.storage import load

    models = sorted(
        f"{item['feature_set']}__{item['model']}"
        for item in available_models()
    )
    prediction_models = set()
    prediction_model_status = {}
    model_dir = RESULTS / "models"
    for item in available_models():
        try:
            bundle = load(model_dir / item["file"])
        except Exception:
            continue
        if (
            bundle.get("dataset_version") == "drugcomb_v1_5_ssum"
            and bundle.get("deployment_status") in {"production", "candidate"}
        ):
            prediction_models.add(item["model"])
            current_status = prediction_model_status.get(item["model"])
            if current_status != "production" or bundle.get("deployment_status") == "production":
                prediction_model_status[item["model"]] = bundle["deployment_status"]

    return {
        "project": "Drug Resistance Signature drug-combination prediction",
        "data_prepared": PROCESSED.exists(),
        "prepared_file": str(PROCESSED)
        if PROCESSED.exists()
        else None,
        "trained_models": models,
        "prediction_models": sorted(prediction_models),
        "prediction_model_status": prediction_model_status,
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
    with _state_lock:
        latest_metrics_file = _latest_metrics_file
    file = Path(latest_metrics_file) if latest_metrics_file else RESULTS / "model_comparison.csv"
    supplementary_file = (
        RESULTS
        / "paper_feature_benchmark"
        / "paper_drs_fallback_model_comparison.csv"
    )
    attention_comparison_file = (
        RESULTS
        / "paper_feature_benchmark"
        / "paper_drs_attention_model_comparison.csv"
    )
    structure_enrichment_comparison_file = (
        RESULTS
        / "pubchem_enrichment_evaluation"
        / "pubchem_structure_model_comparison.csv"
    )
    v1_5_candidate_comparison_file = (
        RESULTS
        / "drugcomb_v1_5_ssum_clip3_pair_cv"
        / "model_comparison.csv"
    )
    v1_5_synergyx_comparison_file = (
        RESULTS
        / "drugcomb_v1_5_ssum_synergyx_clip3_pair_cv"
        / "model_comparison.csv"
    )
    v1_5_full_signature_comparison_file = (
        RESULTS
        / "drugcomb_v1_5_ssum_full_signature_clip3_pair_cv"
        / "model_comparison.csv"
    )
    v1_5_full_pair_cv_comparison_file = (
        RESULTS
        / "drugcomb_v1_5_ssum_full_xgb_pair_cv"
        / "model_comparison.csv"
    )
    v1_5_prism_drs_comparison_file = (
        RESULTS
        / "drugcomb_v1_5_ssum_prism_drs_pair_cv"
        / "model_comparison.csv"
    )
    v1_5_all_models_screening_comparison_file = (
        RESULTS
        / "drugcomb_v1_5_ssum_all_models_5k_pair_cv_review"
        / "model_comparison.csv"
    )
    v1_5_full_all_models_comparison_file = (
        RESULTS
        / "drugcomb_v1_5_ssum_all_models_full_v1_pair_cv"
        / "model_comparison.csv"
    )

    v1_5_xgb_reg_confirmation_file = (
        RESULTS
        / "drugcomb_v1_5_ssum_xgb_reg_confirm_20k_r2"
        / "model_comparison.csv"
    )
    v1_5_xgb_clip_confirmation_file = (
        RESULTS
        / "drugcomb_v1_5_ssum_xgb_clip3_confirm_20k_r2"
        / "model_comparison.csv"
    )
    v1_4_crosswalk_comparison_file = (
        RESULTS
        / "drugcomb_v1_4_crosswalk_pair_cv"
        / "model_comparison.csv"
    )
    prism_drs_model_lab_file = (
        RESULTS
        / "prism_drs_candidate_pair_cv_nested"
        / "model_lab_comparison.csv"
    )

    try:
        tables = []
        if file.exists() and file.stat().st_size > 0:
            tables.append(pd.read_csv(file))
        if supplementary_file.exists() and supplementary_file.stat().st_size > 0:
            tables.append(pd.read_csv(supplementary_file))
        if attention_comparison_file.exists() and attention_comparison_file.stat().st_size > 0:
            tables.append(pd.read_csv(attention_comparison_file))
        if structure_enrichment_comparison_file.exists() and structure_enrichment_comparison_file.stat().st_size > 0:
            tables.append(pd.read_csv(structure_enrichment_comparison_file))
        if v1_5_candidate_comparison_file.exists() and v1_5_candidate_comparison_file.stat().st_size > 0:
            tables.append(pd.read_csv(v1_5_candidate_comparison_file))
        if v1_5_synergyx_comparison_file.exists() and v1_5_synergyx_comparison_file.stat().st_size > 0:
            tables.append(pd.read_csv(v1_5_synergyx_comparison_file))
        if v1_5_full_signature_comparison_file.exists() and v1_5_full_signature_comparison_file.stat().st_size > 0:
            tables.append(pd.read_csv(v1_5_full_signature_comparison_file))
        if v1_5_full_pair_cv_comparison_file.exists() and v1_5_full_pair_cv_comparison_file.stat().st_size > 0:
            tables.append(pd.read_csv(v1_5_full_pair_cv_comparison_file))
        if v1_5_prism_drs_comparison_file.exists() and v1_5_prism_drs_comparison_file.stat().st_size > 0:
            tables.append(pd.read_csv(v1_5_prism_drs_comparison_file))
        if v1_5_all_models_screening_comparison_file.exists() and v1_5_all_models_screening_comparison_file.stat().st_size > 0:
            tables.append(pd.read_csv(v1_5_all_models_screening_comparison_file))
        if v1_5_full_all_models_comparison_file.exists() and v1_5_full_all_models_comparison_file.stat().st_size > 0:
            tables.append(pd.read_csv(v1_5_full_all_models_comparison_file))
        if v1_5_xgb_reg_confirmation_file.exists() and v1_5_xgb_reg_confirmation_file.stat().st_size > 0:
            tables.append(pd.read_csv(v1_5_xgb_reg_confirmation_file))
        if v1_5_xgb_clip_confirmation_file.exists() and v1_5_xgb_clip_confirmation_file.stat().st_size > 0:
            tables.append(pd.read_csv(v1_5_xgb_clip_confirmation_file))
        if v1_4_crosswalk_comparison_file.exists() and v1_4_crosswalk_comparison_file.stat().st_size > 0:
            tables.append(pd.read_csv(v1_4_crosswalk_comparison_file))
        if prism_drs_model_lab_file.exists() and prism_drs_model_lab_file.stat().st_size > 0:
            tables.append(pd.read_csv(prism_drs_model_lab_file))
        if not tables:
            return {
                "rows": [],
                "message": "Train models to create the comparison table.",
            }

        table = pd.concat(tables, ignore_index=True, sort=False).astype(object)

        # Show only the finished full-data v1.5 comparison and its current
        # PRISM DRS coverage candidate. Screening runs and paper-era reports
        # stay out of the website's headline model table.
        if "dataset_label" in table.columns:
            labels = table["dataset_label"].fillna("").astype(str)
            current_runs = labels.str.startswith(
                "DrugComb v1.5 · S_sum · full-data all-model comparison"
            ) | labels.str.startswith(
                "DrugComb v1.5 · S_sum · PRISM DRS candidate"
            )
            table = table.loc[current_runs].copy()

        if {"model", "clip3_rmse"}.issubset(table.columns):
            model_names = table["model"].fillna("").astype(str).str.lower()
            clipped_rmse = pd.to_numeric(table["clip3_rmse"], errors="coerce")
            table["evaluation_status"] = ""
            table.loc[(model_names == "lasso") & (clipped_rmse > 50), "evaluation_status"] = "unstable"

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

def _run_training(settings: TrainRequest, run_dir: Path):
    global _latest_metrics_file

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
                models=",".join(settings.models),
                max_rows=settings.max_rows,
                folds=settings.folds,
                repeats=settings.repeats,
                jobs=-1,
                synergyx_epochs=settings.synergyx_epochs,
                output_dir=str(run_dir),
                data_file=str(PROCESSED_V15_S_SUM),
                dataset_label="DrugComb v1.5 · S_sum · quick run",
                target_description="DrugComb v1.5 S_sum",
                target_clip_sigma=3.0,
            )
        )

        with _state_lock:
            _latest_metrics_file = run_dir / "model_comparison.csv"

            _training_state.update(
                status="complete",
                message=(
                    "Evaluation finished. Its comparison is saved; "
                    "the prediction models were left unchanged."
                ),
                rows=report.get("rows_loaded"),
                run_id=run_dir.name,
            )

    except Exception as exc:

        with _state_lock:

            _training_state.update(
                status="error",
                message=str(exc),
            )


@app.post("/api/train", status_code=202)
def start_training(request: TrainRequest):
    global _latest_metrics_file

    run_id = datetime.now(timezone.utc).strftime("quick_%Y%m%dT%H%M%S_%fZ")
    run_dir = RESULTS / "runs" / run_id

    with _state_lock:

        if _training_state.get("status") in ("running", "queued"):

            raise HTTPException(
                status_code=409,
                detail="A training run is already active.",
            )

        _training_state.update(
            status="queued",
            message="Training is starting.",
            run_id=run_id,
        )
        _latest_metrics_file = None

    thread = threading.Thread(
        target=_run_training,
        args=(request, run_dir),
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
def prediction(payload: PredictRequest, request: Request, response: Response):

    from ML.prediction.predict import predict

    user = _current_user(request)
    visitor_hash = None if user else _get_visitor_hash(request, response)
    predictions_used = None
    if not user:
        predictions_used = _reserve_free_prediction(visitor_hash)

    try:

        result = predict(
            payload.drug_a,
            payload.drug_b,
            payload.cell_line,
            payload.feature_set,
            payload.model,
        )

        result = dict(result)
        prediction_id = uuid.uuid4().hex
        with _db_connect() as connection:
            connection.execute(
                """INSERT INTO predictions
                   (id, user_id, visitor_hash, drug_a, drug_b, cell_line, predicted_score,
                    feature_set, model, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    prediction_id,
                    int(user["id"]) if user else None,
                    visitor_hash,
                    payload.drug_a.strip(),
                    payload.drug_b.strip(),
                    payload.cell_line.strip(),
                    _prediction_score(result),
                    str(result.get("feature_set", payload.feature_set)),
                    str(result.get("model", payload.model or "auto")),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

        result["prediction_id"] = prediction_id
        result["account_authenticated"] = bool(user)
        result["predictions_used"] = predictions_used
        result["free_predictions_left"] = (
            None if user else max(FREE_PREDICTION_LIMIT - int(predictions_used or 0), 0)
        )

        return result

    except (
        ValueError,
        FileNotFoundError,
    ) as exc:

        if visitor_hash and predictions_used is not None:
            _release_free_prediction(visitor_hash)

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        if visitor_hash and predictions_used is not None:
            _release_free_prediction(visitor_hash)

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
