"""Model persistence with a joblib preference and stdlib-pickle fallback."""
from __future__ import annotations

import pickle
from pathlib import Path


def dump(value, path, compress=3) -> Path:
    path = Path(path)
    try:
        import joblib
    except Exception:
        output = path.with_suffix(".pkl")
        with output.open("wb") as stream:
            pickle.dump(value, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return output
    joblib.dump(value, path, compress=compress)
    return path


def load(path):
    path = Path(path)
    if path.suffix.lower() == ".pkl":
        with path.open("rb") as stream:
            return pickle.load(stream)
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        with path.open("rb") as stream:
            return pickle.load(stream)
