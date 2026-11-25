"""Train/evaluate S-score models with sklearn or a NumPy-only fallback.

Run from the project root: python -m ML.training.train --quick
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ML.feature_engineering.features import (
    DATA_DIR, ML_DIR, augment_structure_features, build_context,
    PAIR_SCHEMA_CONCAT, PAIR_SCHEMA_SYMMETRIC_SUM_ABSDIFF,
    make_feature_matrix, make_multimodal_feature_matrix, resolve_drug,
)
from ML.models.multilevel import MultilevelStackedRegressor
from ML.models.native_fallback import (
    AdaBoostR2Fallback, ProximalLasso, RandomForestFallback,
    auc_score, pearson, rankdata, stratified_group_folds,
    stratified_sample_indices,
)
from ML.storage import dump

try:
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
    from sklearn.linear_model import SGDRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import (
        GroupShuffleSplit,
        StratifiedGroupKFold,
    )
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor
    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_NOTE = "scikit-learn"
except Exception as exc:
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_NOTE = f"NumPy fallback ({type(exc).__name__}: {exc})"

PROCESSED_FILE = DATA_DIR / "processed" / "matched_drugcomb.csv"
RESULTS_DIR = ML_DIR / "results"
MODEL_DIR = RESULTS_DIR / "models"


def _atomic_write(path: Path, writer) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        writer(temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _write_csv_atomically(frame: pd.DataFrame, path: Path) -> None:
    _atomic_write(path, lambda temporary_path: frame.to_csv(temporary_path, index=False))


def _write_text_atomically(path: Path, content: str) -> None:
    _atomic_write(
        path,
        lambda temporary_path: temporary_path.write_text(content, encoding="utf-8"),
    )


def _model(
    name: str,
    seed: int,
    jobs: int,
    drug_dim: int = 0,
    cell_dim: int = 0,
    synergyx_epochs: int = 120,
):
    if SKLEARN_AVAILABLE:
        if name == "lasso":
            return TransformedTargetRegressor(
                regressor=make_pipeline(
                    StandardScaler(),
                    SGDRegressor(
                        alpha=0.001,
                        loss="squared_error",
                        penalty="l1",
                        learning_rate="constant",
                        eta0=0.001,
                        max_iter=1000,
                        tol=1e-3,
                        average=True,
                        random_state=seed,
                    ),
                ),
                transformer=StandardScaler(),
            )
        if name == "random_forest":
            return RandomForestRegressor(n_estimators=8, max_depth=8, min_samples_leaf=2, n_jobs=jobs, random_state=seed)
        if name == "adaboost":
            return AdaBoostRegressor(
                estimator=DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, max_features="sqrt", random_state=seed),
                n_estimators=12, learning_rate=0.05, loss="square", random_state=seed,
            )
    else:
        if name == "lasso":
            return ProximalLasso(alpha=0.01, max_iter=70, random_state=seed)
        if name == "random_forest":
            return RandomForestFallback(n_estimators=24, max_depth=4, max_features=24, seed=seed)
        if name == "adaboost":
            return AdaBoostR2Fallback(n_estimators=24, max_depth=2, max_features=24, seed=seed)
    if name in {"xgboost", "xgboost_pairwise", "xgboost_regularized"}:
        try:
            from xgboost import XGBRegressor
        except Exception as exc:
            raise ImportError("XGBoost could not load; install a compatible build: pip install xgboost") from exc
        if name == "xgboost_regularized":
            return XGBRegressor(
                n_estimators=320, max_depth=4, learning_rate=0.04,
                min_child_weight=5.0, subsample=0.8, colsample_bytree=0.8,
                reg_lambda=6.0, objective="reg:squarederror",
                tree_method="hist", n_jobs=jobs, random_state=seed, verbosity=0,
            )
        return XGBRegressor(
            n_estimators=180, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=1.0, objective="reg:squarederror",
            tree_method="hist", n_jobs=jobs, random_state=seed, verbosity=0,
        )
    if name == "synergyx":
        try:
            from ML.models.synergyx import SynergyXRegressor
        except Exception as exc:
            raise ImportError("SynergyX needs a working PyTorch install; see ML/requirements-synergyx.txt") from exc
        return SynergyXRegressor(
            drug_dim=drug_dim,
            cell_dim=cell_dim,
            seed=seed,
            epochs=synergyx_epochs,
        )
    raise ValueError(f"Unknown model: {name}")


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = float(np.mean((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - np.sum((y_true - y_pred) ** 2) / total) if total > 0 else float("nan")
    labels = (y_true >= 10).astype(np.int8)
    predicted_labels = (y_pred >= 10).astype(np.int8)
    true_positive = int(np.sum((labels == 1) & (predicted_labels == 1)))
    false_positive = int(np.sum((labels == 0) & (predicted_labels == 1)))
    false_negative = int(np.sum((labels == 1) & (predicted_labels == 0)))
    true_negative = int(np.sum((labels == 0) & (predicted_labels == 0)))
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)

    if labels.any():
        order = np.argsort(y_pred, kind="mergesort")[::-1]
        ranked_labels = labels[order]
        precision_at_rank = np.cumsum(ranked_labels) / np.arange(1, len(labels) + 1)
        average_precision = float(
            np.sum(precision_at_rank * ranked_labels) / int(labels.sum())
        )
    else:
        average_precision = float("nan")

    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "r2": r2,
        "pearson": pearson(y_true, y_pred),
        "spearman": pearson(rankdata(y_true), rankdata(y_pred)),
        "auc_s_ge_10": auc_score(labels, y_pred),
        "accuracy_s_ge_10": float(np.mean(labels == predicted_labels)),
        "balanced_accuracy_s_ge_10": (recall + specificity) / 2.0,
        "precision_s_ge_10": precision,
        "recall_s_ge_10": recall,
        "f1_s_ge_10": f1,
        "average_precision_s_ge_10": average_precision,
    }


def _target_clip_bounds(y: np.ndarray, sigma: float) -> tuple[float, float] | None:
    """Return fold-local mean +/- sigma standard-deviation target limits."""
    if sigma <= 0 or len(y) < 2:
        return None
    values = np.asarray(y, dtype=float)
    center = float(np.mean(values))
    spread = float(np.std(values))
    if not np.isfinite(center) or not np.isfinite(spread) or spread <= 0:
        return None
    return center - sigma * spread, center + sigma * spread


def _pair_groups(rows: pd.DataFrame, context: dict) -> np.ndarray:
    """Return order-independent drug-pair groups for leakage-resistant CV."""
    groups = []
    for drug_a, drug_b in rows[["drug_row", "drug_col"]].itertuples(
        index=False, name=None
    ):
        pair = sorted((
            resolve_drug(drug_a, context),
            resolve_drug(drug_b, context),
        ))
        groups.append("\x1f".join(pair))
    return np.asarray(groups, dtype=object)


def _grouped_splits(
    X: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    folds: int,
    seed: int,
):
    """Split by drug pair, balancing S-score strata where possible."""
    if SKLEARN_AVAILABLE:
        try:
            splitter = StratifiedGroupKFold(
                n_splits=folds,
                shuffle=True,
                random_state=seed,
            )
            return list(splitter.split(X, labels, groups))
        except ValueError:
            # Some sparse feature views cannot populate every stratum in every
            # group fold. The fallback still guarantees pair separation.
            from sklearn.model_selection import GroupKFold

            return list(GroupKFold(n_splits=folds).split(X, labels, groups))
    return list(stratified_group_folds(labels, groups, folds, seed))


def _group_validation_indices(
    groups: np.ndarray,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Create an inner validation split without sharing drug pairs."""
    rows = np.arange(len(groups))
    if len(np.unique(groups)) < 2:
        return rows, np.asarray([], dtype=np.int64)
    if SKLEARN_AVAILABLE:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=fraction,
            random_state=seed,
        )
        return next(splitter.split(rows, groups=groups))

    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    rng.shuffle(unique)
    target_rows = max(1, int(round(len(rows) * fraction)))
    selected = []
    selected_count = 0
    for group in unique:
        selected.append(group)
        selected_count += int(np.sum(groups == group))
        if selected_count >= target_rows and len(selected) < len(unique):
            break
    validation_mask = np.isin(groups, selected)
    return rows[~validation_mask], rows[validation_mask]


def _implementation_for(name: str) -> str:
    if name == "synergyx":
        return "pytorch"
    if name == "lasso":
        return "scikit-learn SGDRegressor (L1-penalized least squares)"
    if name in {"xgboost", "xgboost_pairwise", "xgboost_regularized"}:
        return "xgboost"
    if SKLEARN_AVAILABLE:
        return "scikit-learn"
    return "numpy_fallback"


def _training_parameters(name: str, synergyx_epochs: int) -> dict:
    if name == "random_forest":
        return {"n_estimators": 8, "max_depth": 8, "min_samples_leaf": 2}
    if name == "adaboost":
        return {
            "n_estimators": 12,
            "learning_rate": 0.05,
            "base_tree_max_depth": 2,
            "base_tree_max_features": "sqrt",
        }
    if name == "synergyx":
        return {"max_epochs": synergyx_epochs, "validation": "pair-grouped early stopping"}
    if name == "lasso":
        return {
            "solver": "SGDRegressor",
            "penalty": "L1",
            "alpha": 0.001,
            "target_scaled_within_fit": True,
        }
    return {"preset": "project default"}


def _sample_frame(frame: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame.reset_index(drop=True)
    target = pd.to_numeric(frame["S"], errors="coerce")
    bins = pd.qcut(target.rank(method="first"), q=min(10, max(2, max_rows // 200)), labels=False, duplicates="drop")
    chosen = stratified_sample_indices(np.asarray(bins), max_rows, seed)
    return frame.iloc[chosen].reset_index(drop=True)


def _stratification_labels(y: np.ndarray, folds: int) -> np.ndarray:
    requested_bins = min(10, max(2, len(y) // (folds * 4)))
    return pd.qcut(pd.Series(y).rank(method="first"), q=requested_bins, labels=False, duplicates="drop").to_numpy()


def _save_model(bundle: dict, path: Path):
    return dump(bundle, path, compress=3)


def train(args) -> dict:
    processed_file = Path(getattr(args, "data_file", None) or PROCESSED_FILE)
    if not processed_file.exists():
        raise FileNotFoundError(f"Prepared DrugComb data missing: {processed_file}. Run the data preparation step first.")
    output_dir = getattr(args, "output_dir", None)
    run_results_dir = Path(output_dir) if output_dir else RESULTS_DIR
    run_model_dir = run_results_dir / "models"
    run_results_dir.mkdir(parents=True, exist_ok=True)
    run_model_dir.mkdir(parents=True, exist_ok=True)
    print("Loading feature sources (structures, LINCS signatures, DepMap expression, GDSC IC50)...", flush=True)
    context = build_context(
        include_prism_drs=bool(getattr(args, "include_prism_drs", False))
    )
    print(f"Estimator backend: {SKLEARN_IMPORT_NOTE}", flush=True)
    print(f"Available drugs: structures={len(context['fingerprints']):,}, DS={len(context['drug_signatures']):,}, DRS={len(context['resistance_signatures']):,}; cell lines={len(context['expression']):,}; genes={len(context['gene_vocab']):,}", flush=True)
    raw = pd.read_csv(processed_file, dtype={"block_id": str, "depmap_id": str}, low_memory=False)
    prepared_rows = int(len(raw))
    raw["S"] = pd.to_numeric(raw["S"], errors="coerce")
    raw = raw.dropna(subset=["S", "depmap_id", "drug_row", "drug_col"])
    target_valid_rows = int(len(raw))
    augment_structure_features(raw, context)
    raw = _sample_frame(raw, args.max_rows, 42)
    print(f"Training input rows after S-score stratified sampling: {len(raw):,}", flush=True)

    feature_sets = [x.strip().lower() for x in args.feature_sets.split(",") if x.strip()]
    model_names = [x.strip().lower() for x in args.models.split(",") if x.strip()]
    synergyx_epochs = int(getattr(args, "synergyx_epochs", 120))
    synergyx_validation_fraction = float(
        getattr(args, "synergyx_validation_fraction", 0.15)
    )
    target_clip_sigma = float(getattr(args, "target_clip_sigma", 0.0))
    clip_metric_prefix = f"clip{target_clip_sigma:g}"
    dataset_label = str(getattr(args, "dataset_label", "") or "").strip()
    target_description = str(
        getattr(args, "target_description", "DrugComb S score")
    ).strip()
    all_metrics: list[dict] = []
    audit: dict[str, dict] = {}

    for feature_set in feature_sets:
        start = time.time()
        try:
            X, y, used_rows, exclusions = make_feature_matrix(raw, feature_set, context)
        except ValueError as exc:
            print(f"Skipping {feature_set}: {exc}", flush=True)
            continue
        model_matrices = {PAIR_SCHEMA_CONCAT: X}
        if feature_set == "structure" and "xgboost_pairwise" in model_names:
            X_pairwise, y_pairwise, pairwise_rows, _ = make_feature_matrix(
                raw,
                feature_set,
                context,
                pair_schema=PAIR_SCHEMA_SYMMETRIC_SUM_ABSDIFF,
            )
            if not np.array_equal(used_rows, pairwise_rows) or not np.array_equal(y, y_pairwise):
                raise RuntimeError("Pairwise structure features changed row alignment.")
            model_matrices[PAIR_SCHEMA_SYMMETRIC_SUM_ABSDIFF] = X_pairwise
        groups = _pair_groups(used_rows, context)
        audit[feature_set] = exclusions | {
            "feature_rows": int(X.shape[0]),
            "feature_columns": int(X.shape[1]),
        "unique_drug_pairs": int(len(np.unique(groups))),
        "validation_group": "unordered drug pair",
        }
        print(f"\n{feature_set.upper()}: {X.shape[0]:,} usable rows x {X.shape[1]:,} features", flush=True)
        if len(y) < args.folds or len(np.unique(groups)) < args.folds:
            print(
                f"Skipping {feature_set}: fewer usable rows or drug-pair groups "
                "than CV folds.",
                flush=True,
            )
            continue
        labels = _stratification_labels(y, args.folds)
        feature_metric_rows: dict[str, list[dict]] = {name: [] for name in model_names}
        best_epochs_by_model: dict[str, list[int]] = {name: [] for name in model_names}

        for repeat in range(args.repeats):
            seed = 42 + repeat
            splits = _grouped_splits(X, labels, groups, args.folds, seed)
            for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                for name in model_names:
                    if name == "xgboost_pairwise" and feature_set != "structure":
                        continue
                    try:
                        model_seed = seed + fold
                        fold_clip_bounds = _target_clip_bounds(
                            y[train_idx], target_clip_sigma
                        )
                        train_target = y[train_idx]
                        test_target = y[test_idx]
                        if fold_clip_bounds is not None:
                            clip_low, clip_high = fold_clip_bounds
                            train_target = np.clip(train_target, clip_low, clip_high)
                            test_target = np.clip(test_target, clip_low, clip_high)
                        pair_schema = (
                            PAIR_SCHEMA_SYMMETRIC_SUM_ABSDIFF
                            if name == "xgboost_pairwise"
                            else PAIR_SCHEMA_CONCAT
                        )
                        X_model = model_matrices[pair_schema]
                        drug_dim = (X_model.shape[1] - context["cell_dim"]) // 2
                        if name == "synergyx":
                            inner_train, inner_valid = _group_validation_indices(
                                groups[train_idx],
                                synergyx_validation_fraction,
                                model_seed,
                            )
                            if len(inner_valid):
                                tuning_model = _model(
                                    name,
                                    model_seed,
                                    args.jobs,
                                    drug_dim=drug_dim,
                                    cell_dim=context["cell_dim"],
                                    synergyx_epochs=synergyx_epochs,
                                )
                                tuning_model.fit(
                                    X_model[train_idx][inner_train],
                                    train_target[inner_train],
                                    validation_data=(
                                        X_model[train_idx][inner_valid],
                                        train_target[inner_valid],
                                    ),
                                )
                                selected_epochs = int(
                                    tuning_model.best_epoch_ or synergyx_epochs
                                )
                            else:
                                selected_epochs = synergyx_epochs
                            best_epochs_by_model[name].append(selected_epochs)
                            model = _model(
                                name,
                                model_seed,
                                args.jobs,
                                drug_dim=drug_dim,
                                cell_dim=context["cell_dim"],
                                synergyx_epochs=selected_epochs,
                            )
                        else:
                            model = _model(
                                name,
                                model_seed,
                                args.jobs,
                                drug_dim=drug_dim,
                                cell_dim=context["cell_dim"],
                            )
                        model.fit(X_model[train_idx], train_target)
                        predicted = model.predict(X_model[test_idx])
                    except ImportError as exc:
                        if repeat == 0 and fold == 1:
                            print(f"Skipping {name}: {exc}", flush=True)
                        continue
                    predicted = np.asarray(predicted, dtype=float)
                    scores = _metrics(y[test_idx], predicted)
                    if fold_clip_bounds is not None:
                        clipped_scores = _metrics(
                            test_target,
                            np.clip(predicted, clip_low, clip_high),
                        )
                        scores.update({
                            f"{clip_metric_prefix}_{metric}": value
                            for metric, value in clipped_scores.items()
                        })
                        scores["target_clip_lower"] = float(clip_low)
                        scores["target_clip_upper"] = float(clip_high)
                    result = {"feature_set": feature_set, "model": name,
                              "pair_schema": pair_schema,
                              "implementation": _implementation_for(name),
                              "split_strategy": "stratified_group_cv_by_unordered_drug_pair",
                              "dataset_label": dataset_label or None,
                              "target_description": target_description,
                              "target_clip_sigma": target_clip_sigma,
                              "rows_evaluated": int(len(y)),
                              "unique_pairs_evaluated": int(len(np.unique(groups))),
                              "repeat": repeat + 1, "fold": fold, **scores}
                    all_metrics.append(result)
                    feature_metric_rows[name].append(result)
                    clipped_note = (
                        f" {clip_metric_prefix}_RMSE={scores[f'{clip_metric_prefix}_rmse']:.3f}"
                        if fold_clip_bounds is not None else ""
                    )
                    print(f"  repeat {repeat + 1}/{args.repeats} fold {fold}/{args.folds} {name}: RMSE={scores['rmse']:.3f}{clipped_note} R2={scores['r2']:.3f}", flush=True)

        best_name = None
        best_score = float("inf")
        selection_metric = (
            f"{clip_metric_prefix}_rmse" if target_clip_sigma > 0 else "rmse"
        )
        for name, entries in feature_metric_rows.items():
            rmse = [
                entry[selection_metric]
                for entry in entries
                if selection_metric in entry and np.isfinite(entry[selection_metric])
            ]
            if rmse and float(np.mean(rmse)) < best_score:
                best_name, best_score = name, float(np.mean(rmse))
        for name, entries in feature_metric_rows.items():
            if not entries:
                continue
            pair_schema = (
                PAIR_SCHEMA_SYMMETRIC_SUM_ABSDIFF
                if name == "xgboost_pairwise"
                else PAIR_SCHEMA_CONCAT
            )
            X_model = model_matrices[pair_schema]
            try:
                selected_epochs = int(np.median(best_epochs_by_model[name])) \
                    if name == "synergyx" and best_epochs_by_model[name] \
                    else synergyx_epochs
                final_model = _model(
                    name,
                    42,
                    args.jobs,
                    drug_dim=(X_model.shape[1] - context["cell_dim"]) // 2,
                    cell_dim=context["cell_dim"],
                    synergyx_epochs=selected_epochs,
                )
                full_clip_bounds = _target_clip_bounds(y, target_clip_sigma)
                final_target = y
                if full_clip_bounds is not None:
                    final_target = np.clip(y, *full_clip_bounds)
                final_model.fit(X_model, final_target)
            except ImportError:
                continue
            bundle = {"model": final_model, "feature_set": feature_set, "model_name": name,
                      "pair_schema": pair_schema,
                      "implementation": _implementation_for(name),
                      "training_parameters": _training_parameters(name, selected_epochs),
                      "deployment_status": (
                          "experimental" if name in {"xgboost_pairwise", "xgboost_regularized"} else "candidate"
                      ),
                      "drug_dim": int((X.shape[1] - context["cell_dim"]) // 2),
                      "cell_dim": int(context["cell_dim"]), "n_train_rows": int(len(y)),
                      "target": target_description,
                      "dataset_label": dataset_label or None,
                      "target_clip": (
                          {
                              "method": "mean +/- n standard deviations",
                              "sigma": target_clip_sigma,
                              "lower": float(full_clip_bounds[0]),
                              "upper": float(full_clip_bounds[1]),
                          }
                          if full_clip_bounds is not None else None
                      )}
            saved = _save_model(bundle, run_model_dir / f"{feature_set}__{name}.joblib")
            print(f"Saved model: {saved}", flush=True)
        print(f"Best observed {feature_set} model: {best_name} (mean CV {selection_metric}={best_score:.3f}); elapsed {time.time()-start:.1f}s", flush=True)

    multilevel_requested = bool(
        getattr(args, "include_multilevel", False)
        and {"structure", "ds", "drs"}.issubset(feature_sets)
    )
    multilevel_trained = False
    if multilevel_requested:
        try:
            X_multi, y_multi, used_multi, multi_coverage = make_multimodal_feature_matrix(raw, context)
        except ValueError as exc:
            print(f"Skipping multilevel stack: {exc}", flush=True)
        else:
            audit["multilevel"] = multi_coverage
            groups_multi = _pair_groups(used_multi, context)
            audit["multilevel"]["unique_drug_pairs"] = int(len(np.unique(groups_multi)))
            audit["multilevel"]["validation_group"] = "unordered drug pair"
            view_dimensions = multi_coverage["view_dimensions"]
            print(
                f"\nMULTILEVEL: {len(y_multi):,} complete-case rows x {X_multi.shape[1]:,} fused features "
                f"(views: {', '.join(map(str, view_dimensions))})",
                flush=True,
            )
            if len(y_multi) < max(3, args.folds) or len(np.unique(groups_multi)) < args.folds:
                print(
                    "Skipping multilevel stack: fewer complete-case rows or "
                    "drug-pair groups than CV folds.",
                    flush=True,
                )
            else:
                labels_multi = _stratification_labels(y_multi, args.folds)
                for repeat in range(args.repeats):
                    seed = 42 + repeat
                    splits = _grouped_splits(
                        X_multi,
                        labels_multi,
                        groups_multi,
                        args.folds,
                        seed,
                    )
                    for fold, (train_idx, test_idx) in enumerate(splits, start=1):
                        model = MultilevelStackedRegressor(
                            view_dimensions=view_dimensions,
                            seed=seed + fold,
                            inner_splits=min(3, len(np.unique(groups_multi[train_idx]))),
                            jobs=args.jobs,
                        )
                        model.fit(
                            X_multi[train_idx],
                            y_multi[train_idx],
                            groups=groups_multi[train_idx],
                        )
                        scores = _metrics(y_multi[test_idx], model.predict(X_multi[test_idx]))
                        all_metrics.append({
                            "feature_set": "multilevel",
                            "model": "stacked_random_forest",
                            "pair_schema": PAIR_SCHEMA_CONCAT,
                            "implementation": model.estimator_backend_ or _implementation_for("stacked_random_forest"),
                            "split_strategy": "stratified_group_cv_by_unordered_drug_pair",
                            "rows_evaluated": int(len(y_multi)),
                            "unique_pairs_evaluated": int(len(np.unique(groups_multi))),
                            "repeat": repeat + 1,
                            "fold": fold,
                            **scores,
                        })
                        print(
                            f"  repeat {repeat + 1}/{args.repeats} fold {fold}/{args.folds} "
                            f"stacked_random_forest: RMSE={scores['rmse']:.3f} R2={scores['r2']:.3f}",
                            flush=True,
                        )

                final_model = MultilevelStackedRegressor(
                    view_dimensions=view_dimensions,
                    seed=42,
                    inner_splits=min(3, len(np.unique(groups_multi))),
                    jobs=args.jobs,
                )
                final_model.fit(X_multi, y_multi, groups=groups_multi)
                bundle = {
                    "model": final_model,
                    "feature_set": "multilevel",
                    "feature_sets": ["structure", "ds", "drs"],
                    "model_name": "stacked_random_forest",
                    "implementation": final_model.estimator_backend_ or _implementation_for("stacked_random_forest"),
                    "view_dimensions": view_dimensions,
                    "n_train_rows": int(len(y_multi)),
                    "target": "DrugComb S score",
                }
                saved = _save_model(bundle, run_model_dir / "multilevel__stacked_random_forest.joblib")
                print(f"Saved multilevel model: {saved}", flush=True)
                multilevel_trained = True

    if not all_metrics:
        raise RuntimeError("No models were trained. Check feature coverage and installed dependencies.")
    metric_frame = pd.DataFrame(all_metrics)
    _write_csv_atomically(metric_frame, run_results_dir / "cv_fold_metrics.csv")
    summary_rows = []
    for (feature_set, model, pair_schema, backend), group in metric_frame.groupby(
        ["feature_set", "model", "pair_schema", "implementation"]
    ):
        entry = {"feature_set": feature_set, "model": model,
                 "pair_schema": pair_schema, "implementation": backend,
                 "deployment_status": (
                     "experimental" if model == "xgboost_pairwise" else "cv_candidate"
                 ),
                 "split_strategy": "stratified_group_cv_by_unordered_drug_pair",
                 "dataset_label": dataset_label or None,
                 "target_description": target_description,
                 "folds_evaluated": int(len(group)),
                 "rows_evaluated": int(group["rows_evaluated"].max())
                     if "rows_evaluated" in group else None,
                 "unique_pairs_evaluated": int(group["unique_pairs_evaluated"].max())
                     if "unique_pairs_evaluated" in group else None}
        metrics_to_summarize = [
            "mse", "rmse", "r2", "pearson", "spearman", "auc_s_ge_10",
            "accuracy_s_ge_10", "balanced_accuracy_s_ge_10",
            "precision_s_ge_10", "recall_s_ge_10", "f1_s_ge_10",
            "average_precision_s_ge_10",
        ]
        if target_clip_sigma > 0:
            metrics_to_summarize.extend(
                f"{clip_metric_prefix}_{metric}"
                for metric in (
                    "mse", "rmse", "r2", "pearson", "spearman",
                    "auc_s_ge_10", "accuracy_s_ge_10",
                    "balanced_accuracy_s_ge_10", "precision_s_ge_10",
                    "recall_s_ge_10", "f1_s_ge_10",
                    "average_precision_s_ge_10",
                )
            )
            entry["target_clip_sigma"] = target_clip_sigma
            for bound in ("target_clip_lower", "target_clip_upper"):
                if bound in group:
                    entry[bound] = float(group[bound].mean())
        for metric in metrics_to_summarize:
            values = group[metric].dropna().to_numpy(dtype=float)
            mean = float(np.mean(values)) if len(values) else float("nan")
            margin = float(1.96 * np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
            entry[metric] = mean
            entry[f"{metric}_ci95_low"] = mean - margin if np.isfinite(margin) else float("nan")
            entry[f"{metric}_ci95_high"] = mean + margin if np.isfinite(margin) else float("nan")
        summary_rows.append(entry)
    _write_csv_atomically(pd.DataFrame(summary_rows), run_results_dir / "model_comparison.csv")
    context_path = dump(context, run_model_dir / "feature_context.joblib", compress=3)
    report = {
        "target": "S", "target_description": target_description,
        "dataset_label": dataset_label or None,
        "target_clip_sigma": target_clip_sigma,
        "rows_loaded": target_valid_rows, "prepared_rows": prepared_rows,
        "target_valid_rows": target_valid_rows, "sampled_rows": int(len(raw)),
        "max_rows_requested": int(args.max_rows),
        "sampling_method": "S-score quantile-stratified sample before feature-specific coverage filtering",
        "feature_coverage": audit,
        "feature_sets": feature_sets + (["multilevel"] if multilevel_trained else []),
        "models_requested": model_names + (["stacked_random_forest"] if multilevel_requested else []),
        "models_trained": sorted(metric_frame["model"].unique().tolist()),
        "experimental_models": [
            name for name in sorted(metric_frame["model"].unique().tolist())
            if name in {"xgboost_pairwise", "xgboost_regularized"}
        ],
        "models_skipped": sorted(
            set(model_names + (["stacked_random_forest"] if multilevel_requested else []))
            - set(metric_frame["model"].unique().tolist())
        ),
        "ds_source": context["ds_source"],
        "drs_source": context.get(
            "drs_source",
            "project-local integrated GDSC columns in DrugComb table",
        ),
        "drs_source_audit": context.get("drs_source_audit", {}),
        "estimator_backend": "mixed; see per-model implementation fields",
        "structure_representation": context["structure_representation"],
        "structure_backend_version": context["structure_backend_version"],
        "folds": args.folds, "repeats": args.repeats, "synergy_threshold": 10,
        "validation_protocol": "StratifiedGroupKFold grouped by unordered canonical drug pair; the same drug pair never appears in both train and validation folds.",
        "grouping_key": "unordered canonical drug pair, across all cell lines",
        "output_directory": str(run_results_dir),
        "notes": [
            "DS uses the official EH3226 quantitative Level 5 signatures when the prepared table is present; profiles are averaged across cell lines over the shared 256-gene vocabulary. The signed GMT is only a fallback.",
            "The paper reports chemical structure-based features but does not identify a fingerprint algorithm; this implementation uses RDKit Morgan fingerprints (radius 2, 1024 bits).",
            "DRS uses fitted GDSC monotherapy ln(IC50) values and baseline DepMap expression; GDSC2 is preferred over GDSC1 and at least 10 matched cell lines are required per drug. With --include-prism-drs, quality-filtered PRISM response signatures are used only as a fallback for drugs without GDSC signatures; the distinct assay values are not pooled.",
            "Scores use pair-grouped cross-validation; they estimate performance on drug pairs absent from training, but do not establish performance on entirely unseen drugs or independent studies.",
            "SynergyX is an attention-based approximation because the paper does not provide enough implementation detail for exact reproduction. It uses shared drug encoding, symmetric pair pooling, training-fold normalization, and pair-grouped validation for early stopping.",
            "SynergyX uses PyTorch; XGBoost uses XGBoost; other estimator backend values are recorded per model.",
            "Threshold metrics classify S >= 10 and are reported separately from continuous S-score regression metrics.",
            "The custom multilevel stack is an optional project extension, not part of the paper's required model comparison; include it only with --include-multilevel.",
            "The pairwise structure XGBoost is experimental: lower CV error alone does not promote it for prediction; confirm improvement on held-out drug pairs first.",
        ],
    }
    if target_clip_sigma > 0:
        report["notes"].append(
            f"Target winsorization at mean +/- {target_clip_sigma:g} standard deviations is fitted within each outer training fold; the report includes raw-score and clipped-score metrics. Each saved candidate bundle records its own fit-data limits."
        )
    _write_text_atomically(run_results_dir / "run_report.json", json.dumps(report, indent=2))
    print(f"\nSaved comparison: {run_results_dir / 'model_comparison.csv'}", flush=True)
    print(f"Saved fold metrics: {run_results_dir / 'cv_fold_metrics.csv'}", flush=True)
    print(f"Saved feature context: {context_path}", flush=True)
    if run_results_dir.resolve() == RESULTS_DIR.resolve():
        try:
            from ML.evaluation.plot_results import main as plot_results
            plot_results()
        except Exception as exc:
            print(f"Plot generation skipped: {exc}", flush=True)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Compare models for DrugComb S-score prediction.")
    parser.add_argument("--feature-sets", default="structure,ds,drs", help="Comma-separated: structure,ds,drs")
    parser.add_argument("--models", default="lasso,random_forest,adaboost,xgboost,xgboost_pairwise,synergyx",
                        help="lasso,random_forest,adaboost,xgboost,xgboost_regularized,xgboost_pairwise,synergyx (PyTorch required for synergyx)")
    parser.add_argument("--include-multilevel", action="store_true",
                        help="Also train the custom multilevel stack; this is an optional extension, not a paper model.")
    parser.add_argument("--include-prism-drs", action="store_true",
                        help="Use quality-filtered DepMap PRISM signatures as DRS fallback features for drugs missing GDSC signatures.")
    parser.add_argument("--max-rows", type=int, default=20000, help="Target-stratified sample size; 0 uses all rows")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--output-dir", default=None,
                        help="Write this evaluation and its models to a separate directory instead of replacing the main saved results.")
    parser.add_argument("--data-file", default=str(PROCESSED_FILE),
                        help="Prepared CSV to evaluate; defaults to ML/data/processed/matched_drugcomb.csv.")
    parser.add_argument("--dataset-label", default="",
                        help="Short dataset/version label recorded in metrics and reports.")
    parser.add_argument("--target-description", default="DrugComb S score",
                        help="Human-readable target name recorded in metrics and bundles.")
    parser.add_argument("--target-clip-sigma", type=float, default=0.0,
                        help="Winsorize targets at fold-local mean +/- this many standard deviations (0 disables).")
    parser.add_argument("--synergyx-epochs", type=int, default=120, help="Maximum SynergyX epochs; grouped validation selects the stopping epoch")
    parser.add_argument("--synergyx-validation-fraction", type=float, default=0.15,
                        help="Fraction of pair groups used for SynergyX early stopping within each CV training fold")
    parser.add_argument("--quick", action="store_true", help="Small check: 2,000 rows, 3 folds, 1 repeat")
    args = parser.parse_args()
    if args.quick:
        args.max_rows, args.folds, args.repeats = 2000, 3, 1
        args.models = "lasso,random_forest,adaboost,xgboost,xgboost_pairwise"
    if args.folds < 2 or args.repeats < 1:
        parser.error("folds must be at least 2 and repeats at least 1")
    if not 0.05 <= args.synergyx_validation_fraction <= 0.4:
        parser.error("synergyx-validation-fraction must be between 0.05 and 0.4")
    if args.synergyx_epochs < 1:
        parser.error("synergyx-epochs must be at least 1")
    return args


if __name__ == "__main__":
    train(parse_args())

