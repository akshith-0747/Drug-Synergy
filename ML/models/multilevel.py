"""Late-fusion model for the three drug-combination feature views.

Level 1 fits one Random Forest specialist to each of structure, DS, and DRS.
Level 2 learns a ridge combiner from out-of-fold specialist predictions so
the meta learner never sees in-sample predictions during its fit.
"""
from __future__ import annotations

import numpy as np

from ML.models.native_fallback import stratified_group_folds


class MultilevelStackedRegressor:
    """Three modality specialists followed by an out-of-fold ridge combiner."""

    def __init__(
        self,
        view_dimensions: list[int] | tuple[int, ...],
        seed: int = 42,
        inner_splits: int = 3,
        jobs: int = -1,
        n_estimators: int = 64,
    ):
        self.view_dimensions = tuple(int(size) for size in view_dimensions)
        self.seed = int(seed)
        self.inner_splits = int(inner_splits)
        self.jobs = int(jobs)
        self.n_estimators = int(n_estimators)
        self.estimators_: list[object] = []
        self.coef_: np.ndarray | None = None
        self.intercept_: float | None = None
        self.estimator_backend_: str | None = None

    def _new_estimator(self, seed: int):
        try:
            from sklearn.ensemble import RandomForestRegressor

            self.estimator_backend_ = "scikit-learn"
            return RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=18,
                min_samples_leaf=2,
                n_jobs=self.jobs,
                random_state=seed,
            )
        except Exception:
            from ML.models.native_fallback import RandomForestFallback

            self.estimator_backend_ = "numpy_fallback"
            return RandomForestFallback(
                n_estimators=24,
                max_depth=4,
                max_features=24,
                seed=seed,
            )

    def _slices(self):
        offsets = np.cumsum((0, *self.view_dimensions))
        return [slice(int(start), int(end)) for start, end in zip(offsets[:-1], offsets[1:])]

    def fit(self, X: np.ndarray, y: np.ndarray, groups=None):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or X.shape[0] != len(y):
            raise ValueError("X must be a 2D matrix with one row per target value.")
        if len(self.view_dimensions) < 2 or any(size < 1 for size in self.view_dimensions):
            raise ValueError("At least two non-empty feature views are required.")
        if sum(self.view_dimensions) != X.shape[1]:
            raise ValueError("Feature view dimensions do not match the input matrix.")
        if len(y) < 3:
            raise ValueError("The multilevel model needs at least three complete-case rows.")

        if groups is not None:
            groups = np.asarray(groups).astype(str)
            if len(groups) != len(y):
                raise ValueError("groups must contain one value per training row.")
            n_splits = min(self.inner_splits, len(np.unique(groups)))
            if n_splits < 2:
                raise ValueError("The multilevel stack needs at least two pair groups for OOF fitting.")
            bin_count = min(10, max(2, len(y) // (n_splits * 4)))
            bins = np.digitize(
                y,
                np.unique(np.quantile(y, np.linspace(0.0, 1.0, bin_count + 1)[1:-1])),
            )
            folds = [
                valid_idx
                for _, valid_idx in stratified_group_folds(
                    bins,
                    groups,
                    n_splits,
                    self.seed,
                )
            ]
        else:
            n_splits = min(self.inner_splits, len(y))
            order = np.random.default_rng(self.seed).permutation(len(y))
            folds = list(np.array_split(order, n_splits))
        view_slices = self._slices()
        oof = np.full((len(y), len(view_slices)), np.nan, dtype=np.float64)

        for fold_number, valid_idx in enumerate(folds):
            train_mask = np.ones(len(y), dtype=bool)
            train_mask[valid_idx] = False
            train_idx = np.flatnonzero(train_mask)
            if not len(train_idx) or not len(valid_idx):
                continue
            for view_index, columns in enumerate(view_slices):
                estimator = self._new_estimator(self.seed + fold_number + view_index * 101)
                estimator.fit(X[train_idx, columns], y[train_idx])
                oof[valid_idx, view_index] = np.asarray(
                    estimator.predict(X[valid_idx, columns]), dtype=np.float64
                )

        if not np.isfinite(oof).all():
            raise RuntimeError("Out-of-fold specialist predictions were incomplete.")

        # Solve ridge regression with an unpenalized intercept. This keeps the
        # meta layer available even when scikit-learn's native extensions fail.
        design = np.column_stack((oof, np.ones(len(oof), dtype=np.float64)))
        penalty = np.eye(design.shape[1], dtype=np.float64)
        penalty[-1, -1] = 0.0
        try:
            weights = np.linalg.solve(design.T @ design + penalty, design.T @ y)
        except np.linalg.LinAlgError:
            weights = np.linalg.lstsq(design.T @ design + penalty, design.T @ y, rcond=None)[0]
        self.coef_ = weights[:-1]
        self.intercept_ = float(weights[-1])

        self.estimators_ = []
        for view_index, columns in enumerate(view_slices):
            estimator = self._new_estimator(self.seed + 10_000 + view_index)
            estimator.fit(X[:, columns], y)
            self.estimators_.append(estimator)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None or self.intercept_ is None or not self.estimators_:
            raise ValueError("The multilevel model has not been fitted.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != sum(self.view_dimensions):
            raise ValueError("Prediction features do not match the fitted view dimensions.")
        specialist_predictions = np.column_stack(
            [estimator.predict(X[:, columns]) for estimator, columns in zip(self.estimators_, self._slices())]
        )
        return specialist_predictions @ self.coef_ + self.intercept_
