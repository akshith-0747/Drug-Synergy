"""Small NumPy-only estimators used when native scikit-learn wheels are blocked.

They implement the same estimator families for local research runs: proximal
L1 regression, randomized regression trees, and AdaBoost.R2-style trees. This
fallback is explicitly reported in results; it does not replace the normal
scikit-learn implementations when those load successfully.
"""
from __future__ import annotations

import numpy as np


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def pearson(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def auc_score(y_true, scores) -> float:
    y = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=float)
    positives = int(y.sum())
    negatives = len(y) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(scores)
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def stratified_folds(labels, n_splits: int, seed: int):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    buckets = [[] for _ in range(n_splits)]
    for label in np.unique(labels):
        members = np.flatnonzero(labels == label)
        rng.shuffle(members)
        for i, member in enumerate(members):
            buckets[i % n_splits].append(int(member))
    all_indices = np.arange(len(labels))
    for test_bucket in buckets:
        test = np.asarray(sorted(test_bucket), dtype=int)
        keep = np.ones(len(labels), dtype=bool)
        keep[test] = False
        yield all_indices[keep], test


def stratified_group_folds(labels, groups, n_splits: int, seed: int):
    """Yield folds that keep every group together and balance target bins."""
    labels = np.asarray(labels)
    groups = np.asarray(groups).astype(str)
    if len(labels) != len(groups):
        raise ValueError("labels and groups must contain the same number of rows.")
    unique_groups, group_ids = np.unique(groups, return_inverse=True)
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Need at least {n_splits} distinct groups for grouped CV; "
            f"found {len(unique_groups)}."
        )

    _, label_ids = np.unique(labels, return_inverse=True)
    class_count = int(label_ids.max()) + 1 if len(label_ids) else 0
    group_class_counts = np.zeros((len(unique_groups), class_count), dtype=np.int64)
    np.add.at(group_class_counts, (group_ids, label_ids), 1)
    group_sizes = group_class_counts.sum(axis=1)
    total_class_counts = group_class_counts.sum(axis=0)
    target_class_counts = total_class_counts / float(n_splits)
    target_size = len(labels) / float(n_splits)

    rng = np.random.default_rng(seed)
    tie_breakers = rng.random(len(unique_groups))
    order = np.lexsort((tie_breakers, -group_sizes))
    fold_class_counts = np.zeros((n_splits, class_count), dtype=np.float64)
    fold_sizes = np.zeros(n_splits, dtype=np.float64)
    assigned_fold = np.full(len(unique_groups), -1, dtype=np.int64)

    for group_id in order:
        costs = np.empty(n_splits, dtype=np.float64)
        for fold in range(n_splits):
            projected_classes = fold_class_counts[fold] + group_class_counts[group_id]
            class_cost = np.mean(
                ((projected_classes - target_class_counts) ** 2)
                / (target_class_counts + 1.0)
            )
            projected_size = fold_sizes[fold] + group_sizes[group_id]
            size_cost = ((projected_size - target_size) / max(target_size, 1.0)) ** 2
            costs[fold] = class_cost + size_cost

        best = np.flatnonzero(np.isclose(costs, costs.min()))
        if len(best) > 1:
            smallest = fold_sizes[best].min()
            best = best[np.isclose(fold_sizes[best], smallest)]
        fold = int(rng.choice(best))
        assigned_fold[group_id] = fold
        fold_class_counts[fold] += group_class_counts[group_id]
        fold_sizes[fold] += group_sizes[group_id]

    row_folds = assigned_fold[group_ids]
    row_indices = np.arange(len(labels))
    for fold in range(n_splits):
        test = row_indices[row_folds == fold]
        train = row_indices[row_folds != fold]
        yield train, test


class ProximalLasso:
    """L1-penalized linear regression solved with proximal gradient descent."""
    def __init__(self, alpha=0.01, max_iter=70, random_state=42):
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        self.random_state = int(random_state)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        self.scale_[self.scale_ < 1e-6] = 1.0
        self.y_mean_ = float(y.mean())
        z = np.asarray((X - self.mean_) / self.scale_, dtype=np.float32)
        target = y - self.y_mean_
        # Power iteration estimates the largest eigenvalue of Z'Z/n.
        rng = np.random.default_rng(self.random_state)
        v = rng.normal(size=z.shape[1]).astype(np.float32)
        v /= max(float(np.linalg.norm(v)), 1e-12)
        eigenvalue = 1.0
        for _ in range(8):
            xv = z @ v
            next_v = z.T @ xv / max(len(y), 1)
            norm = float(np.linalg.norm(next_v))
            if norm < 1e-10:
                break
            v = next_v / norm
            eigenvalue = float(v @ (z.T @ (z @ v)) / max(len(y), 1))
        learning_rate = 0.9 / max(eigenvalue, 1e-4)
        coef = np.zeros(z.shape[1], dtype=np.float32)
        for _ in range(self.max_iter):
            residual = z @ coef - target
            gradient = z.T @ residual / max(len(y), 1)
            updated = coef - learning_rate * gradient
            cutoff = learning_rate * self.alpha
            coef = np.sign(updated) * np.maximum(np.abs(updated) - cutoff, 0.0)
        self.coef_ = coef
        self.n_features_in_ = z.shape[1]
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.mean_) / self.scale_) @ self.coef_ + self.y_mean_


class RandomRegressionTree:
    def __init__(self, max_depth=4, max_features=24, min_leaf=5, seed=42):
        self.max_depth = int(max_depth)
        self.max_features = int(max_features)
        self.min_leaf = int(min_leaf)
        self.rng = np.random.default_rng(seed)
        self.root = None

    @staticmethod
    def _leaf(y, weights, indices):
        w = weights[indices]
        total = float(w.sum())
        return {"value": float(np.dot(w, y[indices]) / total) if total else float(np.mean(y[indices]))}

    def _build(self, X, y, weights, indices, depth):
        if depth <= 0 or len(indices) < 2 * self.min_leaf or np.all(y[indices] == y[indices][0]):
            return self._leaf(y, weights, indices)
        p = X.shape[1]
        features = self.rng.choice(p, size=min(self.max_features, p), replace=False)
        parent_w = weights[indices]
        parent_sum = float(np.dot(parent_w, y[indices]))
        parent_sq = float(np.dot(parent_w, y[indices] ** 2))
        parent_weight = float(parent_w.sum())
        parent_error = parent_sq - parent_sum * parent_sum / max(parent_weight, 1e-12)
        best_gain, best = 0.0, None
        for feature in features:
            values = X[indices, feature]
            low, high = float(values.min()), float(values.max())
            if low >= high:
                continue
            if high - low <= 1.001 and low >= -1e-6 and high <= 1.001:
                thresholds = (0.5,)
            else:
                thresholds = np.unique(np.quantile(values, [0.25, 0.5, 0.75]))
            for threshold in thresholds:
                left_mask = values <= threshold
                if left_mask.sum() < self.min_leaf or (~left_mask).sum() < self.min_leaf:
                    continue
                li, ri = indices[left_mask], indices[~left_mask]
                lw, rw = weights[li], weights[ri]
                lsum, rsum = float(np.dot(lw, y[li])), float(np.dot(rw, y[ri]))
                ltot, rtot = float(lw.sum()), float(rw.sum())
                error = float(np.dot(lw, y[li] ** 2) - lsum * lsum / max(ltot, 1e-12))
                error += float(np.dot(rw, y[ri] ** 2) - rsum * rsum / max(rtot, 1e-12))
                gain = parent_error - error
                if gain > best_gain:
                    best_gain = gain
                    best = (int(feature), float(threshold), li, ri)
        if best is None:
            return self._leaf(y, weights, indices)
        feature, threshold, li, ri = best
        return {"feature": feature, "threshold": threshold,
                "left": self._build(X, y, weights, li, depth - 1),
                "right": self._build(X, y, weights, ri, depth - 1)}

    def fit(self, X, y, sample_weight=None):
        # Keep the training arrays local. Persisting them on every tree makes a
        # fitted forest retain one full copy per estimator and bloats artifacts.
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)
        weights = np.ones(len(y_arr), dtype=np.float64) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        self.root = self._build(X_arr, y_arr, weights, np.arange(len(y_arr)), self.max_depth)
        return self

    @staticmethod
    def _predict_node(node, X, indices, output):
        if len(indices) == 0:
            return
        if "value" in node:
            output[indices] = node["value"]
            return
        left = indices[X[indices, node["feature"]] <= node["threshold"]]
        right = indices[X[indices, node["feature"]] > node["threshold"]]
        RandomRegressionTree._predict_node(node["left"], X, left, output)
        RandomRegressionTree._predict_node(node["right"], X, right, output)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        output = np.empty(len(X), dtype=np.float32)
        self._predict_node(self.root, X, np.arange(len(X)), output)
        return output


class RandomForestFallback:
    def __init__(self, n_estimators=24, max_depth=4, max_features=24, seed=42):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.max_features = int(max_features)
        self.seed = int(seed)
        self.trees = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        rng = np.random.default_rng(self.seed)
        self.trees = []
        for i in range(self.n_estimators):
            sample = rng.integers(0, len(y), size=len(y))
            tree = RandomRegressionTree(self.max_depth, self.max_features, min_leaf=4, seed=self.seed + i)
            tree.fit(X[sample], y[sample])
            self.trees.append(tree)
        return self

    def predict(self, X):
        return np.mean([tree.predict(X) for tree in self.trees], axis=0)


class AdaBoostR2Fallback:
    def __init__(self, n_estimators=24, max_depth=2, max_features=24, seed=42):
        self.n_estimators = int(n_estimators)
        self.max_depth = int(max_depth)
        self.max_features = int(max_features)
        self.seed = int(seed)
        self.trees = []
        self.estimator_weights = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        n = len(y)
        weights = np.full(n, 1.0 / max(n, 1), dtype=np.float64)
        self.trees, self.estimator_weights = [], []
        for i in range(self.n_estimators):
            tree = RandomRegressionTree(self.max_depth, self.max_features, min_leaf=4, seed=self.seed + i)
            tree.fit(X, y, sample_weight=weights)
            prediction = tree.predict(X)
            loss = np.abs(y - prediction)
            max_loss = float(loss.max())
            if max_loss <= 1e-12:
                self.trees.append(tree)
                self.estimator_weights.append(1.0)
                break
            loss /= max_loss
            error = float(np.dot(weights, loss))
            if error >= 0.5:
                if not self.trees:
                    self.trees.append(tree)
                    self.estimator_weights.append(1.0)
                break
            beta = max(error / max(1.0 - error, 1e-12), 1e-8)
            self.trees.append(tree)
            self.estimator_weights.append(float(np.log(1.0 / beta)))
            weights *= np.power(beta, 1.0 - loss)
            weights /= max(float(weights.sum()), 1e-12)
        return self

    def predict(self, X):
        predictions = np.vstack([tree.predict(X) for tree in self.trees])
        weights = np.asarray(self.estimator_weights, dtype=float)
        if predictions.shape[0] == 1:
            return predictions[0]
        order = np.argsort(predictions, axis=0)
        sorted_predictions = np.take_along_axis(predictions, order, axis=0)
        sorted_weights = np.take_along_axis(np.broadcast_to(weights[:, None], predictions.shape), order, axis=0)
        cumulative = np.cumsum(sorted_weights, axis=0)
        midpoint = cumulative[-1] / 2.0
        selected = np.argmax(cumulative >= midpoint, axis=0)
        return sorted_predictions[selected, np.arange(predictions.shape[1])]


def stratified_sample_indices(labels, size: int, seed: int):
    labels = np.asarray(labels)
    if size <= 0 or size >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    exact = [size * len(group) / len(labels) for group in groups]
    quotas = [int(value) for value in exact]
    for i in np.argsort(np.asarray(exact) - np.asarray(quotas))[::-1][:size - sum(quotas)]:
        quotas[int(i)] += 1
    selected = []
    for group, quota in zip(groups, quotas):
        selected.extend(rng.choice(group, size=min(quota, len(group)), replace=False).tolist())
    return np.asarray(sorted(selected), dtype=int)
