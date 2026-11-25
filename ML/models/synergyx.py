
"""SynergyX-inspired cross-modal attention model.

The paper names SynergyX but does not provide enough implementation details
to reproduce the original model exactly. This implementation therefore
provides a transparent attention-based approximation for comparison with
LASSO, Random Forest, AdaBoost, and XGBoost.

Input layout:
    [drug A features, drug B features, cell-line features]

The class follows the sklearn-style fit/predict interface used by ML.training.train.
"""

from __future__ import annotations

import numpy as np
import copy

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError(
        "PyTorch is required for the SynergyX-inspired model. "
        "Install it with: pip install -r ML/requirements-synergyx.txt"
    ) from exc


class _CrossModalNetwork(nn.Module):
    """Small cross-modal attention network."""

    def __init__(
        self,
        drug_dim: int,
        cell_dim: int,
        hidden_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.10,
    ):
        super().__init__()

        # Share weights so both compounds use the same representation.
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.cell_encoder = nn.Sequential(
            nn.Linear(cell_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.attention_norm = nn.LayerNorm(hidden_dim)

        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        drug_a: torch.Tensor,
        drug_b: torch.Tensor,
        cell: torch.Tensor,
    ) -> torch.Tensor:

        if not hasattr(self, "drug_encoder"):
            # Read legacy SynergyX bundles created before the symmetric
            # architecture was introduced. New training always uses the path
            # below; this branch keeps saved predictions loadable in the meantime.
            drug_a_token = self.drug_a_encoder(drug_a)
            drug_b_token = self.drug_b_encoder(drug_b)
            cell_token = self.cell_encoder(cell)
            tokens = torch.stack([drug_a_token, drug_b_token, cell_token], dim=1)
            attended, _ = self.attention(
                tokens,
                tokens,
                tokens,
                need_weights=False,
            )
            fused = self.attention_norm(tokens + attended)
            return self.regression_head(fused.reshape(fused.shape[0], -1)).squeeze(-1)

        drug_a_token = self.drug_encoder(drug_a)
        drug_b_token = self.drug_encoder(drug_b)
        cell_token = self.cell_encoder(cell)

        # Three modalities become three tokens:
        # drug A, drug B, and cell line.
        tokens = torch.stack(
            [
                drug_a_token,
                drug_b_token,
                cell_token,
            ],
            dim=1,
        )

        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            need_weights=False,
        )

        fused = self.attention_norm(tokens + attended)

        # Symmetric pair pooling makes the prediction invariant to drug order.
        drug_tokens = fused[:, :2, :]
        drug_mean = drug_tokens.mean(dim=1)
        drug_difference = torch.abs(drug_tokens[:, 0, :] - drug_tokens[:, 1, :])
        drug_interaction = drug_tokens[:, 0, :] * drug_tokens[:, 1, :]
        cell_context = fused[:, 2, :]
        fused = torch.cat(
            [drug_mean, drug_difference, drug_interaction, cell_context],
            dim=1,
        )

        return self.regression_head(fused).squeeze(-1)


class SynergyXRegressor:
    """SynergyX-inspired attention regressor.

    The class intentionally follows the sklearn-style API:

        model = SynergyXRegressor(...)
        model.fit(X, y)
        predictions = model.predict(X)

    Parameters
    ----------
    drug_dim:
        Number of features representing one drug.

    cell_dim:
        Number of cell-line features.

    seed:
        Random seed for reproducibility.

    epochs:
        Number of training epochs.

    batch_size:
        Mini-batch size.

    learning_rate:
        AdamW learning rate.

    weight_decay:
        AdamW weight-decay value.
    """

    def __init__(
        self,
        drug_dim: int,
        cell_dim: int,
        seed: int = 42,
        epochs: int = 120,
        batch_size: int = 256,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        patience: int = 15,
        min_delta: float = 1e-4,
    ):
        self.drug_dim = int(drug_dim)
        self.cell_dim = int(cell_dim)

        self.seed = int(seed)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)

        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.patience = int(patience)
        self.min_delta = float(min_delta)

        self.network: _CrossModalNetwork | None = None
        self.device_name: str | None = None
        self.input_mean_: np.ndarray | None = None
        self.input_scale_: np.ndarray | None = None
        self.target_mean_: float | None = None
        self.target_scale_: float | None = None
        self.best_epoch_: int | None = None

    @property
    def input_dim(self) -> int:
        """Total number of expected input features."""
        return (2 * self.drug_dim) + self.cell_dim

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Validate and convert an input feature matrix."""

        X = np.asarray(X, dtype=np.float32)

        if X.ndim != 2:
            raise ValueError(
                f"Expected a 2D feature matrix, got shape {X.shape}."
            )

        if X.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} features "
                f"(2 Ã— drug_dim + cell_dim), got {X.shape[1]}."
            )

        if not np.isfinite(X).all():
            raise ValueError(
                "Input feature matrix contains NaN or infinite values."
            )

        return X

    def _split_features(self, X: np.ndarray):
        """Split [drug A, drug B, cell] features."""

        X = self._validate_input(X)

        drug_a_end = self.drug_dim
        drug_b_end = self.drug_dim * 2

        drug_a = X[:, :drug_a_end]
        drug_b = X[:, drug_a_end:drug_b_end]
        cell = X[:, drug_b_end:]

        return drug_a, drug_b, cell

    def _create_network(self) -> _CrossModalNetwork:
        """Create a fresh neural network."""

        return _CrossModalNetwork(
            drug_dim=self.drug_dim,
            cell_dim=self.cell_dim,
            hidden_dim=64,
            num_heads=4,
            dropout=0.10,
        )

    def fit(self, X, y, validation_data=None):
        """Train the model."""

        X = self._validate_input(X)
        y = np.asarray(y, dtype=np.float32).reshape(-1)

        if len(X) != len(y):
            raise ValueError(
                f"X and y have different numbers of rows: "
                f"{len(X)} vs {len(y)}."
            )

        if len(y) == 0:
            raise ValueError("Cannot train SynergyX-inspired model on zero rows.")

        if not np.isfinite(y).all():
            raise ValueError(
                "Target vector contains NaN or infinite values."
            )

        if validation_data is not None:
            X_valid, y_valid = validation_data
            X_valid = self._validate_input(X_valid)
            y_valid = np.asarray(y_valid, dtype=np.float32).reshape(-1)
            if len(X_valid) != len(y_valid) or len(y_valid) == 0:
                raise ValueError("Validation features and targets must have matching non-zero rows.")
            if not np.isfinite(y_valid).all():
                raise ValueError("Validation targets contain NaN or infinite values.")
        else:
            X_valid = None
            y_valid = None

        # Normalize from training data only; validation and prediction reuse
        # these statistics, avoiding information from held-out rows.
        self.input_mean_ = X.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.input_scale_ = X.std(axis=0, dtype=np.float64).astype(np.float32)
        self.input_scale_[self.input_scale_ < 1e-6] = 1.0
        X = (X - self.input_mean_) / self.input_scale_
        self.target_mean_ = float(np.mean(y, dtype=np.float64))
        self.target_scale_ = float(np.std(y, dtype=np.float64))
        if self.target_scale_ < 1e-6:
            self.target_scale_ = 1.0
        y = (y - self.target_mean_) / self.target_scale_
        if X_valid is not None:
            X_valid = (X_valid - self.input_mean_) / self.input_scale_
            y_valid = (y_valid - self.target_mean_) / self.target_scale_

        # Reproducibility.
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.device_name = str(device)

        self.network = self._create_network().to(device)

        optimizer = torch.optim.AdamW(
            self.network.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        loss_function = nn.MSELoss()

        drug_a, drug_b, cell = self._split_features(X)

        drug_a_tensor = torch.as_tensor(
            drug_a,
            dtype=torch.float32,
            device=device,
        )

        drug_b_tensor = torch.as_tensor(
            drug_b,
            dtype=torch.float32,
            device=device,
        )

        cell_tensor = torch.as_tensor(
            cell,
            dtype=torch.float32,
            device=device,
        )

        target_tensor = torch.as_tensor(
            y,
            dtype=torch.float32,
            device=device,
        )

        rng = np.random.default_rng(self.seed)

        if X_valid is not None:
            valid_drug_a, valid_drug_b, valid_cell = self._split_features(X_valid)
            valid_inputs = [
                torch.as_tensor(values, dtype=torch.float32, device=device)
                for values in (valid_drug_a, valid_drug_b, valid_cell)
            ]
            valid_target = torch.as_tensor(y_valid, dtype=torch.float32, device=device)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=max(2, self.patience // 3),
                min_lr=1e-5,
            )
        else:
            valid_inputs = None
            valid_target = None
            scheduler = None

        best_state = None
        best_loss = float("inf")
        epochs_without_improvement = 0
        self.best_epoch_ = self.epochs

        self.network.train()

        for epoch in range(self.epochs):
            order = rng.permutation(len(y))

            for start in range(0, len(order), self.batch_size):
                batch_indices = order[
                    start:start + self.batch_size
                ]

                indices = torch.as_tensor(
                    batch_indices,
                    dtype=torch.long,
                    device=device,
                )

                optimizer.zero_grad(set_to_none=True)

                prediction = self.network(
                    drug_a_tensor[indices],
                    drug_b_tensor[indices],
                    cell_tensor[indices],
                )

                loss = loss_function(
                    prediction,
                    target_tensor[indices],
                )

                loss.backward()

                # Prevent unusually large gradients from destabilizing
                # training on high-dimensional biological features.
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

            if valid_inputs is not None:
                self.network.eval()
                with torch.no_grad():
                    valid_prediction = self.network(*valid_inputs)
                    valid_loss = float(loss_function(valid_prediction, valid_target).item())
                scheduler.step(valid_loss)

                if valid_loss < best_loss - self.min_delta:
                    best_loss = valid_loss
                    best_state = copy.deepcopy(self.network.state_dict())
                    self.best_epoch_ = epoch + 1
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.patience:
                    break

                self.network.train()

        if best_state is not None:
            self.network.load_state_dict(best_state)

        return self

    def predict(self, X):
        """Generate S-score predictions."""

        if self.network is None:
            raise RuntimeError(
                "The SynergyX-inspired model has not been fitted yet."
            )

        X = self._validate_input(X)

        input_mean = getattr(self, "input_mean_", None)
        input_scale = getattr(self, "input_scale_", None)
        if input_mean is not None and input_scale is not None:
            X = (X - input_mean) / input_scale

        device = next(self.network.parameters()).device

        drug_a, drug_b, cell = self._split_features(X)

        drug_a_tensor = torch.as_tensor(
            drug_a,
            dtype=torch.float32,
            device=device,
        )

        drug_b_tensor = torch.as_tensor(
            drug_b,
            dtype=torch.float32,
            device=device,
        )

        cell_tensor = torch.as_tensor(
            cell,
            dtype=torch.float32,
            device=device,
        )

        self.network.eval()

        with torch.no_grad():
            predictions = self.network(
                drug_a_tensor,
                drug_b_tensor,
                cell_tensor,
            )

        output = predictions.detach().cpu().numpy().astype(np.float64)
        target_mean = getattr(self, "target_mean_", None)
        target_scale = getattr(self, "target_scale_", None)
        if target_mean is not None and target_scale is not None:
            output = output * target_scale + target_mean
        return output
