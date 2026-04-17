from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class NeuralCFConfig:
    embedding_dim: int = 16
    hidden_dims: tuple[int, ...] = (32, 16)
    learning_rate: float = 0.01
    epochs: int = 6
    batch_size: int = 1024
    negative_samples: int = 2
    l2_reg: float = 1e-5
    random_state: int = 42


class NeuralCFRecommender:
    """Small NeuMF-style recommender trained on implicit feedback with negative sampling."""

    def __init__(self, config: NeuralCFConfig | None = None):
        self.config = config or NeuralCFConfig()

        self._users: list[str] | None = None
        self._offers: list[str] | None = None
        self._user_index: dict[str, int] | None = None
        self._offer_index: dict[str, int] | None = None
        self._item_popularity: np.ndarray | None = None

        self._user_embeddings: np.ndarray | None = None
        self._item_embeddings: np.ndarray | None = None
        self._hidden_weights: list[np.ndarray] = []
        self._hidden_biases: list[np.ndarray] = []
        self._output_weight: np.ndarray | None = None
        self._output_bias: np.ndarray | None = None

        self._optimizer_state: dict[str, list[np.ndarray] | np.ndarray | int] | None = None
        self.loss_history_: list[float] = []

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _initialize_parameters(self, n_users: int, n_items: int, rng: np.random.Generator) -> None:
        cfg = self.config
        scale = 0.05
        self._user_embeddings = rng.normal(0.0, scale, size=(n_users, cfg.embedding_dim)).astype(np.float32)
        self._item_embeddings = rng.normal(0.0, scale, size=(n_items, cfg.embedding_dim)).astype(np.float32)

        self._hidden_weights = []
        self._hidden_biases = []
        input_dim = cfg.embedding_dim * 2
        for hidden_dim in cfg.hidden_dims:
            weight = rng.normal(0.0, np.sqrt(2.0 / max(1, input_dim)), size=(input_dim, hidden_dim)).astype(
                np.float32
            )
            bias = np.zeros(hidden_dim, dtype=np.float32)
            self._hidden_weights.append(weight)
            self._hidden_biases.append(bias)
            input_dim = hidden_dim

        self._output_weight = rng.normal(0.0, np.sqrt(1.0 / max(1, input_dim)), size=(input_dim, 1)).astype(
            np.float32
        )
        self._output_bias = np.zeros(1, dtype=np.float32)

        self._optimizer_state = {
            "t": 0,
            "m_user": np.zeros_like(self._user_embeddings),
            "v_user": np.zeros_like(self._user_embeddings),
            "m_item": np.zeros_like(self._item_embeddings),
            "v_item": np.zeros_like(self._item_embeddings),
            "m_hidden_w": [np.zeros_like(w) for w in self._hidden_weights],
            "v_hidden_w": [np.zeros_like(w) for w in self._hidden_weights],
            "m_hidden_b": [np.zeros_like(b) for b in self._hidden_biases],
            "v_hidden_b": [np.zeros_like(b) for b in self._hidden_biases],
            "m_out_w": np.zeros_like(self._output_weight),
            "v_out_w": np.zeros_like(self._output_weight),
            "m_out_b": np.zeros_like(self._output_bias),
            "v_out_b": np.zeros_like(self._output_bias),
        }

    def _sample_training_data(
        self,
        positive_pairs: np.ndarray,
        user_positive_items: dict[int, set[int]],
        n_items: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cfg = self.config
        max_rows = len(positive_pairs) * (1 + cfg.negative_samples)
        user_idx = np.empty(max_rows, dtype=np.int32)
        item_idx = np.empty(max_rows, dtype=np.int32)
        labels = np.empty(max_rows, dtype=np.float32)

        cursor = 0
        all_items = np.arange(n_items, dtype=np.int32)
        for pair_user_idx, pair_item_idx in positive_pairs:
            user_idx[cursor] = int(pair_user_idx)
            item_idx[cursor] = int(pair_item_idx)
            labels[cursor] = 1.0
            cursor += 1

            seen = user_positive_items[int(pair_user_idx)]
            if len(seen) >= n_items:
                continue

            negatives_added = 0
            while negatives_added < cfg.negative_samples:
                sampled_item = int(rng.choice(all_items))
                if sampled_item in seen:
                    continue
                user_idx[cursor] = int(pair_user_idx)
                item_idx[cursor] = sampled_item
                labels[cursor] = 0.0
                cursor += 1
                negatives_added += 1

        return user_idx[:cursor], item_idx[:cursor], labels[:cursor]

    def _adam_update(self, param: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray) -> None:
        if self._optimizer_state is None:
            raise RuntimeError("Optimizer state is not initialized.")

        lr = self.config.learning_rate
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t = int(self._optimizer_state["t"])

        m *= beta1
        m += (1.0 - beta1) * grad
        v *= beta2
        v += (1.0 - beta2) * np.square(grad)

        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def _train_batch(self, batch_user_idx: np.ndarray, batch_item_idx: np.ndarray, labels: np.ndarray) -> float:
        if (
            self._user_embeddings is None
            or self._item_embeddings is None
            or self._output_weight is None
            or self._output_bias is None
            or self._optimizer_state is None
        ):
            raise RuntimeError("Model parameters are not initialized.")

        cfg = self.config
        self._optimizer_state["t"] = int(self._optimizer_state["t"]) + 1

        batch_user_emb = self._user_embeddings[batch_user_idx]
        batch_item_emb = self._item_embeddings[batch_item_idx]
        x0 = np.concatenate([batch_user_emb, batch_item_emb], axis=1)

        activations = [x0]
        pre_activations: list[np.ndarray] = []
        current = x0
        for weight, bias in zip(self._hidden_weights, self._hidden_biases):
            z = current @ weight + bias
            current = np.maximum(z, 0.0)
            pre_activations.append(z)
            activations.append(current)

        logits = activations[-1] @ self._output_weight + self._output_bias
        probs = self._sigmoid(logits.reshape(-1))
        eps = 1e-8
        loss = -np.mean(labels * np.log(probs + eps) + (1.0 - labels) * np.log(1.0 - probs + eps))

        batch_size = max(1, len(labels))
        d_logits = ((probs - labels) / batch_size).reshape(-1, 1)

        grad_output_weight = activations[-1].T @ d_logits + cfg.l2_reg * self._output_weight
        grad_output_bias = d_logits.sum(axis=0)
        d_activation = d_logits @ self._output_weight.T

        grad_hidden_weights: list[np.ndarray] = []
        grad_hidden_biases: list[np.ndarray] = []
        for layer_idx in reversed(range(len(self._hidden_weights))):
            d_hidden = d_activation * (pre_activations[layer_idx] > 0.0)
            grad_weight = activations[layer_idx].T @ d_hidden + cfg.l2_reg * self._hidden_weights[layer_idx]
            grad_bias = d_hidden.sum(axis=0)
            grad_hidden_weights.insert(0, grad_weight)
            grad_hidden_biases.insert(0, grad_bias)
            d_activation = d_hidden @ self._hidden_weights[layer_idx].T

        d_user_emb = d_activation[:, : cfg.embedding_dim]
        d_item_emb = d_activation[:, cfg.embedding_dim :]

        grad_user_embeddings = np.zeros_like(self._user_embeddings)
        grad_item_embeddings = np.zeros_like(self._item_embeddings)
        np.add.at(grad_user_embeddings, batch_user_idx, d_user_emb)
        np.add.at(grad_item_embeddings, batch_item_idx, d_item_emb)
        grad_user_embeddings += cfg.l2_reg * self._user_embeddings
        grad_item_embeddings += cfg.l2_reg * self._item_embeddings

        self._adam_update(
            self._user_embeddings,
            grad_user_embeddings,
            self._optimizer_state["m_user"],
            self._optimizer_state["v_user"],
        )
        self._adam_update(
            self._item_embeddings,
            grad_item_embeddings,
            self._optimizer_state["m_item"],
            self._optimizer_state["v_item"],
        )

        for idx in range(len(self._hidden_weights)):
            self._adam_update(
                self._hidden_weights[idx],
                grad_hidden_weights[idx],
                self._optimizer_state["m_hidden_w"][idx],
                self._optimizer_state["v_hidden_w"][idx],
            )
            self._adam_update(
                self._hidden_biases[idx],
                grad_hidden_biases[idx],
                self._optimizer_state["m_hidden_b"][idx],
                self._optimizer_state["v_hidden_b"][idx],
            )

        self._adam_update(
            self._output_weight,
            grad_output_weight,
            self._optimizer_state["m_out_w"],
            self._optimizer_state["v_out_w"],
        )
        self._adam_update(
            self._output_bias,
            grad_output_bias,
            self._optimizer_state["m_out_b"],
            self._optimizer_state["v_out_b"],
        )
        return float(loss)

    def fit(self, interactions_df: pd.DataFrame) -> "NeuralCFRecommender":
        positive = interactions_df[interactions_df["label"] == 1][["user_id", "offer_id"]].drop_duplicates().copy()
        if positive.empty:
            raise ValueError("No positive interactions found for Neural CF training.")

        positive["user_id"] = positive["user_id"].astype(str)
        positive["offer_id"] = positive["offer_id"].astype(str)

        self._users = sorted(positive["user_id"].unique().tolist())
        self._offers = sorted(positive["offer_id"].unique().tolist())
        self._user_index = {user_id: idx for idx, user_id in enumerate(self._users)}
        self._offer_index = {offer_id: idx for idx, offer_id in enumerate(self._offers)}

        positive["user_idx"] = positive["user_id"].map(self._user_index)
        positive["item_idx"] = positive["offer_id"].map(self._offer_index)
        positive_pairs = positive[["user_idx", "item_idx"]].to_numpy(dtype=np.int32)

        user_positive_items: dict[int, set[int]] = {}
        for row in positive_pairs:
            user_positive_items.setdefault(int(row[0]), set()).add(int(row[1]))

        item_popularity = positive.groupby("offer_id").size().reindex(self._offers, fill_value=0).to_numpy(dtype=float)
        max_popularity = float(item_popularity.max()) if len(item_popularity) else 1.0
        if max_popularity <= 0.0:
            max_popularity = 1.0
        self._item_popularity = item_popularity / max_popularity

        rng = np.random.default_rng(self.config.random_state)
        self._initialize_parameters(n_users=len(self._users), n_items=len(self._offers), rng=rng)
        self.loss_history_ = []

        for _ in range(self.config.epochs):
            sampled_user_idx, sampled_item_idx, sampled_labels = self._sample_training_data(
                positive_pairs=positive_pairs,
                user_positive_items=user_positive_items,
                n_items=len(self._offers),
                rng=rng,
            )
            order = rng.permutation(len(sampled_labels))
            sampled_user_idx = sampled_user_idx[order]
            sampled_item_idx = sampled_item_idx[order]
            sampled_labels = sampled_labels[order]

            epoch_loss_sum = 0.0
            epoch_rows = 0
            for start in range(0, len(sampled_labels), self.config.batch_size):
                stop = start + self.config.batch_size
                batch_loss = self._train_batch(
                    batch_user_idx=sampled_user_idx[start:stop],
                    batch_item_idx=sampled_item_idx[start:stop],
                    labels=sampled_labels[start:stop],
                )
                batch_len = len(sampled_labels[start:stop])
                epoch_loss_sum += batch_loss * batch_len
                epoch_rows += batch_len

            self.loss_history_.append(epoch_loss_sum / max(1, epoch_rows))
        return self

    def _score_known_user(self, user_idx: int) -> np.ndarray:
        if (
            self._user_embeddings is None
            or self._item_embeddings is None
            or self._output_weight is None
            or self._output_bias is None
        ):
            raise RuntimeError("Model must be fitted before inference.")

        user_emb = self._user_embeddings[user_idx]
        repeated_user = np.repeat(user_emb[np.newaxis, :], len(self._item_embeddings), axis=0)
        current = np.concatenate([repeated_user, self._item_embeddings], axis=1)
        for weight, bias in zip(self._hidden_weights, self._hidden_biases):
            current = np.maximum(current @ weight + bias, 0.0)
        logits = current @ self._output_weight + self._output_bias
        return self._sigmoid(logits.reshape(-1))

    def recommend_for_users(
        self,
        user_ids: list[str],
        top_k: int = 5,
        exclude_by_user: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        if (
            self._users is None
            or self._offers is None
            or self._user_index is None
            or self._offer_index is None
            or self._item_popularity is None
        ):
            raise RuntimeError("Model must be fitted before inference.")

        exclude_by_user = exclude_by_user or {}
        rows: list[dict[str, str | float | int]] = []

        for user_id in user_ids:
            if user_id in self._user_index:
                score_vec = self._score_known_user(self._user_index[user_id]).astype(float, copy=True)
            else:
                score_vec = self._item_popularity.astype(float, copy=True)

            for offer_id in exclude_by_user.get(user_id, set()):
                item_idx = self._offer_index.get(str(offer_id))
                if item_idx is not None:
                    score_vec[item_idx] = -np.inf

            top_idx = np.argsort(-score_vec)[:top_k]
            for rank, item_idx in enumerate(top_idx, start=1):
                rows.append(
                    {
                        "user_id": str(user_id),
                        "offer_id": self._offers[item_idx],
                        "score": float(score_vec[item_idx]),
                        "rank": rank,
                    }
                )

        return pd.DataFrame(rows)
