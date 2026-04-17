from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class LightGCNConfig:
    embedding_dim: int = 24
    n_layers: int = 2
    learning_rate: float = 0.03
    epochs: int = 25
    batch_size: int = 4096
    samples_per_epoch: int = 30000
    l2_reg: float = 1e-4
    random_state: int = 42


class LightGCNRecommender:
    """LightGCN-style recommender trained with sampled BPR updates on implicit feedback."""

    def __init__(self, config: LightGCNConfig | None = None):
        self.config = config or LightGCNConfig()

        self._users: list[str] | None = None
        self._offers: list[str] | None = None
        self._user_index: dict[str, int] | None = None
        self._offer_index: dict[str, int] | None = None
        self._user_positive_items: dict[int, set[int]] = {}

        self._n_users = 0
        self._n_items = 0
        self._norm_adj: sparse.csr_matrix | None = None
        self._embeddings: np.ndarray | None = None
        self._final_embeddings: np.ndarray | None = None
        self._item_popularity: np.ndarray | None = None
        self._optimizer_state: dict[str, np.ndarray | int] | None = None

        self.loss_history_: list[float] = []

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        clipped = np.clip(x, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _initialize_embeddings(self, n_nodes: int, rng: np.random.Generator) -> None:
        scale = 0.1
        self._embeddings = rng.normal(
            loc=0.0,
            scale=scale,
            size=(n_nodes, self.config.embedding_dim),
        ).astype(np.float32)
        self._optimizer_state = {
            "t": 0,
            "m": np.zeros_like(self._embeddings),
            "v": np.zeros_like(self._embeddings),
        }

    def _build_graph(self, positive_df: pd.DataFrame) -> None:
        users = sorted(positive_df["user_id"].astype(str).unique().tolist())
        offers = sorted(positive_df["offer_id"].astype(str).unique().tolist())
        self._users = users
        self._offers = offers
        self._user_index = {user_id: idx for idx, user_id in enumerate(users)}
        self._offer_index = {offer_id: idx for idx, offer_id in enumerate(offers)}
        self._n_users = len(users)
        self._n_items = len(offers)

        mapped = positive_df.copy()
        mapped["user_idx"] = mapped["user_id"].astype(str).map(self._user_index)
        mapped["item_idx"] = mapped["offer_id"].astype(str).map(self._offer_index)
        user_idx = mapped["user_idx"].to_numpy(dtype=np.int32)
        item_idx = mapped["item_idx"].to_numpy(dtype=np.int32)

        self._user_positive_items = {}
        for u_idx, i_idx in zip(user_idx, item_idx, strict=False):
            self._user_positive_items.setdefault(int(u_idx), set()).add(int(i_idx))

        n_nodes = self._n_users + self._n_items
        bipartite_rows = np.concatenate([user_idx, item_idx + self._n_users]).astype(np.int32, copy=False)
        bipartite_cols = np.concatenate([item_idx + self._n_users, user_idx]).astype(np.int32, copy=False)
        values = np.ones(len(bipartite_rows), dtype=np.float32)
        adjacency = sparse.coo_matrix((values, (bipartite_rows, bipartite_cols)), shape=(n_nodes, n_nodes)).tocsr()

        degrees = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float32)
        inv_sqrt_deg = np.power(np.maximum(degrees, 1.0), -0.5)
        d_inv_sqrt = sparse.diags(inv_sqrt_deg)
        self._norm_adj = (d_inv_sqrt @ adjacency @ d_inv_sqrt).tocsr()

        popularity = (
            positive_df.groupby("offer_id").size().reindex(self._offers, fill_value=0).to_numpy(dtype=np.float32)
        )
        max_popularity = float(popularity.max()) if len(popularity) else 1.0
        if max_popularity <= 0.0:
            max_popularity = 1.0
        self._item_popularity = popularity / max_popularity

    def _propagate_embeddings(self, base_embeddings: np.ndarray) -> np.ndarray:
        if self._norm_adj is None:
            raise RuntimeError("Graph is not initialized.")

        all_layers = [base_embeddings]
        current = base_embeddings
        for _ in range(self.config.n_layers):
            current = self._norm_adj @ current
            all_layers.append(np.asarray(current, dtype=np.float32))
        return np.mean(all_layers, axis=0, dtype=np.float32)

    def _propagate_gradient(self, grad_final: np.ndarray) -> np.ndarray:
        if self._norm_adj is None:
            raise RuntimeError("Graph is not initialized.")

        propagated = grad_final.astype(np.float32, copy=True)
        current = grad_final.astype(np.float32, copy=True)
        for _ in range(self.config.n_layers):
            current = self._norm_adj @ current
            propagated += np.asarray(current, dtype=np.float32)
        propagated /= float(self.config.n_layers + 1)
        return propagated

    def _sample_triples(
        self,
        positive_pairs: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._n_items <= 1:
            raise ValueError("LightGCN requires at least two items.")

        n_samples = max(self.config.batch_size, int(self.config.samples_per_epoch))
        pair_idx = rng.integers(0, len(positive_pairs), size=n_samples)
        sampled_pairs = positive_pairs[pair_idx]
        users = sampled_pairs[:, 0].astype(np.int32, copy=False)
        positives = sampled_pairs[:, 1].astype(np.int32, copy=False)
        negatives = np.empty(n_samples, dtype=np.int32)

        for idx, user_idx in enumerate(users):
            seen_items = self._user_positive_items[int(user_idx)]
            if len(seen_items) >= self._n_items:
                raise ValueError("A user is connected to every item; cannot sample negatives.")

            negative_item = int(rng.integers(0, self._n_items))
            while negative_item in seen_items:
                negative_item = int(rng.integers(0, self._n_items))
            negatives[idx] = negative_item

        return users, positives, negatives

    def _adam_update(self, grad: np.ndarray) -> None:
        if self._embeddings is None or self._optimizer_state is None:
            raise RuntimeError("Optimizer state is not initialized.")

        self._optimizer_state["t"] = int(self._optimizer_state["t"]) + 1
        m = self._optimizer_state["m"]
        v = self._optimizer_state["v"]
        t = int(self._optimizer_state["t"])

        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        m *= beta1
        m += (1.0 - beta1) * grad
        v *= beta2
        v += (1.0 - beta2) * np.square(grad)

        m_hat = m / (1.0 - beta1**t)
        v_hat = v / (1.0 - beta2**t)
        self._embeddings -= self.config.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, interactions_df: pd.DataFrame) -> "LightGCNRecommender":
        positive_df = interactions_df[interactions_df["label"] == 1][["user_id", "offer_id"]].drop_duplicates().copy()
        if positive_df.empty:
            raise ValueError("No positive interactions found for LightGCN training.")

        positive_df["user_id"] = positive_df["user_id"].astype(str)
        positive_df["offer_id"] = positive_df["offer_id"].astype(str)
        self._build_graph(positive_df)

        positive_pairs_df = positive_df.copy()
        positive_pairs_df["user_idx"] = positive_pairs_df["user_id"].map(self._user_index)
        positive_pairs_df["item_idx"] = positive_pairs_df["offer_id"].map(self._offer_index)
        positive_pairs = positive_pairs_df[["user_idx", "item_idx"]].to_numpy(dtype=np.int32)

        rng = np.random.default_rng(self.config.random_state)
        self._initialize_embeddings(self._n_users + self._n_items, rng)
        self.loss_history_ = []

        for _ in range(self.config.epochs):
            if self._embeddings is None:
                raise RuntimeError("Embeddings are not initialized.")

            propagated = self._propagate_embeddings(self._embeddings)
            users, positives, negatives = self._sample_triples(positive_pairs, rng)
            grad_final = np.zeros_like(propagated)

            total_loss = 0.0
            n_samples = len(users)
            for start in range(0, n_samples, self.config.batch_size):
                stop = min(start + self.config.batch_size, n_samples)
                batch_users = users[start:stop]
                batch_pos = positives[start:stop]
                batch_neg = negatives[start:stop]

                user_nodes = batch_users
                pos_nodes = self._n_users + batch_pos
                neg_nodes = self._n_users + batch_neg

                user_emb = propagated[user_nodes]
                pos_emb = propagated[pos_nodes]
                neg_emb = propagated[neg_nodes]

                diff = np.sum(user_emb * (pos_emb - neg_emb), axis=1)
                coeff = self._sigmoid(-diff).astype(np.float32) / float(n_samples)
                total_loss += float(-np.log(self._sigmoid(diff) + 1e-8).sum())

                np.add.at(grad_final, user_nodes, coeff[:, None] * (neg_emb - pos_emb))
                np.add.at(grad_final, pos_nodes, -coeff[:, None] * user_emb)
                np.add.at(grad_final, neg_nodes, coeff[:, None] * user_emb)

            grad_base = self._propagate_gradient(grad_final)
            grad_base += self.config.l2_reg * self._embeddings
            self._adam_update(grad_base.astype(np.float32, copy=False))

            reg_term = 0.5 * self.config.l2_reg * float(np.mean(np.square(self._embeddings)))
            self.loss_history_.append(total_loss / float(n_samples) + reg_term)

        self._final_embeddings = self._propagate_embeddings(self._embeddings)
        return self

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
            or self._final_embeddings is None
            or self._item_popularity is None
        ):
            raise RuntimeError("Model must be fitted before inference.")

        exclude_by_user = exclude_by_user or {}
        user_embeddings = self._final_embeddings[: self._n_users]
        item_embeddings = self._final_embeddings[self._n_users :]
        rows: list[dict[str, str | float | int]] = []

        for user_id in user_ids:
            if user_id in self._user_index:
                score_vec = user_embeddings[self._user_index[user_id]] @ item_embeddings.T
            else:
                score_vec = self._item_popularity.copy()

            score_vec = score_vec.astype(np.float32, copy=True)
            for offer_id in exclude_by_user.get(str(user_id), set()):
                offer_idx = self._offer_index.get(str(offer_id))
                if offer_idx is not None:
                    score_vec[offer_idx] = -np.inf

            top_idx = np.argsort(-score_vec)[:top_k]
            for rank, offer_idx in enumerate(top_idx, start=1):
                rows.append(
                    {
                        "user_id": str(user_id),
                        "offer_id": self._offers[offer_idx],
                        "score": float(score_vec[offer_idx]),
                        "rank": rank,
                    }
                )

        return pd.DataFrame(rows)
