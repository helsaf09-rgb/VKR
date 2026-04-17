from __future__ import annotations

import numpy as np
import pandas as pd


class ImplicitItemKNNRecommender:
    """Item-based kNN recommender for implicit positive interactions."""

    def __init__(self, n_neighbors: int = 50):
        self.n_neighbors = n_neighbors

        self._users: list[str] | None = None
        self._offers: list[str] | None = None
        self._user_index: dict[str, int] | None = None
        self._offer_index: dict[str, int] | None = None
        self._user_item_matrix: np.ndarray | None = None
        self._item_similarity: np.ndarray | None = None
        self._global_scores: np.ndarray | None = None

    def fit(self, interactions_df: pd.DataFrame) -> "ImplicitItemKNNRecommender":
        positive = interactions_df[interactions_df["label"] == 1][["user_id", "offer_id"]].copy()
        if positive.empty:
            raise ValueError("No positive interactions found for item-kNN training.")

        positive["value"] = 1.0
        matrix = (
            positive.pivot_table(
                index="user_id",
                columns="offer_id",
                values="value",
                aggfunc="max",
                fill_value=0.0,
            )
            .sort_index()
            .sort_index(axis=1)
        )

        user_item = matrix.to_numpy(dtype=float)
        item_user = user_item.T
        dot_products = item_user @ item_user.T
        norms = np.linalg.norm(item_user, axis=1)
        denom = np.outer(norms, norms)
        denom[denom == 0.0] = 1.0
        similarity = dot_products / denom
        np.fill_diagonal(similarity, 0.0)

        if 0 < self.n_neighbors < similarity.shape[0]:
            pruned = np.zeros_like(similarity)
            for idx in range(similarity.shape[0]):
                neighbor_idx = np.argpartition(-similarity[idx], self.n_neighbors)[: self.n_neighbors]
                pruned[idx, neighbor_idx] = similarity[idx, neighbor_idx]
            similarity = pruned

        self._users = matrix.index.astype(str).tolist()
        self._offers = matrix.columns.astype(str).tolist()
        self._user_index = {user_id: i for i, user_id in enumerate(self._users)}
        self._offer_index = {offer_id: i for i, offer_id in enumerate(self._offers)}
        self._user_item_matrix = user_item
        self._item_similarity = similarity
        self._global_scores = user_item.sum(axis=0)
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
            or self._user_item_matrix is None
            or self._item_similarity is None
            or self._global_scores is None
        ):
            raise RuntimeError("Model must be fitted before inference.")

        exclude_by_user = exclude_by_user or {}
        rows: list[dict[str, str | float | int]] = []

        for user_id in user_ids:
            if user_id in self._user_index:
                user_vector = self._user_item_matrix[self._user_index[user_id]]
                score_vec = user_vector @ self._item_similarity
                if float(np.abs(score_vec).sum()) == 0.0:
                    score_vec = self._global_scores.copy()
            else:
                score_vec = self._global_scores.copy()

            exclude = set(exclude_by_user.get(user_id, set()))
            if user_id in self._user_index:
                seen_idx = np.flatnonzero(self._user_item_matrix[self._user_index[user_id]] > 0.0)
                exclude.update(self._offers[idx] for idx in seen_idx)

            score_vec = score_vec.astype(float, copy=True)
            for offer_id in exclude:
                idx = self._offer_index.get(offer_id)
                if idx is not None:
                    score_vec[idx] = -np.inf

            top_idx = np.argsort(-score_vec)[:top_k]
            for rank, idx in enumerate(top_idx, start=1):
                rows.append(
                    {
                        "user_id": user_id,
                        "offer_id": self._offers[idx],
                        "score": float(score_vec[idx]),
                        "rank": rank,
                    }
                )

        return pd.DataFrame(rows)
