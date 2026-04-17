from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD


class ImplicitMFRecommender:
    """Implicit-feedback matrix factorization via TruncatedSVD."""

    def __init__(self, n_factors: int = 12, random_state: int = 42):
        self.n_factors = n_factors
        self.random_state = random_state

        self._users: list[str] | None = None
        self._offers: list[str] | None = None
        self._scores: np.ndarray | None = None
        self._user_index: dict[str, int] | None = None
        self._offer_index: dict[str, int] | None = None

    def fit(self, interactions_df: pd.DataFrame) -> "ImplicitMFRecommender":
        positive = interactions_df[interactions_df["label"] == 1][["user_id", "offer_id"]].copy()
        if positive.empty:
            raise ValueError("No positive interactions found for MF training.")

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

        users = matrix.index.astype(str).tolist()
        offers = matrix.columns.astype(str).tolist()

        n_components = max(2, min(self.n_factors, min(matrix.shape) - 1))
        svd = TruncatedSVD(n_components=n_components, random_state=self.random_state)
        user_factors = svd.fit_transform(matrix.to_numpy(dtype=float))
        offer_factors = svd.components_.T
        reconstructed = user_factors @ offer_factors.T

        self._users = users
        self._offers = offers
        self._scores = reconstructed
        self._user_index = {u: i for i, u in enumerate(users)}
        self._offer_index = {o: j for j, o in enumerate(offers)}
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
            or self._scores is None
            or self._user_index is None
            or self._offer_index is None
        ):
            raise RuntimeError("Model must be fitted before inference.")

        exclude_by_user = exclude_by_user or {}
        rows: list[dict[str, str | float | int]] = []

        global_scores = self._scores.mean(axis=0)
        for user_id in user_ids:
            if user_id in self._user_index:
                score_vec = self._scores[self._user_index[user_id]].copy()
            else:
                score_vec = global_scores.copy()

            exclude = exclude_by_user.get(user_id, set())
            for offer_id in exclude:
                j = self._offer_index.get(offer_id)
                if j is not None:
                    score_vec[j] = -np.inf

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
