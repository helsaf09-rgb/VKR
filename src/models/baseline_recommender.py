from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import CATEGORIES


@dataclass(frozen=True)
class BaselineWeights:
    spend_weight: float = 0.70
    freq_weight: float = 0.30


class TransactionSimilarityRecommender:
    """Simple profile-based ranker for transaction -> offer personalization."""

    def __init__(self, categories: list[str] | None = None, weights: BaselineWeights | None = None):
        self.categories = categories or list(CATEGORIES)
        self.weights = weights or BaselineWeights()

        self.user_profiles_: pd.DataFrame | None = None
        self.offer_profiles_: pd.DataFrame | None = None
        self.global_profile_: pd.Series | None = None
        self.offer_meta_: pd.DataFrame | None = None

    def fit(self, transactions_df: pd.DataFrame, offers_df: pd.DataFrame) -> "TransactionSimilarityRecommender":
        self.user_profiles_ = self._build_user_profiles(transactions_df)
        self.offer_profiles_ = self._build_offer_profiles(offers_df)
        self.global_profile_ = self.user_profiles_.mean(axis=0)
        self.offer_meta_ = offers_df[["offer_id", "offer_name", "product_type"]].copy()
        return self

    def _build_user_profiles(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        spend = (
            transactions_df.groupby(["user_id", "category"])["amount"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(columns=self.categories, fill_value=0.0)
        )

        freq = (
            transactions_df.groupby(["user_id", "category"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(columns=self.categories, fill_value=0.0)
        )

        spend_share = spend.div(spend.sum(axis=1).replace(0.0, 1.0), axis=0)
        freq_share = freq.div(freq.sum(axis=1).replace(0.0, 1.0), axis=0)
        profile = (
            self.weights.spend_weight * spend_share + self.weights.freq_weight * freq_share
        ).astype(float)
        return profile

    def _build_offer_profiles(self, offers_df: pd.DataFrame) -> pd.DataFrame:
        category_columns = [f"cat_{c}" for c in self.categories]
        if set(category_columns).issubset(set(offers_df.columns)):
            matrix = offers_df[category_columns].to_numpy(dtype=float)
        else:
            matrix = np.zeros((len(offers_df), len(self.categories)), dtype=float)
            for i, target in enumerate(offers_df["target_categories"].astype(str).tolist()):
                targets = set(target.split("|"))
                for j, category in enumerate(self.categories):
                    matrix[i, j] = float(category in targets)

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        normalized = matrix / row_sums
        return pd.DataFrame(normalized, index=offers_df["offer_id"].tolist(), columns=self.categories)

    def _score_user(self, user_id: str) -> pd.Series:
        if self.user_profiles_ is None or self.offer_profiles_ is None or self.global_profile_ is None:
            raise RuntimeError("Model must be fitted before scoring.")

        if user_id in self.user_profiles_.index:
            user_profile = self.user_profiles_.loc[user_id].to_numpy(dtype=float)
        else:
            user_profile = self.global_profile_.to_numpy(dtype=float)

        scores = self.offer_profiles_.to_numpy(dtype=float) @ user_profile
        return pd.Series(scores, index=self.offer_profiles_.index, name="score").sort_values(ascending=False)

    def recommend(
        self,
        user_id: str,
        top_k: int = 5,
        exclude_offer_ids: set[str] | None = None,
    ) -> pd.DataFrame:
        exclude_offer_ids = exclude_offer_ids or set()
        scores = self._score_user(user_id)

        if exclude_offer_ids:
            scores = scores[~scores.index.isin(exclude_offer_ids)]

        top_scores = scores.head(top_k)
        result = pd.DataFrame(
            {
                "user_id": user_id,
                "offer_id": top_scores.index.astype(str),
                "score": top_scores.values,
                "rank": np.arange(1, len(top_scores) + 1, dtype=int),
            }
        )
        return result

    def recommend_for_users(
        self,
        user_ids: list[str],
        top_k: int = 5,
        exclude_by_user: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        rows: list[pd.DataFrame] = []
        exclude_by_user = exclude_by_user or {}

        for user_id in user_ids:
            rows.append(
                self.recommend(
                    user_id=user_id,
                    top_k=top_k,
                    exclude_offer_ids=exclude_by_user.get(user_id),
                )
            )

        if not rows:
            return pd.DataFrame(columns=["user_id", "offer_id", "score", "rank"])
        return pd.concat(rows, ignore_index=True)
