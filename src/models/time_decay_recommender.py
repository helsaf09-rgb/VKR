from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CATEGORIES


class TimeDecayRecommender:
    """Time-aware profile recommender with recency decay and short-term blending."""

    def __init__(
        self,
        decay_rate: float = 0.02,
        short_term_days: int = 60,
        short_term_weight: float = 0.35,
        spend_weight: float = 0.70,
        freq_weight: float = 0.30,
    ):
        self.decay_rate = decay_rate
        self.short_term_days = short_term_days
        self.short_term_weight = short_term_weight
        self.spend_weight = spend_weight
        self.freq_weight = freq_weight

        self.user_profiles_: pd.DataFrame | None = None
        self.offer_profiles_: pd.DataFrame | None = None
        self.global_profile_: pd.Series | None = None

    @staticmethod
    def _build_offer_profiles(offers_df: pd.DataFrame) -> pd.DataFrame:
        category_columns = [f"cat_{c}" for c in CATEGORIES]
        if set(category_columns).issubset(set(offers_df.columns)):
            matrix = offers_df[category_columns].to_numpy(dtype=float)
        else:
            matrix = np.zeros((len(offers_df), len(CATEGORIES)), dtype=float)
            for i, target in enumerate(offers_df["target_categories"].astype(str).tolist()):
                targets = set(target.split("|"))
                for j, category in enumerate(CATEGORIES):
                    matrix[i, j] = float(category in targets)

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        normalized = matrix / row_sums
        return pd.DataFrame(normalized, index=offers_df["offer_id"].astype(str), columns=CATEGORIES)

    def _build_weighted_profile(self, tx_df: pd.DataFrame, weight_col: str) -> pd.DataFrame:
        spend = (
            tx_df.assign(weighted_amount=tx_df["amount"] * tx_df[weight_col])
            .groupby(["user_id", "category"])["weighted_amount"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(columns=CATEGORIES, fill_value=0.0)
        )
        freq = (
            tx_df.groupby(["user_id", "category"])[weight_col]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(columns=CATEGORIES, fill_value=0.0)
        )

        spend_share = spend.div(spend.sum(axis=1).replace(0.0, 1.0), axis=0)
        freq_share = freq.div(freq.sum(axis=1).replace(0.0, 1.0), axis=0)
        profile = self.spend_weight * spend_share + self.freq_weight * freq_share
        return profile.astype(float)

    def fit(self, transactions_df: pd.DataFrame, offers_df: pd.DataFrame) -> "TimeDecayRecommender":
        tx = transactions_df.copy()
        tx["timestamp"] = pd.to_datetime(tx["timestamp"])

        reference_time = tx["timestamp"].max()
        age_days = (reference_time - tx["timestamp"]).dt.total_seconds() / 86400.0
        tx["recency_weight"] = np.exp(-self.decay_rate * age_days)

        long_term_profile = self._build_weighted_profile(tx, "recency_weight")

        short_mask = (reference_time - tx["timestamp"]).dt.days <= self.short_term_days
        short_tx = tx[short_mask].copy()
        if short_tx.empty:
            short_term_profile = long_term_profile.copy()
        else:
            short_tx["short_weight"] = 1.0
            short_term_profile = self._build_weighted_profile(short_tx, "short_weight")

        all_users = sorted(set(long_term_profile.index).union(set(short_term_profile.index)))
        long_term_profile = long_term_profile.reindex(all_users, fill_value=0.0)
        short_term_profile = short_term_profile.reindex(all_users, fill_value=0.0)

        final_profile = (1.0 - self.short_term_weight) * long_term_profile + self.short_term_weight * short_term_profile

        self.user_profiles_ = final_profile.astype(float)
        self.offer_profiles_ = self._build_offer_profiles(offers_df)
        self.global_profile_ = self.user_profiles_.mean(axis=0)
        return self

    def recommend_for_users(
        self,
        user_ids: list[str],
        top_k: int = 5,
        exclude_by_user: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        if self.user_profiles_ is None or self.offer_profiles_ is None or self.global_profile_ is None:
            raise RuntimeError("Model must be fitted before inference.")

        exclude_by_user = exclude_by_user or {}
        offer_matrix = self.offer_profiles_.to_numpy(dtype=float)
        offer_ids = self.offer_profiles_.index.astype(str).tolist()

        rows: list[dict[str, str | float | int]] = []
        for user_id in user_ids:
            if user_id in self.user_profiles_.index:
                user_vec = self.user_profiles_.loc[user_id].to_numpy(dtype=float)
            else:
                user_vec = self.global_profile_.to_numpy(dtype=float)

            scores = offer_matrix @ user_vec
            for offer_id in exclude_by_user.get(user_id, set()):
                if offer_id in self.offer_profiles_.index:
                    idx = self.offer_profiles_.index.get_loc(offer_id)
                    scores[idx] = -np.inf

            top_idx = np.argsort(-scores)[:top_k]
            for rank, idx in enumerate(top_idx, start=1):
                rows.append(
                    {
                        "user_id": user_id,
                        "offer_id": offer_ids[idx],
                        "score": float(scores[idx]),
                        "rank": rank,
                    }
                )

        return pd.DataFrame(rows)
