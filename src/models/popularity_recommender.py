from __future__ import annotations

import pandas as pd


class PopularityRecommender:
    """Non-personalized popularity baseline for implicit feedback."""

    def __init__(self) -> None:
        self._scores: pd.Series | None = None

    def fit(self, interactions_df: pd.DataFrame) -> "PopularityRecommender":
        positive = interactions_df[interactions_df["label"] == 1].copy()
        if positive.empty:
            raise ValueError("No positive interactions found for popularity baseline.")

        scores = (
            positive.groupby("offer_id")
            .size()
            .astype(float)
            .sort_values(ascending=False)
        )
        self._scores = scores
        return self

    def recommend_for_users(
        self,
        user_ids: list[str],
        top_k: int = 5,
        exclude_by_user: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        if self._scores is None:
            raise RuntimeError("Model must be fitted before inference.")

        exclude_by_user = exclude_by_user or {}
        rows: list[dict[str, str | float | int]] = []

        for user_id in user_ids:
            scores = self._scores.copy()
            exclude = exclude_by_user.get(user_id, set())
            if exclude:
                scores = scores[~scores.index.isin(exclude)]

            top_scores = scores.head(top_k)
            for rank, (offer_id, score) in enumerate(top_scores.items(), start=1):
                rows.append(
                    {
                        "user_id": user_id,
                        "offer_id": str(offer_id),
                        "score": float(score),
                        "rank": rank,
                    }
                )

        if not rows:
            return pd.DataFrame(columns=["user_id", "offer_id", "score", "rank"])
        return pd.DataFrame(rows)
