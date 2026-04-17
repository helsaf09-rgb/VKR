from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from src.config import CATEGORIES


class HybridSemanticRecommender:
    """Hybrid ranker: category-profile score + TF-IDF semantic score."""

    def __init__(
        self,
        profile_weight: float = 0.70,
        semantic_weight: float = 0.30,
        spend_weight: float = 0.70,
        freq_weight: float = 0.30,
    ):
        self.profile_weight = profile_weight
        self.semantic_weight = semantic_weight
        self.spend_weight = spend_weight
        self.freq_weight = freq_weight

        self._user_profiles: pd.DataFrame | None = None
        self._offer_profiles: pd.DataFrame | None = None
        self._global_profile: pd.Series | None = None

        self._user_semantic: np.ndarray | None = None
        self._offer_semantic: np.ndarray | None = None
        self._user_ids: list[str] | None = None
        self._offer_ids: list[str] | None = None
        self._user_index: dict[str, int] | None = None
        self._offer_index: dict[str, int] | None = None

    @staticmethod
    def _build_user_profiles(transactions_df: pd.DataFrame) -> pd.DataFrame:
        spend = (
            transactions_df.groupby(["user_id", "category"])["amount"]
            .sum()
            .unstack(fill_value=0.0)
            .reindex(columns=CATEGORIES, fill_value=0.0)
        )
        freq = (
            transactions_df.groupby(["user_id", "category"])
            .size()
            .unstack(fill_value=0.0)
            .reindex(columns=CATEGORIES, fill_value=0.0)
        )
        return spend, freq

    @staticmethod
    def _build_offer_profiles(offers_df: pd.DataFrame) -> pd.DataFrame:
        cat_columns = [f"cat_{c}" for c in CATEGORIES]
        if set(cat_columns).issubset(set(offers_df.columns)):
            matrix = offers_df[cat_columns].to_numpy(dtype=float)
        else:
            matrix = np.zeros((len(offers_df), len(CATEGORIES)), dtype=float)
            for i, target in enumerate(offers_df["target_categories"].astype(str).tolist()):
                targets = set(target.split("|"))
                for j, category in enumerate(CATEGORIES):
                    matrix[i, j] = float(category in targets)

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0.0] = 1.0
        normalized = matrix / row_sums
        return pd.DataFrame(normalized, index=offers_df["offer_id"].astype(str).tolist(), columns=CATEGORIES)

    @staticmethod
    def _weighted_category_text(row: pd.Series, prefix: str) -> str:
        tokens: list[str] = []
        for category in CATEGORIES:
            value = float(row[f"{prefix}{category}"])
            repeats = max(1, int(round(value * 30)))
            tokens.extend([category] * repeats)
        return " ".join(tokens)

    def fit(self, transactions_df: pd.DataFrame, offers_df: pd.DataFrame) -> "HybridSemanticRecommender":
        spend, freq = self._build_user_profiles(transactions_df)
        spend_share = spend.div(spend.sum(axis=1).replace(0.0, 1.0), axis=0)
        freq_share = freq.div(freq.sum(axis=1).replace(0.0, 1.0), axis=0)
        user_profiles = self.spend_weight * spend_share + self.freq_weight * freq_share

        offer_profiles = self._build_offer_profiles(offers_df)
        self._user_profiles = user_profiles.astype(float)
        self._offer_profiles = offer_profiles.astype(float)
        self._global_profile = self._user_profiles.mean(axis=0)

        user_text_df = self._user_profiles.reset_index().rename(columns={"index": "user_id"})
        for c in CATEGORIES:
            user_text_df[f"pref_{c}"] = user_text_df[c]
        user_text_df["semantic_text"] = user_text_df.apply(
            lambda r: self._weighted_category_text(r, "pref_"),
            axis=1,
        )
        user_text_df = user_text_df[["user_id", "semantic_text"]].copy()

        offer_text_df = offers_df[["offer_id", "offer_name", "description", "target_categories"]].copy()
        offer_text_df["semantic_text"] = (
            offer_text_df["offer_name"].astype(str)
            + " "
            + offer_text_df["description"].astype(str)
            + " "
            + offer_text_df["target_categories"].astype(str).str.replace("|", " ", regex=False)
        )

        user_corpus = user_text_df["semantic_text"].tolist()
        offer_corpus = offer_text_df["semantic_text"].tolist()

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        tfidf = vectorizer.fit_transform(user_corpus + offer_corpus)
        user_vec = tfidf[: len(user_corpus)]
        offer_vec = tfidf[len(user_corpus) :]

        self._user_semantic = normalize(user_vec, norm="l2", axis=1).toarray()
        self._offer_semantic = normalize(offer_vec, norm="l2", axis=1).toarray()

        self._user_ids = user_text_df["user_id"].astype(str).tolist()
        self._offer_ids = offer_text_df["offer_id"].astype(str).tolist()
        self._user_index = {u: i for i, u in enumerate(self._user_ids)}
        self._offer_index = {o: i for i, o in enumerate(self._offer_ids)}
        return self

    def recommend_for_users(
        self,
        user_ids: list[str],
        top_k: int = 5,
        exclude_by_user: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        if (
            self._user_profiles is None
            or self._offer_profiles is None
            or self._global_profile is None
            or self._user_semantic is None
            or self._offer_semantic is None
            or self._user_ids is None
            or self._offer_ids is None
            or self._user_index is None
            or self._offer_index is None
        ):
            raise RuntimeError("Model must be fitted before inference.")

        exclude_by_user = exclude_by_user or {}
        offer_profile_matrix = self._offer_profiles.loc[self._offer_ids].to_numpy(dtype=float)

        rows: list[dict[str, str | float | int]] = []
        for user_id in user_ids:
            if user_id in self._user_profiles.index:
                user_profile = self._user_profiles.loc[user_id].to_numpy(dtype=float)
            else:
                user_profile = self._global_profile.to_numpy(dtype=float)

            profile_scores = offer_profile_matrix @ user_profile

            if user_id in self._user_index:
                semantic_scores = self._offer_semantic @ self._user_semantic[self._user_index[user_id]]
            else:
                semantic_scores = np.zeros(len(self._offer_ids), dtype=float)

            combined = self.profile_weight * profile_scores + self.semantic_weight * semantic_scores

            for offer_id in exclude_by_user.get(user_id, set()):
                idx = self._offer_index.get(offer_id)
                if idx is not None:
                    combined[idx] = -np.inf

            top_idx = np.argsort(-combined)[:top_k]
            for rank, idx in enumerate(top_idx, start=1):
                rows.append(
                    {
                        "user_id": user_id,
                        "offer_id": self._offer_ids[idx],
                        "score": float(combined[idx]),
                        "rank": rank,
                    }
                )

        return pd.DataFrame(rows)
