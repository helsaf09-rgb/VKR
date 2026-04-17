from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CATEGORIES, REPORTS_DIR, SYNTHETIC_DATA_DIR
from src.models.time_decay_recommender import TimeDecayRecommender
from src.pipelines.run_baseline_pipeline import build_exclusion_map, build_test_ground_truth
from src.service.localization import (
    translate_category,
    translate_channel,
    translate_model,
    translate_offer,
    translate_product_type,
    translate_segment,
)


class RecommendationBackend:
    def __init__(
        self,
        data_dir: Path = SYNTHETIC_DATA_DIR,
        reports_dir: Path = REPORTS_DIR,
    ) -> None:
        self.data_dir = data_dir
        self.reports_dir = reports_dir

        self.users_df: pd.DataFrame | None = None
        self.transactions_df: pd.DataFrame | None = None
        self.offers_df: pd.DataFrame | None = None
        self.interactions_df: pd.DataFrame | None = None
        self.overall_metrics_df: pd.DataFrame | None = None
        self.segment_metrics_df: pd.DataFrame | None = None
        self.synthetic_manifest: dict[str, Any] | None = None
        self.model: TimeDecayRecommender | None = None
        self.exclusion_map: dict[str, set[str]] = {}

    def load(self) -> None:
        users_path = self.data_dir / "users.csv"
        transactions_path = self.data_dir / "transactions.csv"
        offers_path = self.data_dir / "offers.csv"
        interactions_path = self.data_dir / "interactions.csv"

        required_paths = [users_path, transactions_path, offers_path, interactions_path]
        if any(not path.exists() for path in required_paths):
            raise FileNotFoundError(
                "Файлы синтетических данных не найдены. Сначала запустите baseline-пайплайн: "
                "`python -m src.pipelines.run_baseline_pipeline`."
            )

        self.users_df = pd.read_csv(users_path)
        self.transactions_df = pd.read_csv(transactions_path)
        self.offers_df = pd.read_csv(offers_path)
        self.interactions_df = pd.read_csv(interactions_path)

        if (self.reports_dir / "analysis_overall_metrics.csv").exists():
            self.overall_metrics_df = pd.read_csv(self.reports_dir / "analysis_overall_metrics.csv")
        if (self.reports_dir / "analysis_segment_metrics.csv").exists():
            self.segment_metrics_df = pd.read_csv(self.reports_dir / "analysis_segment_metrics.csv")
        if (self.data_dir / "manifest.json").exists():
            self.synthetic_manifest = json.loads((self.data_dir / "manifest.json").read_text(encoding="utf-8"))

        test_gt_df = build_test_ground_truth(self.interactions_df)
        self.exclusion_map = build_exclusion_map(self.interactions_df, test_gt_df)

        self.model = TimeDecayRecommender(
            decay_rate=0.01,
            short_term_days=30,
            short_term_weight=0.2,
            spend_weight=0.6,
            freq_weight=0.4,
        ).fit(self.transactions_df, self.offers_df)

    def known_users(self) -> list[str]:
        if self.users_df is None:
            return []
        return sorted(self.users_df["user_id"].astype(str).unique().tolist())

    def get_segments(self) -> list[str]:
        if self.users_df is None:
            return []
        return sorted(self.users_df["segment"].dropna().astype(str).unique().tolist())

    def get_user_options(self, segment: str | None = None) -> list[str]:
        if self.users_df is None:
            return []

        users = self.users_df.copy()
        if segment:
            users = users[users["segment"].astype(str) == segment]
        return users.sort_values("user_id")["user_id"].astype(str).tolist()

    def get_user_label(self, user_id: str) -> str:
        if self.users_df is None:
            return user_id
        match = self.users_df[self.users_df["user_id"].astype(str) == user_id]
        if match.empty:
            return f"{user_id} · холодный старт"
        segment = str(match.iloc[0]["segment"])
        return f"{user_id} · {translate_segment(segment)}"

    def get_overall_summary(self) -> dict[str, Any]:
        if self.users_df is None or self.transactions_df is None or self.offers_df is None:
            raise RuntimeError("Сервис рекомендаций не инициализирован.")

        best_model = "time_decay"
        best_ndcg = None
        if self.overall_metrics_df is not None and not self.overall_metrics_df.empty:
            ordered = self.overall_metrics_df.sort_values("ndcg_at_k", ascending=False)
            best_model = str(ordered.iloc[0]["model"])
            best_ndcg = float(ordered.iloc[0]["ndcg_at_k"])

        return {
            "n_users": int(self.users_df["user_id"].nunique()),
            "n_transactions": int(len(self.transactions_df)),
            "n_offers": int(self.offers_df["offer_id"].nunique()),
            "best_model": best_model,
            "best_model_label": translate_model(best_model),
            "best_ndcg_at_k": best_ndcg,
            "positive_rate": (
                float(self.synthetic_manifest["interaction_model"]["positive_rate"])
                if self.synthetic_manifest is not None
                else None
            ),
        }

    def _get_user_preferences(self, user_id: str) -> pd.Series:
        if self.users_df is None:
            raise RuntimeError("Сервис рекомендаций не инициализирован.")

        pref_columns = [f"pref_{category}" for category in CATEGORIES]
        match = self.users_df[self.users_df["user_id"].astype(str) == user_id]
        if not match.empty and set(pref_columns).issubset(match.columns):
            preferences = match.iloc[0][pref_columns].astype(float).copy()
            preferences.index = [name.replace("pref_", "") for name in preferences.index]
            return preferences.sort_values(ascending=False)

        category_mix = self.get_user_category_mix(user_id)
        if category_mix.empty:
            return pd.Series(dtype=float)
        return category_mix.set_index("category")["spend_share"].sort_values(ascending=False)

    def _build_offer_explanation(self, user_id: str, target_categories: str) -> tuple[str, list[str]]:
        preferences = self._get_user_preferences(user_id)
        targets = [category.strip() for category in str(target_categories).split("|") if category.strip()]
        matched = [category for category in preferences.head(5).index.tolist() if category in targets]
        matched_ru = [translate_category(category) for category in matched]
        targets_ru = [translate_category(category) for category in targets]

        if matched:
            reason = f"Сильное совпадение с интересом клиента к категориям: {', '.join(matched_ru[:2])}."
        else:
            reason = f"Хорошее общее соответствие по направлениям: {', '.join(targets_ru[:2])}."
        return reason, matched_ru

    def recommend(self, user_id: str, top_k: int = 5) -> pd.DataFrame:
        if self.model is None or self.offers_df is None:
            raise RuntimeError("Сервис рекомендаций не инициализирован.")

        rec_df = self.model.recommend_for_users(
            user_ids=[user_id],
            top_k=top_k,
            exclude_by_user=self.exclusion_map,
        )
        rec_df = rec_df.merge(
            self.offers_df[["offer_id", "offer_name", "product_type", "target_categories", "description"]],
            on="offer_id",
            how="left",
        )
        rec_df["product_type_label"] = rec_df["product_type"].astype(str).map(translate_product_type)

        translated_names: list[str] = []
        translated_descriptions: list[str] = []
        target_categories_labels: list[str] = []

        reasons: list[str] = []
        matched_categories: list[str] = []
        for row in rec_df.itertuples(index=False):
            offer_name_ru, description_ru = translate_offer(
                str(row.offer_id),
                str(row.offer_name),
                str(row.description),
            )
            translated_names.append(offer_name_ru)
            translated_descriptions.append(description_ru)
            target_categories_labels.append(
                ", ".join(
                    translate_category(category)
                    for category in str(row.target_categories).split("|")
                    if category.strip()
                )
            )
            reason, matched = self._build_offer_explanation(user_id, str(row.target_categories))
            reasons.append(reason)
            matched_categories.append(", ".join(matched[:3]))

        if not rec_df.empty:
            score_min = float(rec_df["score"].min())
            score_max = float(rec_df["score"].max())
            denom = score_max - score_min if score_max != score_min else 1.0
            rec_df["fit_score_pct"] = ((rec_df["score"] - score_min) / denom * 100.0).round(1)
        else:
            rec_df["fit_score_pct"] = pd.Series(dtype=float)

        rec_df["offer_name_label"] = translated_names
        rec_df["description_label"] = translated_descriptions
        rec_df["target_categories_label"] = target_categories_labels
        rec_df["reason"] = reasons
        rec_df["matched_categories"] = matched_categories
        return rec_df

    def get_user_snapshot(self, user_id: str) -> dict[str, Any]:
        if self.users_df is None or self.transactions_df is None or self.interactions_df is None:
            raise RuntimeError("Сервис рекомендаций не инициализирован.")

        user_row = self.users_df[self.users_df["user_id"].astype(str) == user_id]
        tx = self.transactions_df[self.transactions_df["user_id"].astype(str) == user_id].copy()
        tx["timestamp"] = pd.to_datetime(tx["timestamp"])

        total_spend = float(tx["amount"].sum()) if not tx.empty else 0.0
        avg_ticket = float(tx["amount"].mean()) if not tx.empty else 0.0
        tx_count = int(len(tx))
        last_tx_date = tx["timestamp"].max() if not tx.empty else None

        category_mix = self.get_user_category_mix(user_id)
        recent_transactions = self.get_recent_transactions(user_id)
        monthly_spend = self.get_user_monthly_spend(user_id)
        accepted_offers = self.get_user_positive_offers(user_id)
        preferences = self._get_user_preferences(user_id)

        snapshot = {
            "user_id": user_id,
            "segment": str(user_row.iloc[0]["segment"]) if not user_row.empty else "cold-start",
            "segment_label": translate_segment(str(user_row.iloc[0]["segment"])) if not user_row.empty else "Холодный старт",
            "total_spend": round(total_spend, 2),
            "avg_ticket": round(avg_ticket, 2),
            "tx_count": tx_count,
            "last_tx_date": last_tx_date,
            "top_preference_categories": preferences.head(5).rename_axis("category").reset_index(name="value"),
            "category_mix": category_mix,
            "recent_transactions": recent_transactions,
            "monthly_spend": monthly_spend,
            "accepted_offers": accepted_offers,
        }
        return snapshot

    def get_user_category_mix(self, user_id: str) -> pd.DataFrame:
        if self.transactions_df is None:
            raise RuntimeError("Сервис рекомендаций не инициализирован.")

        tx = self.transactions_df[self.transactions_df["user_id"].astype(str) == user_id].copy()
        if tx.empty:
            return pd.DataFrame(columns=["category", "total_amount", "tx_count", "spend_share"])

        grouped = (
            tx.groupby("category")
            .agg(total_amount=("amount", "sum"), tx_count=("transaction_id", "count"))
            .reset_index()
            .sort_values("total_amount", ascending=False)
            .reset_index(drop=True)
        )
        total_amount = float(grouped["total_amount"].sum())
        grouped["spend_share"] = grouped["total_amount"] / total_amount if total_amount else 0.0
        grouped["category_label"] = grouped["category"].astype(str).map(translate_category)
        return grouped

    def get_recent_transactions(self, user_id: str, limit: int = 8) -> pd.DataFrame:
        if self.transactions_df is None:
            raise RuntimeError("Сервис рекомендаций не инициализирован.")

        tx = self.transactions_df[self.transactions_df["user_id"].astype(str) == user_id].copy()
        if tx.empty:
            return pd.DataFrame(columns=["timestamp", "category", "amount", "channel"])

        tx["timestamp"] = pd.to_datetime(tx["timestamp"])
        recent = (
            tx.sort_values("timestamp", ascending=False)[["timestamp", "category", "amount", "channel"]]
            .head(limit)
            .reset_index(drop=True)
        )
        recent["category_label"] = recent["category"].astype(str).map(translate_category)
        recent["channel_label"] = recent["channel"].astype(str).map(translate_channel)
        return recent

    def get_user_monthly_spend(self, user_id: str) -> pd.DataFrame:
        if self.transactions_df is None:
            raise RuntimeError("Сервис рекомендаций не инициализирован.")

        tx = self.transactions_df[self.transactions_df["user_id"].astype(str) == user_id].copy()
        if tx.empty:
            return pd.DataFrame(columns=["month", "amount"])

        tx["timestamp"] = pd.to_datetime(tx["timestamp"])
        monthly = (
            tx.set_index("timestamp")
            .resample("MS")["amount"]
            .sum()
            .rename("amount")
            .reset_index()
            .rename(columns={"timestamp": "month"})
        )
        return monthly

    def get_user_positive_offers(self, user_id: str, limit: int = 5) -> pd.DataFrame:
        if self.interactions_df is None or self.offers_df is None:
            raise RuntimeError("Сервис рекомендаций не инициализирован.")

        accepted = self.interactions_df[
            (self.interactions_df["user_id"].astype(str) == user_id) & (self.interactions_df["label"] == 1)
        ].copy()
        if accepted.empty:
            return pd.DataFrame(columns=["timestamp", "offer_name", "product_type"])

        accepted["timestamp"] = pd.to_datetime(accepted["timestamp"])
        accepted = accepted.merge(
            self.offers_df[["offer_id", "offer_name", "product_type"]],
            on="offer_id",
            how="left",
        )
        accepted[["offer_name_label", "description_label"]] = accepted.apply(
            lambda row: pd.Series(
                translate_offer(str(row["offer_id"]), str(row["offer_name"]), "")
            ),
            axis=1,
        )
        accepted["product_type_label"] = accepted["product_type"].astype(str).map(translate_product_type)
        accepted = (
            accepted.sort_values("timestamp", ascending=False)[
                ["timestamp", "offer_name_label", "product_type_label"]
            ]
            .head(limit)
            .reset_index(drop=True)
        )
        return accepted

    def get_segment_benchmark(self, segment: str) -> pd.DataFrame:
        if self.segment_metrics_df is None:
            return pd.DataFrame(
                columns=["model", "segment", "precision_at_k", "recall_at_k", "map_at_k", "ndcg_at_k"]
            )
        return (
            self.segment_metrics_df[self.segment_metrics_df["segment"].astype(str) == segment]
            .sort_values("ndcg_at_k", ascending=False)
            .reset_index(drop=True)
        )
