from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import CATEGORY_AVG_AMOUNT, CATEGORY_SIGMA
from src.data.generate_synthetic_transactions import SEGMENT_CONFIG, SEGMENT_PROBABILITIES


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _round_dict(values: dict[str, float], digits: int = 4) -> dict[str, float]:
    return {key: round(float(value), digits) for key, value in values.items()}


def build_synthetic_data_manifest(
    users_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    offers_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    generation_params: dict[str, Any],
    interaction_params: dict[str, Any],
) -> dict[str, Any]:
    users = users_df.copy()
    transactions = transactions_df.copy()
    offers = offers_df.copy()
    interactions = interactions_df.copy()

    transactions["timestamp"] = pd.to_datetime(transactions["timestamp"])
    interactions["timestamp"] = pd.to_datetime(interactions["timestamp"])

    tx_per_user = transactions.groupby("user_id").size()
    segment_mix = (
        users["segment"]
        .value_counts(dropna=False)
        .rename_axis("segment")
        .reset_index(name="n_users")
        .sort_values("segment")
        .reset_index(drop=True)
    )
    segment_mix["share"] = segment_mix["n_users"] / max(len(users), 1)

    category_mix = (
        transactions.groupby("category")
        .agg(
            n_transactions=("transaction_id", "count"),
            mean_amount=("amount", "mean"),
            median_amount=("amount", "median"),
        )
        .reset_index()
        .sort_values("n_transactions", ascending=False)
        .reset_index(drop=True)
    )
    category_mix["share"] = category_mix["n_transactions"] / max(len(transactions), 1)

    impressions_per_user = interactions.groupby("user_id").size()
    positives_per_user = interactions.groupby("user_id")["label"].sum()
    product_type_distribution = (
        offers["product_type"]
        .value_counts(dropna=False)
        .sort_index()
        .astype(int)
        .to_dict()
    )

    manifest: dict[str, Any] = {
        "generation": generation_params,
        "interaction_simulation": interaction_params,
        "artifacts": {
            "users_file": "users.csv",
            "transactions_file": "transactions.csv",
            "offers_file": "offers.csv",
            "interactions_file": "interactions.csv",
            "holdout_file": "test_positive_ground_truth.csv",
        },
        "segments": {
            "probabilities": _round_dict(SEGMENT_PROBABILITIES),
            "boosted_categories": SEGMENT_CONFIG,
            "observed_mix": [
                {
                    "segment": str(row.segment),
                    "n_users": int(row.n_users),
                    "share": round(float(row.share), 4),
                }
                for row in segment_mix.itertuples(index=False)
            ],
        },
        "transaction_model": {
            "category_avg_amount": _round_dict(CATEGORY_AVG_AMOUNT, digits=2),
            "category_sigma": _round_dict(CATEGORY_SIGMA, digits=2),
            "summary": {
                "n_users": int(len(users)),
                "n_transactions": int(len(transactions)),
                "avg_transactions_per_user": round(float(tx_per_user.mean()), 2),
                "median_transactions_per_user": round(float(tx_per_user.median()), 2),
                "date_min": transactions["timestamp"].min().isoformat(),
                "date_max": transactions["timestamp"].max().isoformat(),
                "n_categories_observed": int(transactions["category"].nunique()),
            },
            "category_mix": [
                {
                    "category": str(row.category),
                    "n_transactions": int(row.n_transactions),
                    "share": round(float(row.share), 4),
                    "mean_amount": round(float(row.mean_amount), 2),
                    "median_amount": round(float(row.median_amount), 2),
                }
                for row in category_mix.itertuples(index=False)
            ],
        },
        "offer_catalog": {
            "n_offers": int(len(offers)),
            "product_type_distribution": product_type_distribution,
            "avg_target_categories_per_offer": round(float(offers["n_target_categories"].mean()), 2),
        },
        "interaction_model": {
            "n_interactions": int(len(interactions)),
            "positive_rate": round(float(interactions["label"].mean()), 4),
            "avg_impressions_per_user": round(float(impressions_per_user.mean()), 2),
            "avg_positive_offers_per_user": round(float(positives_per_user.mean()), 2),
            "holdout_users": int(holdout_df["user_id"].nunique()),
            "holdout_pairs": int(len(holdout_df)),
        },
    }
    return manifest


def save_synthetic_data_manifest(manifest: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def write_synthetic_data_report(manifest: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generation = manifest["generation"]
    interaction = manifest["interaction_simulation"]
    tx_summary = manifest["transaction_model"]["summary"]
    interaction_summary = manifest["interaction_model"]
    offers = manifest["offer_catalog"]
    segment_rows = manifest["segments"]["observed_mix"]
    category_rows = manifest["transaction_model"]["category_mix"][:5]

    lines = [
        "# Synthetic Data Report",
        "",
        "## Why this artifact exists",
        "- Documents how the synthetic banking dataset was generated.",
        "- Makes the simulation assumptions explicit for thesis defense and reproducibility.",
        "",
        "## Generation Parameters",
        f"- Users: {generation['n_users']}",
        f"- Average transactions per user: {generation['avg_transactions']}",
        f"- History window (months): {generation['months']}",
        f"- Random seed: {generation['seed']}",
        "",
        "## Interaction Simulation Parameters",
        f"- Impression range per user: {interaction['min_impressions']}..{interaction['max_impressions']}",
        f"- Score noise std: {interaction['score_noise_std']}",
        f"- Logistic slope: {interaction['logistic_slope']}",
        f"- Logistic center: {interaction['logistic_center']}",
        f"- Interaction seed: {interaction['seed']}",
        "",
        "## Observed Dataset Summary",
        f"- Users generated: {tx_summary['n_users']}",
        f"- Transactions generated: {tx_summary['n_transactions']}",
        f"- Avg transactions per user: {tx_summary['avg_transactions_per_user']}",
        f"- Median transactions per user: {tx_summary['median_transactions_per_user']}",
        f"- Observed categories: {tx_summary['n_categories_observed']}",
        f"- Transaction range: {tx_summary['date_min']} .. {tx_summary['date_max']}",
        f"- Offers in catalog: {offers['n_offers']}",
        f"- Avg target categories per offer: {offers['avg_target_categories_per_offer']}",
        f"- Simulated interactions: {interaction_summary['n_interactions']}",
        f"- Positive interaction rate: {interaction_summary['positive_rate']}",
        f"- Avg impressions per user: {interaction_summary['avg_impressions_per_user']}",
        f"- Holdout users: {interaction_summary['holdout_users']}",
        "",
        "## Observed Segment Mix",
    ]

    for row in segment_rows:
        lines.append(f"- {row['segment']}: {row['n_users']} users ({row['share']:.2%})")

    lines += [
        "",
        "## Top Transaction Categories",
    ]

    for row in category_rows:
        lines.append(
            (
                f"- {row['category']}: {row['n_transactions']} tx "
                f"({row['share']:.2%}), mean amount={row['mean_amount']:.2f}, "
                f"median amount={row['median_amount']:.2f}"
            )
        )

    product_types = ", ".join(
        f"{name}={count}" for name, count in manifest["offer_catalog"]["product_type_distribution"].items()
    )
    lines += [
        "",
        "## Offer Catalog Mix",
        f"- Product types: {product_types}",
        "",
        "## Notes",
        "- The dataset remains synthetic and should be reported as a controlled simulation, not as an empirical banking sample.",
        "- Real-data validation is still needed to support external validity claims.",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
