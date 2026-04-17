from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import REPORTS_DIR, SYNTHETIC_DATA_DIR
from src.data.build_offer_catalog import build_offer_catalog, save_offer_catalog
from src.data.generate_synthetic_transactions import (
    generate_transactions,
    generate_users,
    save_synthetic_transactions,
)
from src.data.simulate_interactions import InteractionSimulationParams, save_interactions, simulate_interactions
from src.data.synthetic_reporting import (
    build_synthetic_data_manifest,
    save_synthetic_data_manifest,
    write_synthetic_data_report,
)
from src.evaluation.ranking_metrics import evaluate_ranking
from src.models.baseline_recommender import TransactionSimilarityRecommender


def build_test_ground_truth(interactions_df: pd.DataFrame) -> pd.DataFrame:
    positives = interactions_df[interactions_df["label"] == 1].copy()
    if positives.empty:
        return pd.DataFrame(columns=["user_id", "offer_id", "timestamp"])

    positives["timestamp"] = pd.to_datetime(positives["timestamp"])
    test_gt = (
        positives.sort_values("timestamp")
        .groupby("user_id", as_index=False)
        .tail(1)[["user_id", "offer_id", "timestamp"]]
        .sort_values(["user_id", "timestamp"])
        .reset_index(drop=True)
    )
    return test_gt


def build_exclusion_map(
    interactions_df: pd.DataFrame,
    test_ground_truth_df: pd.DataFrame,
) -> dict[str, set[str]]:
    positives = interactions_df[interactions_df["label"] == 1][["user_id", "offer_id"]].copy()
    holdout_pairs = set(
        tuple(x) for x in test_ground_truth_df[["user_id", "offer_id"]].itertuples(index=False, name=None)
    )

    exclusion_map: dict[str, set[str]] = {}
    for user_id, offer_id in positives.itertuples(index=False):
        pair = (str(user_id), str(offer_id))
        if pair in holdout_pairs:
            continue
        exclusion_map.setdefault(str(user_id), set()).add(str(offer_id))
    return exclusion_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run synthetic baseline recommendation pipeline.")
    parser.add_argument("--n-users", type=int, default=500, help="Number of simulated users.")
    parser.add_argument(
        "--avg-transactions",
        type=int,
        default=120,
        help="Average number of transactions per user.",
    )
    parser.add_argument("--months", type=int, default=12, help="Length of transaction history in months.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K recommendations for evaluation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SYNTHETIC_DATA_DIR,
        help="Directory for synthetic csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for metrics and recommendation outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    users_df = generate_users(args.n_users, rng)
    transactions_df = generate_transactions(
        users_df=users_df,
        avg_transactions=args.avg_transactions,
        months=args.months,
        rng=rng,
    )
    save_synthetic_transactions(users_df, transactions_df, args.data_dir)

    offers_df = build_offer_catalog()
    save_offer_catalog(offers_df, args.data_dir)

    interaction_params = InteractionSimulationParams(seed=args.seed + 1)
    interactions_df = simulate_interactions(users_df, offers_df, seed=interaction_params.seed)
    save_interactions(interactions_df, args.data_dir)

    test_gt_df = build_test_ground_truth(interactions_df)
    test_gt_df.to_csv(args.data_dir / "test_positive_ground_truth.csv", index=False)

    synthetic_manifest = build_synthetic_data_manifest(
        users_df=users_df,
        transactions_df=transactions_df,
        offers_df=offers_df,
        interactions_df=interactions_df,
        holdout_df=test_gt_df,
        generation_params={
            "n_users": int(args.n_users),
            "avg_transactions": int(args.avg_transactions),
            "months": int(args.months),
            "seed": int(args.seed),
        },
        interaction_params={
            "seed": int(interaction_params.seed),
            "min_impressions": int(interaction_params.min_impressions),
            "max_impressions": int(interaction_params.max_impressions),
            "lookback_days": int(interaction_params.lookback_days),
            "score_noise_std": float(interaction_params.score_noise_std),
            "logistic_slope": float(interaction_params.logistic_slope),
            "logistic_center": float(interaction_params.logistic_center),
        },
    )
    manifest_path = save_synthetic_data_manifest(
        synthetic_manifest,
        args.data_dir / "manifest.json",
    )
    synthetic_report_path = write_synthetic_data_report(
        synthetic_manifest,
        args.output_dir / "synthetic_data_report.md",
    )

    recommender = TransactionSimilarityRecommender().fit(transactions_df, offers_df)
    user_ids = sorted(test_gt_df["user_id"].astype(str).unique().tolist())
    exclusion_map = build_exclusion_map(interactions_df, test_gt_df)
    recommendations_df = recommender.recommend_for_users(
        user_ids=user_ids,
        top_k=args.top_k,
        exclude_by_user=exclusion_map,
    )

    metrics, per_user_metrics_df = evaluate_ranking(
        recommendations_df=recommendations_df,
        ground_truth_df=test_gt_df[["user_id", "offer_id"]],
        k=args.top_k,
    )

    with (args.output_dir / "baseline_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)

    per_user_metrics_df.to_csv(args.output_dir / "per_user_metrics.csv", index=False)
    recommendations_df.to_csv(args.output_dir / "sample_recommendations.csv", index=False)

    print("[ok] pipeline finished")
    print(f"[ok] users: {len(users_df)}")
    print(f"[ok] transactions: {len(transactions_df)}")
    print(f"[ok] offers: {len(offers_df)}")
    print(f"[ok] interactions: {len(interactions_df)}")
    print(f"[ok] evaluated users: {metrics['n_users_evaluated']}")
    print(
        "[ok] metrics: "
        f"Precision@{args.top_k}={metrics['precision_at_k']:.4f}, "
        f"Recall@{args.top_k}={metrics['recall_at_k']:.4f}, "
        f"MAP@{args.top_k}={metrics['map_at_k']:.4f}, "
        f"NDCG@{args.top_k}={metrics['ndcg_at_k']:.4f}"
    )
    print(f"[ok] synthetic data dir: {args.data_dir}")
    print(f"[ok] reports dir: {args.output_dir}")
    print(f"[ok] synthetic manifest: {manifest_path}")
    print(f"[ok] synthetic report: {synthetic_report_path}")


if __name__ == "__main__":
    main()
