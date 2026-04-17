from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR, SYNTHETIC_DATA_DIR
from src.evaluation.ranking_metrics import evaluate_ranking
from src.models.hybrid_semantic_recommender import HybridSemanticRecommender
from src.pipelines.run_baseline_pipeline import build_exclusion_map, build_test_ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid semantic baseline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SYNTHETIC_DATA_DIR,
        help="Directory with synthetic users/transactions/offers/interactions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for output reports.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-K recommendations.")
    parser.add_argument(
        "--profile-weight",
        type=float,
        default=0.70,
        help="Weight for profile score in final blend.",
    )
    parser.add_argument(
        "--semantic-weight",
        type=float,
        default=0.30,
        help="Weight for semantic score in final blend.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    users_df = pd.read_csv(args.data_dir / "users.csv")
    transactions_df = pd.read_csv(args.data_dir / "transactions.csv")
    offers_df = pd.read_csv(args.data_dir / "offers.csv")
    interactions_df = pd.read_csv(args.data_dir / "interactions.csv")

    test_gt_df = build_test_ground_truth(interactions_df)
    exclusion_map = build_exclusion_map(interactions_df, test_gt_df)
    user_ids = sorted(test_gt_df["user_id"].astype(str).unique().tolist())

    model = HybridSemanticRecommender(
        profile_weight=args.profile_weight,
        semantic_weight=args.semantic_weight,
    ).fit(transactions_df, offers_df)

    recommendations_df = model.recommend_for_users(
        user_ids=user_ids,
        top_k=args.top_k,
        exclude_by_user=exclusion_map,
    )
    metrics, per_user_df = evaluate_ranking(
        recommendations_df=recommendations_df,
        ground_truth_df=test_gt_df[["user_id", "offer_id"]],
        k=args.top_k,
    )

    with (args.output_dir / "hybrid_baseline_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)

    recommendations_df.to_csv(args.output_dir / "hybrid_recommendations.csv", index=False)
    per_user_df.to_csv(args.output_dir / "hybrid_per_user_metrics.csv", index=False)

    print("[ok] hybrid baseline finished")
    print(f"[ok] users loaded: {len(users_df)}")
    print(
        "[ok] metrics: "
        f"Precision@{args.top_k}={metrics['precision_at_k']:.4f}, "
        f"Recall@{args.top_k}={metrics['recall_at_k']:.4f}, "
        f"MAP@{args.top_k}={metrics['map_at_k']:.4f}, "
        f"NDCG@{args.top_k}={metrics['ndcg_at_k']:.4f}"
    )


if __name__ == "__main__":
    main()
