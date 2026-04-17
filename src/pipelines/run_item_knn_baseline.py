from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR, SYNTHETIC_DATA_DIR
from src.evaluation.ranking_metrics import evaluate_ranking
from src.models.item_knn_recommender import ImplicitItemKNNRecommender
from src.pipelines.run_mf_baseline import (
    build_exclusion_map,
    build_test_ground_truth,
    build_train_interactions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run item-kNN baseline on synthetic interactions.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SYNTHETIC_DATA_DIR,
        help="Directory containing interactions.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory for metrics and predictions.",
    )
    parser.add_argument("--n-neighbors", type=int, default=10, help="Number of neighbors per item.")
    parser.add_argument("--top-k", type=int, default=5, help="Top-K recommendations.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    interactions_path = args.data_dir / "interactions.csv"
    if not interactions_path.exists():
        raise FileNotFoundError(
            f"{interactions_path} not found. Run run_baseline_pipeline first to generate synthetic data."
        )

    interactions_df = pd.read_csv(interactions_path)
    test_gt_df = build_test_ground_truth(interactions_df)
    train_interactions_df = build_train_interactions(interactions_df, test_gt_df)
    exclusion_map = build_exclusion_map(interactions_df, test_gt_df)
    user_ids = sorted(test_gt_df["user_id"].astype(str).unique().tolist())

    model = ImplicitItemKNNRecommender(n_neighbors=args.n_neighbors).fit(train_interactions_df)
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
    metrics["n_neighbors"] = int(args.n_neighbors)

    with (args.output_dir / "item_knn_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)

    recommendations_df.to_csv(args.output_dir / "item_knn_recommendations.csv", index=False)
    per_user_df.to_csv(args.output_dir / "item_knn_per_user_metrics.csv", index=False)

    print("[ok] item-kNN baseline finished")
    print(f"[ok] evaluated users: {metrics['n_users_evaluated']}")
    print(
        "[ok] metrics: "
        f"Precision@{args.top_k}={metrics['precision_at_k']:.4f}, "
        f"Recall@{args.top_k}={metrics['recall_at_k']:.4f}, "
        f"MAP@{args.top_k}={metrics['map_at_k']:.4f}, "
        f"NDCG@{args.top_k}={metrics['ndcg_at_k']:.4f}"
    )


if __name__ == "__main__":
    main()
