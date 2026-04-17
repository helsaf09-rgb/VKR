from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR, SYNTHETIC_DATA_DIR
from src.evaluation.ranking_metrics import evaluate_ranking
from src.models.neural_cf_recommender import NeuralCFConfig, NeuralCFRecommender
from src.pipelines.run_mf_baseline import build_exclusion_map, build_test_ground_truth, build_train_interactions


def _parse_hidden_dims(raw_value: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in raw_value.split(",") if part.strip())
    if not values:
        raise ValueError("At least one hidden layer size must be provided.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Neural Collaborative Filtering baseline.")
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
    parser.add_argument("--top-k", type=int, default=5, help="Top-K recommendations.")
    parser.add_argument("--embedding-dim", type=int, default=16, help="Embedding size for users and items.")
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="32,16",
        help="Comma-separated hidden dimensions, e.g. '32,16'.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1024, help="Mini-batch size.")
    parser.add_argument("--negative-samples", type=int, default=2, help="Negative samples per positive pair.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
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

    config = NeuralCFConfig(
        embedding_dim=int(args.embedding_dim),
        hidden_dims=_parse_hidden_dims(args.hidden_dims),
        learning_rate=float(args.learning_rate),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        negative_samples=int(args.negative_samples),
        random_state=int(args.seed),
    )
    model = NeuralCFRecommender(config=config).fit(train_interactions_df)
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
    metrics.update(
        {
            "embedding_dim": int(config.embedding_dim),
            "hidden_dims": list(config.hidden_dims),
            "learning_rate": float(config.learning_rate),
            "epochs": int(config.epochs),
            "batch_size": int(config.batch_size),
            "negative_samples": int(config.negative_samples),
            "training_loss_final": float(model.loss_history_[-1]) if model.loss_history_ else None,
        }
    )

    with (args.output_dir / "ncf_baseline_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)

    recommendations_df.to_csv(args.output_dir / "ncf_recommendations.csv", index=False)
    per_user_df.to_csv(args.output_dir / "ncf_per_user_metrics.csv", index=False)

    print("[ok] neural CF baseline finished")
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
