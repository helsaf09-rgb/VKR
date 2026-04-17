from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR, SYNTHETIC_DATA_DIR
from src.evaluation.ranking_metrics import evaluate_ranking
from src.models.time_decay_recommender import TimeDecayRecommender
from src.pipelines.run_baseline_pipeline import build_exclusion_map, build_test_ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run time-decay recommendation model.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SYNTHETIC_DATA_DIR,
        help="Directory with synthetic users/transactions/offers/interactions files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPORTS_DIR,
        help="Directory where outputs are stored.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-K recommendations.")
    parser.add_argument("--decay-rate", type=float, default=0.02, help="Exponential time-decay rate.")
    parser.add_argument("--short-term-days", type=int, default=60, help="Recent window length in days.")
    parser.add_argument(
        "--short-term-weight",
        type=float,
        default=0.35,
        help="Blending weight for short-term profile.",
    )
    parser.add_argument("--spend-weight", type=float, default=0.70, help="Weight of spend share.")
    parser.add_argument("--freq-weight", type=float, default=0.30, help="Weight of frequency share.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    transactions_df = pd.read_csv(args.data_dir / "transactions.csv")
    offers_df = pd.read_csv(args.data_dir / "offers.csv")
    interactions_df = pd.read_csv(args.data_dir / "interactions.csv")

    test_gt_df = build_test_ground_truth(interactions_df)
    exclusion_map = build_exclusion_map(interactions_df, test_gt_df)
    user_ids = sorted(test_gt_df["user_id"].astype(str).unique().tolist())

    model = TimeDecayRecommender(
        decay_rate=args.decay_rate,
        short_term_days=args.short_term_days,
        short_term_weight=args.short_term_weight,
        spend_weight=args.spend_weight,
        freq_weight=args.freq_weight,
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
    metrics.update(
        {
            "decay_rate": args.decay_rate,
            "short_term_days": args.short_term_days,
            "short_term_weight": args.short_term_weight,
            "spend_weight": args.spend_weight,
            "freq_weight": args.freq_weight,
        }
    )

    with (args.output_dir / "time_decay_metrics.json").open("w", encoding="utf-8") as fp:
        json.dump(metrics, fp, ensure_ascii=False, indent=2)

    recommendations_df.to_csv(args.output_dir / "time_decay_recommendations.csv", index=False)
    per_user_df.to_csv(args.output_dir / "time_decay_per_user_metrics.csv", index=False)

    print("[ok] time-decay model finished")
    print(
        "[ok] params: "
        f"decay_rate={args.decay_rate}, "
        f"short_term_days={args.short_term_days}, "
        f"short_term_weight={args.short_term_weight}"
    )
    print(
        "[ok] metrics: "
        f"Precision@{args.top_k}={metrics['precision_at_k']:.4f}, "
        f"Recall@{args.top_k}={metrics['recall_at_k']:.4f}, "
        f"MAP@{args.top_k}={metrics['map_at_k']:.4f}, "
        f"NDCG@{args.top_k}={metrics['ndcg_at_k']:.4f}"
    )


if __name__ == "__main__":
    main()
