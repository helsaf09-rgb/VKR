from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR, SYNTHETIC_DATA_DIR
from src.evaluation.ranking_metrics import evaluate_ranking
from src.models.time_decay_recommender import TimeDecayRecommender
from src.pipelines.run_baseline_pipeline import build_exclusion_map, build_test_ground_truth


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hyperparameter sweep for time-decay model.")
    parser.add_argument("--data-dir", type=Path, default=SYNTHETIC_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--top-k", type=int, default=5)
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

    decay_rates = [0.005, 0.01, 0.02, 0.04]
    short_term_days_values = [30, 60, 90]
    short_term_weights = [0.20, 0.35, 0.50]
    spend_weights = [0.60, 0.70, 0.80]

    rows: list[dict[str, float | int]] = []
    configs = list(product(decay_rates, short_term_days_values, short_term_weights, spend_weights))
    for decay_rate, short_days, short_weight, spend_weight in configs:
        freq_weight = 1.0 - spend_weight
        model = TimeDecayRecommender(
            decay_rate=float(decay_rate),
            short_term_days=int(short_days),
            short_term_weight=float(short_weight),
            spend_weight=float(spend_weight),
            freq_weight=float(freq_weight),
        ).fit(transactions_df, offers_df)

        recommendations_df = model.recommend_for_users(
            user_ids=user_ids,
            top_k=args.top_k,
            exclude_by_user=exclusion_map,
        )
        metrics, _ = evaluate_ranking(
            recommendations_df=recommendations_df,
            ground_truth_df=test_gt_df[["user_id", "offer_id"]],
            k=args.top_k,
        )
        rows.append(
            {
                "decay_rate": float(decay_rate),
                "short_term_days": int(short_days),
                "short_term_weight": float(short_weight),
                "spend_weight": float(spend_weight),
                "freq_weight": float(freq_weight),
                "precision_at_k": float(metrics["precision_at_k"]),
                "recall_at_k": float(metrics["recall_at_k"]),
                "map_at_k": float(metrics["map_at_k"]),
                "ndcg_at_k": float(metrics["ndcg_at_k"]),
            }
        )

    results_df = pd.DataFrame(rows).sort_values(
        ["ndcg_at_k", "map_at_k", "recall_at_k"], ascending=False
    )
    results_df.to_csv(args.output_dir / "time_decay_sweep_results.csv", index=False)

    best = results_df.iloc[0].to_dict()
    with (args.output_dir / "time_decay_best_config.txt").open("w", encoding="utf-8") as fp:
        for k, v in best.items():
            fp.write(f"{k}: {v}\n")

    print("[ok] time-decay sweep finished")
    print(f"[ok] tried configs: {len(results_df)}")
    print(
        "[ok] best: "
        f"decay_rate={best['decay_rate']}, "
        f"short_term_days={int(best['short_term_days'])}, "
        f"short_term_weight={best['short_term_weight']}, "
        f"spend_weight={best['spend_weight']}, "
        f"freq_weight={best['freq_weight']}"
    )
    print(
        "[ok] best metrics: "
        f"Precision@{args.top_k}={best['precision_at_k']:.4f}, "
        f"Recall@{args.top_k}={best['recall_at_k']:.4f}, "
        f"MAP@{args.top_k}={best['map_at_k']:.4f}, "
        f"NDCG@{args.top_k}={best['ndcg_at_k']:.4f}"
    )


if __name__ == "__main__":
    main()
