from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CATEGORIES, SYNTHETIC_DATA_DIR


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _offer_category_matrix(offers_df: pd.DataFrame) -> np.ndarray:
    cat_columns = [f"cat_{category}" for category in CATEGORIES]
    if not set(cat_columns).issubset(set(offers_df.columns)):
        matrix = np.zeros((len(offers_df), len(CATEGORIES)), dtype=float)
        for i, target in enumerate(offers_df["target_categories"].astype(str).tolist()):
            targets = set(target.split("|"))
            for j, category in enumerate(CATEGORIES):
                matrix[i, j] = float(category in targets)
    else:
        matrix = offers_df[cat_columns].to_numpy(dtype=float)

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return matrix / row_sums


def _user_preference_matrix(users_df: pd.DataFrame) -> np.ndarray:
    pref_columns = [f"pref_{category}" for category in CATEGORIES]
    if not set(pref_columns).issubset(set(users_df.columns)):
        raise ValueError("users_df must include pref_<category> columns.")

    matrix = users_df[pref_columns].to_numpy(dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return matrix / row_sums


@dataclass(frozen=True)
class InteractionSimulationParams:
    seed: int = 42
    min_impressions: int = 6
    max_impressions: int = 10
    lookback_days: int = 120
    score_noise_std: float = 0.08
    logistic_slope: float = 7.5
    logistic_center: float = 0.30


def simulate_interactions(
    users_df: pd.DataFrame,
    offers_df: pd.DataFrame,
    seed: int = 42,
    min_impressions: int = 6,
    max_impressions: int = 10,
) -> pd.DataFrame:
    params = InteractionSimulationParams(
        seed=seed,
        min_impressions=min_impressions,
        max_impressions=max_impressions,
    )
    rng = np.random.default_rng(seed)
    user_matrix = _user_preference_matrix(users_df)
    offer_matrix = _offer_category_matrix(offers_df)

    n_offers = len(offers_df)
    offer_ids = offers_df["offer_id"].astype(str).to_numpy()
    user_ids = users_df["user_id"].astype(str).to_numpy()

    end_ts = pd.Timestamp.utcnow().tz_localize(None).floor("h")
    start_ts = end_ts - pd.Timedelta(days=params.lookback_days)
    total_seconds = int((end_ts - start_ts).total_seconds())

    rows: list[dict[str, str | int | float]] = []
    interaction_idx = 1
    max_sample = min(max_impressions, n_offers)

    for user_idx, user_id in enumerate(user_ids):
        user_vector = user_matrix[user_idx]
        base_scores = offer_matrix @ user_vector
        noisy_scores = np.clip(
            base_scores + rng.normal(0.0, params.score_noise_std, size=n_offers),
            0.0,
            1.0,
        )
        conversion_probs = _sigmoid(
            params.logistic_slope * (noisy_scores - params.logistic_center)
        )

        n_impressions = int(rng.integers(params.min_impressions, max_sample + 1))
        shown_indices = rng.choice(n_offers, size=n_impressions, replace=False)
        shown_probs = conversion_probs[shown_indices]
        labels = (rng.random(n_impressions) < shown_probs).astype(int)

        if labels.sum() == 0:
            best_local_idx = int(np.argmax(noisy_scores[shown_indices]))
            labels[best_local_idx] = 1

        offsets = np.sort(rng.integers(0, total_seconds, size=n_impressions))
        timestamps = start_ts + pd.to_timedelta(offsets, unit="s")

        for local_idx, offer_idx in enumerate(shown_indices):
            rows.append(
                {
                    "interaction_id": f"I{interaction_idx:09d}",
                    "user_id": user_id,
                    "offer_id": str(offer_ids[offer_idx]),
                    "timestamp": timestamps[local_idx],
                    "label": int(labels[local_idx]),
                    "true_score": round(float(noisy_scores[offer_idx]), 6),
                    "conversion_probability": round(float(conversion_probs[offer_idx]), 6),
                }
            )
            interaction_idx += 1

    interactions_df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return interactions_df


def save_interactions(interactions_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "interactions.csv"
    interactions_df.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate user-offer interactions.")
    parser.add_argument(
        "--users-path",
        type=Path,
        default=SYNTHETIC_DATA_DIR / "users.csv",
        help="Path to users.csv",
    )
    parser.add_argument(
        "--offers-path",
        type=Path,
        default=SYNTHETIC_DATA_DIR / "offers.csv",
        help="Path to offers.csv",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SYNTHETIC_DATA_DIR,
        help="Directory where interactions.csv will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    users_df = pd.read_csv(args.users_path)
    offers_df = pd.read_csv(args.offers_path)
    interactions_df = simulate_interactions(users_df, offers_df, seed=args.seed)
    output_path = save_interactions(interactions_df, args.output_dir)

    positive_rate = float(interactions_df["label"].mean())
    print(f"[ok] interactions: {len(interactions_df)}")
    print(f"[ok] positive-rate: {positive_rate:.3f}")
    print(f"[ok] saved to: {output_path}")


if __name__ == "__main__":
    main()
