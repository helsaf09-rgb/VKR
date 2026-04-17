from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CATEGORIES,
    CATEGORY_AVG_AMOUNT,
    CATEGORY_SIGMA,
    CATEGORY_TO_MCCS,
    SYNTHETIC_DATA_DIR,
)


@dataclass(frozen=True)
class SimulationParams:
    n_users: int = 500
    avg_transactions: int = 120
    months: int = 12
    seed: int = 42


SEGMENT_CONFIG: dict[str, list[str]] = {
    "daily_life": ["groceries", "restaurants", "transport", "utilities"],
    "traveler": ["travel", "transport", "restaurants", "entertainment"],
    "family": ["groceries", "home", "healthcare", "education", "insurance"],
    "digital_pro": ["electronics", "money_transfer", "entertainment", "investments"],
    "investor": ["investments", "insurance", "money_transfer", "education"],
    "student": ["education", "entertainment", "transport", "restaurants", "electronics"],
}

SEGMENT_PROBABILITIES: dict[str, float] = {
    "daily_life": 0.26,
    "traveler": 0.14,
    "family": 0.20,
    "digital_pro": 0.14,
    "investor": 0.11,
    "student": 0.15,
}


def _build_alpha(segment: str) -> np.ndarray:
    alpha = np.full(len(CATEGORIES), 0.70, dtype=float)
    boost_categories = SEGMENT_CONFIG[segment]
    for category in boost_categories:
        alpha[CATEGORIES.index(category)] += 2.50
    return alpha


def generate_users(n_users: int, rng: np.random.Generator) -> pd.DataFrame:
    segment_names = list(SEGMENT_PROBABILITIES)
    segment_probs = np.array(list(SEGMENT_PROBABILITIES.values()), dtype=float)
    segment_probs = segment_probs / segment_probs.sum()

    rows: list[dict[str, float | str]] = []
    for i in range(n_users):
        segment = str(rng.choice(segment_names, p=segment_probs))
        alpha = _build_alpha(segment)
        preferences = rng.dirichlet(alpha)

        row: dict[str, float | str] = {
            "user_id": f"U{i + 1:05d}",
            "segment": segment,
        }
        for category, value in zip(CATEGORIES, preferences):
            row[f"pref_{category}"] = float(value)
        rows.append(row)

    return pd.DataFrame(rows)


def _sample_channel(category: str, rng: np.random.Generator) -> str:
    if category == "cash_withdrawal":
        return "atm"
    if category == "money_transfer":
        return "transfer"
    return str(rng.choice(["card", "online", "mobile"], p=[0.55, 0.30, 0.15]))


def generate_transactions(
    users_df: pd.DataFrame,
    avg_transactions: int,
    months: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    end_ts = pd.Timestamp.utcnow().tz_localize(None).floor("h")
    start_ts = end_ts - pd.DateOffset(months=months)
    total_seconds = int((end_ts - start_ts).total_seconds())

    rows: list[dict[str, str | int | float]] = []
    tx_id = 1

    pref_columns = [f"pref_{category}" for category in CATEGORIES]
    for user in users_df.itertuples(index=False):
        n_tx = max(20, int(rng.poisson(avg_transactions)))
        user_preferences = np.array([getattr(user, col) for col in pref_columns], dtype=float)

        chosen_categories = rng.choice(CATEGORIES, size=n_tx, p=user_preferences)
        offsets = rng.integers(0, total_seconds, size=n_tx)
        timestamps = start_ts + pd.to_timedelta(offsets, unit="s")

        for category, timestamp in zip(chosen_categories, timestamps):
            mean_amount = CATEGORY_AVG_AMOUNT[category]
            sigma = CATEGORY_SIGMA[category]
            amount = float(rng.lognormal(mean=np.log(mean_amount), sigma=sigma))
            amount = round(max(amount, 30.0), 2)

            rows.append(
                {
                    "transaction_id": f"T{tx_id:09d}",
                    "user_id": user.user_id,
                    "timestamp": timestamp,
                    "mcc": int(rng.choice(CATEGORY_TO_MCCS[category])),
                    "category": category,
                    "amount": amount,
                    "channel": _sample_channel(category, rng),
                }
            )
            tx_id += 1

    transactions_df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return transactions_df


def save_synthetic_transactions(
    users_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    users_df.to_csv(output_dir / "users.csv", index=False)
    transactions_df.to_csv(output_dir / "transactions.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic bank transaction data.")
    parser.add_argument("--n-users", type=int, default=500, help="Number of users to simulate.")
    parser.add_argument(
        "--avg-transactions",
        type=int,
        default=120,
        help="Average number of transactions per user.",
    )
    parser.add_argument("--months", type=int, default=12, help="Length of history in months.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SYNTHETIC_DATA_DIR,
        help="Directory where users.csv and transactions.csv will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = SimulationParams(
        n_users=args.n_users,
        avg_transactions=args.avg_transactions,
        months=args.months,
        seed=args.seed,
    )

    rng = np.random.default_rng(params.seed)
    users_df = generate_users(params.n_users, rng)
    transactions_df = generate_transactions(
        users_df=users_df,
        avg_transactions=params.avg_transactions,
        months=params.months,
        rng=rng,
    )
    save_synthetic_transactions(users_df, transactions_df, args.output_dir)

    print(f"[ok] users: {len(users_df)}")
    print(f"[ok] transactions: {len(transactions_df)}")
    print(f"[ok] saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
