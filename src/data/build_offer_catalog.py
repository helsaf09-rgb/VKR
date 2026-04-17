from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import CATEGORIES, OFFER_BLUEPRINTS, SYNTHETIC_DATA_DIR


def build_offer_catalog() -> pd.DataFrame:
    offers_df = pd.DataFrame(OFFER_BLUEPRINTS).copy()

    for category in CATEGORIES:
        col = f"cat_{category}"
        offers_df[col] = offers_df["target_categories"].str.contains(category).astype(int)

    offers_df["n_target_categories"] = offers_df[[f"cat_{c}" for c in CATEGORIES]].sum(axis=1)
    return offers_df


def save_offer_catalog(offers_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "offers.csv"
    offers_df.to_csv(output_path, index=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build synthetic offer catalog.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SYNTHETIC_DATA_DIR,
        help="Directory where offers.csv will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    offers_df = build_offer_catalog()
    output_path = save_offer_catalog(offers_df, args.output_dir)
    print(f"[ok] offers: {len(offers_df)}")
    print(f"[ok] saved to: {output_path}")


if __name__ == "__main__":
    main()
