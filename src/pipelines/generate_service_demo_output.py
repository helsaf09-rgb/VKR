from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.config import REPORTS_DIR, SYNTHETIC_DATA_DIR
from src.service.app import get_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate demo output from recommendation service logic.")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--n-users", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    service = get_service()
    tx_df = pd.read_csv(SYNTHETIC_DATA_DIR / "transactions.csv")
    sample_users = sorted(tx_df["user_id"].astype(str).unique().tolist())[: args.n_users]

    output: dict[str, list[dict[str, str | float | int]]] = {}
    for user_id in sample_users:
        recommendation_df = service.recommend(user_id=user_id, top_k=args.top_k)
        output[user_id] = recommendation_df.to_dict(orient="records")

    output_path = args.output_dir / "service_demo_output.json"
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] demo output saved: {output_path}")


if __name__ == "__main__":
    main()
