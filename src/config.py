from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DATA_DIR = DATA_DIR / "synthetic"
REAL_DATA_DIR = DATA_DIR / "real"
REPORTS_DIR = PROJECT_ROOT / "reports"

CATEGORIES: list[str] = [
    "groceries",
    "restaurants",
    "transport",
    "travel",
    "utilities",
    "healthcare",
    "education",
    "entertainment",
    "electronics",
    "fashion",
    "home",
    "investments",
    "insurance",
    "cash_withdrawal",
    "money_transfer",
]

CATEGORY_TO_MCCS: dict[str, list[int]] = {
    "groceries": [5411, 5499],
    "restaurants": [5812, 5814],
    "transport": [4111, 4121, 4131],
    "travel": [4511, 7011, 4722],
    "utilities": [4900],
    "healthcare": [8062, 5912],
    "education": [8220, 8299],
    "entertainment": [7832, 7995, 5941],
    "electronics": [5732, 5734],
    "fashion": [5651, 5691],
    "home": [5712, 5200],
    "investments": [6211],
    "insurance": [6300],
    "cash_withdrawal": [6011],
    "money_transfer": [4829],
}

MCC_TO_CATEGORY: dict[int, str] = {
    mcc: category
    for category, mcc_list in CATEGORY_TO_MCCS.items()
    for mcc in mcc_list
}

CATEGORY_AVG_AMOUNT: dict[str, float] = {
    "groceries": 1800.0,
    "restaurants": 1400.0,
    "transport": 650.0,
    "travel": 18000.0,
    "utilities": 4200.0,
    "healthcare": 3100.0,
    "education": 6800.0,
    "entertainment": 2200.0,
    "electronics": 15000.0,
    "fashion": 5200.0,
    "home": 7300.0,
    "investments": 13000.0,
    "insurance": 9600.0,
    "cash_withdrawal": 4200.0,
    "money_transfer": 3600.0,
}

CATEGORY_SIGMA: dict[str, float] = {
    "groceries": 0.45,
    "restaurants": 0.50,
    "transport": 0.40,
    "travel": 0.75,
    "utilities": 0.35,
    "healthcare": 0.55,
    "education": 0.80,
    "entertainment": 0.60,
    "electronics": 0.90,
    "fashion": 0.85,
    "home": 0.70,
    "investments": 1.10,
    "insurance": 0.65,
    "cash_withdrawal": 0.50,
    "money_transfer": 0.60,
}

OFFER_BLUEPRINTS: list[dict[str, str]] = [
    {
        "offer_id": "O001",
        "offer_name": "Smart Daily Cashback Card",
        "product_type": "card",
        "target_categories": "groceries|restaurants|transport",
        "description": "Cashback for everyday spending with bonus categories for food and city mobility.",
    },
    {
        "offer_id": "O002",
        "offer_name": "Travel Miles Premium Card",
        "product_type": "card",
        "target_categories": "travel|transport|restaurants",
        "description": "Miles and airport perks for frequent travelers and transport-heavy users.",
    },
    {
        "offer_id": "O003",
        "offer_name": "Beginner Investment Account",
        "product_type": "investment",
        "target_categories": "investments|money_transfer|education",
        "description": "Entry-level brokerage account with guided portfolio and learning path.",
    },
    {
        "offer_id": "O004",
        "offer_name": "Flexible Personal Loan",
        "product_type": "credit",
        "target_categories": "home|electronics|fashion",
        "description": "Unsecured loan with flexible repayment for lifestyle and household purchases.",
    },
    {
        "offer_id": "O005",
        "offer_name": "Mortgage Refinance Program",
        "product_type": "credit",
        "target_categories": "home|utilities|insurance",
        "description": "Refinancing option for clients with stable home and utility-related spending.",
    },
    {
        "offer_id": "O006",
        "offer_name": "Health and Family Insurance Bundle",
        "product_type": "insurance",
        "target_categories": "healthcare|insurance|home",
        "description": "Comprehensive health and family protection package with digital claims.",
    },
    {
        "offer_id": "O007",
        "offer_name": "Auto and Mobility Protection",
        "product_type": "insurance",
        "target_categories": "transport|insurance|travel",
        "description": "Insurance solution for active drivers and high transportation spenders.",
    },
    {
        "offer_id": "O008",
        "offer_name": "Family Savings Deposit",
        "product_type": "deposit",
        "target_categories": "groceries|home|utilities",
        "description": "Goal-based deposit product for stable monthly household planners.",
    },
    {
        "offer_id": "O009",
        "offer_name": "EdTech Learning Subscription",
        "product_type": "partner",
        "target_categories": "education|electronics|money_transfer",
        "description": "Discounted learning platform access for upskilling and digital education.",
    },
    {
        "offer_id": "O010",
        "offer_name": "Premium Lifestyle Subscription",
        "product_type": "subscription",
        "target_categories": "entertainment|restaurants|fashion",
        "description": "Subscription with partner benefits in lifestyle, dining, and entertainment.",
    },
    {
        "offer_id": "O011",
        "offer_name": "Utility AutoPay Cashback",
        "product_type": "card",
        "target_categories": "utilities|home|money_transfer",
        "description": "Auto-payment card with bonuses for utilities and recurring bills.",
    },
    {
        "offer_id": "O012",
        "offer_name": "Digital Security Package",
        "product_type": "service",
        "target_categories": "electronics|money_transfer|investments",
        "description": "Fraud monitoring and transaction protection for digitally active clients.",
    },
    {
        "offer_id": "O013",
        "offer_name": "Student Smart Start Card",
        "product_type": "card",
        "target_categories": "education|transport|entertainment",
        "description": "Student-focused card with discounts in education, commuting, and media.",
    },
    {
        "offer_id": "O014",
        "offer_name": "Cash Management Plus",
        "product_type": "service",
        "target_categories": "cash_withdrawal|money_transfer|groceries",
        "description": "Fee optimization package for cash-heavy and transfer-heavy activity.",
    },
    {
        "offer_id": "O015",
        "offer_name": "Balanced Finance Bundle",
        "product_type": "bundle",
        "target_categories": "investments|insurance|utilities",
        "description": "Bundle for financially disciplined users balancing risk and stability.",
    },
]
