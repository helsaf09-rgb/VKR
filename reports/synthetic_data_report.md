# Synthetic Data Report

## Why this artifact exists
- Documents how the synthetic banking dataset was generated.
- Makes the simulation assumptions explicit for thesis defense and reproducibility.

## Generation Parameters
- Users: 800
- Average transactions per user: 140
- History window (months): 12
- Random seed: 42

## Interaction Simulation Parameters
- Impression range per user: 6..10
- Score noise std: 0.08
- Logistic slope: 7.5
- Logistic center: 0.3
- Interaction seed: 43

## Observed Dataset Summary
- Users generated: 800
- Transactions generated: 112626
- Avg transactions per user: 140.78
- Median transactions per user: 141.0
- Observed categories: 15
- Transaction range: 2025-04-04T06:01:01 .. 2026-04-04T05:52:48
- Offers in catalog: 15
- Avg target categories per offer: 3.0
- Simulated interactions: 6442
- Positive interaction rate: 0.202
- Avg impressions per user: 8.05
- Holdout users: 800

## Observed Segment Mix
- daily_life: 205 users (25.62%)
- digital_pro: 147 users (18.37%)
- family: 162 users (20.25%)
- investor: 80 users (10.00%)
- student: 99 users (12.37%)
- traveler: 107 users (13.38%)

## Top Transaction Categories
- transport: 10672 tx (9.48%), mean amount=705.88, median amount=648.34
- restaurants: 10584 tx (9.40%), mean amount=1588.23, median amount=1392.63
- groceries: 9849 tx (8.74%), mean amount=1971.47, median amount=1783.32
- entertainment: 9789 tx (8.69%), mean amount=2628.08, median amount=2214.94
- education: 9222 tx (8.19%), mean amount=9342.43, median amount=6717.81

## Offer Catalog Mix
- Product types: bundle=1, card=4, credit=2, deposit=1, insurance=2, investment=1, partner=1, service=2, subscription=1

## Notes
- The dataset remains synthetic and should be reported as a controlled simulation, not as an empirical banking sample.
- Real-data validation is still needed to support external validity claims.