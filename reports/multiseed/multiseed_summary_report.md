# Multi-Seed Benchmark Summary

## Setup
- Seeds: 7, 13, 21, 42, 77
- Users per seed: 800
- Models: hybrid_semantic, time_decay, profile_baseline, implicit_mf, item_knn

## Best Model (by mean NDCG@K)
- hybrid_semantic
- mean NDCG@K = 0.2833 (std 0.0050)
- mean MAP@K = 0.2275 (std 0.0041)

## Mean Metrics by Model
- hybrid_semantic: NDCG@K=0.2833 (+/- 0.0050), MAP@K=0.2275 (+/- 0.0041), Recall@K=0.4550 (+/- 0.0122)
- time_decay: NDCG@K=0.2824 (+/- 0.0116), MAP@K=0.2254 (+/- 0.0137), Recall@K=0.4577 (+/- 0.0096)
- profile_baseline: NDCG@K=0.2815 (+/- 0.0081), MAP@K=0.2262 (+/- 0.0066), Recall@K=0.4515 (+/- 0.0151)
- implicit_mf: NDCG@K=0.2155 (+/- 0.0145), MAP@K=0.1680 (+/- 0.0132), Recall@K=0.3618 (+/- 0.0193)
- item_knn: NDCG@K=0.2153 (+/- 0.0149), MAP@K=0.1680 (+/- 0.0125), Recall@K=0.3612 (+/- 0.0232)

## Time-Decay vs Profile Baseline (pooled user-level diffs across seeds)
- delta_ndcg_at_k: mean=0.000935, 95% CI=[-0.004915, 0.006779], P(diff>0)=0.627
- delta_map_at_k: mean=-0.000754, 95% CI=[-0.006100, 0.004425], P(diff>0)=0.380
- delta_recall_at_k: mean=0.006250, 95% CI=[-0.004500, 0.016500], P(diff>0)=0.866
- delta_precision_at_k: mean=0.001250, 95% CI=[-0.000900, 0.003350], P(diff>0)=0.872

## Notes
- This benchmark measures stability of conclusions across random seeds.
- If CIs still include zero, gains should be reported as directional rather than definitive.