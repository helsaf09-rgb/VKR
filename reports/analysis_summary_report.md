# Analysis Summary Report

## Overall Best Model
- Best by NDCG@5: `time_decay`
- NDCG@5 = 0.2936
- MAP@5 = 0.2387

## Model Leaderboard
- time_decay: Precision@5=0.0922, Recall@5=0.4612, MAP@5=0.2387, NDCG@5=0.2936
- profile_baseline: Precision@5=0.0922, Recall@5=0.4612, MAP@5=0.2349, NDCG@5=0.2906
- hybrid_semantic: Precision@5=0.0918, Recall@5=0.4587, MAP@5=0.2334, NDCG@5=0.2888
- neural_cf: Precision@5=0.0678, Recall@5=0.3387, MAP@5=0.1518, NDCG@5=0.1975
- lightgcn: Precision@5=0.0628, Recall@5=0.3137, MAP@5=0.1533, NDCG@5=0.1926
- implicit_mf: Precision@5=0.0660, Recall@5=0.3300, MAP@5=0.1460, NDCG@5=0.1911
- item_knn: Precision@5=0.0640, Recall@5=0.3200, MAP@5=0.1474, NDCG@5=0.1895

## Time-Decay vs Profile Baseline
- Profile baseline NDCG@5 = 0.2906
- Time-decay NDCG@5 = 0.2936
- Profile baseline MAP@5 = 0.2349
- Time-decay MAP@5 = 0.2387

## Bootstrap Confidence (Time-Decay minus Profile Baseline)
- ndcg_at_k: mean_diff=0.003010, 95% CI=[-0.010062, 0.015470], P(diff>0)=0.680
- map_at_k: mean_diff=0.003813, 95% CI=[-0.007459, 0.015189], P(diff>0)=0.751
- recall_at_k: mean_diff=0.000000, 95% CI=[-0.022500, 0.022500], P(diff>0)=0.477
- precision_at_k: mean_diff=-0.000000, 95% CI=[-0.004500, 0.004500], P(diff>0)=0.499

## Segment-wise Delta in NDCG@5 (Time-Decay - Profile Baseline)
- daily_life: -0.0040
- digital_pro: -0.0053
- family: 0.0164
- investor: 0.0088
- student: 0.0175
- traveler: -0.0101

## Practical Note
- Improvements are modest in top-rank quality metrics (MAP/NDCG).
- Bootstrap confidence intervals include zero, so gains should be treated as directional, not yet definitive.
- Recency-aware modeling is still a reasonable advanced step, but stronger evidence requires larger data or stronger sequential models.