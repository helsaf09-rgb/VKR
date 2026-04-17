# Real-Data Validation Report

## Dataset
- Name: Online Retail
- Source: https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx
- Kaggle mirror: https://www.kaggle.com/datasets/ineubytes/online-retail-ecommerce-dataset
- Filtering: min_user_interactions=5, min_item_interactions=10
- Users after filtering: 2992
- Items after filtering: 1499
- Positive interactions after filtering: 201226
- Time range: 2010-12-01T08:34:00 .. 2011-12-09T12:50:00

## Best Model
- Best by NDCG@10: `implicit_mf`
- NDCG@10 = 0.1296
- MAP@10 = 0.1025

## Metrics
| model | precision@k | recall@k | map@k | ndcg@k |
| --- | ---: | ---: | ---: | ---: |
| implicit_mf | 0.0219 | 0.2186 | 0.1025 | 0.1296 |
| item_knn | 0.0165 | 0.1648 | 0.0742 | 0.0952 |
| neural_cf | 0.0085 | 0.0846 | 0.0330 | 0.0449 |
| popularity | 0.0049 | 0.0491 | 0.0165 | 0.0241 |
| lightgcn | 0.0024 | 0.0244 | 0.0075 | 0.0114 |

## Interpretation
- The benchmark now uses real transactional purchases instead of a media-rating dataset, which is a closer validation setting for recommendation from behavioral signals.
- The domain is still retail, not banking, so results should be presented as proof that the pipeline transfers to real transaction logs rather than as a direct estimate of banking uplift.
- The benchmark now includes both a nonlinear neural baseline (Neural CF) and an implemented graph-based SOTA branch (LightGCN), which makes the validation less dependent on purely linear baselines.