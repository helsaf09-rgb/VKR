# SASRec Real-Data Validation Report

## Configuration
- embedding_dim: 32
- num_heads: 4
- num_blocks: 2
- max_seq_len: 50
- window_stride: 1
- dropout: 0.1
- learning_rate: 0.002
- epochs: 8
- batch_size: 256
- samples_per_epoch: 50000

## Dataset
- users: 2992
- items: 1499
- positives: 201226

## Metrics
- Precision@10: 0.0019
- Recall@10: 0.0191
- MAP@10: 0.0063
- NDCG@10: 0.0092
- Final training loss: 0.6293

## Interpretation
- This run checks a true sequence-aware branch on the real transaction log rather than on the synthetic offer benchmark.
- In the current implementation and split, SASRec underperforms the strongest non-sequential baselines, so it should be presented as an implemented next-stage model rather than as the new leader.