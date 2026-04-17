from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class SASRecConfig:
    embedding_dim: int = 32
    num_heads: int = 4
    num_blocks: int = 2
    dropout: float = 0.1
    max_seq_len: int = 50
    window_stride: int = 1
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    epochs: int = 8
    batch_size: int = 256
    samples_per_epoch: int = 50000
    random_state: int = 42


class _SASRecModule(nn.Module):
    def __init__(self, n_items: int, config: SASRecConfig) -> None:
        super().__init__()
        self.config = config
        self.item_embedding = nn.Embedding(n_items + 1, config.embedding_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.num_heads,
            dim_feedforward=config.embedding_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_blocks)
        self.output_norm = nn.LayerNorm(config.embedding_dim)
        self.scale = math.sqrt(float(config.embedding_dim))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_len)
        hidden = self.item_embedding(input_ids) * self.scale
        hidden = hidden + self.position_embedding(positions)
        hidden = self.dropout(hidden)

        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=input_ids.device, dtype=torch.bool),
            diagonal=1,
        )
        padding_mask = input_ids.eq(0)

        hidden = self.encoder(hidden, mask=causal_mask, src_key_padding_mask=padding_mask)
        hidden = self.output_norm(hidden)
        return hidden


class SASRecRecommender:
    """Lightweight SASRec-style recommender for next-item prediction on implicit sequences."""

    def __init__(self, config: SASRecConfig | None = None):
        self.config = config or SASRecConfig()

        self._users: list[str] | None = None
        self._offers: list[str] | None = None
        self._user_index: dict[str, int] | None = None
        self._offer_index: dict[str, int] | None = None
        self._history_by_user: dict[str, list[int]] = {}
        self._item_popularity: np.ndarray | None = None
        self._module: _SASRecModule | None = None

        self.loss_history_: list[float] = []

    def _build_sequences(self, interactions_df: pd.DataFrame) -> pd.DataFrame:
        positive = interactions_df[interactions_df["label"] == 1][["user_id", "offer_id", "timestamp"]].copy()
        if positive.empty:
            raise ValueError("No positive interactions found for SASRec training.")

        positive["user_id"] = positive["user_id"].astype(str)
        positive["offer_id"] = positive["offer_id"].astype(str)
        positive["timestamp"] = pd.to_datetime(positive["timestamp"])
        return positive.sort_values(["user_id", "timestamp", "offer_id"]).reset_index(drop=True)

    def _encode_prefix(self, item_ids: list[int]) -> np.ndarray:
        max_len = self.config.max_seq_len
        input_seq = np.zeros(max_len, dtype=np.int64)
        input_tail = item_ids[-max_len:]
        input_seq[-len(input_tail) :] = np.asarray(input_tail, dtype=np.int64)
        return input_seq

    def _build_training_samples(self, user_sequences: dict[str, list[int]]) -> tuple[np.ndarray, np.ndarray]:
        input_rows: list[np.ndarray] = []
        target_rows: list[int] = []

        for sequence in user_sequences.values():
            if len(sequence) < 2:
                continue

            positions = list(range(1, len(sequence), max(1, self.config.window_stride)))
            if positions[-1] != len(sequence) - 1:
                positions.append(len(sequence) - 1)
            for end in positions:
                prefix = sequence[max(0, end - self.config.max_seq_len) : end]
                target_item = sequence[end]
                input_rows.append(self._encode_prefix(prefix))
                target_rows.append(int(target_item))

        if not input_rows:
            raise ValueError("SASRec requires at least one user sequence with two or more interactions.")

        return np.stack(input_rows), np.asarray(target_rows, dtype=np.int64)

    def fit(self, interactions_df: pd.DataFrame) -> "SASRecRecommender":
        positive = self._build_sequences(interactions_df)

        self._users = sorted(positive["user_id"].unique().tolist())
        self._offers = sorted(positive["offer_id"].unique().tolist())
        self._user_index = {user_id: idx for idx, user_id in enumerate(self._users)}
        self._offer_index = {offer_id: idx for idx, offer_id in enumerate(self._offers)}

        offer_index_1based = {offer_id: idx + 1 for idx, offer_id in enumerate(self._offers)}
        grouped = positive.groupby("user_id")["offer_id"].apply(list).to_dict()
        self._history_by_user = {
            user_id: [offer_index_1based[offer_id] for offer_id in offer_ids]
            for user_id, offer_ids in grouped.items()
        }

        input_rows, target_rows = self._build_training_samples(self._history_by_user)
        input_tensor = torch.tensor(input_rows, dtype=torch.long)
        target_tensor = torch.tensor(target_rows, dtype=torch.long)

        item_popularity = positive.groupby("offer_id").size().reindex(self._offers, fill_value=0).to_numpy(dtype=float)
        max_popularity = float(item_popularity.max()) if len(item_popularity) else 1.0
        if max_popularity <= 0.0:
            max_popularity = 1.0
        self._item_popularity = item_popularity / max_popularity

        torch.manual_seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        self._module = _SASRecModule(n_items=len(self._offers), config=self.config)
        optimizer = torch.optim.Adam(
            self._module.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.loss_history_ = []
        self._module.train()
        n_samples = int(input_tensor.shape[0])
        for _ in range(self.config.epochs):
            permutation = torch.randperm(n_samples)
            if 0 < self.config.samples_per_epoch < n_samples:
                permutation = permutation[: self.config.samples_per_epoch]
            epoch_loss = 0.0
            epoch_rows = 0

            for start in range(0, len(permutation), self.config.batch_size):
                stop = min(start + self.config.batch_size, len(permutation))
                batch_idx = permutation[start:stop]
                batch_inputs = input_tensor[batch_idx]
                batch_targets = target_tensor[batch_idx]
                batch_negatives = torch.randint(1, len(self._offers) + 1, size=batch_targets.shape, dtype=torch.long)
                same_mask = batch_negatives.eq(batch_targets)
                while same_mask.any():
                    batch_negatives[same_mask] = torch.randint(
                        1,
                        len(self._offers) + 1,
                        size=(int(same_mask.sum().item()),),
                        dtype=torch.long,
                    )
                    same_mask = batch_negatives.eq(batch_targets)

                optimizer.zero_grad()
                hidden = self._module(batch_inputs)
                sequence_lengths = batch_inputs.ne(0).sum(dim=1)
                last_positions = sequence_lengths - 1
                last_hidden = hidden[torch.arange(hidden.shape[0]), last_positions]
                positive_embeddings = self._module.item_embedding(batch_targets)
                negative_embeddings = self._module.item_embedding(batch_negatives)

                positive_scores = (last_hidden * positive_embeddings).sum(dim=1)
                negative_scores = (last_hidden * negative_embeddings).sum(dim=1)
                loss = F.softplus(negative_scores - positive_scores).mean()
                reg_term = self.config.weight_decay * (
                    last_hidden.pow(2).mean()
                    + positive_embeddings.pow(2).mean()
                    + negative_embeddings.pow(2).mean()
                )
                total_loss = loss + reg_term
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._module.parameters(), 1.0)
                optimizer.step()

                batch_len = int(stop - start)
                epoch_loss += float(total_loss.item()) * batch_len
                epoch_rows += batch_len

            self.loss_history_.append(epoch_loss / max(1, epoch_rows))

        self._module.eval()
        return self

    def _score_known_user(self, user_id: str) -> np.ndarray:
        if self._module is None or self._offers is None:
            raise RuntimeError("Model must be fitted before inference.")

        history = self._history_by_user.get(str(user_id), [])
        if not history:
            if self._item_popularity is None:
                raise RuntimeError("Popularity fallback is not initialized.")
            return self._item_popularity.copy()

        encoded = np.zeros(self.config.max_seq_len, dtype=np.int64)
        tail = history[-self.config.max_seq_len :]
        encoded[-len(tail) :] = np.asarray(tail, dtype=np.int64)

        with torch.no_grad():
            hidden = self._module(torch.tensor(encoded[np.newaxis, :], dtype=torch.long))
            last_hidden = hidden[0, -1]
            scores = (last_hidden @ self._module.item_embedding.weight[1:].transpose(0, 1)).cpu().numpy()
        return scores.astype(np.float32, copy=False)

    def recommend_for_users(
        self,
        user_ids: list[str],
        top_k: int = 5,
        exclude_by_user: dict[str, set[str]] | None = None,
    ) -> pd.DataFrame:
        if self._offers is None or self._offer_index is None:
            raise RuntimeError("Model must be fitted before inference.")

        exclude_by_user = exclude_by_user or {}
        rows: list[dict[str, str | float | int]] = []

        for user_id in user_ids:
            if user_id in self._history_by_user:
                score_vec = self._score_known_user(user_id).copy()
            else:
                if self._item_popularity is None:
                    raise RuntimeError("Popularity fallback is not initialized.")
                score_vec = self._item_popularity.copy()

            for offer_id in exclude_by_user.get(str(user_id), set()):
                offer_idx = self._offer_index.get(str(offer_id))
                if offer_idx is not None:
                    score_vec[offer_idx] = -np.inf

            top_idx = np.argsort(-score_vec)[:top_k]
            for rank, offer_idx in enumerate(top_idx, start=1):
                rows.append(
                    {
                        "user_id": str(user_id),
                        "offer_id": self._offers[offer_idx],
                        "score": float(score_vec[offer_idx]),
                        "rank": rank,
                    }
                )

        return pd.DataFrame(rows)
