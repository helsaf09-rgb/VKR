from __future__ import annotations

from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

from src.service.backend import RecommendationBackend


@lru_cache(maxsize=1)
def get_service() -> RecommendationBackend:
    service = RecommendationBackend()
    service.load()
    return service


app = FastAPI(
    title="Сервис персонализации банковских предложений",
    version="0.1.0",
    description="Прототип API для выдачи топ-N рекомендаций банковских продуктов на основе транзакционного поведения.",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Сервис запущен",
        "health": "/health",
        "docs": "/docs",
        "recommend_example": "/recommend/U00001?top_k=5",
    }


@app.get("/favicon.ico")
def favicon() -> Response:
    # Не засоряем логи запросами favicon без добавления статических файлов.
    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/recommend/{user_id}")
def recommend(
    user_id: str,
    top_k: int = Query(default=5, ge=1, le=20),
) -> dict[str, Any]:
    service = get_service()
    if service.transactions_df is None:
        raise HTTPException(status_code=500, detail="Сервис не инициализирован.")

    is_known = user_id in set(service.known_users())
    recommendations = service.recommend(user_id=user_id, top_k=top_k).to_dict(orient="records")

    if not recommendations:
        raise HTTPException(status_code=404, detail="Для этого пользователя нет доступных рекомендаций.")

    return {
        "user_id": user_id,
        "known_user": is_known,
        "top_k": top_k,
        "model": "time_decay_profile_v1",
        "recommendations": recommendations,
    }
