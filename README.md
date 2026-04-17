# Персонализация клиентских предложений на основе транзакционной активности пользователя банковских услуг

Студент: Сафонова Елена Михайловна  
Научный руководитель: Ильвовский Дмитрий Алексеевич  
Программа: НИУ ВШЭ, ФКН, образовательная программа «Магистр наук о данных»

## Цель проекта
Собрать и проверить воспроизводимый пайплайн рекомендательной системы, который на основе транзакционной активности пользователя формирует персональные банковские предложения и отдает top-K рекомендаций через API-сервис.

## План работ по чекпойнтам
- **Чекпойнт 1 (14.12.2025)**: создан репозиторий, зафиксирован план работ по ВКР.
- **Чекпойнт 2 (28.12.2025)**: проведён обзор источников (см. `docs/06_sources_review_2026-03-14.md`), выделены базовые и SOTA-модели рекомендательных систем.
- **Чекпойнт 3 (31.01.2026)**: сформирован синтетический датасет (800 пользователей, 112 626 транзакций, 15 офферов, 6 442 взаимодействия), проведён EDA, зафиксирован манифест (`data/synthetic/manifest.json`, `reports/synthetic_data_report.md`).
- **Чекпойнт 4 (22.02.2026)**: реализован baseline — profile similarity, выбраны метрики top-K ранжирования (Precision@K, Recall@K, MAP@K, NDCG@K), построены первые FastAPI-эндпоинты.
- **Чекпойнт 5 (31.03.2026)**: реализованы продвинутые модели — implicit MF, Neural CF, hybrid semantic, time-decay, LightGCN, SASRec. Проведено сравнение с baseline. Выполнена minimal real-data валидация на UCI Online Retail. Собран черновой сервисный прототип (FastAPI + Streamlit UI).
- **Предзащита (1–15.04.2026)**: подготовлены презентация (`deliverables/predefense_presentation_ru.pptx`) и шпаргалка (`docs/28_predefense_talk_track_ru.md`).
- **Чекпойнт 6 (10.05.2026)**: доработка по замечаниям научного руководителя, финализация текста ВКР, доработка сервиса.
- **Защита ВКР**: 8–11.06.2026.

## Основные артефакты
- Обзор источников: `docs/06_sources_review_2026-03-14.md`
- Финальный черновик ВКР (RU): `docs/25_thesis_final_ru.md`
- Финальный черновик ВКР (EN): `docs/24_thesis_final_assembled_en.md`
- Материалы предзащиты: `docs/27_predefense_presentation_ru.md`, `docs/28_predefense_talk_track_ru.md` (шпаргалка к защите)
- Word-версии ВКР: `deliverables/VKR_draft_gost.docx`, `deliverables/VKR_draft_gost_en.docx`
- PDF-версии ВКР: `deliverables/VKR_draft_gost.pdf`, `deliverables/VKR_draft_gost_en.pdf`
- Презентация предзащиты: `deliverables/predefense_presentation_ru.pptx`, `deliverables/predefense_presentation_ru.pdf`
- Streamlit UI: `src/ui/streamlit_app.py`
- Manifest синтетики: `data/synthetic/manifest.json`
- Отчёт по synthetic-данным: `reports/synthetic_data_report.md`
- Сводный synthetic-отчёт: `reports/analysis_summary_report.md`
- Multi-seed summary: `reports/multiseed/multiseed_summary_report.md`
- Валидация на реальном датасете: `reports/real_validation/real_validation_report.md`
- Отдельный SASRec-прогон: `reports/real_validation/sasrec_real_report.md`

## Структура проекта
```
├── data/                  # Данные (синтетические, реальные, промежуточные)
├── deliverables/          # Word- и PDF-версии ВКР, презентация
├── docs/                  # Обзор источников, финальные тексты ВКР RU/EN, материалы предзащиты
├── notebooks/             # Jupyter-ноутбуки (EDA)
├── reports/               # Отчёты, графики, метрики
│   ├── figures/           # PNG-графики EDA и моделей
│   ├── multiseed/         # Multi-seed benchmark
│   └── real_validation/   # Валидация на Online Retail
├── scripts/               # PowerShell-скрипты запуска пайплайнов
├── src/
│   ├── data/              # Генерация синтетики, загрузчики real-data
│   ├── evaluation/        # Метрики top-K ранжирования
│   ├── models/            # Рекомендательные модели (9 реализаций)
│   ├── pipelines/         # Сквозные сценарии воспроизведения
│   ├── service/           # FastAPI-прототип (app, backend, localization)
│   └── ui/                # Streamlit-интерфейс
├── requirements.txt       # Runtime-зависимости
├── requirements-dev.txt   # Dev-зависимости
└── pyproject.toml         # Конфигурация проекта
```

**Основные модели из сравнительной таблицы ВКР (раздел 3.2):** profile similarity, implicit MF, Neural CF, hybrid semantic, time-decay.  
**Продвинутые ветки, оцененные отдельно:** LightGCN (graph-based), SASRec (sequence-aware).

## Быстрый запуск
```powershell
# Создание и наполнение виртуального окружения
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

# Полный прогон synthetic-пайплайна
powershell -ExecutionPolicy Bypass -File .\scripts\run_all.ps1

# Отдельная валидация на реальном транзакционном датасете (Online Retail)
powershell -ExecutionPolicy Bypass -File .\scripts\run_real_validation.ps1

# Отдельный sequence-aware прогон SASRec на реальном transaction log
powershell -ExecutionPolicy Bypass -File .\scripts\run_sasrec_real_validation.ps1

# Streamlit пользовательский интерфейс
.\run_ui.bat

# Запуск API-сервиса
powershell -ExecutionPolicy Bypass -File .\scripts\run_service.ps1
```

## API
- Swagger UI: `http://127.0.0.1:8000/docs`
- Проверка состояния: `GET /health`
- Рекомендации: `GET /recommend/{user_id}?top_k=5`

Пример:
```text
http://127.0.0.1:8000/recommend/U00001?top_k=5
```

## Последний полный прогон
- Пользователей: 800
- Транзакций: 112 626
- Лучшая модель: `time_decay`
- Precision@5: `0.0922`
- Recall@5: `0.4612`
- MAP@5: `0.2387`
- NDCG@5: `0.2936`

## Последняя валидация на реальном датасете
- Датасет: `UCI Online Retail`
- Пользователей после фильтрации: `2992`
- Объектов после фильтрации: `1499`
- Положительных взаимодействий: `201 226`
- Лучшая модель: `implicit MF`
- NDCG@10: `0.1296`
- MAP@10: `0.1025`

## SASRec на реальных транзакциях
- NDCG@10: `0.0092`
- MAP@10: `0.0063`
- Конфигурация: `embedding_dim=32`, `num_heads=4`, `num_blocks=2`, `max_seq_len=50`, `epochs=8`, `samples_per_epoch=50000`
