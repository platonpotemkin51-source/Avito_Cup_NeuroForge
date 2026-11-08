# Бейзлайны NeuroForge — 8 ноября 2026

Оба исходных бейзлайна обучены и измерены. Новые гипотезы, признаки, архитектуры и ансамбли не проверялись. Это исходная точка **на 50 000 лент train**, а не результат полного обучения на всём датасете.

## Общие данные и метрика

- Train: первые 50 000 полных лент при сортировке имён Parquet-файлов; **3 896 275 кандидатов**. Это около 1,6% от 3 151 208 лент полного train. Случайная репрезентативная выборка не формировалась.
- Val: все **14 423 ленты**, **1 389 971 кандидат**, одинаковые для обеих моделей.
- Основная метрика проекта: полный NDCG, одинаковый вес лент, бинарный gain, негативы раньше позитивов при равных scores; ленты без позитивов дают 1. @10 и @30 — дополнительные метрики.
- Val использовалась для выбора лучшей модели. Отдельный независимый holdout не выделен. Это локальные development-результаты, не public/private leaderboard.
- Данные на этом компьютере: `baseline/data/competition`.

## Измеренные результаты

| Модель | NDCG | NDCG@10 | NDCG@30 | Сохранённая версия |
|---|---:|---:|---:|---|
| CatBoost YetiRank, GPU | 0.3524205385 | 0.2197542885 | 0.2888227546 | 234 дерева |
| Исходный Transformer, GPU | 0.3326732174 | 0.1952267511 | 0.2647034744 | Эпоха 4 из 5 |

CatBoost: до 1000 деревьев, early stopping 50, выбор по полному NDCG, learning rate 0,4, seed 666. Нейросеть: исходные 84 841 параметр, 5 эпох, batch 32, AdamW, learning rate 0,001, seed 42; выбор чекпойнта оставлен по исходному val Logloss. Лучший Logloss: **0.0747915750**. NDCG нейросети посчитан отдельно для сохранённого чекпойнта после завершения обучения.

Разницу результатов на этой подвыборке нельзя считать окончательным выводом о потенциале архитектур. Для будущих изменений ветки A контроль — CatBoost из этой таблицы, ветки B — Transformer; данные и правило оценки должны совпадать. При смене объёма train нужен контрольный бейзлайн на новом объёме.

## Что проверено

- Обучение обеих моделей реально завершилось, веса сохранены.
- Для каждой сохранены прогнозы всей val: уникальные `(feed_id, item_id)`, точное покрытие reference, конечные float32 scores.
- NDCG, NDCG@10, NDCG@30 рассчитаны общей независимой реализацией `src/neuroforge/metrics.py`.
- Все три независимые метрики CatBoost совпали с результатами его обучения с точностью округления; полный NDCG дополнительно проверен программно с допуском 1e-12.
- SHA256 исходных CatBoost/Transformer и lock-файла совпадают с зафиксированными до обучения. Исходники `baseline/` не менялись.
- Списки обучающих и проверочных лент и их SHA256 сохранены. Обучающие данные и веса в Git не добавлялись. Сабмит на соревнование не отправлялся.

## Ресурсы и окружение

Python 3.11.18, CatBoost 1.2.10, NumPy 1.26.4, Polars 1.14.0. Обе модели обучались на RTX 4060 Laptop 8 ГБ; системная RAM — около 16 ГБ.

| Запуск | Время процесса с загрузкой данных | Наблюдавшийся пик RAM |
|---|---:|---:|
| CatBoost | 8.01 мин | 5.373 ГиБ |
| Transformer | 23.38 мин | 6.211 ГиБ |

RAM — максимум доступных замеров Windows PeakWorkingSet64; более поздний пик мог не попасть в замеры. Часть памяти относится к отображённой таблице item-эмбеддингов. В отдельных замерах nvidia-smi показывал около 4709 MiB занятой VRAM для CatBoost и до 515 MiB для Transformer; это наблюдения, не гарантированные пиковые значения.

Первоначально PyTorch в `baseline/.venv` был CPU-only. Официальная сборка **torch 2.10.0+cu128 / CUDA 12.8** установлена отдельно в `runs/torch-cu128`, базовое окружение и lock-файл сохранены. Проверены доступность CUDA и прямой/обратный проход исходной модели на batch 32 с 299 кандидатами в ленте. Источник установки: [официальные инструкции PyTorch](https://pytorch.org/get-started/previous-versions/).

Предварительный CPU-запуск нейросети `runs/B-cpu-50k` остановлен после первой эпохи и в итоговую таблицу не включён. Финальный GPU-запуск начат заново, прошёл все 5 эпох. Время CPU-пробы и установки CUDA не включено во время итогового Transformer.

Полный train содержит 246 745 836 кандидатов. На всём train модели не обучались: исходные загрузчики материализуют кандидатов в RAM; для полного запуска нужны отдельное профилирование и подготовка загрузки данных. В рамках этого расчёта загрузчики не изменялись.

## Артефакты

- `runs/A-gpu-50k/`: `model.cbm`, `run.json`, `predictions_val.parquet`, `metrics_val.json`, `metrics_per_feed.parquet`.
- `runs/B-gpu-50k/`: `model.pt`, `run.json`, `predictions_val.parquet`, `metrics_val.json`, `metrics_per_feed.parquet`.
- `runs/baseline-50k/`: `summary.json`, `manifest.json`, `train_feeds.parquet`, `val_feeds.parquet`, `data_stats.json`, `cuda_probe.json`.
- `runs/baseline_evaluation.py`: воспроизводимый расчёт прогнозов и метрик для исходных моделей.
- `configs/catboost_gpu_50k.json`, `configs/neural_gpu_50k.json`: итоговые конфигурации.
- `runs/A-gpu-50k-resources.jsonl`, `runs/B-gpu-50k-resources.jsonl`: замеры ресурсов.

## Повторение

Из корня репозитория, PowerShell. Для нового запуска выберите ещё не существующие папки результатов.

На этом компьютере CUDA-слой уже установлен. Для переноса на другое окружение сначала выполните базовый `uv sync`, затем установите отдельный слой той же командой, которая использована здесь:

```powershell
uv pip install --python baseline/.venv/Scripts/python.exe --target runs/torch-cu128 --no-deps --cache-dir runs/uv-cache --index-url https://download.pytorch.org/whl/cu128 "torch==2.10.0+cu128"
```

После этого команды обучения:

```powershell
uv run --project baseline python scripts/run_baseline.py configs/catboost_gpu_50k.json train --data baseline/data/competition --output runs/A-gpu-50k-repeat

$env:PYTHONPATH = "$PWD/runs/torch-cu128;$PWD/src"
uv run --project baseline python scripts/run_baseline.py configs/neural_gpu_50k.json train --data baseline/data/competition --output runs/B-gpu-50k-repeat
```

Для повторения оценки уже сохранённых моделей, после установки того же PYTHONPATH:

```powershell
uv run --project baseline python runs/baseline_evaluation.py catboost runs/A-gpu-50k-repeat
uv run --project baseline python runs/baseline_evaluation.py neural runs/B-gpu-50k-repeat --device cuda
```

Существующие файлы прогнозов намеренно не перезаписываются. Все зависимости базового окружения устанавливаются исходной командой `uv sync --project baseline --locked --python 3.11`; CUDA-сборка — отдельный слой, указанный выше.
