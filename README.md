# NeuroForge — Avito ML Cup 2026

Цель: максимальный Private NDCG и борьба за победу. Капитан — участник 3.

## С чего начать

Человеку: [план команды](docs/HUMAN_PLAN.md).
ИИ: [правила и выбор роли](AGENTS.md), затем [задания](docs/AI_PLAYBOOK.md).
Все эксперименты: [реестр гипотез](docs/HYPOTHESES.md).

## Архитектура

- `baseline/`: исходные решения организаторов, сохранены без изменений.
- `src/neuroforge/metrics.py`: независимая точная метрика, без ML-зависимостей.
- `src/neuroforge/cli.py`: оценка val, проверка сабмита, объединение прогнозов.
- `scripts/run_baseline.py`: запуск исходных моделей с небольшими конфигурациями.
- `configs/`: стартовые конфигурации для проверки работоспособности.
- `tests/`: проверки метрики и контракта предсказаний.
- `experiments/registry.csv`: общий журнал; `docs/`: протокол и роли.
- `data/competition/`: датасет (не Git); `runs/`: результаты (не Git).

## Окружение

Бейзлайны требуют Python 3.11. Используем исходный lock-файл организаторов.
Установите uv по https://docs.astral.sh/uv/getting-started/installation/ .
Из корня репозитория:

```powershell
uv sync --project baseline --locked --python 3.11
$env:PYTHONPATH = "$PWD/src"
uv run --project baseline python -m unittest discover -s tests -v
uv run --project baseline python scripts/run_baseline.py configs/catboost_smoke.json train --data data/competition --output runs/A-smoke
uv run --project baseline python scripts/run_baseline.py configs/catboost_smoke.json predict --data data/competition --model runs/A-smoke/model.cbm --output runs/A-smoke/test.parquet
uv run --project baseline python -m neuroforge.cli validate --predictions runs/A-smoke/test.parquet --reference template/sample_submission.parquet
```

Для нейросети используйте `configs/neural_smoke.json`, модель `model.pt`.
Smoke-прогоны проверяют только работу кода. Первые ленты файла не являются честной выборкой для научных выводов.

Оценка собственных val-прогнозов:
```powershell
uv run --project baseline python -m neuroforge.cli evaluate --predictions runs/example/val.parquet --reference data/competition/val/impressions
uv run --project baseline python -m neuroforge.cli blend --inputs runs/A/val.parquet runs/B/val.parquet --weights 0.5 0.5 --output runs/blend/val.parquet
```

Evaluate ожидает точное покрытие переданной reference; для подвыборки передайте соответствующий файл меток. Blend делает прямое взвешивание scores, масштаб моделей сначала согласуйте на development. Это исходная операция, не автоматический подбор весов.

## Ограничения стартового кода

8 ноября 2026 оба бейзлайна обучены на 50 000 лент train и измерены на всей val: [результаты и команды повторения](docs/BASELINE_RESULTS_2026-11-08.md). На этом компьютере данные находятся в `baseline/data/competition`; используйте этот путь в `--data` вместо `data/competition` в примерах выше. Исходные модели материализуют кандидатов в RAM: полный запуск на 16 ГБ требует профилирования и доработки загрузчиков. Smoke-конфиги ограничены целыми лентами. Нейросетевой baseline пока выбирает checkpoint по Logloss; исправление — B01. CatBoost в конфиге выбирается по полному NDCG. Валидатор и blend также читают прогнозы в RAM; для больших артефактов требуется порционная обработка.
