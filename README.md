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

## Результат нейросети B01 + B02

B02 закрыта: приняты user cosine и top3 cosine по недавней истории. NDCG0.35617/0.35524 на двух seed, full val; Private не измерен. [Итог B02](docs/B02_RESULTS.md), [исторический B01](docs/B01_RESULTS.md).

**ИИ капитана: начни с [CAPTAIN_AI_HANDOFF.md](docs/CAPTAIN_AI_HANDOFF.md).** Это задание на интеграцию в его код, а не новый готовый CUDA launcher. Сначала согласовать GPU-окно и объём, измерить память/скорость и оценить время. Проверки NVIDIA пока не выполнялись.

Сохранены рабочие блоки SignalRanker, recent32 reader и только три сигнала (user/mean/top3), а также спецификация configs/B02_top3.json. Принимающий ИИ самостоятельно пишет свой цикл profile/train/evaluate/predict по инструкции. Старые отрицательные modes/launcher/configs/weights/caches удалены; их числа и настройки сохранены в experiments/B02/.

Веса и данные не коммитятся. Два лучших checkpoint находятся локально у участника2; пути и SHA256 указаны в передаче капитану. Очищенная модель сохраняет совместимость с ними.

Исходные baseline/ не менялись и материализуют кандидатов в RAM. Не запускать полный RAM-loader на16GB. Для будущего полного обучения использовать порционное чтение shards и int8 memmap; не строить expanded float cache на~170GB. Архитектурный следующий этап участника2 — B03.
