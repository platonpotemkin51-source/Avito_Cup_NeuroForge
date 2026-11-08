Проверено 7 ноября 2026:
- 9 тестов passed: NDCG, ties, cutoff, invalid values, duplicate keys, Parquet evaluate/validate/blend, shuffled key alignment, missing coverage, overwrite protection.
- Python compileall: src и scripts прошли.
- Тестовое окружение Python 3.13 + Polars 1.14.0; ML-окружение организаторов требует Python 3.11 и ещё не установлено.
- Датасет отсутствует. Обучение и GPU-инференс не проверялись; метрика соревнования не измерена.
- Baseline сохранён без изменений; инструкции установки находятся в README.


Проверено 8 ноября 2026:
- Оба исходных бейзлайна обучены на 50 000 полных лент train и всей val (14 423 ленты).
- CatBoost: NDCG 0.35242053847515664; Transformer: NDCG 0.3326732173555679.
- Проверены точное покрытие val, уникальность ключей и конечность float32 scores; сохранены прогнозы и независимые метрики.
- Исходники baseline и lock-файл сохранены; полный train не обучался, гипотезы не проверялись.
- Подробности: docs/BASELINE_RESULTS_2026-11-08.md; машиночитаемый итог: runs/baseline-50k/summary.json.
