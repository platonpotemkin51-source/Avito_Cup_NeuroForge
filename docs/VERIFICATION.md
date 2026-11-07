Проверено 7 ноября 2026:
- 9 тестов passed: NDCG, ties, cutoff, invalid values, duplicate keys, Parquet evaluate/validate/blend, shuffled key alignment, missing coverage, overwrite protection.
- Python compileall: src и scripts прошли.
- Тестовое окружение Python 3.13 + Polars 1.14.0; ML-окружение организаторов требует Python 3.11 и ещё не установлено.
- Датасет отсутствует. Обучение и GPU-инференс не проверялись; метрика соревнования не измерена.
- Baseline сохранён без изменений; инструкции установки находятся в README.
