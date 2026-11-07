Проверено 7 ноября 2026:
- 9 тестов passed: NDCG, ties, cutoff, invalid values, duplicate keys, Parquet evaluate/validate/blend, shuffled key alignment, missing coverage, overwrite protection.
- Python compileall: src и scripts прошли.
- Тестовое окружение Python 3.13 + Polars 1.14.0; ML-окружение организаторов требует Python 3.11 и ещё не установлено.
- Датасет отсутствует. Обучение и GPU-инференс не проверялись; метрика соревнования не измерена.
- Baseline сохранён без изменений; инструкции установки находятся в README.

Проверка ветки участника 2, 8 ноября 2026:
- B01: 50 000 train, пять эпох, seed=42 и 123; вся val: 14 423 ленты / 1 389 971 кандидат.
- Разница NDCG против выбора по Logloss: +0.00083286 и -0.00094614. Текущий вариант не принят; см. docs/B01_RESULTS.md.
- 14 тестов выполнены в baseline/.venv (Python 3.11.15, PyTorch 2.10.0).
- Проверены два критерия выбора, padding, ключи предсказаний, сохранение RNG и исключение выборки выбора из сравнения.
- Полная NDCG всех четырёх моделей сверена с CatBoost, расхождение меньше 1e-10.
- baseline/ и src/neuroforge/metrics.py не менялись. Веса/данные остаются вне Git.
