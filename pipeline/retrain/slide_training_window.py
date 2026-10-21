"""Сдвиг трёхмесячного окна обучения на один месяц вперёд."""

import logging
from datetime import date

import pandas as pd

from pipeline.settings import CURRENT_DATASET, PENDING_WINDOW

logger = logging.getLogger(__name__)


def slide_training_window(batch_info: dict) -> dict:
    """Добавляет новый месяц к текущему окну и два предыдущих месяца.

    Читает current.parquet и parquet батча из batch_info.
    Новое окно пишет в pending_window.parquet и не заменяет current.parquet: замена происходит
    только после сохранения кандидата, чтобы повтор задачи не сдвинул окно дважды.

    Окно формируется относительно месяца батча и двух предыдущих месяцев.
    Если данные батча уже присутствуют, они заменяются,
    что предотвращает дублирование при повторных запусках.

    Возвращает путь к новому окну и месяц добавленного батча.
    """
    if not CURRENT_DATASET.exists():
        raise FileNotFoundError(f"Нет исходного датасета {CURRENT_DATASET}")

    batch_month = date.fromisoformat(batch_info["batch_month"])
    current = pd.read_parquet(CURRENT_DATASET)
    batch = pd.read_parquet(batch_info["batch_path"])
    current["transaction_time"] = pd.to_datetime(current["transaction_time"])
    batch["transaction_time"] = pd.to_datetime(batch["transaction_time"])

    batch_period = pd.Period(batch_month, freq="M")
    current_periods = current["transaction_time"].dt.to_period("M")
    current = current.loc[current_periods != batch_period]

    combined = pd.concat([current, batch], ignore_index=True)
    periods = combined["transaction_time"].dt.to_period("M")
    updated = combined.loc[
        (periods >= batch_period - 2) & (periods <= batch_period)
    ].sort_values("transaction_time").reset_index(drop=True)

    updated.to_parquet(PENDING_WINDOW, index=False)

    logger.info(
        "Окно после сдвига: %s строк, %s — %s.",
        len(updated),
        updated["transaction_time"].min(),
        updated["transaction_time"].max(),
    )

    return {
        "dataset_path": str(PENDING_WINDOW),
        "batch_month": batch_month.isoformat(),
    }
