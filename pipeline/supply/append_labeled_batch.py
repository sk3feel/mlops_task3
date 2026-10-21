"""Дозапись подготовленного месячного батча в размеченную витрину."""

import logging
from pathlib import Path
import pandas as pd
import io

from pipeline.db import LABELED_COLUMNS, get_conn
logger = logging.getLogger(__name__)


def append_labeled_batch(batch_info: dict) -> dict:
    """Загружает строки из parquet в labeled_transactions.

    Сами данные и факт загрузки месяца в supplied_batches 
    сохраняются одной транзакцией. Если была ошибка - все откатывается.
    Отмеченный месяц пропускается при повторной загрузке.
    Возвращает путь, месяц и число строк батча.
    """
    batch_path = Path(batch_info["batch_path"])
    if not batch_path.exists():
        raise FileNotFoundError(f"Нет файла батча для загрузки: {batch_path}")
    
    frame = pd.read_parquet(batch_path)
    batch_month = pd.Timestamp(batch_info["batch_month"])
    frame = frame.loc[:, LABELED_COLUMNS].copy()
    frame["transaction_time"] = pd.to_datetime(frame["transaction_time"])

    if batch_month != batch_month.to_period("M").start_time:
        raise ValueError("Месяц батча должен быть задан первым числом месяца")
    if frame.empty:
        raise ValueError("Батч пуст")
    if not frame["transaction_time"].dt.to_period("M").eq(batch_month.to_period("M")).all():
        raise ValueError("В батче есть пропущенные даты или строки другого месяца")



    conn = get_conn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS supplied_batches (
                        batch_month DATE PRIMARY KEY,
                        rows_count BIGINT NOT NULL
                    )
                    """
                )
                cur.execute(
                    """
                    INSERT INTO supplied_batches (batch_month, rows_count)
                    VALUES (%s, %s)
                    ON CONFLICT (batch_month) DO NOTHING
                    RETURNING rows_count
                    """,
                    (batch_month.date(), len(frame)),
                )
                inserted = cur.fetchone()

                if inserted is None:
                    cur.execute(
                        """
                        SELECT rows_count FROM supplied_batches
                        WHERE batch_month = %s
                        """,
                        (batch_month.date(),))
                    rows = cur.fetchone()[0]
                    logger.info("Месяц %s уже загружен, повтор пропущен", batch_info["batch_month"])
                else:
                    buffer = io.StringIO()
                    frame.to_csv(buffer, index=False, header=False, date_format="%Y-%m-%d %H:%M:%S")
                    buffer.seek(0)
                    columns = ", ".join(LABELED_COLUMNS)
                    cur.copy_expert(
                        f"COPY labeled_transactions ({columns}) "
                        "FROM STDIN WITH (FORMAT CSV)",
                        buffer)
                    rows = len(frame)
    finally:
        conn.close()

    return {
        "batch_path": batch_info["batch_path"],
        "batch_month": batch_info["batch_month"],
        "rows": int(rows),
    }
