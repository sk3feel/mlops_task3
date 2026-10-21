from datetime import datetime, timezone

import numpy as np
import pandas as pd
from airflow.exceptions import AirflowException
from airflow.operators.python import get_current_context

from pipeline.db import get_conn
from pipeline.settings import MODELS_DIR


def generate_dq_report() -> dict:
    """Проверяет последний месяц витрины, пишет HTML, возвращает нарушения и путь.

    Повтор заменяет отчёт запуска, данные не меняем.
    """
    conn = get_conn()
    try:
        frame = pd.read_sql(
            """
            WITH latest AS (
                SELECT date_trunc('month', max(transaction_time)) AS month
                FROM labeled_transactions
            )
            SELECT transaction_time, amount, target, population_city,
                   lat, lon, merchant_lat, merchant_lon
            FROM labeled_transactions
            WHERE transaction_time >= (SELECT month FROM latest)
              AND transaction_time < (
                  SELECT month + interval '1 month' FROM latest
              )
            """,
            conn,
        )
    finally:
        conn.close()

    results = []

    def add_rule(name, violations):
        count = int(violations)
        results.append({
            "Правило": name,
            "Статус": "PASS" if count == 0 else "FAIL",
            "Нарушения": count,
        })

    add_rule("Выборка не пустая", int(frame.empty))
    required = [
        "transaction_time", "amount", "target", "population_city",
        "lat", "lon", "merchant_lat", "merchant_lon",
    ]
    add_rule("Обязательные поля заполнены", frame[required].isna().any(axis=1).sum())
    amount = pd.to_numeric(frame["amount"], errors="coerce")
    add_rule("Сумма конечная и неотрицательная", (~(np.isfinite(amount) & amount.ge(0))).sum())
    add_rule("Метка target равна 0 или 1", (~frame["target"].isin([0, 1])).sum())
    coordinates_valid = pd.Series(True, index=frame.index)
    for column, limit in [("lat", 90), ("lon", 180), ("merchant_lat", 90), ("merchant_lon", 180)]:
        values = pd.to_numeric(frame[column], errors="coerce")
        coordinates_valid &= values.between(-limit, limit)

    add_rule("Координаты заполнены и допустимы", (~coordinates_valid).sum())
    failed = [row["Правило"] for row in results if row["Статус"] == "FAIL"]
    month = (
        pd.Timestamp(frame["transaction_time"].min()).strftime("%Y-%m")
        if not frame.empty else "нет данных"
    )
    checked_at = datetime.now(timezone.utc).isoformat()
    table = pd.DataFrame(results).to_html(index=False, escape=True)
    status = "FAIL" if failed else "PASS"

    html = f"""<!doctype html>
<html lang="ru">
<head><meta charset="utf-8"><title>Data Quality</title></head>
<body>
<h1>Data Quality: {status}</h1>
<p>Таблица: labeled_transactions. Месяц: {month}.</p>
<p>Строк: {len(frame)}. Время проверки UTC: {checked_at}.</p>
{table}
<p>Нарушения: число некорректных строк.
Для пустой выборки — 1, если она пуста.</p>
</body>
</html>
"""

    reports_dir = MODELS_DIR.parent / "dq_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"report_{get_current_context()['ts_nodash']}.html"
    temporary_path = report_path.with_suffix(".tmp")
    temporary_path.write_text(html, encoding="utf-8")
    temporary_path.replace(report_path)

    return {"report_path": str(report_path), "failed_rules": failed}


def validate_dq(result: dict) -> None:
    """Читает результат проверки и вызывает ошибку при нарушениях.

    
    """
    if result["failed_rules"]:
        raise AirflowException(
            f"Нарушены правила: {', '.join(result['failed_rules'])}. "
            f"Отчёт: {result['report_path']}"
        )
