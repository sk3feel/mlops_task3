from datetime import datetime, timedelta

from airflow.decorators import dag, task

from pipeline.data_quality import generate_dq_report, validate_dq


@dag(
    dag_id="data_quality",
    schedule="*/5 * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=10),
    tags=["data_quality"],
)
def data_quality():
    """Создаёт DQ-отчёт, потом проверяет наличие нарушений."""
    report = task(generate_dq_report, task_id="generate_dq_report")()
    task(validate_dq, task_id="validate_dq")(report)


data_quality()
