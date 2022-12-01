"""Идея dag в том, что он вроде стороннего сервиса, который грузить данные ежедневные на s3."""


import airflow
from airflow import DAG
from airflow.operators.email_operator import EmailOperator
from airflow.operators.bash import BashOperator


default_args = {
    "owner": "Theo",
    "email": ["godloverus@gmail.com"],
    "email_on_failure": True,
}


with DAG(
    dag_id="test_email",
    start_date=airflow.utils.dates.days_ago(3),
    schedule_interval="30 00 * * *",
    default_args=default_args,
) as dag:

    info_messege = BashOperator(
        task_id="info_messege",
        bash_command=f"echo 'Uploading data for the {{ ds }}'",
        dag=dag,
    )

    test_email = EmailOperator(
        task_id="send_email",
        to="godloverus@gmail.com",
        subject="ingestion complete",
        html_content="Date: {{ ds }}",
        dag=dag,
    )

    info_messege >> test_email
