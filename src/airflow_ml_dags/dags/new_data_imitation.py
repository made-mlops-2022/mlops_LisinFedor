"""Идея dag в том, что он вроде стороннего сервиса, который грузить данные ежедневные на s3."""
from pathlib import Path

import airflow
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from docker.types import Mount


TMP_FILES = Path(__file__).parent.parent / "tmp"
ORIGIN_DATA = TMP_FILES / "origin.csv"
TIME_RANDOM_DELAY_S = 60
NUM_OF_SAMPLES = 300

TMP_FILES_STR = str(TMP_FILES)

# default_args = {'owner': 'Theo'}

default_args = {
    "owner": "Theo",
    "email": ["godloverus@gmail.com"],
    "email_on_failure": True,
}

docker_env = {
    "TIME_RANDOM_DELAY_S": 60,
    "NUM_OF_SAMPLES": 300,
    "DS": "{{ ds }}",
    "DS_NODASH": "{{ ds_nodash }}",
    "AWS_ACCESS_KEY_ID": "{{ var.value.AWS_ACCESS_KEY_ID }}",
    "AWS_SECRET_ACESS_KEY": "{{ var.value.AWS_SECRET_ACESS_KEY }}",
    "ENDPOINT_URL": "{{ var.value.ENDPOINT_URL }}",
    "bucket_name": "{{ var.value.bucket_name }}",
}

with DAG(
    dag_id="new_data_load_imitation",
    start_date=airflow.utils.dates.days_ago(3),
    schedule_interval="30 00 * * *",
    default_args=default_args,
) as dag:

    info_messege = BashOperator(
        task_id="info_messege",
        bash_command=f"echo 'Uploading data for the {{ ds }}'",
        dag=dag,
    )

    daily_push = DockerOperator(
        task_id="s3_daily_push",
        image="airflow-s3-push:latest",
        network_mode="bridge",
        mounts=[
            Mount(
                source="/home/theo/mlops/mlops_LisinFedor/src/airflow_ml_dags/tmp",
                target="/assets",
                type="bind",
            ),
        ],
        environment=docker_env,
        dag=dag,
    )

    info_messege >> daily_push
