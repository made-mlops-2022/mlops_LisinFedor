import airflow
from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.operators.bash import BashOperator
from airflow.providers.amazon.aws.sensors.s3_key import S3KeySensor
from docker.types import Mount


docker_env = {
    "DS": "{{ ds }}",
    "DS_NODASH": "{{ ds_nodash }}",
    "AWS_ACCESS_KEY_ID": "{{ var.value.AWS_ACCESS_KEY_ID }}",
    "AWS_SECRET_ACESS_KEY": "{{ var.value.AWS_SECRET_ACESS_KEY }}",
    "ENDPOINT_URL": "{{ var.value.ENDPOINT_URL }}",
    "bucket_name": "{{ var.value.bucket_name }}",
    "MLFLOW_URI": "{{ var.value.MLFLOW_URI }}",
    "REGISTRY_MODEL_NAME": "{{ var.value.REGISTRY_MODEL_NAME }}",
}

# default_args = {'owner': 'Theo'}

default_args = {
    "owner": "Theo",
    "email": ["godloverus@gmail.com"],
    "email_on_failure": True,
}

with DAG(
    dag_id="predict_on_prod",
    catchup=False,
    default_args=default_args,
    start_date=airflow.utils.dates.days_ago(2),
    schedule_interval="30 00 * * *",
) as dag:

    s3_check = S3KeySensor(
        task_id="sheck_last_day_data",
        bucket_name="{{ var.value.bucket_name }}",
        bucket_key="{{ ds }}/{{ ds_nodash }}.csv",
        aws_conn_id="s3_airflow",
    )

    info_messege = BashOperator(
        task_id="info_messege",
        bash_command=f"echo 'Data for the {{ ds }} was founded'",
        dag=dag,
    )

    last_day_load = DockerOperator(
        task_id="last_day_load",
        image="airflow-preprocess:latest",
        network_mode="bridge",
        command="--no-target",
        mounts=[
            Mount(
                source="/home/theo/mlops/mlops_LisinFedor/src/airflow_ml_dags/data",
                target="/data",
                type="bind",
            ),
        ],
        environment=docker_env,
    )

    prod_model_predict = DockerOperator(
        task_id="predict_on_prod",
        image="airflow-model-eval:latest",
        network_mode="bridge",
        mounts=[
            Mount(
                source="/home/theo/mlops/mlops_LisinFedor/src/airflow_ml_dags/data",
                target="/data",
                type="bind",
            ),
        ],
        environment=docker_env,
    )

    s3_check >> info_messege >> last_day_load >> prod_model_predict
