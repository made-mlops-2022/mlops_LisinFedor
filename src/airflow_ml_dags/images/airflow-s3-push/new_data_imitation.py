"""Идея dag в том, что он вроде стороннего сервиса, который грузить данные ежедневные на s3."""
import os
import pandas as pd
from boto3 import Session
from time import sleep
from random import randint
from sdv.tabular import GaussianCopula
from pathlib import Path


TMP_FILES = Path("/assets")

ORIGIN_DATA = TMP_FILES / "origin.csv"
TIME_RANDOM_DELAY_S = int(os.environ.get("TIME_RANDOM_DELAY_S")) or 60
NUM_OF_SAMPLES = int(os.environ.get("NUM_OF_SAMPLES")) or 300

DS = os.environ.get("DS")
DS_NODASH = os.environ.get("DS_NODASH")


def synt_data() -> pd.DataFrame:
    origin = pd.read_csv(str(ORIGIN_DATA))
    ub = NUM_OF_SAMPLES // 10
    num_of_samples = NUM_OF_SAMPLES + randint(-ub, ub)
    synt_model = GaussianCopula()
    synt_model.fit(origin)

    return synt_model.sample(num_of_samples)


def dump_last_day_data():
    file_name = f"{DS_NODASH}.csv"

    last_day_data = synt_data()
    last_day_data.to_csv(TMP_FILES / file_name, index=False)


def load_to_s3():
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACESS_KEY")
    endpoint_url = os.environ.get("ENDPOINT_URL")

    s3 = Session()
    s3_client = s3.client(
        service_name="s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
    )

    file_name = f"{DS_NODASH}.csv"
    folder_name = f"{DS}"
    bucket_name = os.environ.get("bucket_name")
    file_path = TMP_FILES / file_name
    bucket_path = f"{folder_name}/{file_name}"

    sleep(randint(0, TIME_RANDOM_DELAY_S))

    try:
        s3_client.upload_file(str(file_path), bucket_name, bucket_path)
    finally:
        os.remove(file_path)


if __name__ == "__main__":
    dump_last_day_data()
    load_to_s3()
