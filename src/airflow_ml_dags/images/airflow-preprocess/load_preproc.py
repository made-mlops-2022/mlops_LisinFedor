import click
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from boto3 import Session
from pathlib import Path

DS = os.environ.get("DS")
DS_NODASH = os.environ.get("DS_NODASH")


def load_last_day_data():
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
    out_path = Path("/data") / "raw" / f"{folder_name}"
    out_path.mkdir(parents=True, exist_ok=True)
    out_file = out_path / file_name
    bucket_path = f"{folder_name}/{file_name}"

    s3_client.download_file(bucket_name, bucket_path, str(out_file))


@click.command()
@click.option(
    "--no-target",
    is_flag=True,
    show_default=True,
    default=False,
    help="Save full last day data with no target.",
)
def train_val_gen(no_target):
    random_state = int(DS_NODASH)
    file_name = f"{DS_NODASH}.csv"
    folder_name = f"{DS}"
    in_path = f"/data/raw/{folder_name}/{file_name}"
    out_path = Path("/data") / "processed" / folder_name
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    if not no_target:
        train_ind, val_ind = train_test_split(
            df.index, test_size=0.3, random_state=random_state
        )

        df.iloc[train_ind].to_csv(out_path / f"{DS_NODASH}_train.csv", index=False)
        df.iloc[val_ind].to_csv(out_path / f"{DS_NODASH}_val.csv", index=False)
    else:
        df.drop("condition", axis=1).to_csv(
            out_path / f"{DS_NODASH}_notarget.csv", index=False
        )


if __name__ == "__main__":
    load_last_day_data()
    train_val_gen()
