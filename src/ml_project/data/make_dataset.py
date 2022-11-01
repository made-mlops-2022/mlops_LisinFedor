import os
import logging
from dotenv import load_dotenv
from typing import Tuple, Union, Optional

import pandas as pd
from boto3.session import Session
from sklearn.model_selection import train_test_split
from pathlib import Path

from ml_project.enities.splitting_params import SplittingParams
from ml_project.enities.s3_params import AwsS3Params

logger = logging.getLogger(__name__)
load_dotenv()


def download_data_from_s3(
    params: AwsS3Params,
):
    aws_access_key_id = os.environ.get("AWS_ACCESS_KEY_ID")
    aws_secret_access_key = os.environ.get("AWS_SECRET_ACESS_KEY")
    endpoint_url = os.environ.get("ENDPOINT_URL")

    cred: bool = True
    cred = cred & _check_credentials(aws_access_key_id, "aws_access_key_id")
    cred = cred & _check_credentials(aws_secret_access_key, "aws_secret_access_key")
    cred = cred & _check_credentials(endpoint_url, "endpoint_url")

    if not cred:
        return

    s3 = Session()
    s3_client = s3.client(
        service_name="s3",
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        endpoint_url=endpoint_url,
    )

    inpath = "/".join([params.path, params.defaultfile])
    oufile = "/".join([params.defaultout, params.defaultfile])

    s3_client.download_file(params.bucket, inpath, oufile)


def _check_credentials(cred: Optional[str], name: str) -> bool:
    if cred is None:
        logger.warning("No %s in .env founded.", name)
        logger.warning("Skip loading raw data.")
        return False
    return True

def read_data(path: Union[str, Path]) -> pd.DataFrame:
    return pd.read_csv(path)


def split_train_test_data(
    df: pd.DataFrame, params: SplittingParams,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    return train_test_split(
        df, test_size=params.val_size, random_state=params.random_state,
    )
