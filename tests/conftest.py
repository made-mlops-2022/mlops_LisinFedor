import pytest
import os
import pandas as pd
from dotenv import load_dotenv
from typing import Tuple, List
from pathlib import Path
from sdv.tabular import GaussianCopula
from mlflow.exceptions import MlflowException


load_dotenv()

ASSETS = Path(__file__).parent / "assets"
DATA_PATH = ASSETS / "data.csv"
MODEL_ID = "5bc0d340c40444fa95ed74ec2a9d82ac"
ARTIFACT_ID = "1edb406391354719b7fbfbdc1fb08f1b"
MLFLOW_URI = os.environ.get("MLFLOW_URI")
NUM_OF_SAMPLES = 300


@pytest.fixture(scope="session")
def synt_full_data() -> pd.DataFrame:

    base_artifacts = os.listdir(ASSETS)

    df = pd.read_csv(DATA_PATH)
    model = GaussianCopula()
    model.fit(df)

    yield model.sample(NUM_OF_SAMPLES)

    delete_artifacts(base_artifacts)


@pytest.fixture(scope="session")
def synt_data_target(synt_full_data) -> Tuple[pd.DataFrame, pd.Series]:
    synt_data_with_target = synt_full_data
    return (
        synt_data_with_target.drop("condition", axis=1),
        synt_data_with_target["condition"],
    )


@pytest.fixture(scope="session")
def assets() -> Path:
    return ASSETS


@pytest.fixture(scope="session")
def model_id() -> str:
    return MODEL_ID


@pytest.fixture(scope="session")
def artifact_id() -> str:
    return ARTIFACT_ID


@pytest.fixture(scope="session")
def mlflow_uri() -> str:
    if MLFLOW_URI is None:
        raise MlflowException("Cant find MLFLOW_URI in environments.")
    return MLFLOW_URI


def delete_artifacts(base_artifacts: List[str]):
    assets_path = ASSETS
    for filename in os.listdir(assets_path):
        if filename not in base_artifacts:
            os.remove(assets_path / filename)
