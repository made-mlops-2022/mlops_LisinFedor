import yaml
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional
from marshmallow_dataclass import class_schema
from mlflow.exceptions import MlflowException
from dotenv import load_dotenv

from ml_project.enities import (
    feature_params,
    model_params,
    s3_params,
    splitting_params,
)


load_dotenv()
MLFLOW_URI = os.environ.get("MLFLOW_URI")


logger = logging.getLogger(__name__)


@dataclass
class TrainigParams:
    expname: str
    features: feature_params.FeatureParams
    splitting: splitting_params.SplittingParams
    s3: s3_params.AwsS3Params
    model: model_params.ModelParams
    data_path: Optional[str]
    use_mlflow: bool = field(default=True)
    mlflow_url: Optional[str] = None
    mlflow_exp: str = "hw1"


TrainingPipelineSchema = class_schema(TrainigParams)


def read_train_params(path: Union[str, Path]) -> TrainigParams:
    with open(path, "r") as param_file:
        schema = TrainingPipelineSchema()
        tparams = schema.load(yaml.safe_load(param_file))

        if tparams.mlflow_url is not None:
            logger.warning("Remove mlflow_url parameter from config before push!")

        if MLFLOW_URI is None:
            raise MlflowException("Cant find MLFLOW_URI in environments.")

        tparams.mlflow_url = MLFLOW_URI
        return tparams


def dump_train_params(new_config: TrainigParams, path: Union[str, Path]) -> None:
    with open(path, "w") as param_file:
        logger.info("Removing mlflow_url parameter from config.")
        new_config.mlflow_url = None
        schema = TrainingPipelineSchema()
        yaml.safe_dump(schema.dump(new_config), param_file)
