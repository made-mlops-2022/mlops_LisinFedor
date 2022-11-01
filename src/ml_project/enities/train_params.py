import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Optional
from marshmallow_dataclass import class_schema

from ml_project.enities import (
    feature_params,
    model_params,
    s3_params,
    splitting_params,
)


@dataclass
class TrainigParams:
    expname: str
    features: feature_params.FeatureParams
    splitting: splitting_params.SplittingParams
    s3: s3_params.AwsS3Params
    model: model_params.ModelParams
    data_path: Optional[str]
    use_mlflow: bool = field(default=True)
    mlflow_url: str = "http://185.208.207.159:5000"
    mlflow_exp: str = "hw1"


TrainingPipelineSchema = class_schema(TrainigParams)


def read_train_params(path: Union[str, Path]) -> TrainigParams:
    with open(path, "r") as param_file:
        schema = TrainingPipelineSchema()
        return schema.load(yaml.safe_load(param_file))
