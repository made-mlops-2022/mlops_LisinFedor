import logging
import pickle
import numpy as np
import pandas as pd
import mlflow

from pathlib import Path
from typing import Any, Dict, Tuple, Union
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

from ml_project.enities import model_params
from ml_project.project_paths import TRAINED_PATH

logger = logging.getLogger(__name__)


def train_model(
    feats: pd.DataFrame, target: pd.Series, train_params: model_params.ModelParams
) -> Any:
    """Import model class, init model and train.

    Args:
        feats (pd.DataFrame): X data.
        target (pd.Series): y data.
        train_params (model_params.ModelParams): model params.

    Returns:
        Any: fitted model.
    """
    model_class = model_params.create_clear_model(train_params)
    model = model_params.init_model(model_class, train_params)
    model.fit(feats, target)
    return model


def predict_model(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    return model.predict(features)


def evaluate_model(
    predicts: np.ndarray,
    target: pd.Series,
) -> Union[Dict[str, Any], str]:
    return classification_report(target, predicts, output_dict=True)


def serialize_pipline(model: object, file_name: str):
    path = TRAINED_PATH / file_name

    with open(path, "wb") as model_file:
        pickle.dump(model, model_file)

    logger.info("Save model: %s", str(path))


def load_local_model(file_name_or_path: Union[Path, str], is_path: bool = False):
    path: Union[Path, str] = ""

    if is_path:
        path = file_name_or_path
    else:
        path = Path(__file__).parent / "trained" / file_name_or_path

    logger.info("Load model from %s", path)
    with open(path, "rb") as model_file:
        return pickle.load(model_file)


def set_mlflow_params(uri: str, exp_name: str):
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp_name)


def _load_mlflow_sklearn_model(path: str):
    return mlflow.pyfunc.load_model(path)


def _load_mlflow_model_from_artifacts(path: str):
    tmp_model_path = mlflow.artifacts.download_artifacts(path)
    return load_local_model(tmp_model_path, is_path=True)


def load_nonlocal_model(path: str, is_dir: bool):
    if is_dir:
        model = _load_mlflow_sklearn_model(path)
    else:
        model = _load_mlflow_model_from_artifacts(path)
    return model


def get_mlflow_model_artifact_path(model_name: str, run_id: str):
    return "mlflow-artifacts:/{id}/artifacts/{name}".format(id=run_id, name=model_name)


def get_mlflow_model_runs_path(model_name: str, run_id: str):
    return "runs:/{id}/{name}".format(id=run_id, name=model_name)


def get_model_mlflow_path_by_id(uri: str, model_id: str) -> Tuple[str, bool]:
    mlflow.tracking.set_tracking_uri(uri)
    client = mlflow.tracking.MlflowClient(uri)
    client_info = client.list_artifacts(model_id)

    for file_info in client_info:
        if ".yml" in file_info.path:
            continue
        elif file_info.is_dir:
            return (
                get_mlflow_model_runs_path(
                    file_info.path,
                    model_id,
                ),
                True,
            )
        return (
            get_mlflow_model_artifact_path(
                file_info.path,
                model_id,
            ),
            False,
        )
