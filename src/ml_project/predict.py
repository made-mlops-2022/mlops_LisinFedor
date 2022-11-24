import csv
import logging
import mlflow
from pathlib import Path
from typing import Any, Optional, Union

from ml_project.enities import train_params
from ml_project.data.make_dataset import read_data
from ml_project.models import model_fit_predict
from ml_project.project_paths import CONFIG_PATH, RAW_DATA_PATH


logger = logging.getLogger(__name__)


def load_and_predict(
    data_path: Union[Path, str],
    out_csv_file_path: Union[Path, str],
    model_id: str = "last",
    model_path: Optional[Union[Path, str]] = None,
) -> None:

    config_path = CONFIG_PATH / "config.yml"
    config_params = train_params.read_train_params(config_path)

    if model_path is not None:
        try:
            model = model_fit_predict.load_local_model(model_path, is_path=True)
        except FileNotFoundError:
            logger.warning("File not found. Load model from mlflow.")

    if model_id == "last":
        model = load_last_model(config_params)
    elif config_params.mlflow_url is not None:
        model_path, is_dir = model_fit_predict.get_model_mlflow_path_by_id(
            config_params.mlflow_url,
            model_id,
        )
        model = model_fit_predict.load_nonlocal_model(model_path, is_dir)
    else:
        raise mlflow.exceptions.MlflowException("Cant find MLFLOW_URI in environments.")

    return predict(model, data_path, out_csv_file_path)


def predict(
    model: Any, data_path: Union[Path, str], out_csv_file_path: Union[Path, str]
) -> None:
    df = read_data(data_path)
    prediction = model_fit_predict.predict_model(model, df)
    with open(out_csv_file_path, "w") as ofile:
        csv_write = csv.writer(ofile)
        prediction = prediction.reshape([len(prediction), 1])
        csv_write.writerows(prediction)


def try_load_local_model(
    model_path: Union[Path, str], config_params: train_params.TrainigParams
) -> Any:

    try:
        return model_fit_predict.load_local_model(model_path, is_path=True)
    except FileNotFoundError:
        logger.warning("File not found. Load last model from mlflow.")

    return load_last_model(config_params)


def load_last_model(config_params: train_params.TrainigParams):
    mlflow.set_tracking_uri(config_params.mlflow_url)
    mlflow.set_experiment(config_params.mlflow_exp)
    last_model_path = config_params.model.last_model

    if last_model_path is not None:

        is_dir = False
        if "runs" in last_model_path:
            is_dir = True

        try:
            model = model_fit_predict.load_nonlocal_model(last_model_path, is_dir)
        except mlflow.exceptions.MlflowException as ex:
            logger.error("No last model founded. Fit any model first.")
            raise ex
    else:
        logger.error("No last model founded. Fit any model first.")
        raise ValueError("No last model founded.")

    return model


if __name__ == "__main__":
    path = RAW_DATA_PATH / "heart_cleveland_upload.csv"
    load_and_predict(
        data_path=path,
        out_csv_file_path="preds.csv",
        model_id="5bc0d340c40444fa95ed74ec2a9d82ac",
    )
