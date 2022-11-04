import logging
import os
import mlflow
from typing import Any, Tuple, Union
from pathlib import Path
from sklearn.pipeline import Pipeline
from dotenv import load_dotenv

from ml_project.enities.train_params import (
    read_train_params,
    TrainigParams,
    dump_train_params,
)
from ml_project.data import make_dataset
from ml_project.features import build_features
from ml_project.models import model_fit_predict
from ml_project.project_paths import CONFIG_PATH, TRAINED_PATH

load_dotenv()

logger = logging.getLogger(__name__)


def train_pipeline(config_path: Union[str, Path]):
    config_params = read_train_params(config_path)

    if config_params.use_mlflow:

        mlflow.set_tracking_uri(config_params.mlflow_url)
        mlflow.set_experiment(config_params.mlflow_exp)

        with mlflow.start_run(run_name=config_params.expname) as run:
            mlflow.log_artifact(local_path=config_path)
            model_name, metrics, pipe = run_train_pipeline(config_params)
            mlflow.log_metrics(metrics.get("macro avg"))
            mlflow.log_metric("accuracy", metrics.get("accuracy"))

            if config_params.model.module == "sklearn":
                model_info = mlflow.sklearn.log_model(pipe, model_name)
                logger.info(f"Model info: {model_info.model_uri}")
                config_params.model.last_model = model_info.model_uri
            else:
                mlflow.log_artifact(TRAINED_PATH / model_name)
                model_path = model_fit_predict.get_mlflow_model_artifact_path(
                    model_name, run.info.run_id,
                )
                config_params.model.last_model = model_path

        dump_train_params(config_params, config_path)

    else:
        return run_train_pipeline(config_params)


def run_train_pipeline(train_params: TrainigParams) -> Tuple[str, Any, Pipeline]:
    raw_data = train_params.data_path

    if not raw_data:
        logger.info("Using default raw data.")
        raw_data = "/".join([train_params.s3.defaultout, train_params.s3.defaultfile])

        if not os.path.exists(raw_data):
            logger.info("Raw data not founded.")
            logger.info("Trying to load data from s3 using credetials from `.env`.")

            make_dataset.download_data_from_s3(train_params.s3)
            logger.info("Data loaded.")

    df = make_dataset.read_data(raw_data)
    train, test = make_dataset.split_train_test_data(df, train_params.splitting)

    train_target = build_features.extract_target(train, train_params.features)
    test_target = build_features.extract_target(test, train_params.features)

    train = train.drop(train_params.features.target_col, axis=1)
    test = test.drop(train_params.features.target_col, axis=1)

    pipe = build_features.build_data_preproc_pipline(train_params.features)

    if pipe is not None:
        train_features = pipe.fit_transform(train)
    else:
        train_features = train
        pipe = Pipeline(steps=[])

    model = model_fit_predict.train_model(
        feats=train_features, target=train_target, train_params=train_params.model,
    )

    pipe.steps.append(
        (
            "Prediction",
            model,
        ),
    )

    preds = model_fit_predict.predict_model(pipe, test)
    metrics = model_fit_predict.evaluate_model(
        preds,
        test_target,
        odct=train_params.use_mlflow,
    )

    model_name = f"{train_params.model.model_name}_{train_params.expname}"
    model_fit_predict.serialize_pipline(pipe, model_name)

    return model_name, metrics, pipe


if __name__ == "__main__":
    config_path = CONFIG_PATH / "config.yml"
    train_pipeline(config_path)
