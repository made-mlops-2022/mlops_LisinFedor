import click
import csv
import os
import logging
import sys
import importlib
import mlflow
import pandas as pd
import yaml
import inspect
from pathlib import Path
from typing import Any, Dict
from sklearn.metrics import classification_report


logger = logging.getLogger("model_eval")
str_handler = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter(
    "%(asctime)s\t%(levelname)s\t[%(name)s]: %(message)s",
)
logger.setLevel(logging.INFO)
str_handler.setFormatter(fmt)
logger.addHandler(str_handler)


DS = os.environ.get("DS")
DS_NODASH = os.environ.get("DS_NODASH")
pred_path = Path("/data") / "predictions" / f"{DS}"
model_path = Path("/data") / "models" / f"{DS}"
configs = Path("/data") / "configs"
data_path = Path("/data") / "processed" / f"{DS}"


def get_model_class(model_path: str) -> Any:
    """Create model from pretrained import from lib.

    Args:
        model_path (str): model path in format 'sklearn.submodule.modelname'.

    Returns:
        Any: sklearn class of model with `fit` and `predict` methods.
    """
    if "sklearn" not in model_path:
        logger.error("Only sklearn models supported.")
        raise ValueError

    parts = model_path.split(".")
    model_name = parts[-1]
    path = ".".join(parts[:-1])
    module = importlib.import_module(path)
    return getattr(module, model_name)


def init_new_model(model_class, model_path: str) -> Any:
    """Fit new model from provided class.

    Args:
        model_class (_type_): sklearn model class.
        model_path (str): model path in format 'sklearn.submodule.modelname'.

    Returns:
        Any: fitted model.
    """
    configs.mkdir(parents=True, exist_ok=True)
    config_name = model_path.replace(".", "_")
    config_path = configs / f"{config_name}.yml"

    if config_path.exists():
        logger.info(f"Use provided config '{config_name}.yml'")
        with open(config_path, "r") as model_conf:
            kwargs = yaml.safe_load(model_conf)
        return model_class(**kwargs)

    logger.warning(f"Configuration file for {model_path} not found.")
    logger.warning("Use default. See in 'data/configs'.")

    signature = inspect.signature(model_class)
    def_kwargs = {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

    save_def_kwargs(def_kwargs, config_path)
    return model_class(**def_kwargs)


def save_def_kwargs(kwargs: Dict[str, Any], file_path: Path):
    logger.warning("Create file with default kwars.")
    logger.warning(f"File created: {file_path}")
    with open(file_path, "w") as kw_file:
        yaml.safe_dump(kwargs, kw_file)


def load_mlflow_model(model_name: str):
    """Load model from mlflow registry.

    Args:
        model_name (str): name of model in registry.
    """

    url = f"models:/{model_name}/production"
    mlflow_uri = os.environ.get("MLFLOW_URI")

    mlflow.set_tracking_uri(mlflow_uri)
    return mlflow.sklearn.load_model(url)


def predict_on_model(model, df: pd.DataFrame, save: bool = False):
    pred = model.predict(df)
    pred = pred.reshape(-1, 1)
    pred_path.mkdir(parents=True, exist_ok=True)
    if save:
        with open(pred_path / f"{DS_NODASH}_preds.csv", "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerows(pred)

    return pred


def get_score(y_true, y_pred) -> Dict[str, float]:
    full_metrics = classification_report(y_true, y_pred, output_dict=True)
    metrics = full_metrics.get("macro avg")
    metrics["accuracy"] = full_metrics.get("accuracy")
    return metrics


def log_to_mlflow(metrics: Dict[str, float], model: Any):
    mlflow_uri = os.environ.get("MLFLOW_URI")
    if mlflow_uri is None:
        logger.error("No 'mlflow_uri' founded in environs.")
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name="mlops")

    with mlflow.start_run():
        mlflow.set_tag("mlflow.runName", f"day_{DS_NODASH}")
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, f"{DS_NODASH}_model")


def eval_new_model():
    model_path = os.environ.get("model_path")
    if model_path is None:
        logger.error("No model_path founded in environs.")
        raise ValueError

    model_class = get_model_class(model_path)
    model = init_new_model(model_class, model_path)

    train_path = data_path / f"{DS_NODASH}_train.csv"
    val_path = data_path / f"{DS_NODASH}_val.csv"

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)

    X, y = train.drop("condition", axis=1), train["condition"]
    X_val, y_val = val.drop("condition", axis=1), val["condition"]

    model.fit(X, y)
    preds = predict_on_model(model, X_val, save=False)
    metrics = get_score(y_val, preds)
    log_to_mlflow(metrics, model)


def eval_mlflow_model():
    model_name = os.environ.get("REGISTRY_MODEL_NAME")
    if model_name is None:
        logger.error("No registry model name fonded in environs.")
        raise ValueError
    model = load_mlflow_model(model_name)
    df = pd.read_csv(data_path / f"{DS_NODASH}_notarget.csv")
    predict_on_model(model, df, save=True)


@click.command()
@click.option(
    "--new",
    is_flag=True,
    show_default=True,
    default=False,
    help="Use new model or registry.",
)
def main(new):
    if new:
        eval_new_model()
    else:
        eval_mlflow_model()


if __name__ == "__main__":
    main()
