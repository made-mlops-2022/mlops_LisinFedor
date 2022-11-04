import sys
import os
import pandas as pd
from typing import Tuple
from pathlib import Path
from ml_project.models import model_fit_predict as mfp
from ml_project import predict


def test_predict_with_model_id(model_id, synt_data_target, mlflow_uri):
    """If the model was removed from storage, this test might not pass."""
    sdata, starget = synt_data_target
    path, is_dir = mfp.get_model_mlflow_path_by_id(mlflow_uri, model_id)

    assert "runs" in path
    assert is_dir is True

    model = mfp.load_nonlocal_model(path, is_dir=is_dir)
    prediction = mfp.predict_model(model, sdata)

    cls_rep = mfp.evaluate_model(prediction, starget, odct=False)

    sys.stdout.write("\n")
    sys.stdout.write(cls_rep)


def test_predict_with_artifact_id(artifact_id, synt_data_target, mlflow_uri):
    """If the model was removed from storage, this test might not pass."""
    sdata, starget = synt_data_target
    path, is_dir = mfp.get_model_mlflow_path_by_id(mlflow_uri, artifact_id)

    assert "artifacts" in path
    assert is_dir is False

    model = mfp.load_nonlocal_model(path, is_dir=is_dir)
    prediction = mfp.predict_model(model, sdata)

    cls_rep = mfp.evaluate_model(prediction, starget, odct=False)

    sys.stdout.write("\n")
    sys.stdout.write(cls_rep)


def test_load_predict_last(
    monkeypatch, assets: Path, synt_data_target: Tuple[pd.DataFrame, pd.Series],
):
    sdata, _ = synt_data_target
    monkeypatch.setattr(predict, "CONFIG_PATH", assets)
    monkeypatch.setattr(predict, "read_data", lambda _: sdata)
    out_path = assets / "pred.csv"
    num_samples = sdata.shape[0]

    predict.load_and_predict("", out_path)

    assert "pred.csv" in os.listdir(assets)
    assert pd.read_csv(out_path, header=None).shape[0] == num_samples


def test_load_predict_artifact(
    monkeypatch, assets: Path, synt_data_target: Tuple[pd.DataFrame, pd.Series], artifact_id: str,
):
    sdata, _ = synt_data_target
    monkeypatch.setattr(predict, "CONFIG_PATH", assets)
    monkeypatch.setattr(predict, "read_data", lambda _: sdata)
    out_path = assets / "pred.csv"
    num_samples = sdata.shape[0]

    predict.load_and_predict("", out_path, model_id=artifact_id)

    assert "pred.csv" in os.listdir(assets)
    assert pd.read_csv(out_path, header=None).shape[0] == num_samples


def test_load_predict_path(
    monkeypatch, assets: Path, synt_data_target: Tuple[pd.DataFrame, pd.Series], artifact_id: str,
):
    sdata, _ = synt_data_target
    model_path = assets / "RandomForestClassifier_experiment1"
    monkeypatch.setattr(predict, "CONFIG_PATH", assets)
    monkeypatch.setattr(predict, "read_data", lambda _: sdata)
    out_path = assets / "pred.csv"
    num_samples = sdata.shape[0]

    predict.load_and_predict("", out_path, model_path=model_path)

    assert "pred.csv" in os.listdir(assets)
    assert pd.read_csv(out_path, header=None).shape[0] == num_samples
