import os
import pandas as pd
from pathlib import Path
from ml_project.train import run_train_pipeline
from ml_project.enities.train_params import read_train_params
from ml_project.data import make_dataset
from ml_project.models import model_fit_predict
from ml_project.enities import model_params


def test_train_s3_data(monkeypatch, synt_full_data: pd.DataFrame, assets: Path):
    # This test will fail if no s3 parameters founded in .env file
    monkeypatch.setattr(make_dataset, "read_data", lambda _: synt_full_data)
    monkeypatch.setattr(model_fit_predict, "TRAINED_PATH", assets)
    monkeypatch.setattr(model_params, "MODEL_CONFIGS_PATH", assets)

    train_params = read_train_params(assets / "config.yml")
    expected_name = "{mn}_{en}".format(mn=train_params.model.model_name, en=train_params.expname)
    exp_kwargs = "{mn}_kwargs.yml".format(mn=train_params.model.model_name)
    name, metrics, pipe = run_train_pipeline(train_params)

    assert name == expected_name
    assert name in os.listdir(assets)
    assert exp_kwargs in os.listdir(assets)
    assert train_params.s3.defaultfile in os.listdir(assets)
