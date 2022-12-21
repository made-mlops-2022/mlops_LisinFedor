import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from ml_project.features.transformers import ScalerTransformer


def test_simple_pd():
    train = pd.DataFrame({"A": [0, 2, 4, 8]})
    test = pd.DataFrame({"A": [0.0, 0.25, 0.5, 1.0]})
    test_scaler = MinMaxScaler()
    scale_trans = ScalerTransformer(test_scaler, 1)
    trans_train = scale_trans.fit_transform(train)
    assert np.isclose(trans_train["A"].to_numpy(), test["A"].to_numpy()).all()


def test_pd_after_numpy():
    train_pd = pd.DataFrame({"A": [0, 2, 4, 8]})
    train_np = np.array([[0], [2], [4], [8]])
    test = np.array([0.0, 0.25, 0.5, 1.0])
    test_scaler = MinMaxScaler()
    scale_trans = ScalerTransformer(test_scaler, 1)
    trans_train = scale_trans.fit(train_np).transform(train_pd)
    assert np.isclose(trans_train["A"].to_numpy(), test).all()


def test_numpy_after_pd():
    train_pd = pd.DataFrame({"A": [0, 2, 4, 8]})
    train_np = np.array([[0], [2], [4], [8]])
    test = np.array([[0.0], [0.25], [0.5], [1.0]])
    test_scaler = MinMaxScaler()
    scale_trans = ScalerTransformer(test_scaler, 1)
    trans_train = scale_trans.fit(train_pd).transform(train_np)
    assert np.isclose(trans_train, test).all()


def test_np_skip_str():
    train = np.array([[0, 0], [2, 1], [4, 0], [8, 1]])
    test = np.array([[0, 0], [0.25, 1], [0.5, 0], [1.0, 1]])
    test_scaler = MinMaxScaler()
    scale_trans = ScalerTransformer(test_scaler, 3)
    trans_train = scale_trans.fit_transform(train)
    assert np.isclose(trans_train, test).all()
